#!/usr/bin/env bash
# Prepare both Sparks to serve Qwen3.8-Flash-Next with TensorFold's Zig engine (tensorfold-native, two ranks), from the
# checkpoint QUANT names (scripts/config.sh: nvfp4, NVIDIA's NVFP4, by default; int4ar, the INT4-AutoRound one):
#   1. preflight checks: docker and the GPU on both nodes, key-based ssh to the worker, the RoCE link, disk space
#   2. the image on the head: tensorfold-native built from TensorFold (TF_REF, branch zig-flashnext) plus
#      patches/*.patch with Zig, on NVIDIA's PyTorch container, with the kernel set; pulled prebuilt from $GHCR_IMAGE
#      when a matching tag is reachable (PULL=0 skips that), else built locally
#   3. the same image on the worker: pulled there, else streamed from the head (docker save | ssh docker load)
#   4. download that checkpoint (only it) on the head into the Hugging Face cache (~124 GiB NVFP4, ~122 GiB INT4-AR,
#      resumable), at its pinned revision (MODEL_REVISION)
#   5. verify the checkpoint: every file its index names is there, and it is the model this engine serves
#   6. the same files on the worker, copied from the head over the Sparks' link (rsync), checked file by file; or, with
#      WORKER_WEIGHTS=nfs, a read-only NFS volume of the head's cache on the worker, checked the same way
# ./start.sh runs this by itself when needed. Safe to re-run: every step skips work that is already done.
# Pass --rebuild to rebuild the image from scratch, --quant nvfp4|int4ar to prepare that checkpoint (QUANT).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."     # the repository root
_args=()
while (( $# )); do
  case "$1" in
    --quant) [[ $# -ge 2 ]] || { echo "[prepare.sh] ERROR: --quant needs nvfp4 or int4ar" >&2; exit 1; }; export QUANT=$2; shift 2 ;;
    --quant=*) export QUANT=${1#*=}; shift ;;
    *) _args+=("$1"); shift ;;
  esac
done
set -- "${_args[@]}"; unset _args
source ./scripts/config.sh
source ./scripts/nodes.sh

REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------- 1. preflight
mkdir -p "$KERNEL_CACHE" "$STATE_DIR" "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running (it holds the download locks); wait for it or stop it: pgrep -af prepare.sh"
log "Preflight checks on both Sparks"
command -v docker >/dev/null || die "docker is not installed"
command -v rsync >/dev/null || die "rsync is not installed on this node (sudo apt install rsync)"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is your user in the docker group?)"
if ! command -v nvidia-smi >/dev/null; then warn "nvidia-smi not found on this node"
elif ! nvidia-smi -L >/dev/null 2>&1; then warn "nvidia-smi failed on this node: is the NVIDIA driver working?"; fi
docker info 2>/dev/null | grep -qi nvidia || warn "docker does not list an nvidia runtime on this node; --gpus all may fail"
check_workers
need_worker 1
worker 1 'docker info >/dev/null 2>&1' || die "the worker ($WORKER) cannot talk to its docker daemon (docker group?)"
worker 1 'nvidia-smi -L >/dev/null 2>&1' || warn "nvidia-smi failed on the worker: is the NVIDIA driver working?"
worker 1 'docker info 2>/dev/null | grep -qi nvidia' || warn "docker does not list an nvidia runtime on the worker; --gpus all may fail"
[[ "$WORKER_WEIGHTS" == nfs ]] || worker 1 'command -v rsync >/dev/null' ||
  die "rsync is not installed on the worker (sudo apt install rsync)"
detect_link
log "Link: head $HEAD_ADDR ($HEAD_DEV, $HEAD_HCAS, GID $HEAD_GID) <-> worker $WORKER_ADDR ($WORKER_DEV, $WORKER_HCAS, GID $WORKER_GID)"
WORKER_HF=$(worker_hf_cache)
[[ "$WORKER_WEIGHTS" == nfs ]] || worker 1 "mkdir -p '$WORKER_HF/hub' && test -w '$WORKER_HF/hub'" ||
  die "the worker's $WORKER_HF/hub is not writable (left root-owned by a container? fix its ownership there)"

PATCHES_HASH=$(image_hash)
built_hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'; }
worker_free_gb() { worker 1 "df -BG --output=avail '$2' | tail -1 | tr -dc '0-9'"; }
models=("$MODEL_ID")
# hub_missing_gb: GB (GiB, as df counts) the download still needs on the head: the files of the revision to serve
# whose blob is not in the cache yet (by name, the LFS sha256 or else the git blob id, and by size), from the Hub's
# file list. A new pin that shares its blobs with a cached revision needs next to nothing. Asked with huggingface_hub
# from this host's python3 or the hf CLI's own; fails without either, or without the Hub.
HUB_MISSING_PY='
import math, os, sys
from huggingface_hub import HfApi
hub, args, missing = sys.argv[1], sys.argv[2:], 0
for repo, rev in zip(args[::2], args[1::2]):
    blobs = os.path.join(hub, "models--" + repo.replace("/", "--"), "blobs")
    for f in HfApi().model_info(repo, revision=rev or None, files_metadata=True).siblings:
        lfs = f.lfs
        name = (getattr(lfs, "sha256", None) or lfs["sha256"]) if lfs else f.blob_id
        size = (getattr(lfs, "size", None) or lfs["size"]) if lfs else (f.size or 0)
        p = os.path.join(blobs, name)
        missing += 0 if os.path.isfile(p) and os.path.getsize(p) == size else size
print(math.ceil(missing / 2**30))'
hub_missing_gb() {
  local py id args=() out tried=""
  for id in "${models[@]}"; do args+=("$id" "$(model_revision "$id")"); done
  for py in "$(command -v python3)" "$(sed -n '1s/^#! *\(\/[^ ]*python[0-9.]*\)$/\1/p' "$(command -v hf || echo /dev/null)" 2>/dev/null)"; do
    [[ -n "$py" && -x "$py" && "$py" != "${tried:-}" ]] || continue
    tried=$py
    out=$(timeout 30 "$py" -c "$HUB_MISSING_PY" "$HF_CACHE/hub" "${args[@]}" 2>/dev/null) && [[ "$out" =~ ^[0-9]+$ ]] &&
      { echo "$out"; return 0; }
  done
  return 1
}
# Disk on the head: what the download still needs (plus 5 GB), and the image (unless built from these patches), both
# on one filesystem when Docker's root shares it with HF_CACHE. Without the Hub's file list: MIN_FREE_GB, unless the
# snapshot to serve is already here.
DOCKER_ROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
if missing=$(hub_missing_gb); then
  need_ckpt=0; (( missing == 0 )) || need_ckpt=$((missing + 5))
  ckpt_what="the download needs ~${missing} GB that the cache does not hold yet"
else
  need_ckpt=0
  for id in "${models[@]}"; do [[ -d "$(model_cache_dir "$id")/snapshots/$(snapshot_rev "$id")" ]] || need_ckpt=$MIN_FREE_GB; done
  ckpt_what="the checkpoint needs MIN_FREE_GB (the Hub's file list could not be read to count what is missing)"
fi
(( need_ckpt )) || ckpt_what="nothing to download"
need_img=0; [[ $REBUILD -eq 0 && "$built_hash" == "$PATCHES_HASH" ]] || need_img=$IMAGE_FREE_GB
if [[ "$(stat -c %d "$HF_CACHE")" == "$(stat -c %d "$DOCKER_ROOT" 2>/dev/null)" ]]; then
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt + need_img )) ||
    die "only ${have} GB free under $HF_CACHE (also Docker's root), ~$((need_ckpt + need_img)) GB needed: $ckpt_what; the image needs ${need_img} GB (IMAGE_FREE_GB)"
else
  have=$(free_gb "$HF_CACHE"); (( have >= need_ckpt )) ||
    die "only ${have} GB free under $HF_CACHE, ~${need_ckpt} GB needed: $ckpt_what"
  have=$(free_gb "$DOCKER_ROOT"); (( have >= need_img )) ||
    die "only ${have} GB free under Docker's root ($DOCKER_ROOT); the image needs ~${need_img} GB (IMAGE_FREE_GB)"
fi
WORKER_DOCKER_ROOT=$(worker 1 "docker info -f '{{.DockerRootDir}}'" 2>/dev/null || echo /var/lib/docker)
if [[ "$WORKER_WEIGHTS" == nfs ]]; then d=$WORKER_DOCKER_ROOT; else d=$WORKER_HF; fi
log "Disk: $(free_gb "$HF_CACHE") GB free under $HF_CACHE here, $(worker_free_gb 1 "$d") GB under $d on the worker"
# The memory floor: MemAvailable of at least MEM_FLOOR_GIB (TENSORFOLD_MEMORY_RESERVE_GIB, 10) on each Spark; the
# image build and the checkpoint copy run beside whatever else the Spark does. MEM_CHECK=0 warns instead.
_mem_here=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
_mem_there=$(worker 1 "awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo" 2>/dev/null || echo 0)
for _m in "here:$_mem_here" "the worker:$_mem_there"; do
  if (( ${_m#*:} < MEM_FLOOR_GIB )); then
    _msg="only ${_m#*:} GiB MemAvailable on ${_m%%:*} (the floor is MEM_FLOOR_GIB=$MEM_FLOOR_GIB): stop other work first"
    if [[ "${MEM_CHECK:-1}" == 0 ]]; then warn "$_msg"; else die "$_msg (MEM_CHECK=0 continues anyway)"; fi
  fi
done
log "Memory: ${_mem_here} GiB MemAvailable here, ${_mem_there} GiB on the worker (floor $MEM_FLOOR_GIB)"

# ---------------------------------------------------------------- 2. image (head)
# The Zig engine, built from TensorFold's checkout at TF_REF with ./patches applied (git apply in the checkout's root,
# in filename order), and the kernel set. Only /opt/tensorfold (the Zig build and the kernel set), TensorFold's
# license files and its Python tree (src/tensorfold, for the vision helper, with transformers and PyAV) reach the image. The image is rebuilt when the patches,
# the kernel set or the Dockerfile change (image_hash); docker's layer cache keeps the Zig download and the checkout.
#
# The kernel set. The engine replays Triton cubins listed in aot.json from TENSORFOLD_CUDA_KERNELS
# (/opt/tensorfold/share/tensorfold/cuda/sm121). The build stage compiles every specialization the engine launches, one
# GPU and two ranks, with triton.compile for sm_121: no GPU and no checkpoint (TensorFold's
# tools/zig/flashnext_aot.py over the spec zig/tests/cuda/flashnext/kernels.json). Triton's line info puts the
# Python source path into each cubin, so the checkout sits at /tensorfold, the path the spec was captured with, and
# its DWARF line table records each source file's modification time and size, so the spec's Python files get the
# mtime they had when the reference set was compiled (config.sh KERNEL_SOURCE_MTIME): the cubins are then byte for
# byte the ones the oracle run compiled (the tool checks the captured ones and stops the build on a difference).
prebuilt=$(prebuilt_image)            # the pinned digest (config.sh's IMAGE_TAG / IMAGE_DIGEST), else the hash's tag
if [[ $REBUILD -eq 0 && "${PULL:-1}" == 1 && "$built_hash" != "$PATCHES_HASH" ]]; then
  log "Pulling the prebuilt image $prebuilt (PULL=0 builds instead)"
  if docker pull "$prebuilt" &&
     [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$prebuilt")" == "$PATCHES_HASH" ]]; then
    docker tag "$prebuilt" "$IMAGE"; built_hash=$PATCHES_HASH
    log "Using $prebuilt as $IMAGE"
  else
    warn "could not pull $prebuilt (no image for these patches, the package is not public, or no network): building it locally"
  fi
fi
if [[ $REBUILD -eq 1 || "$built_hash" != "$PATCHES_HASH" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 && [[ $REBUILD -eq 0 ]] || { log "Pulling base image $BASE_IMAGE"; docker pull "$BASE_IMAGE"; }
  kernels=present
  log "Building $IMAGE (TensorFold ${TF_REF:0:12} with Zig $ZIG_VERSION, patches $PATCHES_HASH: $(compgen -G 'patches/*.patch' | wc -l) patches; the engine, then its Triton kernel set)"
  # the build context: patches/ only (never the rest of the repository), in a fresh private directory
  ctx=$(mktemp -d "$STATE_DIR/build.XXXXXX")
  trap 'rm -rf -- "$ctx"' EXIT
  mkdir -p "$ctx/patches"
  compgen -G 'patches/*.patch' >/dev/null && cp -- patches/*.patch "$ctx/patches/"
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  docker build "${nocache[@]}" -t "$IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" \
    --build-arg ZIG_VERSION="$ZIG_VERSION" --build-arg ZIG_SHA256="$ZIG_SHA256" \
    --build-arg PATCHES_HASH="$PATCHES_HASH" --build-arg KERNELS="$kernels" \
    --build-arg KERNEL_SOURCE_MTIME="$KERNEL_SOURCE_MTIME" \
    -f - "$ctx" <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE} AS build
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG TF_REPO
ARG TF_REF
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
COPY patches /opt/tf-patches
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins install native && \
    test -x /opt/tensorfold/native/bin/tensorfold-native
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done
# every Triton specialization, compiled for sm_121 without a GPU (byte-equal to the oracle's: the tree is at
# /tensorfold, the kernel sources carry the reference build's mtime)
ARG KERNEL_SOURCE_MTIME
RUN cd /tensorfold && python -B -c 'import json, os, sys; t = int(sys.argv[1]); \
      [os.utime(os.path.join("src", f), (t, t)) for f in sorted({k["source"]["file"] for k in \
       json.load(open("zig/tests/cuda/flashnext/kernels.json"))["kernels"]})]' "${KERNEL_SOURCE_MTIME}" && \
    PYTHONPATH=/tensorfold/src python -B tools/zig/flashnext_aot.py build \
      --spec zig/tests/cuda/flashnext/kernels.json --jit zig/tests/cuda/flashnext/jit.json --tp 1,2 \
      --out /opt/tensorfold/share/tensorfold/cuda/sm121 && \
    test -f /opt/tensorfold/share/tensorfold/cuda/sm121/aot.json

FROM ${BASE_IMAGE}
COPY --from=build /opt/tensorfold /opt/tensorfold
# TensorFold's Python tree for the vision helper (python3 -m tensorfold.vision.native_helper, started by --vision)
COPY --from=build /tensorfold/src/tensorfold /opt/tensorfold/python/tensorfold
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native && \
    pip install --no-cache-dir "huggingface_hub>=1.0" && \
    pip install --no-cache-dir --no-deps "transformers==5.17.0" "av==19.0.1"
ARG PATCHES_HASH
ARG KERNELS
LABEL tf.patches=${PATCHES_HASH} tf.kernels=${KERNELS}
ENV HF_HOME=/root/.cache/huggingface TENSORFOLD_CUDA_KERNELS=/opt/tensorfold/share/tensorfold/cuda/sm121 \
    CUDA_CACHE_PATH=/cache/nv TENSORFOLD_VISION_PYTHONPATH=/opt/tensorfold/python
WORKDIR /workspace
DOCKERFILE
  rm -rf -- "$ctx"; trap - EXIT
else
  log "Image $IMAGE already built with patches $PATCHES_HASH (use --rebuild to force)"
fi
docker run --rm --network none --entrypoint test "$IMAGE" -x /opt/tensorfold/native/bin/tensorfold-native ||
  die "$IMAGE has no tensorfold-native"
log "Image $IMAGE: tensorfold-native, kernel set $(docker image inspect -f '{{index .Config.Labels "tf.kernels"}}' "$IMAGE")"

# ---------------------------------------------------------------- 3. image (worker)
image_id=$(image_ident "$IMAGE")                    # by content: .Id differs between image stores
if [[ "$(worker_image_ident 1 "$IMAGE")" != "$image_id" ]]; then
  wfree=$(worker_free_gb 1 "$WORKER_DOCKER_ROOT")
  (( wfree >= IMAGE_FREE_GB )) ||
    die "only ${wfree} GB free under the worker's Docker root ($WORKER_DOCKER_ROOT); the image needs ~${IMAGE_FREE_GB} GB (IMAGE_FREE_GB)"
  if [[ "${PULL:-1}" == 1 ]] && worker 1 docker pull "$prebuilt" >/dev/null 2>&1 &&
     [[ "$(worker_image_ident 1 "$prebuilt")" == "$image_id" ]]; then
    worker 1 docker tag "$prebuilt" "$IMAGE"
    log "Using $prebuilt as $IMAGE on the worker"
  else
    log "Copying $IMAGE to the worker (docker save | docker load; only missing layers are stored)"
    docker save "$IMAGE" | worker 1 docker load >/dev/null
  fi
  [[ "$(worker_image_ident 1 "$IMAGE")" == "$image_id" ]] || die "the worker's $IMAGE differs from the head's"
fi
log "Image $IMAGE identical on both Sparks"

# ---------------------------------------------------------------- 4. download (head)
# One revision on both ranks: MODEL_REVISION (config.sh; empty: what the Hub calls main when first downloaded, then
# kept). HF_TOKEN (optional: the checkpoint is public) reaches the download by name only, never on a command line.
command -v hf >/dev/null || warn "host 'hf' CLI not found, downloading from inside the container"
download() {  # <repo id> <revision or empty>
  if command -v hf >/dev/null; then
    # Host CLI: resumable, parallel, writes the standard HF cache layout.
    hf download "$1" ${2:+--revision "$2"} --cache-dir "$HF_CACHE/hub" >/dev/null
  else
    # Keep downloads owned by the host user, in the same HF_CACHE/hub layout as the host CLI.
    # (HF_HUB_OFFLINE by name too: offline, it only finds the cached snapshot and changes nothing)
    docker run --rm --user "$(id -u):$(id -g)" --network host --entrypoint python ${HF_TOKEN:+-e HF_TOKEN} \
      ${HF_HUB_OFFLINE:+-e HF_HUB_OFFLINE} \
      -v "$HF_CACHE":/hf -e HF_HOME=/hf -e HOME=/tmp "$IMAGE" -c \
      'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2] or None)' "$1" "$2"
  fi
}
for id in "${models[@]}"; do
  pin=$(model_revision "$id"); dir=$(model_cache_dir "$id")
  had=0; [[ -n "$pin" && -f "$dir/snapshots/$pin/config.json" ]] && had=1   # a snapshot already cached is left as it is
  log "Downloading $id${pin:+ @ ${pin:0:8}} into $HF_CACHE/hub (resumes if interrupted)"
  if ! download "$id" "$pin"; then
    # no network: a pinned snapshot already here is enough (serving reads the local cache only)
    [[ -n "$pin" && -f "$dir/snapshots/$pin/config.json" ]] || die "$id: the download failed"
    warn "$id: could not reach Hugging Face; using the snapshot already here (${pin:0:8})"
  fi
  # a download by commit sha writes no refs/main: name the pin there when nothing else is (tools that take repo ids),
  # for a snapshot this run brought (a cache that already held it is not changed)
  (( had )) || [[ -z "$pin" || -f "$dir/refs/main" ]] || { mkdir -p "$dir/refs"; printf %s "$pin" > "$dir/refs/main"; }
  rev=$(snapshot_rev "$id")
  [[ -n "$rev" && -d "$dir/snapshots/$rev" ]] || die "$id: no snapshot${rev:+ $rev} after the download"
  log "Checkpoint: $dir/snapshots/$rev ($(du -shL "$dir/snapshots/$rev" | cut -f1))"
done

# ---------------------------------------------------------------- 5. verify (head)
# before the copy: the worker gets only a complete checkpoint of the model this engine serves (model_type qwen4_exp),
# of the format QUANT names (nvfp4: ModelOpt; int4ar: GPTQ-format int4 from AutoRound, with its n-gram table in
# ple-table/), every shard its index names present and not empty
log "Verifying the checkpoint"
CHECK_PY='
import glob, json, os, sys
d, quant = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(d, "config.json")))
if cfg.get("model_type") != "qwen4_exp":
    sys.exit("model_type is %r, not qwen4_exp (Qwen3.8-Flash-Next)" % cfg.get("model_type"))
q = cfg.get("quantization_config") or {}
producer = str(q.get("producer", {}).get("name", "") if isinstance(q.get("producer"), dict) else q.get("quant_method", ""))
if os.path.isfile(os.path.join(d, "hf_quant_config.json")):
    h = json.load(open(os.path.join(d, "hf_quant_config.json")))
    producer = str(h.get("producer", {}).get("name", producer))
    q = h.get("quantization", q)
algo = str(q.get("quant_algo") or "")
if quant == "nvfp4" and "modelopt" not in producer.lower():
    sys.exit("not a ModelOpt checkpoint (producer %r), as QUANT=nvfp4 needs" % producer)
if quant == "int4ar":
    if str(q.get("quant_method", "")).lower() != "gptq" or q.get("bits") != 4:
        sys.exit("not a GPTQ-format int4 checkpoint (quant_method %r, bits %r), as QUANT=int4ar needs" % (q.get("quant_method"), q.get("bits")))
    table = [f for f in glob.glob(os.path.join(d, "ple-table", "*.safetensors")) if os.path.getsize(f) > 0]
    if not table:
        sys.exit("no n-gram table files in ple-table/ (INT4-AR keeps its FP8 n-gram table there)")
    producer, algo = "AutoRound", "int4 g%s + %d n-gram table files" % (q.get("group_size"), len(table))
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
shards = sorted(set(idx["weight_map"].values()))
bad = [s for s in shards if not os.path.isfile(os.path.join(d, s)) or os.path.getsize(os.path.join(d, s)) == 0]
if bad:
    sys.exit("missing or empty shards: " + ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else ""))
print("%s, %s %s, %d tensors in %d shards" % (cfg.get("model_type"), producer, algo, len(idx["weight_map"]), len(shards)))'
snap="$(model_cache_dir "$MODEL_ID")/snapshots/$(snapshot_rev "$MODEL_ID")"
info=$(python3 -I -c "$CHECK_PY" "$snap" "$QUANT" 2>&1) || die "the checkpoint at $snap is not ready: $info"
log "Checkpoint OK: $info"

# ---------------------------------------------------------------- 6. the same files on the worker
# The manifests below ("<file> <size>" a line) are sorted in byte order on both sides (LC_ALL=C, also inside the
# commands sent over ssh, which carries no locale of ours): each host's own collation would order the same files
# differently. On a mismatch, the first differing lines show what differs.
manifest_diff() {  # <head's manifest> <worker's manifest>
  warn "first differences (< head, > worker):"
  diff <(printf '%s\n' "$1") <(printf '%s\n' "$2") | head -20 >&2 || true
}
if [[ "$WORKER_WEIGHTS" == nfs ]]; then  # no copy: rank 1 reads the head's cache over NFS
  server=$(nfs_server)
  ensure_nfs_volume
  # the mount first: a refused export says so here, not as a missing file below
  if ! out=$(worker_nfs true 2>&1); then
    die "the worker cannot mount the head's :$NFS_PATH from $server over NFS ($NFS_VOLUME): $(tail -1 <<<"$out")
    The head must export $NFS_PATH to the worker's address on the link ($WORKER_ADDR, or its subnet), read-only;
    or set NFS_SERVER to a head address the export allows, or WORKER_WEIGHTS=copy (README: Worker weights over NFS)"
  fi
  for id in "${models[@]}"; do
    dir=$(model_cache_dir "$id"); rev=$(snapshot_rev "$id")
    manifest=$(cd "$dir/snapshots/$rev" && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort)
    have=$(worker_nfs find -L "/hf/hub/${dir##*/}/snapshots/$rev" -type f -printf '%P %s\n' 2>/dev/null | LC_ALL=C sort || true)
    [[ "$have" == "$manifest" ]] || { manifest_diff "$manifest" "$have"
      die "the worker does not see $id @ ${rev:0:8} over NFS ($NFS_VOLUME: :$NFS_PATH from $server); is HF_CACHE exported to it? (README: Worker weights over NFS)"; }
    log "Worker reads $id @ ${rev:0:8} from the head over NFS ($NFS_VOLUME)"
  done
else
  for id in "${models[@]}"; do
    dir=$(model_cache_dir "$id"); rev=$(snapshot_rev "$id")
    # every file of the snapshot, with its size (links followed), as the head has it
    manifest=$(cd "$dir/snapshots/$rev" && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort)
    wdir="$WORKER_HF/hub/${dir##*/}"
    have=$(worker 1 "cd '$wdir/snapshots/$rev' 2>/dev/null && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort" || true)
    if [[ "$have" == "$manifest" ]]; then log "Worker has $id @ ${rev:0:8}"; continue; fi
    # the blobs this revision uses (the cache layout huggingface_hub keeps), and how much rsync must send: the blobs (and
    # any plain file in the snapshot) that the worker lacks or holds at another size (an earlier revision may have
    # brought it the rest)
    lists=$(mktemp -d "$STATE_DIR/lists.XXXXXX")
    (cd "$dir" && find "snapshots/$rev" -type l -printf '%l\n' | sed 's#^\(\.\./\)*##' | LC_ALL=C sort -u) > "$lists/blobs"
    (cat "$lists/blobs"; cd "$dir" && find "snapshots/$rev" -type f) > "$lists/files"
    send=$(awk 'FILENAME == ARGV[1] { w[$2] = $1; next } w[$2] != $1 { s += $1 } END { printf "%d", (s + 2^30 - 1) / 2^30 }' \
             <(worker 1 "cd '$wdir' 2>/dev/null && xargs -r stat -Lc '%s %n' 2>/dev/null; true" < "$lists/files") \
             <(cd "$dir" && xargs -r stat -Lc '%s %n' < "$lists/files"))
    need=0; (( send == 0 )) || need=$((send + 5))
    wfree=$(worker_free_gb 1 "$WORKER_HF")
    (( wfree >= need )) ||
      die "only ${wfree} GB free under $WORKER_HF on the worker; $id needs ~${need} GB there (~${send} GB to send; or set WORKER_WEIGHTS=nfs)"
    log "Copying $id @ ${rev:0:8} to the worker over the Sparks' link (~${send} GB to send, resumes)"
    worker 1 "mkdir -p '$wdir/refs' '$wdir/snapshots'"
    # the blobs, then the snapshot's links and refs/main
    # -L: a blob may itself be a link into the cache root's blobs/ (huggingface_hub's xet backend)
    rsync -a -L --partial --files-from="$lists/blobs" "$dir/" "$WORKER:$wdir/" \
      -e "ssh -o BatchMode=yes" ${RSYNC_OPTS:-}
    rsync -a "$dir/snapshots/$rev" "$WORKER:$wdir/snapshots/" -e "ssh -o BatchMode=yes"
    rm -rf -- "$lists"
    # refs/main as the head has it (with a pin: only when the worker has none)
    if [[ -z "$(model_revision "$id")" ]]; then worker 1 "printf %s '$rev' > '$wdir/refs/main'"
    else worker 1 "test -f '$wdir/refs/main' || printf %s '$rev' > '$wdir/refs/main'"; fi
    have=$(worker 1 "cd '$wdir/snapshots/$rev' && find -L . -type f -printf '%P %s\n' | LC_ALL=C sort")
    [[ "$have" == "$manifest" ]] ||
      { manifest_diff "$manifest" "$have"; die "the worker's copy of $id differs from the head's after the copy"; }
  done
fi

prepared_state > "$PREPARED_MARKER"
log "Done: both Sparks are ready. Start the server with ./start.sh (port $PORT)."
