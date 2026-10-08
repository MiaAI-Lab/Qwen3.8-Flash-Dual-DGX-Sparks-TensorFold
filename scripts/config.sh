# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`, `PULL=0 scripts/prepare.sh`
#   2. scripts/local.sh (this setup's own values, above all WORKER; sourced as bash), then ./.env (KEY=value lines,
#      read, never run): both are yours, not the repository's; where both set a key, local.sh wins
#   3. the defaults below
# The defaults follow TensorFold's Zig engine (tensorfold-native, branch zig-flashnext plus this repository's patches).
_cfg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$_cfg_root/scripts/local.sh" ]]; then
  declare -A _cfg_env=()
  while IFS= read -r _n; do _cfg_env[$_n]=${!_n}; done < <(compgen -e)
  source "$_cfg_root/scripts/local.sh"
  # the environment wins over local.sh: put back any variable it had that local.sh changed
  for _n in "${!_cfg_env[@]}"; do [[ "${!_n-}" == "${_cfg_env[$_n]}" ]] || export "$_n=${_cfg_env[$_n]}"; done
  unset _cfg_env
fi
if [[ -f "$_cfg_root/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < "$_cfg_root/.env"
fi
unset _n _line _key _value

# The Sparks: this machine serves rank 0 and the API; WORKER (ssh target, key-based) runs rank 1. Two Sparks only
# (tensor parallel over two ranks, TP=2).
TP=2
WORKER="${WORKER:-}"                 # e.g. user@<worker address>; set it in scripts/local.sh
FABRIC_PEER="${FABRIC_PEER:-}"       # the worker's CX7 address when WORKER is reached over another network
WORKER_HF_CACHE="${WORKER_HF_CACHE:-}"  # the worker's Hugging Face cache when it is not its HF_HOME (absolute path)
MASTER_PORT="${MASTER_PORT:-29551}"  # the ranks' rendezvous port (--master-port; keep it on the private link)
# The rendezvous address (--master): the head's address on the link to the worker (scripts/nodes.sh, detect_link)
MASTER_ADDR="${MASTER_ADDR:-}"

# The checkpoint, chosen by QUANT (one image serves both; the engine reads the format from config.json):
#   nvfp4   NVIDIA's NVFP4 quantization of Qwen3.8-Flash-Next, the original model: top-10 routing, ~6.0B active
#           parameters a token (routed experts NVFP4, the n-gram table FP8, the rest bf16; MTP head), ~124 GiB
#   int4ar  azampatti's INT4-AutoRound checkpoint: top-5 routing with a healed shared expert, ~4.8B active, the
#           experts in Intel's AutoRound int4 (GPTQ format, W4A16), the n-gram table FP8 (ple-table/), ~122 GiB;
#           faster, and a few points lower on its authors' benchmarks (its model card)
# Set QUANT in scripts/local.sh or .env, or `./start.sh --quant int4ar`. prepare.sh downloads only the chosen one.
QUANT="${QUANT:-nvfp4}"
case "$QUANT" in
  nvfp4)  _model=nvidia/Qwen3.8-Flash-Next-NVFP4; _pin=fc694b54fb0174e0913e6adf86691ef85a4ead47
          _name=Qwen3.8-Flash-Next; QUANT_LABEL=NVFP4; CKPT_GIB=124; _mem_base=48 ;;
  int4ar) _model=azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound; _pin=1464274120d36a4d8fcaa934552334a7d83ce0fd
          _name=Qwen3.8-Flash-Next-INT4-AR; QUANT_LABEL=INT4-AutoRound; CKPT_GIB=122; _mem_base=44 ;;
  *) printf '[%s] ERROR: QUANT is nvfp4 or int4ar, not %s\n' "$(basename "$0")" "$QUANT" >&2; exit 1 ;;
esac
MODEL_ID="${MODEL_ID:-$_model}"
# Its revision (a Hugging Face commit sha): the one this recipe is built for. prepare.sh downloads exactly it, start.sh
# serves that snapshot from the local cache (no network), and a new upstream commit changes nothing here until the pin
# does. Empty: the Hub's main when first downloaded. The pin belongs to the quant's checkpoint; another MODEL_ID gets
# no pin unless you set one.
_rev=""; [[ "$MODEL_ID" == "$_model" ]] && _rev=$_pin
MODEL_REVISION="${MODEL_REVISION-$_rev}"

# The engine: TensorFold's Zig engine, tensorfold-native, built from TF_REPO at TF_REF (branch zig-flashnext) with
# patches/*.patch applied (git apply in the checkout's root), with Zig ZIG_VERSION. Only the Zig build goes into the
# image (/opt/tensorfold); TensorFold's Python package is not installed.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-db281878ddb836fd0df510d8771ecb7e0fe47d26}"   # zig-flashnext
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"   # zig-aarch64-linux-0.17.0.tar.xz
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
TF_TAG="zig-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-qwen38fn:${TF_TAG}}"
# The kernel set the engine replays (aot.json + cubins), in the image at TENSORFOLD_CUDA_KERNELS: compiled in the
# image build from TensorFold's spec (prepare.sh, "the kernel set").
KERNELS_PATH=/opt/tensorfold/share/tensorfold/cuda/sm121
# Triton's line info records each kernel source file's modification time (with its path and size) in the cubin: the
# image build gives the spec's Python files this mtime, the one they had when the reference kernel set (the oracle's
# capture, and the model-free set the engine was checked with) was compiled, so its cubins are byte for byte those.
KERNEL_SOURCE_MTIME="${KERNEL_SOURCE_MTIME:-1791318675}"
# The image's hash: the patches, the kernel set, the Dockerfile (in prepare.sh) and what it builds from (TensorFold's
# commit, Zig); a change rebuilds the image (or pulls the published one of that hash). prepare.sh writes it into the
# tf.patches label.
image_hash() {
  ( cd "$_cfg_root" || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh   # the Dockerfile itself
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE kernel-mtime-$KERNEL_SOURCE_MTIME"
  ) | sha256sum | cut -c1-12
}
GHCR_IMAGE="${GHCR_IMAGE:-ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold}"
# The published image of this release's patches, pinned: prepare.sh pulls it by digest (a tag can be moved, a digest
# cannot) while image_hash still gives IMAGE_TAG's hash. Other hashes pull $GHCR_IMAGE:<TF_TAG>-<hash> when one is
# published, else build locally. scripts/publish-image.sh prints both. Empty: prepare.sh builds the image on the head.
IMAGE_TAG="${IMAGE_TAG:-zig-db28187-f3e1300f4f62}"
IMAGE_DIGEST="${IMAGE_DIGEST:-sha256:e188128ddabdc67ac02c05781f5c0c6b348db5f050cac297861d82b7f086e18c}"
# the registry reference prepare.sh pulls for these patches: the pinned digest, or the hash's tag
prebuilt_image() {
  local tag="${TF_TAG}-$(image_hash)"
  if [[ "$tag" == "$IMAGE_TAG" && -n "$IMAGE_DIGEST" ]]; then echo "$GHCR_IMAGE@$IMAGE_DIGEST"; else echo "$GHCR_IMAGE:$tag"; fi
}
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-fn-tf}"           # the same name on both Sparks

SERVED_NAME="${SERVED_NAME:-$_name}"          # the API model id: Qwen3.8-Flash-Next, or -INT4-AR for int4ar
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# Prompt + reply window per request (--context). The checkpoint's native window is 262,144; past it the engine applies
# Qwen's documented YaRN (rope_type yarn, factor 4.0, original_max_position_embeddings 262144), so the default 1,048,576
# is four native windows. TF_FLASHNEXT_YARN (the engine's variable, read by both ranks): the YaRN factor, 4 when
# CONTEXT is past 262,144, else 0 (off: plain RoPE, as the checkpoint ships). With 4 the engine admits windows up to
# 1,048,576; without it, up to the native 262,144.
CONTEXT="${CONTEXT:-1048576}"
if [[ "$CONTEXT" =~ ^[0-9]+$ ]] && (( CONTEXT > 262144 )); then _yarn=4; else _yarn=0; fi
export TF_FLASHNEXT_YARN="${TF_FLASHNEXT_YARN:-$_yarn}"
# Requests decoded together (--parallel): up to 16 streams share each forward (the engine's shared rounds; each
# stream's tokens equal its solo run). Each request's caches grow as it goes, inside the engine's memory budget; a
# request that would not fit beside the others is refused rather than squeezed.
PARALLEL="${PARALLEL:-16}"
# Decode speed-ups, on by default in the engine and set here explicitly (both ranks get every TF_FLASHNEXT_*):
# TF_FLASHNEXT_ROCE: the ranks' small all-gathers as one-shot RDMA writes over the CX7 link instead of NCCL (both
# ranks must agree; a rank that cannot open RoCE leaves both on NCCL); TF_FLASHNEXT_L2PF: prefetch the next weights
# into L2 while a gather waits. 0 turns either off. Exact either way: same bits.
export TF_FLASHNEXT_ROCE="${TF_FLASHNEXT_ROCE:-1}"
export TF_FLASHNEXT_L2PF="${TF_FLASHNEXT_L2PF:-1}"
# MTP drafts: up to TF_FLASHNEXT_DEPTH a round (15: 16-row windows, also for copy drafts); when a draft stops is the
# engine's own (hybrid) rule. Drafts only propose, so replies are the same either way.
export TF_FLASHNEXT_DEPTH="${TF_FLASHNEXT_DEPTH:-15}"
# The reply budget of a request that sets no max_tokens, reasoning and answer together (--max-tokens; the engine's own
# default is 4,096, which can end a reply inside its thinking). A request's own value wins.
MAX_TOKENS="${MAX_TOKENS:-32768}"
# Think before answering by default (--thinking; 0: --no-thinking, answer directly unless a request asks to think)
THINKING="${THINKING:-1}"
# Drafts from the checkpoint's own MTP head (1); 0 serves without drafts (--no-drafts). Drafts only propose: every
# drafted token is checked against the model's own sample, so the replies are the same either way.
DRAFTS="${DRAFTS:-1}"
# KV cache dtype (--kv-dtype). fp8 (the default here) stores each cached key and value in FP8 with a scale: the KV pool
# is about 1.8-1.9x larger than bf16's, and the cache is LOSSY: about 98.8% top-1 agreement with a bf16 cache on
# teacher-forced replies, so a free-running reply can diverge from the bf16 one. bf16 is the exact cache: KV_DTYPE=bf16.
KV_DTYPE="${KV_DTYPE:-fp8}"
# Image and video (--vision, TensorFold's own vision frontend and tower in a Python helper on rank 0). VISION=0 serves
# text only and returns the helper's workspace (4 GiB) to the KV pool; VISION_URLS=1 also lets the server fetch public
# https:// URLs (data URLs only otherwise).
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
export TENSORFOLD_MAX_IMAGES="${TENSORFOLD_MAX_IMAGES:-50}"          # images a request (--vision-max-images)
MAX_VIDEOS="${MAX_VIDEOS:-4}"                                        # videos a request (--vision-max-videos)
export TENSORFOLD_IMAGE_TOKENS="${TENSORFOLD_IMAGE_TOKENS:-16384}"   # image tokens a request, at most 4,096 an image
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"   # video tokens a request
export TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-2048}"
# The memory floor: the engine keeps this much MemAvailable free when it sizes the KV pool (and refuses what would
# eat into it); start.sh and prepare.sh refuse a Spark that has less than this free.
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-10}"
# Memory. The engine budgets the KV caches itself (MemAvailable at load less TENSORFOLD_MEMORY_RESERVE_GIB and the
# vision workspace; a request whose prompt plus reply budget would not fit beside the others is refused), but not a prompt's prefill scratch, which grows with the
# prompt. The figures below were measured with an earlier engine (bd895c3) that held the whole window's caches.
# On GB10 unified memory, running out freezes the machine instead of failing an allocation, so start.sh refuses to
# start a rank on a Spark with less
# MemAvailable than MEM_NEED_GIB (empty: mem_need_gib below, what a rank of this window holds under a prompt that
# fills it, measured, plus what the machine needs to stay responsive). MEM_CHECK=0 makes it a warning (your risk).
MEM_NEED_GIB="${MEM_NEED_GIB:-}"
MEM_FLOOR_GIB="${MEM_FLOOR_GIB:-$TENSORFOLD_MEMORY_RESERVE_GIB}"   # while loading, start.sh stops both ranks when a Spark's MemAvailable falls below it
# mem_need_gib <context> <parallel>: the MemAvailable a rank needs at start, measured on two Sparks with nvfp4 (2026-10-07,
# the worker's rank 1, a 1,048,576-token window, PARALLEL 1): ~48 GiB fixed (half of the weights with its half of the n-gram
# table on the GPU, scratch, CUDA's context), plus ~17 GiB of caches per million tokens of window (held from the start)
# and ~31 GiB per million tokens of prompt while it is prefilled (freed after): 113 GiB free at start, 48 at idle, 16.5
# at the low point of a 998,986-token prompt. Plus MEM_FLOOR_GIB and 2 GiB of slack: ~106 GiB for the default window.
mem_need_gib() {
  local ctx=${1:-$CONTEXT} par=${2:-$PARALLEL}
  [[ "$ctx" =~ ^[0-9]+$ ]] || ctx=$CONTEXT
  [[ "$par" =~ ^[0-9]+$ ]] || par=1
  # (the engine admits further streams only inside its own budget, so the check covers one request of the full window)
  # int4ar's fixed part is ~4 GiB smaller (58.3 GiB of weights a rank against NVFP4's 62.3)
  # (an fp8 cache holds ~9 GiB per million tokens instead of 17; the vision helper's workspace adds
  # TENSORFOLD_VISION_WORKSPACE_MIB, 2 GiB by default)
  local kv=17 vis=0
  [[ "$KV_DTYPE" == fp8 ]] && kv=9
  [[ "$VISION" == 1 ]] && vis=$(( (TENSORFOLD_VISION_WORKSPACE_MIB + 1023) / 1024 ))
  echo $(( _mem_base + vis + (kv * ctx + 31 * ctx + 1048575) / 1048576 + MEM_FLOOR_GIB + 2 ))
}

export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
# Where rank 1 reads the checkpoint: copy (default) keeps a copy in the worker's own Hugging Face cache (prepare.sh
# copies ~124 GiB over the link); nfs reads the head's HF_CACHE over NFS instead (no copy, no disk on the worker),
# through a read-only docker volume NFS_VOLUME on the worker that prepare.sh creates (no sudo there). The head must
# export NFS_PATH (default: HF_CACHE) to the worker; NFS_SERVER defaults to the head's address on the link.
WORKER_WEIGHTS="${WORKER_WEIGHTS:-copy}"
NFS_PATH="${NFS_PATH:-$HF_CACHE}"
NFS_SERVER="${NFS_SERVER:-}"
NFS_VOLUME="${NFS_VOLUME:-qwen38fn-hf}"
KERNEL_CACHE="${KERNEL_CACHE:-$HOME/.cache/tensorfold-qwen38fn}"   # the ranks' /cache (CUDA's JIT cache), a folder per image hash
STATE_DIR="${STATE_DIR:-$HOME/.local/state/qwen38fn-tensorfold}"   # this recipe's locks and setup marker
# Server logs: stop.sh (and start.sh, before it removes a stopped container left from an earlier run) saves each rank's
# container log, stdout and stderr with timestamps, gzipped, as <date>-<time>-rank<N>.log.gz in LOG_DIR here and in
# ~/.cache/tensorfold-qwen38fn/logs on the worker, and keeps the newest LOG_KEEP (0: saves none). docker rm deletes a
# container's own log, so without this a crash's log is gone at the next stop or start.
LOG_DIR="${LOG_DIR:-$HOME/.cache/tensorfold-qwen38fn/logs}"
LOG_KEEP="${LOG_KEEP:-10}"
# Free disk prepare.sh asks for before it downloads or copies: under HF_CACHE, what the download still needs (the
# revision's files whose blobs are not cached yet, from the Hub's file list, plus 5 GB), or MIN_FREE_GB for the
# checkpoint (~133 GB) when that list cannot be read (no huggingface_hub on the host, or no network); on the worker,
# what rsync must send plus 5 GB; and an image build or copy under Docker's root on each Spark (IMAGE_FREE_GB); both
# together when they share a filesystem.
MIN_FREE_GB="${MIN_FREE_GB:-140}"
IMAGE_FREE_GB="${IMAGE_FREE_GB:-35}"

# Colours only on a terminal.
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

# HF_TOKEN from scripts/local.sh reaches the hf CLI and the download container only when exported (by name, never as
# a value on a command line)
[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
model_cache_dir() { local id=${1:-$MODEL_ID}; echo "$HF_CACHE/hub/models--${id//\//--}"; }
# model_revision <id>: the pinned revision of MODEL_ID (empty: none, the cache's refs/main counts)
model_revision() { [[ "$1" != "$MODEL_ID" ]] || echo "$MODEL_REVISION"; }
# snapshot_rev <id>: the snapshot this setup serves: the pin, else what refs/main names on this Spark
snapshot_rev() { local rev; rev=$(model_revision "$1"); [[ -n "$rev" ]] || rev=$(cat "$(model_cache_dir "$1")/refs/main" 2>/dev/null); echo "$rev"; }

# What scripts/prepare.sh last left ready on both Sparks (it writes this line to PREPARED_MARKER when it succeeds);
# start.sh runs prepare.sh again whenever the current line differs: a missing or different image on either Spark, new
# patches or kernels, another model or revision, another worker. Needs scripts/nodes.sh (the worker's image).
PREPARED_MARKER="$STATE_DIR/prepared"
prepared_state() {
  local hash label wlabel
  hash=$(image_hash)
  label=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  wlabel=$(worker 1 docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null || echo missing)
  echo "model=$MODEL_ID@$MODEL_REVISION image=$label worker=$wlabel patches=$hash worker_host=$WORKER weights=$WORKER_WEIGHTS"
}
