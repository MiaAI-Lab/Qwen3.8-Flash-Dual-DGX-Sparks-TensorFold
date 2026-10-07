#!/usr/bin/env bash
# Serve Qwen3.8 Flash Next (NVIDIA's NVFP4, or the INT4-AutoRound checkpoint: QUANT) with TensorFold's Zig engine on two DGX Sparks, end to
# end: runs scripts/prepare.sh on both Sparks when the image or the checkpoint is not ready yet (first run, or after
# patches change), starts rank 1 on the worker and rank 0 here, which serves the API on port 8888, waits until the
# OpenAI API answers, then runs a smoke test. Stop it with ./stop.sh.
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                         # scripts/config.sh defaults: NVFP4, 16 requests at once, FP8 KV, image and video input, a 1,048,576-token
#                                      # window (YaRN past the native 262,144), MTP drafts, thinking on
#   ./start.sh --quant int4ar          # the INT4-AutoRound checkpoint instead (QUANT; with restart to switch)
#                                      # (if the server already runs, says so and leaves it alone)
#   ./start.sh restart                 # stop both ranks (./stop.sh), then start them again, e.g. to apply changed
#                                      # settings or patches; the new arguments are checked before stopping
#   ./start.sh restart --parallel 2 --context 262144
#   CONTEXT=262144 ./start.sh restart  # the native window, without YaRN
#   DRAFTS=0 ./start.sh restart        # no MTP drafts (--no-drafts)
#   DRY_RUN=1 ./start.sh               # print both ranks' docker commands and exit, stopping and starting nothing
# Extra arguments go to both ranks after the defaults, so they win (the last value of a flag counts).
# Setup: WORKER=user@<worker address> in scripts/local.sh (key-based ssh).
# Settings, from the environment, scripts/local.sh or ./.env (defaults in scripts/config.sh):
#   serving  CONTEXT, PARALLEL (1-16), MAX_TOKENS, THINKING, DRAFTS, KV_DTYPE (fp8 | bf16), SERVED_NAME, HOST, PORT
#   vision   VISION (1 | 0), VISION_URLS, MAX_VIDEOS, TENSORFOLD_MAX_IMAGES / _IMAGE_TOKENS / _VIDEO_TOKENS
#   nodes    WORKER, FABRIC_PEER, WORKER_HF_CACHE, MASTER_PORT, MASTER_ADDR, NCCL_RAILS (1: one RoCE device),
#            NCCL_CHANNELS, NCCL_DEBUG
#   files    QUANT (nvfp4 | int4ar), MODEL_ID, MODEL_REVISION, HF_CACHE (default: HF_HOME), KERNEL_CACHE,
#            WORKER_WEIGHTS (copy | nfs: rank 1 reads the head's HF_CACHE over NFS), NFS_PATH, NFS_SERVER, NFS_VOLUME,
#            STATE_DIR, HF_HUB_OFFLINE=0 (let the engine reach the Hub; default serves from the local cache only)
#   image    IMAGE, TF_REPO, TF_REF, ZIG_VERSION, BASE_IMAGE, GHCR_IMAGE, IMAGE_TAG / IMAGE_DIGEST (the
#            pinned published image), CONTAINER_NAME
#   setup    PREPARE (auto | 1 | 0), PULL, MIN_FREE_GB, IMAGE_FREE_GB, RSYNC_OPTS, HF_TOKEN (prepare.sh's download);
#            FOREGROUND=1 (stay attached to rank 0's log, exit with its code); WAIT_TIMEOUT (seconds, default 1800);
#            DRY_RUN=1 (print the docker commands, change nothing); MEM_NEED_GIB, MEM_CHECK=0 (the memory check);
#            STOP_TIMEOUT (stop.sh); LOG_DIR, LOG_KEEP (saved server logs)
#   ranks    every TENSORFOLD_*, TF_FLASHNEXT_*, TF_TP_* and TF_CUDA_* variable goes to both ranks
#            (TENSORFOLD_API_KEY by name only, to rank 0)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
# --quant Q (or --quant=Q) picks the checkpoint (QUANT) before the settings are read; the other arguments stay
_args=()
while (( $# )); do
  case "$1" in
    --quant) [[ $# -ge 2 ]] || { echo "[start.sh] ERROR: --quant needs nvfp4 or int4ar" >&2; exit 1; }; export QUANT=$2; shift 2 ;;
    --quant=*) export QUANT=${1#*=}; shift ;;
    *) _args+=("$1"); shift ;;
  esac
done
set -- "${_args[@]}"; unset _args
source ./scripts/config.sh
source ./scripts/nodes.sh

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help) usage; exit 0 ;;
esac
for arg in "$@"; do [[ "$arg" == -h || "$arg" == --help ]] && { usage; exit 0; }; done

# The serve arguments both ranks share: scripts/config.sh's defaults first, then the command line's (the last value
# of a flag counts).
check_workers
[[ "$CONTEXT" =~ ^[1-9][0-9]*$ && "$CONTEXT" -le 1048576 ]] || die "CONTEXT is a token count up to 1048576, not $CONTEXT"
[[ "$PARALLEL" =~ ^([1-9]|1[0-6])$ ]] || die "PARALLEL is 1 to 16, not $PARALLEL"
[[ "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "MAX_TOKENS is a token count, not $MAX_TOKENS"
for v in THINKING DRAFTS VISION VISION_URLS; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$TF_FLASHNEXT_YARN" =~ ^(0|4)$ ]] || die "TF_FLASHNEXT_YARN is 0 (off) or 4 (Qwen's YaRN factor), not $TF_FLASHNEXT_YARN"
(( CONTEXT <= 262144 || TF_FLASHNEXT_YARN != 0 )) || die "CONTEXT=$CONTEXT is past the native 262,144-token window: it needs YaRN (TF_FLASHNEXT_YARN=4)"
[[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || die "MASTER_PORT is a port number, not $MASTER_PORT"
DRY=0; [[ "${DRY_RUN:-0}" == 1 ]] && DRY=1
SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS")
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ "$DRAFTS" == 1 ]] || SERVE_ARGS+=(--no-drafts)
[[ "$KV_DTYPE" =~ ^(bf16|fp8)$ ]] || die "KV_DTYPE is bf16 or fp8, not $KV_DTYPE"
SERVE_ARGS+=(--kv-dtype "$KV_DTYPE")
[[ "$MAX_VIDEOS" =~ ^[0-9]+$ ]] || die "MAX_VIDEOS is a count, not $MAX_VIDEOS"
if [[ "$VISION" == 1 ]]; then
  SERVE_ARGS+=(--vision --vision-max-images "$TENSORFOLD_MAX_IMAGES" --vision-max-videos "$MAX_VIDEOS"
               --vision-image-tokens "$TENSORFOLD_IMAGE_TOKENS")
  [[ "$VISION_URLS" == 1 ]] && SERVE_ARGS+=(--vision-urls)
fi
USER_ARGS=("$@")
# The effective value of a flag (its last occurrence, as --flag value or --flag=value).
arg_value() {
  local flag=$1 value="" i
  local -a all=("${SERVE_ARGS[@]}" "${USER_ARGS[@]}")
  for (( i = 0; i < ${#all[@]}; i++ )); do
    case "${all[i]}" in
      "$flag") value="${all[i + 1]:-}" ;;
      "$flag="*) value="${all[i]#*=}" ;;
    esac
  done
  echo "$value"
}
# Where to reach the server from this machine: a wildcard bind answers on loopback.
API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
[[ "$API_HOST" == *:* ]] && API_HOST="[$API_HOST]"
URL="http://$API_HOST:$PORT"

running_here()   { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
running_worker() { [[ "$(worker 1 docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
served_name() {
  curl -s --max-time 5 "$URL/v1/models" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}

# ---------------------------------------------------------------- banner and progress
B=$'\033[1m'; M=$'\033[1;35m'; G=$'\033[1;32m'; D=$'\033[2m'; R=$'\033[0m'
[[ -t 1 ]] || { B=; M=; G=; D=; R=; }
source ./scripts/banner.sh
echo
banner "Qwen3.8-Flash-Next · $QUANT_LABEL"          # the TensorFold ribbon and MIA AI LAB, and the quant (terminals only)
printf '\n%s  Mia'"'"'s TensorFold Start Script%s\n' "$M" "$R"
printf '%s  %s (%s) · 2 x DGX Spark · %s at once · %s-token window · drafts %s · port %s%s\n\n' "$D" "$MODEL_ID" "$QUANT_LABEL" \
  "$(arg_value --parallel)" "$(arg_value --context)" "$( [[ "$DRAFTS" == 1 ]] && echo MTP || echo off)" "$PORT" "$R"
printf '%s  KV %s · vision %s · memory reserve %s GiB%s\n\n' "$D" "$KV_DTYPE" "$( [[ "$VISION" == 1 ]] && echo on || echo off)" \
  "$TENSORFOLD_MEMORY_RESERVE_GIB" "$R"
STEPS=5
step() { printf '%s[%s/%s]%s %s%s%s\n' "$M" "$1" "$STEPS" "$R" "$B" "$2" "$R"; }

command -v docker >/dev/null || die "docker is not installed"
mkdir -p "$KERNEL_CACHE" "$STATE_DIR"
exec 8>"$STATE_DIR/start.lock"
flock -n 8 || die "another ./start.sh is already running; wait for it to finish"
need_workers
(( DRY )) && log "DRY_RUN=1: printing the docker commands; nothing is stopped, started or prepared"

# ---------------------------------------------------------------- already running?
all_running() { running_here && running_worker; }
if (( ! DRY )) && [[ "$MODE" == start ]] && all_running; then
  log "$CONTAINER_NAME is already running on both Sparks (model: $(served_name || echo "not answering yet"), port $PORT): nothing to do."
  running=$(served_name || true)
  [[ -z "$running" || "$running" == "$SERVED_NAME" ]] ||
    log "It serves $running, not $SERVED_NAME (QUANT=$QUANT): ./start.sh restart${QUANT:+ --quant $QUANT} switches."
  log "Use ./start.sh restart to restart it (e.g. with new settings), or ./stop.sh to stop it."
  exit 0
fi

# ---------------------------------------------------------------- 1. setup
# scripts/prepare.sh (image and checkpoint on both Sparks) runs whenever what it last prepared differs from now: the
# first run, new patches or kernels, another model, revision or worker. PREPARE=1 forces it, PREPARE=0 skips it.
step 1 "Setup: image and checkpoint on both Sparks"
if [[ "${PREPARE:-auto}" == 1 || ( "${PREPARE:-auto}" != 0 && "$(prepared_state 2>/dev/null)" != "$(cat "$PREPARED_MARKER" 2>/dev/null)" ) ]]; then
  if (( DRY )); then log "DRY_RUN: scripts/prepare.sh would run now (not ready yet)"
  else
    log "Not ready yet: running scripts/prepare.sh (the first time this builds or pulls the image, downloads ~$CKPT_GIB GiB and copies both to the worker)"
    ./scripts/prepare.sh
  fi
else
  log "Ready: $IMAGE (patches $(image_hash)) and $MODEL_ID on both Sparks${PREPARE:+ (PREPARE=$PREPARE)}"
fi
why="scripts/prepare.sh did not"; [[ "${PREPARE:-auto}" == 0 ]] && why="PREPARE=0 skipped scripts/prepare.sh, which would"
(( DRY )) && why="DRY_RUN: scripts/prepare.sh would"
# (in a dry run, a missing image or checkpoint is a warning: the commands still print, with placeholders)
missing() { if (( DRY )); then warn "$1"; else die "$1"; fi; }
have_image=1
docker image inspect "$IMAGE" >/dev/null 2>&1 || { have_image=0; missing "image $IMAGE missing: $why build it"; }
if [[ -z "$WORKER_DOWN" ]]; then
  worker 1 docker image inspect "$IMAGE" >/dev/null 2>&1 || missing "image $IMAGE missing on the worker: $why copy it there"
fi
# The kernel set the engine replays (scripts/prepare.sh, "the kernel set"): an image without one (an older build)
# cannot serve, so it is refused here, before anything is stopped
if (( have_image )); then
  kernels=$(docker image inspect -f '{{index .Config.Labels "tf.kernels"}}' "$IMAGE" 2>/dev/null || true)
  [[ "$kernels" == present ]] ||
    missing "$IMAGE has no kernel set (tf.kernels=${kernels:-?}): rebuild it with scripts/prepare.sh --rebuild"
fi
# The ranks' /cache (CUDA's JIT cache), a folder per image hash
KCACHE=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
[[ "$KCACHE" =~ ^[0-9a-f]{12}$ ]] || KCACHE=$(image_hash)
# The snapshot both ranks serve (config.sh's pin, else refs/main), as a path under the containers' cache mount: the
# ranks read it offline, whatever the Hub's main is now. The worker's cache is its own HF_HOME (prepare.sh copies into
# the same place), or the head's over NFS (WORKER_WEIGHTS=nfs).
WORKER_HF=$(worker_hf_cache)
WORKER_MOUNT="$WORKER_HF:/root/.cache/huggingface:ro"
[[ "$WORKER_WEIGHTS" == nfs ]] && WORKER_MOUNT="$NFS_VOLUME:/root/.cache/huggingface:ro"
snapshot() {  # <repo id>: its snapshot path in the container, checked on both Sparks (DRY_RUN: here only)
  local id=$1 rev sub
  rev=$(snapshot_rev "$id")
  [[ -n "$rev" ]] || die "$id not in $HF_CACHE: $why download it"
  sub="hub/models--${id//\//--}/snapshots/$rev"
  if [[ ! -f "$HF_CACHE/$sub/config.json" ]]; then missing "$id @ ${rev:0:8} not in $HF_CACHE: $why download it"
  elif (( ! DRY )); then
    if [[ "$WORKER_WEIGHTS" == nfs ]]; then
      worker_nfs test -f "/hf/$sub/config.json" ||
        die "the worker does not see $id @ ${rev:0:8} over NFS ($NFS_VOLUME): $why set it up"
    else
      worker 1 "test -f '$WORKER_HF/$sub/config.json'" ||
        die "$id @ ${rev:0:8} not on the worker ($WORKER_HF): $why copy it there"
    fi
  fi
  echo "/root/.cache/huggingface/$sub"
}
MODEL_ARG=$(snapshot "$MODEL_ID")

# ---------------------------------------------------------------- 2. checks
step 2 "Checks: arguments, link, previous server, port, memory"
detect_link
log "Link: $HEAD_ADDR ($HEAD_DEV) <-> $WORKER_ADDR ($WORKER_DEV), RoCE $HEAD_HCAS / $WORKER_HCAS, rendezvous $MASTER_ADDR:$MASTER_PORT"
# rank r's tensorfold-native serve command (the CLI contract): rank 1 runs the same command with --rank 1 and serves
# no HTTP; it joins rank 0 at --master:--master-port (the TCP control link and NCCL's id), and both ranks check they
# load the same snapshot, kernel set, window and drafts before NCCL starts.
rank_argv() {
  RANK_ARGV=(tensorfold-native serve "$MODEL_ARG" --name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}"
             --tp 2 --rank "$1" --master "$MASTER_ADDR" --master-port "$MASTER_PORT" "${USER_ARGS[@]}")
}
# The engine's own parser, in a throwaway container without the GPU or the network: a typo fails here, before anything
# is stopped. tensorfold-native has no parse-only mode: it is given a model path that does not exist, so arguments it
# rejects end it with code 2 (its usage error, flags it does not serve included), and arguments it takes end it with
# code 1 at the model ("... is neither a directory nor a Hugging Face repo id").
if (( have_image )); then
  rank_argv 0
  RANK_ARGV[2]=/nonexistent/argument-check
  code=0
  docker run --rm --network none --entrypoint /opt/tensorfold/native/bin/tensorfold-native "$IMAGE" "${RANK_ARGV[@]:1}" \
    >/dev/null 2>"$STATE_DIR/args.err" || code=$?
  if (( code == 2 )) || ! grep -q 'neither a directory' "$STATE_DIR/args.err"; then
    if (( DRY )); then warn "DRY_RUN: tensorfold-native in $IMAGE rejects these arguments: $(grep -v '^usage:' "$STATE_DIR/args.err" | tail -1)"
    else cat "$STATE_DIR/args.err" >&2; die "tensorfold-native serve rejects these arguments (see above); nothing was changed"; fi
  fi
fi
here_up=0; running_here && here_up=1
worker_up=0; running_worker && worker_up=1
if (( DRY )); then
  (( here_up || worker_up )) && log "DRY_RUN: $CONTAINER_NAME is running; a real start would stop it first (./stop.sh)"
elif (( here_up || worker_up )); then               # after the setup and the checks: down only while restarting,
  if [[ "$MODE" == start ]]; then                   # or when a start that failed halfway left one rank up
    if (( here_up )); then log "Only rank 0 is running (here): stopping it, then starting both ranks"
    else log "Only rank 1 is running (on $WORKER): stopping it, then starting both ranks"; fi
  fi
  ./stop.sh
fi
left_here=0; docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" && left_here=1
left_worker=0
worker 1 "docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'" 2>/dev/null && left_worker=1
if (( ! DRY && ( left_here || left_worker ) )); then
  where="on both Sparks"; (( left_worker )) || where="here"; (( left_here )) || where="on the worker"
  log "Removing the previous (stopped) container $CONTAINER_NAME $where, its log saved first (a crash's evidence)"
  if (( left_here )); then
    saved=$(save_log "$LOG_DIR" 0 "$CONTAINER_NAME" "$LOG_KEEP") || warn "could not save rank 0's log to $LOG_DIR"
    [[ -z "${saved:-}" ]] || log "Rank 0's log: $saved"
    docker rm -f "$CONTAINER_NAME" >/dev/null
  fi
  if (( left_worker )); then
    saved=$(worker_save_log) || warn "could not save rank 1's log on $WORKER"
    [[ -z "${saved:-}" ]] || log "Rank 1's log (on $WORKER): $saved"
    worker 1 "docker rm -f '$CONTAINER_NAME' >/dev/null"
  fi
fi
if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  (( DRY )) && log "DRY_RUN: port $PORT is in use now" ||
    die "port $PORT is already in use: $(ss -ltnp "sport = :$PORT" 2>/dev/null | tail -n +2)"
fi
# Free memory at start on each Spark. Each rank holds its half of the weights and its caches, and a long prompt's
# prefill takes more while it runs (scripts/config.sh, "Memory"); on GB10's unified memory running out freezes the
# machine rather than failing an allocation. So a rank starts only where MemAvailable is at least MEM_NEED_GIB (scripts/config.sh
# mem_need_gib: what a rank of this window holds under a long prompt, measured, plus MEM_FLOOR_GIB for the machine).
# MEM_CHECK=0 warns instead of stopping here.
need_gib=${MEM_NEED_GIB:-$(mem_need_gib "$(arg_value --context)" "$(arg_value --parallel)")}
here_gb=$(free -g | awk '/^Mem:/ {print $7}')
there_gb=$(worker 1 "free -g | awk '/^Mem:/ {print \$7}'" 2>/dev/null || echo 0)
low_mem() {  # <where> <GiB available> <how to see what holds it>
  local msg="only $2 GiB memory available $1; a rank of a $(arg_value --context)-token window needs ~$need_gib (MemAvailable): stop other GPU work ($3) or lower CONTEXT"
  if (( DRY )) || [[ "${MEM_CHECK:-1}" == 0 ]]; then warn "$msg"; else die "$msg (MEM_CHECK=0 starts anyway)"; fi
}
if (( here_gb >= need_gib && there_gb >= need_gib )); then
  log "Arguments OK, port $PORT free, ${here_gb} GiB memory available here, ${there_gb} GiB on the worker (a rank needs ~$need_gib)"
else
  (( here_gb >= need_gib )) || low_mem here "$here_gb" "docker ps"
  [[ -n "$WORKER_DOWN" ]] || (( there_gb >= need_gib )) || low_mem "on the worker" "$there_gb" "ssh $WORKER docker ps"
fi

# TensorFold's own switches (TENSORFOLD_*, TF_FLASHNEXT_*, TF_TP_*, TF_CUDA_*) reach both ranks with the same values:
# the worker's docker command runs over ssh, where this shell's environment does not reach. TENSORFOLD_API_KEY is a
# secret: it never goes on a command line, only by name to rank 0 (docker reads its value from this environment).
env_args() {
  local name
  ENV_ARGS=(-e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" -e TENSORFOLD_CUDA_KERNELS="$KERNELS_PATH")
  while IFS='=' read -r name _; do
    [[ "$name" == TENSORFOLD_API_KEY || "$name" == TENSORFOLD_CUDA_KERNELS ]] && continue
    ENV_ARGS+=(-e "$name=${!name}")
  done < <(env | grep -E '^(TENSORFOLD|TF_FLASHNEXT|TF_TP|TF_CUDA)_[A-Z0-9_]+=' || true)
}
env_args
KEY_ARGS=(); [[ -z "${TENSORFOLD_API_KEY:-}" ]] || { export TENSORFOLD_API_KEY; KEY_ARGS=(-e TENSORFOLD_API_KEY); }
RUN_ARGS=(--gpus all --ipc=host --network host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1)

# ---------------------------------------------------------------- 3. launch, 4. load (a second try when the window does not fit)
# No token goes into the containers: the ranks read only the local cache (HF_HUB_OFFLINE=1).
# Rank 1 on the worker first, then rank 0 here, which serves the API. DRY_RUN=1 prints each rank's command exactly as
# it would run (the worker's as the string ssh sends) and exits.
launch() {
  local remote a
  local -a worker_cmd here_cmd
  rank_argv 1
  log "Rank 1 on $WORKER: ${RANK_ARGV[*]}"
  worker_cmd=(docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" "${ENV_ARGS[@]}"
              $(rank_nccl_env 1)
              -v "$WORKER_MOUNT" -v "\$HOME/.cache/tensorfold-qwen38fn/$KCACHE:/cache"
              "$IMAGE" "${RANK_ARGV[@]}")
  remote=""; for a in "${worker_cmd[@]}"; do
    case "$a" in '$HOME'*) remote+=" \"$a\"" ;; *) remote+=" $(printf '%q' "$a")" ;; esac
  done
  if (( DRY )); then
    printf '[dry-run] rank 1 on %s:\n  %s\n' "$WORKER" "mkdir -p \$HOME/.cache/tensorfold-qwen38fn &&$remote"
  else
    worker 1 "mkdir -p \$HOME/.cache/tensorfold-qwen38fn &&$remote" >/dev/null || die "could not start rank 1 on $WORKER"
  fi
  rank_argv 0
  log "Rank 0 here: ${RANK_ARGV[*]}"
  here_cmd=(docker run -d --name "$CONTAINER_NAME" "${RUN_ARGS[@]}" "${ENV_ARGS[@]}" "${KEY_ARGS[@]}"
            $(rank_nccl_env 0)
            -v "$HF_CACHE":/root/.cache/huggingface:ro -v "$KERNEL_CACHE/$KCACHE":/cache
            "$IMAGE" "${RANK_ARGV[@]}")
  if (( DRY )); then
    printf '[dry-run] rank 0 here:\n  %s\n' "$(printf '%q ' "${here_cmd[@]}" | sed 's/ $//')"
    exit 0
  fi
  "${here_cmd[@]}" >/dev/null
}
# FOREGROUND=1: stay attached to rank 0's log and exit with its code (systemd's Restart=on-failure). Either rank ending
# takes the other one down: a lone rank would wait for its peer forever.
foreground() {
  local watch w code
  trap './stop.sh; exit 130' INT TERM
  ( exec 8>&-                                        # not holding start.sh's lock once start.sh has exited
    while sleep 30; do
      running_here || exit 0
      worker 1 true 2>/dev/null || continue         # a worker out of reach for a moment says nothing about its rank
      running_worker && continue
      warn "rank 1 on $WORKER exited: stopping rank 0"
      docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1
      exit 1
    done ) &
  watch=$!
  # rank 0's log as it comes, without the container banner and the status polls (NOISE, below)
  docker logs -f "$CONTAINER_NAME" > >(grep --line-buffered -v -E "$NOISE") 2>&1 || true
  kill "$watch" 2>/dev/null || true
  w=0; wait "$watch" || w=$?
  code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo 1)
  [[ "$w" != 1 || "$code" != 0 ]] || code=1         # the worker's rank failed, even if rank 0 then shut down cleanly
  worker 1 "docker stop -t ${STOP_TIMEOUT:-30} '$CONTAINER_NAME'" >/dev/null 2>&1 || true
  exit "$code"
}
# NVIDIA's container banner, without its license notice (GOVERNING TERMS ...), which stays visible; and the server's
# access lines for status polls (start.sh's own /health, dashboards polling /health, /v1/models, /metrics, /stats)
NOISE='^\s*$|^=+$|^== PyTorch ==|^NVIDIA Release|Copyright|All rights reserved|PyTorch Version|Various files include|NOTE: CUDA Forward|Using CUDA|cuda-compatibility|Container image|^\[tensorfold\] [^ ]+ "GET /(health|v1/models|models|metrics|stats|slots)[ ?]'
LOGS_PID=""
trap 'kill $LOGS_PID 2>/dev/null || true' EXIT
# GPU memory a container's processes hold so far (GiB); on the worker through ssh, with this definition
gpu_gib() {
  local pids
  pids=$(docker top "$1" -eo pid 2>/dev/null | tail -n +2 | paste -sd'|')
  [[ -n "$pids" ]] || { echo 0; return; }
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' -v re="^($pids)$" '$1 ~ re { s += $2 } END { printf "%.1f", s / 1024 }'
}
fail() {
  kill $LOGS_PID 2>/dev/null || true
  sleep 0.5
  printf '\n%s── rank 0 (here): last server log lines ──%s\n' "$D" "$R"
  docker logs --tail 25 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  printf '%s── rank 1 (%s): last server log lines ──%s\n' "$D" "$WORKER" "$R"
  worker 1 docker logs --tail 25 "$CONTAINER_NAME" 2>&1 | sed 's/^/  │ /'
  die "$1"
}
step 3 "Launch: container $CONTAINER_NAME, rank 1 on $WORKER, then rank 0 here"
launch
[[ "${FOREGROUND:-0}" == 1 ]] && foreground
step 4 "Loading: half of the weights on each Spark (about two minutes)"
# docker logs is the background job, so killing it ends the whole pipeline (no orphaned `docker logs -f`)
docker logs -f "$CONTAINER_NAME" > >(grep --line-buffered -v -E "$NOISE" | sed -u "s/^/  ${D}│${R} /") 2>&1 &
LOGS_PID=$!
start=$SECONDS; next_beat=15
# A rank that takes the last of a Spark's memory freezes it: below MEM_FLOOR_GIB both ranks are stopped (not removed:
# their logs stay for fail's lines, and the next start saves them)
mem_stop() {
  docker stop -t 5 "$CONTAINER_NAME" >/dev/null 2>&1 || true
  worker 1 "docker stop -t 5 '$CONTAINER_NAME' >/dev/null 2>&1" || true
  fail "$1 had only $2 GiB memory left while loading (MEM_FLOOR_GIB $MEM_FLOOR_GIB): both ranks stopped; lower CONTEXT, or stop other work there"
}
until curl -sf --max-time 5 "$URL/health" >/dev/null 2>&1; do     # /health answers without an API key, once loaded
  if ! running_here; then
    docker logs "$CONTAINER_NAME" 2>&1 | grep -qiE 'OutOfMemory|out of memory|CUDA_ERROR_OUT_OF_MEMORY' &&
      fail "rank 0 ran out of memory while loading a $(arg_value --context)-token window: lower CONTEXT, or free memory here"
    fail "rank 0 exited (code $(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME")) before it was ready"
  fi
  (( SECONDS - start < WAIT_TIMEOUT )) ||
    fail "not ready after ${WAIT_TIMEOUT}s (WAIT_TIMEOUT); the ranks are still running: docker logs -f $CONTAINER_NAME"
  a=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  (( a >= MEM_FLOOR_GIB )) || mem_stop "this Spark" "$a"
  if (( SECONDS - start >= next_beat )); then
    running_worker ||
      fail "rank 1 on $WORKER exited (code $(worker 1 docker inspect -f '{{.State.ExitCode}}' "$CONTAINER_NAME" 2>/dev/null || echo "?")) before the server was ready"
    read -r g wa < <(worker 1 "$(declare -f gpu_gib); echo \$(gpu_gib $CONTAINER_NAME) \$(awk '/MemAvailable/ {print int(\$2 / 1048576)}' /proc/meminfo)" 2>/dev/null || echo "? ?")
    [[ ! "${wa:-}" =~ ^[0-9]+$ ]] || (( wa >= MEM_FLOOR_GIB )) || mem_stop "the worker" "$wa"
    printf '  %s⋯ %ss elapsed, %s GiB on the GPU here (%s GiB free), %s on the worker (%s free)%s\n' "$D" \
      "$((SECONDS - start))" "$(gpu_gib "$CONTAINER_NAME")" "$a" "${g:-?}" "${wa:-?}" "$R"
    next_beat=$((next_beat + 15))
  fi
  sleep 3
done
kill $LOGS_PID 2>/dev/null || true
sleep 0.3
LOADED_S=$((SECONDS - start))
log "Server answered after ${LOADED_S}s"

# ---------------------------------------------------------------- 5. smoke test
# Thinking off and greedy, so that a short reply has text (the model thinks first otherwise); no text fails the start.
step 5 "Smoke test: one chat completion through both ranks"
SERVED=$(served_name || echo "$SERVED_NAME")
# (with TENSORFOLD_API_KEY set, its header goes to curl on stdin, never on the command line)
smoke_request() {
  local body
  body="{\"model\": \"$SERVED\", \"max_tokens\": 32, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}"
  if [[ -n "${TENSORFOLD_API_KEY:-}" ]]; then
    printf 'Authorization: Bearer %s\n' "$TENSORFOLD_API_KEY" |
      curl -s --max-time 180 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -H @- -d "$body"
  else
    curl -s --max-time 180 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$body"
  fi
}
if smoke=$(smoke_request |
           python3 -c 'import json,sys; r = json.load(sys.stdin); c = r["choices"][0]["message"].get("content") or ""; assert c.strip(); print(repr(c.strip()[:40]) + ",", r["usage"]["completion_tokens"], "tokens")' 2>/dev/null); then
  log "OK: $smoke"
else
  fail "the smoke test request failed (no reply text); the ranks are still running"
fi

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] || IP="$HOST"
printf '\n%s  ✔ %s is now LIVE! on port %s%s\n\n' "$G" "$SERVED" "$PORT" "$R"
cat <<EOF
    API      http://${IP:-<spark-address>}:$PORT/v1   (model: $SERVED)
    Loaded   in ${LOADED_S}s · ${here_gb} GiB was free here, ${there_gb} GiB on the worker
    Model    $MODEL_ID @ ${MODEL_REVISION:0:8} ($QUANT_LABEL, QUANT=$QUANT)
    Window   $(arg_value --context) tokens$( [[ "$TF_FLASHNEXT_YARN" != 0 ]] && echo " (YaRN x$TF_FLASHNEXT_YARN)") · $(arg_value --parallel) at once · drafts $( [[ "$DRAFTS" == 1 ]] && echo MTP || echo off) · thinking $( [[ "$THINKING" == 1 ]] && echo on || echo off)
    KV       $KV_DTYPE$( [[ "$KV_DTYPE" == fp8 ]] && echo " (lossy, ~1.8x the pool; KV_DTYPE=bf16 is exact)") · vision $( [[ "$VISION" == 1 ]] && echo "on (up to $TENSORFOLD_MAX_IMAGES images, $MAX_VIDEOS videos)" || echo off) · memory reserve $TENSORFOLD_MEMORY_RESERVE_GIB GiB
    Logs     docker logs -f $CONTAINER_NAME   (rank 1: ssh $WORKER docker logs -f $CONTAINER_NAME)
    Restart  ./start.sh restart
    Stop     ./stop.sh

EOF
