#!/usr/bin/env bash
# Stop the server that ./start.sh started and remove its containers on both Sparks, freeing their GPU memory: here
# (rank 0) and on the worker (WORKER, rank 1).
# The ranks get STOP_TIMEOUT seconds (default 30) to shut down; requests still running are cut off (it does not drain
# them), so stop.sh says when there are any. Each rank's log is saved first (docker rm deletes it), gzipped, in LOG_DIR
# here (~/.cache/tensorfold-qwen38fn/logs) and ~/.cache/tensorfold-qwen38fn/logs on the worker; the newest LOG_KEEP
# (10) of each rank stay.
# Usage: ./stop.sh      Env: CONTAINER_NAME, PORT, WORKER (see scripts/config.sh), STOP_TIMEOUT, LOG_DIR, LOG_KEEP
#   DRY_RUN=1 ./stop.sh   says what it would stop, and on which Spark, and stops nothing
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
source ./scripts/nodes.sh

# Where the containers are: here (rank 0) and on the worker (rank 1)
here=0; docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME" && here=1
there=0
if [[ -z "$WORKER" ]]; then
  warn "WORKER is not set (scripts/local.sh): rank 1 was left alone"
elif ! worker 1 true 2>/dev/null; then
  warn "cannot reach the worker ($WORKER) over ssh: rank 1 was left alone"
elif worker 1 "docker ps -a --format '{{.Names}}' | grep -qx '$CONTAINER_NAME'"; then
  there=1
fi
if (( here == 0 && there == 0 )); then
  log "No container named $CONTAINER_NAME on either Spark: nothing to stop"
  exit 0
fi
where="on both Sparks"; (( there )) || where="here"; (( here )) || where="on the worker"

# Requests in flight: the Zig server's /health "live" part counts every open request (running or queued) as
# "connections" ("waiting" is the queued share of it). With API keys on, /health names nothing: no count, no warning.
if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; then
  api_host="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && api_host=127.0.0.1
  [[ "$api_host" == *:* ]] && api_host="[$api_host]"
  busy=$(curl -s --max-time 3 "http://$api_host:$PORT/health" 2>/dev/null |
         python3 -c 'import json,sys; l = json.load(sys.stdin).get("live") or {}; print(int(l.get("connections") or 0))' 2>/dev/null || echo 0)
  (( busy == 0 )) || warn "$busy request(s) still running will be cut off"
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  log "DRY_RUN=1: would stop and remove $CONTAINER_NAME $where (up to ${STOP_TIMEOUT:-30}s); nothing is stopped"
  if (( here )); then printf '[dry-run] rank 0 here:\n  docker stop -t %s %s; (its log saved to %s); docker rm -f %s\n' "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" "$LOG_DIR" "$CONTAINER_NAME"; fi
  if (( there )); then printf '[dry-run] rank 1 on %s:\n  docker stop -t %s %s; (its log saved to ~/.cache/tensorfold-qwen38fn/logs there); docker rm -f %s\n' "$WORKER" "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" "$CONTAINER_NAME"; fi
  exit 0
fi
log "Stopping $CONTAINER_NAME $where (up to ${STOP_TIMEOUT:-30}s)"

# each rank: stop, save its log (with the shutdown lines), then remove the container
stop_here() {
  local saved
  docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER_NAME" >/dev/null 2>&1 || true
  saved=$(save_log "$LOG_DIR" 0 "$CONTAINER_NAME" "$LOG_KEEP") || warn "could not save rank 0's log to $LOG_DIR"
  docker rm -f "$CONTAINER_NAME" >/dev/null
  log "Stopped and removed rank 0 here${saved:+; its log: $saved}"
}
stop_worker() {
  local saved
  worker 1 "docker stop -t ${STOP_TIMEOUT:-30} '$CONTAINER_NAME' >/dev/null 2>&1" || true
  saved=$(worker_save_log) || warn "could not save rank 1's log on $WORKER"
  worker 1 "docker rm -f '$CONTAINER_NAME' >/dev/null" || { warn "could not stop rank 1 on $WORKER"; return 1; }
  log "Stopped and removed rank 1 on $WORKER${saved:+; its log there: $saved}"
}
wpid=""
if (( there )); then stop_worker & wpid=$!; fi
if (( here )); then stop_here; fi
failed=0
[[ -z "$wpid" ]] || wait "$wpid" || failed=1
(( failed == 0 )) || exit 0                             # a warning above said what is left
log "Stopped and removed $CONTAINER_NAME $where; its GPU memory is free again"
