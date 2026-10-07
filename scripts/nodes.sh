# The two Sparks: this machine (rank 0), and the worker over ssh (WORKER, rank 1), and the link between them: a direct
# cable between their ConnectX-7 ports, found from the route to the worker (detect_link).
# Sourced by start.sh, stop.sh and prepare.sh after config.sh.

worker_host() { echo "$WORKER"; }

# check_workers: WORKER is set, and WORKER_WEIGHTS is copy or nfs.
check_workers() {
  [[ -n "$WORKER" ]] || die "WORKER is not set: put WORKER=user@<worker address> in scripts/local.sh (see scripts/local.sh.example)"
  [[ "$WORKER_WEIGHTS" =~ ^(copy|nfs)$ ]] || die "WORKER_WEIGHTS is copy or nfs, not $WORKER_WEIGHTS"
}

# DRY_RUN=1 goes on without a worker it cannot reach: its values print as <WORKER:...>.
WORKER_DOWN=""
# worker <i> <command...>: run a command on the worker (rank 1; i is always 1). Key-based ssh only, no prompts.
worker() {
  shift
  [[ -z "$WORKER_DOWN" ]] || return 255
  ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$WORKER" "$@"
}

# save_log <dir> <rank> <container> <keep>: the container's log (stdout and stderr, with timestamps) gzipped into
# <dir>/<date>-<time>-rank<N>.log.gz before the container is removed (docker rm deletes it), keeping the newest <keep>
# of that rank; prints the file. Runs on the worker too (its definition is sent over ssh): no config.sh there.
save_log() {
  local dir=$1 rank=$2 c=$3 keep=$4 stamp f i=0
  [[ "$keep" =~ ^[0-9]+$ ]] && (( keep > 0 )) || return 0
  docker inspect "$c" >/dev/null 2>&1 || return 0
  mkdir -p "$dir" || return 1
  stamp=$(date +%Y%m%d-%H%M%S); f="$dir/$stamp-rank$rank.log.gz"
  while [[ -e "$f" ]]; do f="$dir/$stamp.$((++i))-rank$rank.log.gz"; done   # same second: still sorts as newer
  if ! docker logs --timestamps "$c" 2>&1 | gzip > "$f"; then rm -f "$f"; return 1; fi
  ls -1 "$dir"/*-rank"$rank".log.gz 2>/dev/null | { grep -vxF "$f" || true; } | LC_ALL=C sort -r | tail -n +"$keep" | xargs -r rm -f --
  echo "$f"
}
# worker_save_log 1: the same, for rank 1 on the worker (into ~/.cache/tensorfold-qwen38fn/logs there)
worker_save_log() {
  worker 1 "$(declare -f save_log); save_log \"\$HOME/.cache/tensorfold-qwen38fn/logs\" 1 '$CONTAINER_NAME' '$LOG_KEEP'"
}

need_worker() {
  [[ -n "$WORKER" ]] || die "WORKER is not set: put WORKER=user@<worker address> in scripts/local.sh (see scripts/local.sh.example)"
  worker 1 true 2>/dev/null && return 0
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    warn "DRY_RUN: cannot ssh to $WORKER (WORKER): its values print as <WORKER:...>"
    WORKER_DOWN=1
    return 0
  fi
  die "cannot ssh to $WORKER without a password: set up key-based ssh (ssh-copy-id $WORKER)"
}
need_workers() { need_worker 1; }

# The worker's Hugging Face cache: its own HF_HOME, else ~/.cache/huggingface there. prepare.sh copies the checkpoint
# into it and start.sh mounts it into rank 1. WORKER_HF_CACHE wins: a worker whose cache is not its HF_HOME (e.g. a
# shared models disk).
worker_hf_cache() {
  if [[ -n "$WORKER_HF_CACHE" ]]; then echo "$WORKER_HF_CACHE"; return; fi
  [[ -z "$WORKER_DOWN" ]] || { echo "<WORKER:hf-cache>"; return 0; }
  worker 1 'echo "${HF_HOME:-$HOME/.cache/huggingface}"'
}

# Weights over NFS: rank 1 mounts the head's HF_CACHE read-only through the docker volume NFS_VOLUME on the worker, from
# nfs_server: NFS_SERVER, else the head's address on the link (detect_link). ensure_nfs_volume creates it (or checks
# the one there names the same export); worker_nfs <cmd...> runs a command in a throwaway container on the worker with
# the volume at /hf.
nfs_server() { echo "${NFS_SERVER:-${HEAD_ADDR:-}}"; }
ensure_nfs_volume() {
  local server have
  server=$(nfs_server)
  [[ -n "$server" ]] || die "no NFS server address for $WORKER: set NFS_SERVER to the head's address on the link"
  have=$(worker 1 "docker volume inspect -f '{{index .Options \"device\"}} {{index .Options \"o\"}}' '$NFS_VOLUME'" 2>/dev/null || true)
  if [[ -z "$have" ]]; then
    worker 1 "docker volume create --driver local --opt type=nfs --opt device=':$NFS_PATH' \
      --opt o='addr=$server,nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600' '$NFS_VOLUME'" >/dev/null ||
      die "could not create the docker volume $NFS_VOLUME on $WORKER"
  elif [[ "${have%% *}" != ":$NFS_PATH" || "$have" != *"addr=$server,"* ]]; then
    die "$WORKER's docker volume $NFS_VOLUME is ${have%% *} from ${have#* }, not :$NFS_PATH from $server: remove it there (docker volume rm $NFS_VOLUME) or set NFS_VOLUME / NFS_PATH / NFS_SERVER"
  fi
}
worker_nfs() { worker 1 "docker run --rm --network none --entrypoint '$1' -v '$NFS_VOLUME:/hf:ro' '$IMAGE' $(printf '%q ' "${@:2}")"; }

# link_info <peer address>: this node's side of the link to <peer>: "<address> <netdev> <rdma device> <gid index>".
# The netdev and source address come from the route to the peer; the RDMA device from sysfs; the GID index is the
# RoCE v2 entry that holds this node's IPv4 address (an all-zero or v1 entry makes NCCL fail about a minute in).
link_info() {
  local peer=$1 route dev src hca idx type gid want
  route=$(ip -o -4 route get "$peer" 2>/dev/null) || return 1
  dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
  src=$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$route")
  hca=$(ls /sys/class/net/"$dev"/device/infiniband 2>/dev/null | head -1)
  idx=""
  if [[ -n "$hca" ]]; then
    want=$(printf '0000:0000:0000:0000:0000:ffff:%02x%02x:%02x%02x' $(tr '.' ' ' <<<"$src"))
    for f in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
      gid=$(cat "$f" 2>/dev/null) || continue
      [[ "$gid" == "$want" ]] || continue
      type=$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"${f##*/}" 2>/dev/null)
      [[ "$type" == *"v2"* ]] && { idx=${f##*/}; break; }
    done
  fi
  echo "$src $dev ${hca:--} ${idx:--}"
}

# rails <netdev> <gid index>: the RoCE devices this node can run the link on - the link's own device first, then every
# other RoCE device whose RoCE v2 IPv4 GID sits at the same index: one on the link's subnet (a second CX7 port, when it
# is up and addressed there), and the second PCIe link of the same QSFP port (a DGX Spark's "twin", which lives in its
# own subnet, so the subnet scan below cannot see it).
rails() {
  local dev=$1 gid=$2 net cidr hcas other ip idx
  cidr=$(ip -o -4 addr show dev "$dev" | awk '{print $4}' | head -1)
  net=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "$cidr")
  hcas=$(ls /sys/class/net/"$dev"/device/infiniband | head -1)
  for n in /sys/class/net/*; do
    other=${n##*/}
    [[ "$other" == "$dev" || ! -d $n/device/infiniband || "$(cat $n/operstate 2>/dev/null)" != up ]] && continue
    ip=$(ip -o -4 addr show dev "$other" | awk '{print $4}' | head -1)
    [[ -n "$ip" ]] || continue
    python3 -c "import ipaddress,sys; sys.exit(ipaddress.ip_interface(sys.argv[1]) not in ipaddress.ip_network(sys.argv[2]) and ipaddress.ip_interface(sys.argv[1]).ip not in ipaddress.ip_network(sys.argv[2]))" "$ip" "$net" || continue
    idx=$(ls $n/device/infiniband | head -1)
    [[ "$(cat /sys/class/infiniband/$idx/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$idx"
  done
  # A DGX Spark's QSFP port reaches the GB10 over two independent PCIe Gen5 x4 links, so one cabled port shows up as
  # two netdevs and two RoCE devices (rocep1s0f0 and roceP2p1s0f0), and NVIDIA's own two-Spark playbook gives the two
  # twins different subnets - so the scan above never sees the twin and NCCL gets one x4: ~112 Gb/s of the port's 200.
  # Pair the twin by name instead of by subnet: same fN/npM tail, different PCIe prefix. Same conditions as above
  # (link up, a RoCE v2 GID at $gid), and never add a device twice.
  # (only for Spark-style names, <prefix>f<N>np<M>; the check below compares whole comma-separated names, so a twin the
  # subnet scan already added, as when both twins share the link's subnet, is not listed twice)
  local tail=${dev##*f} other2 ib2
  [[ "$dev" =~ f[0-9]+np[0-9]+$ ]] || { echo "$hcas"; return 0; }
  for n in /sys/class/net/*; do
    other2=${n##*/}
    [[ "$other2" == "$dev" || "$other2" != *f"$tail" ]] && continue
    [[ -d $n/device/infiniband && "$(cat $n/operstate 2>/dev/null)" == up ]] || continue
    ib2=$(ls $n/device/infiniband | head -1)
    [[ -n "$ib2" && ",$hcas," != *",$ib2,"* ]] || continue
    [[ "$(cat /sys/class/infiniband/$ib2/ports/1/gid_attrs/types/$gid 2>/dev/null)" == *v2* ]] || continue
    hcas+=",$ib2"
  done
  echo "$hcas"
}

# The same function on the worker (its definition is sent over ssh).
worker_link_info() { worker 1 "$(declare -f link_info); link_info $2"; }

# An image's identity by content (its layers' diffIDs and runtime config), the same under Docker's overlay2 and
# containerd image stores: .Id is the config digest under one and the manifest digest under the other, so it never
# matches across a mixed pair. The template holds no spaces: worker() passes it through ssh, which re-splits
# arguments. A missing image is "missing". worker_image_ident 1 <image>: on the worker.
IMAGE_IDENT='{{.RootFS.Layers}}{{.Config.Env}}{{.Config.Entrypoint}}{{.Config.Cmd}}{{.Config.WorkingDir}}'
image_ident() { local s; s=$(docker image inspect -f "$IMAGE_IDENT" "$1" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing; }
worker_image_ident() {
  local s; s=$(worker 1 docker image inspect -f "$IMAGE_IDENT" "$2" 2>/dev/null) && sha256sum <<<"$s" | cut -c1-64 || echo missing
}

# cx7_peer: the worker's address on a CX7 port that one of this node's CX7 ports reaches directly (same subnet, no
# gateway), for a worker given by its LAN address. Empty when there is none.
cx7_peer() {
  local a route dev
  for a in $(worker 1 'for d in /sys/class/net/*; do [ -d "$d/device/infiniband" ] && ip -o -4 addr show dev "${d##*/}"; done' 2>/dev/null |
             awk '{print $4}' | cut -d/ -f1); do
    route=$(ip -o -4 route get "$a" 2>/dev/null) || continue
    [[ "$route" == *" via "* ]] && continue
    dev=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
    [[ -d /sys/class/net/$dev/device/infiniband ]] && { echo "$a"; return 0; }
  done
  return 1
}

# HEAD_ADDR / HEAD_DEV / HEAD_HCA / HEAD_GID and WORKER_ADDR / WORKER_DEV / WORKER_HCA / WORKER_GID: the link the two
# ranks talk over (NCCL and the rendezvous); HEAD_HCAS / WORKER_HCAS: every rail of it (both PCIe twins of the cabled
# port, a second cabled port). FABRIC_PEER overrides the worker's link address when WORKER is reached over another
# network. DRY_RUN with a worker out of reach: the worker's side prints as <WORKER:...>.
detect_link() {
  local peer=${FABRIC_PEER:-${WORKER#*@}} cx7 v
  # a host name (an /etc/hosts alias of the CX7 address, say) is resolved first: ``ip route get`` takes addresses only
  if [[ ! "$peer" =~ ^[0-9]+(\.[0-9]+){3}$ ]]; then
    peer=$(getent ahostsv4 "$peer" | awk 'NR == 1 {print $1}') || true
    [[ -n "$peer" ]] || die "cannot resolve ${FABRIC_PEER:-${WORKER#*@}} to an IPv4 address"
  fi
  read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  [[ -n "${HEAD_ADDR:-}" ]] || die "no route from this node to $peer"
  # WORKER given by a LAN address (the route goes out a port without RoCE): use the worker's CX7 address instead
  if [[ -z "${FABRIC_PEER:-}" && "${HEAD_HCA:--}" == "-" ]] && cx7=$(cx7_peer); then
    log "$peer is reached over $HEAD_DEV, not a CX7 port: using the worker's CX7 address $cx7 for the link (FABRIC_PEER)"
    peer=$cx7
    read -r HEAD_ADDR HEAD_DEV HEAD_HCA HEAD_GID <<<"$(link_info "$peer")" || true
  fi
  HEAD_HCAS=$(rails "$HEAD_DEV" "$HEAD_GID")
  if [[ -n "$WORKER_DOWN" ]]; then
    WORKER_ADDR="<WORKER:address>"; WORKER_DEV="<WORKER:netdev>"; WORKER_HCA="<WORKER:hca>"; WORKER_GID="<WORKER:gid>"
    WORKER_HCAS="<WORKER:hcas>"
  else
    read -r WORKER_ADDR WORKER_DEV WORKER_HCA WORKER_GID <<<"$(worker_link_info 1 "$HEAD_ADDR")" || true
    [[ -n "${WORKER_ADDR:-}" ]] || die "the worker has no route back to $HEAD_ADDR"
    # every rail (both PCIe twins of the cabled port, a second cabled port): a chunk's all-gather is ~1.8x faster on two
    WORKER_HCAS=$(worker 1 "$(declare -f rails); rails $WORKER_DEV $WORKER_GID")
    [[ "$(tr ',' '\n' <<<"$HEAD_HCAS" | wc -l)" == "$(tr ',' '\n' <<<"$WORKER_HCAS" | wc -l)" ]] ||
      { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  fi
  [[ "${NCCL_RAILS:-2}" == 1 ]] && { HEAD_HCAS=$HEAD_HCA; WORKER_HCAS=$WORKER_HCA; }
  for v in HEAD_HCA HEAD_GID WORKER_HCA WORKER_GID; do
    [[ "${!v}" != "-" ]] || die "no RoCE device or RoCE v2 GID for the link ($v): is $HEAD_DEV / $WORKER_DEV the CX7 port between the Sparks? Set FABRIC_PEER to the worker's CX7 address"
  done
  MASTER_ADDR=${MASTER_ADDR:-$HEAD_ADDR}
}

# The NCCL settings of a rank's container (docker -e arguments): the RoCE link's netdev, RoCE devices (both rails) and
# GID index, 4 channels. Measured between two Sparks in the GLM-5.3 recipe this comes from: a decode-sized all-gather
# 32 us (80 with NCCL's defaults) and a 32 MB one 1.9 ms on two rails (3.5 on one). With this engine's exchanges (fp32
# all-gathers on two rails): a decode row 19.5 us, 16 rows 34 us, a 2,048-row prompt chunk (21 MB a rank) 1.18 ms.
nccl_env() {
  local dev=$1 hcas=$2 gid=$3
  echo "-e NCCL_SOCKET_IFNAME=$dev -e NCCL_IB_HCA=$hcas -e NCCL_IB_GID_INDEX=$gid" \
       "-e NCCL_MIN_NCHANNELS=${NCCL_CHANNELS:-4} -e NCCL_MAX_NCHANNELS=${NCCL_CHANNELS:-4}" \
       ${NCCL_DEBUG:+-e NCCL_DEBUG=$NCCL_DEBUG}
}
rank_nccl_env() {
  if (( $1 == 0 )); then nccl_env "$HEAD_DEV" "$HEAD_HCAS" "$HEAD_GID"
  else nccl_env "$WORKER_DEV" "$WORKER_HCAS" "$WORKER_GID"; fi
}
