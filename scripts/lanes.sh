# shellcheck shell=bash
# Compute lanes (ENGINE=lanes, README "Compute lanes"): the lane engine's image, built on top of the default image.
# Sourced by scripts/prepare.sh after the default image ($MIA_IMAGE: TensorFold plus ./patches and the kernel set) is
# ready. ENGINE=patches (the default) never reads this file, so the default image and its hash are unchanged.
#
# The lane engine is upstream TensorFold's Zig engine at LANES_REF (LANES_REPO, unmodified) with the patches in lanes/
# applied by git am in lanes/series order: this recipe's patches ported onto it (TF_FLASHNEXT_ENGINE=mia serves
# MiaAI-Lab's Flash Next engine), plus the compute lane and the in-model runner. The build fails unless the patched
# tree is the one lanes/series pins. Its image keeps everything of $MIA_IMAGE (NVIDIA's PyTorch container, the
# kernel set, the vision helper's files) and replaces only tensorfold-native, then adds:
#   - tf-runner-frontend-check, and clang-18 / lld-18 (Ubuntu's): the runner's RV32 frontend, checked at build time;
#   - the runner's circuit (calc.tfpc) and graph bank (graph-v7-powmod, MIT notice included), from the release
#     RUNNER_ASSETS_URL at RUNNER_ASSETS_SHA256, every file checked against the engine's packaging/runner/assets.sha256.
# The build fails unless the runner computes the engine's probe cells on that circuit.

# lanes_hash: what the lane image is built from (this file's Dockerfile, the default image's hash, the lane engine's
# base commit and patches, the runner assets, Zig); a change rebuilds the lane image
lanes_hash() {
  ( sed -n "/<<'LANES_DOCKERFILE'/,/^LANES_DOCKERFILE\$/p" scripts/lanes.sh
    cat lanes/series && sha256sum lanes/*.patch
    echo "$(image_hash) $LANES_REPO $LANES_REF $RUNNER_ASSETS_URL $RUNNER_ASSETS_SHA256 zig-$ZIG_VERSION $ZIG_SHA256"
  ) | sha256sum | cut -c1-12
}

# lanes_image: build $IMAGE (the lane image) from $MIA_IMAGE unless it is already built from the same inputs
lanes_image() {
  [[ "$LANES_REF" =~ ^[0-9a-f]{40}$ ]] ||
    die "ENGINE=lanes needs LANES_REF, the full commit sha in $LANES_REPO the lane patches apply to (README: Compute lanes)"
  [[ "$LANES_TREE" =~ ^[0-9a-f]{40}$ ]] || die "lanes/series pins no tree"
  local hash built base
  hash=$(lanes_hash)
  built=$(docker image inspect -f '{{index .Config.Labels "tf.lanes"}}' "$IMAGE" 2>/dev/null || true)
  if [[ $REBUILD -eq 0 && "$built" == "$hash" ]]; then
    log "Lane image $IMAGE already built ($hash, engine ${LANES_REF:0:12}; use --rebuild to force)"
    return 0
  fi
  base=$(docker image inspect -f '{{.Id}}' "$MIA_IMAGE")
  log "Building the lane image $IMAGE on $MIA_IMAGE (TensorFold ${LANES_REF:0:12} from $LANES_REPO + lanes/, tree ${LANES_TREE:0:12}, lanes $hash)"
  local nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  # build context lanes/ (the patches and their series); the Dockerfile fetches TensorFold at LANES_REF, applies them,
  # and fetches the runner assets at their sha256
  docker build "${nocache[@]}" -t "$IMAGE" --build-arg MIA_IMAGE="$base" \
    --build-arg ZIG_VERSION="$ZIG_VERSION" --build-arg ZIG_SHA256="$ZIG_SHA256" \
    --build-arg LANES_REPO="$LANES_REPO" --build-arg LANES_REF="$LANES_REF" --build-arg LANES_TREE="$LANES_TREE" \
    --build-arg RUNNER_ASSETS_URL="$RUNNER_ASSETS_URL" --build-arg RUNNER_ASSETS_SHA256="$RUNNER_ASSETS_SHA256" \
    --build-arg LANES_HASH="$hash" -f - lanes <<'LANES_DOCKERFILE'
ARG MIA_IMAGE
FROM ${MIA_IMAGE} AS build
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz
ARG LANES_REPO
ARG LANES_REF
RUN git init -q /lanes && cd /lanes && git remote add origin "${LANES_REPO}" && \
    git fetch -q --depth 1 origin "${LANES_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${LANES_REF}"
COPY . /lanes-patches/
RUN cd /lanes && for p in $(awk '!/^#/ && NF && $1 != "base" && $1 != "tree"' /lanes-patches/series); do \
      git -c user.name=lanes -c user.email=lanes@localhost -c core.autocrlf=false am --quiet --keep-cr "/lanes-patches/$p" || exit 1; \
    done && test "$(git rev-parse 'HEAD^{tree}')" = "$(awk '$1 == "tree" {print $2}' /lanes-patches/series)"
# ReleaseFast for sm_121, stripped, host code for this Spark's own CPU (as her engine is built: the load repacks the
# checkpoint on the CPU); the runner frontend's boot check
RUN cd /lanes && /opt/zig/zig build native \
      -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast -Dstrip=true --prefix /opt/lanes -j"$(nproc)" && \
    /opt/zig/zig build runner-frontend-check --prefix /opt/lanes -j"$(nproc)" && mkdir -p /opt/lanes-out && \
    for f in tensorfold-native tf-runner-frontend-check; do \
      p=$(find /opt/lanes -type f -name "$f" | head -1); test -x "$p" && cp "$p" /opt/lanes-out/ || exit 1; done
ARG RUNNER_ASSETS_URL
ARG RUNNER_ASSETS_SHA256
RUN curl -fsSL -o /tmp/assets.tar.gz "${RUNNER_ASSETS_URL}" && \
    echo "${RUNNER_ASSETS_SHA256}  /tmp/assets.tar.gz" | sha256sum -c - && \
    mkdir -p /opt/runner && tar -xzf /tmp/assets.tar.gz -C /opt/runner --no-same-owner --no-same-permissions && \
    cd /opt/runner && grep -v '^#' /lanes/packaging/runner/assets.sha256 | sha256sum -c --strict --quiet -

FROM ${MIA_IMAGE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-18 lld-18 && \
    ln -sf clang-18 /usr/bin/clang && ln -sf ld.lld-18 /usr/bin/ld.lld && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/lanes-out/tensorfold-native /opt/tensorfold/native/bin/tensorfold-native
COPY --from=build /opt/lanes-out/tf-runner-frontend-check /opt/tensorfold/bin/tf-runner-frontend-check
COPY --from=build /opt/runner/calc.tfpc /opt/tensorfold/share/runner/calc.tfpc
COPY --from=build /opt/runner/graph-v7-powmod/bank/ /opt/tensorfold/share/runner/graph-v7-powmod/bank/
COPY --from=build /opt/runner/graph-v7-powmod/bank/Qwenputer-MIT.txt /opt/tensorfold/licenses/runtime-deps/Qwenputer-MIT.txt
COPY --from=build /lanes/LICENSE /lanes/NOTICE /lanes/THIRD_PARTY_NOTICES.md /opt/tensorfold/share/doc/tensorfold-lanes/
COPY --from=build /lanes/LICENSES/ /opt/tensorfold/share/doc/tensorfold-lanes/LICENSES/
RUN /opt/tensorfold/bin/tf-runner-frontend-check --require --circuit /opt/tensorfold/share/runner/calc.tfpc
ARG LANES_REF
ARG LANES_TREE
ARG LANES_HASH
LABEL tf.patches=${LANES_HASH} tf.lanes=${LANES_HASH} tf.lanes.ref=${LANES_REF} tf.lanes.tree=${LANES_TREE}
LANES_DOCKERFILE
  docker run --rm --network none --entrypoint /opt/tensorfold/bin/tf-runner-frontend-check "$IMAGE" --require ||
    die "$IMAGE: the runner frontend check failed"
  log "Lane image $IMAGE: TensorFold ${LANES_REF:0:12} + lanes/ (tree ${LANES_TREE:0:12}), runner circuit and graph bank, lanes $hash"
}
