## Description and Motivation

<!--

    Please write a description of what this PR is changing, removing or adding, and why.
    Consider including before/after comparisons.

    For this kit, a good description usually covers:
      * which setting in scripts/config.sh, script behavior or patch changes
      * whether the change affects measured numbers (prefill/decode throughput, TTFT,
        concurrent requests, each rank's memory) and in which direction
      * for a patch: whether it keeps the same bits, and that drafted replies still
        equal serial ones

-->

## Related Issues

<!--

    Add the list of issues related to this PR from the [issue tracker](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues).
    Indicate which of these issues are resolved or fixed by this PR, like #XXXX, where XXXX is the issue number.

-->

---

## Testing

<!--

    Tell us how you verified this change. For this kit that usually means:

      * `bash -n start.sh stop.sh scripts/*.sh` (syntax check)
      * `shellcheck start.sh stop.sh scripts/*.sh` if available
      * `DRY_RUN=1 ./start.sh` (both ranks' docker commands, nothing started)
      * `scripts/prepare.sh` (a patch change rebuilds the image on the head and copies it
        to the worker; every patch must apply with `git apply` to TensorFold's source at
        the commit pinned in scripts/config.sh)
      * an actual launch on two Sparks, plus `docker logs qwen38-fn-tf 2>&1 | head -50`
      * sparkDash (https://github.com/MiaAI-Lab/sparkDash) for speed
      * if behavior changed, the measured numbers with the new settings, stating
        which configuration they came from (see README "Performance")

    If you changed a patch, confirm drafted replies still equal the engine's serial
    reference: the same request with "draft": false and a fixed "seed" (or
    temperature 0) must give the same reply. Say whether the patch keeps the same
    bits as before (and measure quality if it does not).

-->

---

## Checklist:

<!--

    Thanks for contributing to Mia's AI Lab!

    Before you file this pull request, please follow the items on this checklist and
    put an x in each of the boxes, like this: [x].

-->

- [ ] I have read the README and `scripts/config.sh` and kept my changes consistent with them.
- [ ] My pull request has a sound title and description (not something vague like `Update README.md`).
- [ ] My change is reproducible and verified (script syntax check, `DRY_RUN=1 ./start.sh`, `scripts/prepare.sh`, a launch, or a re-measurement).
- [ ] A patch change keeps drafted replies equal to serial ones, and I said how I checked it (and whether its bits change).
- [ ] I updated the README and/or `scripts/config.sh` if a setting, default, or measured number changed.
- [ ] If my change affects memory, I checked both ranks still fit the default 4 x 1,048,576-token configuration.
- [ ] Defaults in `scripts/config.sh` still work out of the box; a new setting has a sane fallback like the existing ones.
