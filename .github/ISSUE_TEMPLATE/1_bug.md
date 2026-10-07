---
name: Bug report
about: The model kit is not behaving as expected, is producing an error, or the numbers do not match the README.
title: ""
labels: "bug"
assignees: ""
---

<!-- Thank you for using this model kit!

     If you are looking for support, please check the README and scripts/config.sh first,
     or reach out on X:
      * https://x.com/MiaAI_lab

     If you have found a bug, then fill out the template below.
-->

---

## Environment

<!-- Fill in what applies to your setup. The README's "Performance" and
     "Configuration" sections list the settings that affect behavior and the defaults
     shipped in scripts/config.sh. -->

- Hardware: <!-- e.g. 2x DGX Spark (GB10, 128 GB unified memory each), one QSFP cable between the CX7 ports -->
- Memory available before launch, on both Sparks: <!-- `free -g`; other GPU workloads running? -->
- Image: <!-- `docker images | grep tensorfold-qwen38fn`, or the ghcr.io tag you pulled -->
- Image labels: <!-- `docker image inspect -f '{{json .Config.Labels}}' tensorfold-qwen38fn:<tag>` (tf.patches, tf.kernels) -->
- `start.sh` invocation: <!-- e.g. `./start.sh`, or `PARALLEL=2 ./start.sh restart --max-tokens 16384` -->
- Changed settings: <!-- environment, scripts/local.sh or .env: PARALLEL, CONTEXT, MAX_TOKENS, THINKING, DRAFTS, WORKER_WEIGHTS, TENSORFOLD_* / TF_FLASHNEXT_* -->
- Startup lines: <!-- the first lines of `docker logs qwen38-fn-tf 2>&1` after NVIDIA's banner -->

---

## Steps to Reproduce

<!-- Please include full steps so that we can reproduce the problem. -->

1. Run `./start.sh` <!-- describe any overrides and what it printed up to the failure -->
2. ... <!-- describe steps to demonstrate the bug -->
3. ... <!-- for example "a reply stops in the middle of a sentence" -->

**Expected results:** <!-- what did you expect to happen? -->

**Actual results:** <!-- what did you actually see happen? -->

---

### Additional context

Add any other context here: a minimal request that reproduces bad output, JSON
responses, `docker inspect` output, and so on.

<details>
<summary>Minimal reproduction sample</summary>

<!--
      If the bug is about model output or API behavior, attach a minimal reproducible
      request below between the lines with the backticks.

      NOTE: the model thinks before it answers, in "reasoning_content" rather than
      "content". Budget enough max_tokens (~2,000 for anything non-trivial): with a
      small budget the reply is often still in its reasoning and "content" comes back
      empty on a perfectly healthy server. That is not a bug.

      To tell an engine bug from model behavior, send the same request with
      "draft": false (the engine's serial reference) and a fixed "seed": drafted and
      serial replies must be identical.
-->

```bash
curl -s http://<head-address>:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-Flash-Next",
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 2000
  }'
```

</details>

<details>
  <summary>Logs</summary>

<!--
      Paste the log output below between the backticks, and mention whether it came
      from `start.sh`, `scripts/prepare.sh`, `docker logs qwen38-fn-tf` (rank 0, on the
      head), `ssh <worker> docker logs qwen38-fn-tf` (rank 1), a saved log from
      ~/.cache/tensorfold-qwen38fn/logs, or a client.

      Common culprits worth checking before filing:
        * "this start's memory budget holds a N-token window" -> another GPU workload is
          using memory on one of the Sparks (start.sh retries once with the window that fits).
        * start.sh warns "only N GiB memory available here" -> stop other GPU
          containers on that Spark first (`docker ps`).
        * "no RoCE device or RoCE v2 GID for the link" -> WORKER is reached over another
          network: set FABRIC_PEER to the worker's CX7 address.
        * `start.sh` refuses port 8888 -> something else listens there; set PORT.
        * "has no kernel set" -> the image was built without kernels/sm121 (README:
          "What start.sh and scripts/prepare.sh do").
        * prepare.sh fails applying a patch -> TF_REF was changed; the patches are
          made for the commit pinned in scripts/config.sh.
-->

```

```

</details>
