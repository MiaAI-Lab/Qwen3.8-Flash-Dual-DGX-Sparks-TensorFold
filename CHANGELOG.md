# Changelog

Every change to this recipe, newest first. Each release names the image it serves: `scripts/prepare.sh` pulls
`ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold` by the digest pinned in `scripts/config.sh`.

## Image prompt reuse

`patches/0010-image-prompt-reuse.patch`. A prompt that contains a picture is kept and resumed.
The cache key mixes each picture row with a hash of its features, so a different picture does not
reuse the old rows, and the engine refills only the rows past the resume point. Text prompts are
unchanged. `python3 tools/image_prompt_reuse.py 12k 3` checks it: a follow-up with the same picture
should resume at least 85% of the prompt, and the same words with different pixels should not.


## v1.0 (2026-10-07): Qwen3.8-Flash-Next on two DGX Sparks with TensorFold's Zig engine

Image: `ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold:zig-db28187-57a020c8d666`
(`sha256:ab25d95979709261a2ac1cd50d9d14e8e29f775c784a4629230358ad94ffefab`): TensorFold `zig-flashnext` at `db281878`
with `patches/0001`-`0009` (the engine at `zig-next` `20e709a`), Zig 0.17.0, on NVIDIA's PyTorch 26.07 container
(`PULL=0` builds it locally instead). The image build compiles the engine's Triton kernel set itself (no GPU, no checkpoint), byte for byte the gated set.

### The engine (`patches/`)
- **TensorFold's Flash Next CUDA engine ported to the Zig engine** (`tensorfold-native`): bit-exact against TensorFold's
  Python engine on one Spark; on two Sparks (tensor parallel, one rank a Spark, NCCL over both CX7 rails, a one-shot
  RoCE all-gather for small exchanges, L2 prefetch during them) both ranks compute the same bits.
- **Two checkpoints** (`QUANT`): NVIDIA's NVFP4 (default) and azampatti's INT4-AutoRound (GPTQ int4 experts and head,
  block-FP8 projections), one image and one kernel set for both.
- **1,048,576-token window** (YaRN factor 4 over the native 262,144).
- **Up to 16 requests at once** in shared rounds, each request's reply equal to its own solo run; requests that arrive
  together prefill in one pass; the first token goes out before the first drafts.
- **Drafts:** the checkpoint's MTP head with a running-product stop (asking deep while few requests run, confidence
  0.5 above) and copy drafts from the prompt; drafted replies equal serial ones.
- **FP8 KV cache** (`--kv-dtype fp8`): ~1.8-1.9x the KV pool, lossy (98.8% top-1 agreement with bf16); bf16 stays
  available and exact. The format is adapted from MiaAI-Lab's GLM recipe patch 0038-glm-kv-fp8.
- **Image and video input** (`--vision`): TensorFold's vision helper and the checkpoint's vision tower, up to 50 images
  and 4 videos a request, with FP8 or bf16 KV. Adapted from MiaAI-Lab's single-Spark recipe patches 0008 and 0009;
  TensorFold's vision code is by Ash Hart and the TensorFold contributors.
- **`tool_choice: "required"`** and a named function: the reply always opens a tool call.
- **Prompt reuse:** a conversation's next turn resumes from its kept state.
- **The served prompt path as fast as the engine's own:** the kept state is copied after the first token, the page
  cache is dropped after load, fused hyper-connection kernels, a faster long-prompt indexer.
- **A KV pool sized by the engine** from the memory free after load, less `TENSORFOLD_MEMORY_RESERVE_GIB` and the
  vision workspace; caches grow on demand inside it.

### The recipe
- **`./start.sh` and `./stop.sh`** for two Sparks, the wrapper of MiaAI-Lab's GLM-5.3-Flash two-Spark recipe: setup on
  first run, the checks before anything is stopped (the arguments, the link, the previous server, the port, free
  memory), rank 1 on the worker then rank 0 here, the loading progress, a smoke test, the LIVE message;
  `FOREGROUND=1`, `DRY_RUN=1`, logs saved on stop.
- **Defaults** (`scripts/config.sh`): `QUANT=nvfp4`, `CONTEXT=1048576`, `PARALLEL=16`, `KV_DTYPE=fp8`, `VISION=1`
  (50 images, 4 videos, 16,384 image and 16,384 video tokens a request, a 2 GiB tower workspace),
  `TENSORFOLD_MEMORY_RESERVE_GIB=10`, 32,768-token replies, thinking on, drafts on; checkpoints pinned (`fc694b54`,
  `1464274`).
- **Memory checks:** `start.sh` and `prepare.sh` refuse a Spark below the memory floor (`MEM_FLOOR_GIB`, 10 GiB) or
  below what the chosen window needs (`MEM_NEED_GIB`); `MEM_CHECK=0` warns instead.
- **`scripts/prepare.sh`**: both Sparks checked, the image (pulled by a pinned digest once published, else built from
  TensorFold's source with `patches/*.patch`), copied to the worker and checked identical by content, the chosen
  checkpoint downloaded at its pin, verified, and copied to the worker or read over NFS (`WORKER_WEIGHTS=nfs`).
- **`scripts/publish-image.sh`**: the image to GHCR (`<TensorFold commit>-<image hash>` and `latest`).
- **Checks** in `tools/`: `exact.py`, `toolcheck.py`, `needle.py`, `long_context.py`, `prompt_reuse.py`, `client.py`.
