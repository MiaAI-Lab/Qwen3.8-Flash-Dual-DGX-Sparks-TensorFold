<h1 align="center">Qwen3.8-Flash-Next on two DGX Sparks with TensorFold</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve **Qwen3.8-Flash-Next** from two NVIDIA DGX Sparks (GB10, 128 GB each, linked by their ConnectX-7 ports) through
an OpenAI-compatible API, with a **1,048,576-token context** (Qwen's YaRN over the
native 262,144). It runs [TensorFold](https://github.com/ashhart/TensorFold)'s Zig engine, `tensorfold-native`
(branch [`zig-flashnext`](https://github.com/ashhart/TensorFold/tree/zig-flashnext)), on both Sparks (one rank on
each) in NVIDIA's PyTorch container, with no Python engine in the serving path (Python only decodes images and video).

Up to **16 requests at once**, an **FP8 KV cache** (about 3.9M tokens of KV with NVFP4, 4.6M with INT4-AR), **image
and video input**, tool calls (including `tool_choice: "required"`), and two checkpoints to choose from: NVIDIA's
NVFP4 or the faster INT4-AutoRound ([Choose a quant](#choose-a-quant)).

**Credits.** [TensorFold](https://github.com/ashhart/TensorFold) and its Zig engine are by Ash Hart
([ashhart](https://github.com/ashhart)) and the [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors), branch [`zig-flashnext`](https://github.com/ashhart/TensorFold/tree/zig-flashnext).
The Zig CUDA serving path and the CUDA family registry this recipe runs on were authored by Jürgen Schmied
([jschmied](https://github.com/jschmied)) in [TensorFold PR #443](https://github.com/ashhart/TensorFold/pull/443)
(commit [`59e77e8`](https://github.com/ashhart/TensorFold/commit/59e77e8f4b875ce0e863a8c896fc8e424bc539ac)).
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is by the Qwen team (Alibaba); the NVFP4
checkpoint, [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), is by NVIDIA,
quantized with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (ModelOpt). The INT4-AR
checkpoint ([`QUANT=int4ar`](#choose-a-quant)) is by azampatti ([azampatti](https://huggingface.co/azampatti)), who
authored its top-5 expert cut, the shared-expert healing and the checkpoint,
[`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound);
its AutoRound int4 quantization is by Intel
([`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)), and
its hybrid checkpoint and FP8 n-gram table are by Saren-Arterius ([Saren-Arterius](https://github.com/Saren-Arterius),
[`Saren/Qwen3.8-Flash-Next-ple-table-fp8`](https://huggingface.co/Saren/Qwen3.8-Flash-Next-ple-table-fp8)).
The one-shot RoCE all-gather (`TF_FLASHNEXT_ROCE`) implements the RoCEnante protocol of
[b12x](https://github.com/local-inference-lab/b12x) by local-inference-lab (Apache-2.0); our implementation is new code
written for this engine. Image and video input is adapted from the patches 0008 and 0009 of MiaAI-Lab's own single-Spark recipe, [Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) (by MiaAI-Lab), and runs TensorFold's vision code, authored by Ash Hart and the TensorFold contributors; the FP8 KV cache format is adapted from MiaAI-Lab's GLM recipe patch 0038-glm-kv-fp8 ([GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)). [Zig](https://ziglang.org) is by the Zig Software Foundation and the Zig contributors. More in
[Credits](#credits) and
[`CREDITS.md`](CREDITS.md).

- Checkpoint ([choose a quant](#choose-a-quant)): [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
  at revision `fc694b54` by default: routed experts in NVFP4, the n-gram table and the MTP experts in FP8, bf16
  elsewhere (~124 GiB); or INT4-AR (`QUANT=int4ar`, ~122 GiB)
- Drafts: the checkpoint's own MTP head, plus copy drafts from the prompt (`DRAFTS`); drafted replies equal serial ones
- API model id: `Qwen3.8-Flash-Next` (`Qwen3.8-Flash-Next-INT4-AR` with `QUANT=int4ar`)
- Context: **1,048,576 tokens** a request (YaRN factor 4), up to 16 requests at once; the KV cache is FP8 by default (lossy, see
  [KV cache](#kv-cache-fp8-by-default)), and images and video are accepted ([Images and video](#images-and-video))
- One command on the first Spark: `./start.sh` sets up both Sparks and starts both ranks; `./stop.sh` stops them

## Performance

Measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) (its own prompts and protocol) through the OpenAI API
on two Sparks, 2026-10-07.

**Decode, prose**, at the defaults (16 requests at once, FP8 KV, image input on, a 1,048,576-token window, MTP and copy
drafts). Tokens a second: all requests together, per request, and the time to first token.

| Requests at once | NVFP4 | per request | TTFT | INT4-AR | per request | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 66.8 | 66.8 | 101 ms | 89.3 | 89.3 | 76 ms |
| 2 | 97.1 | 50.2 | 190 ms | 135.0 | 73.7 | 113 ms |
| 3 | 125.5 | 45.0 | 179 ms | - | - | - |
| 4 | 150.8 | 40.3 | 262 ms | 220.2 | 58.8 | 167 ms |
| 8 | 221.3 | 30.9 | 344 ms | 284.9 | 43.0 | 288 ms |
| 16 | - | - | - | 483.6 | 35.1 | 461 ms |

**Prefill** (tokens a second, and the time to first token), measured earlier the same day (bf16 KV, 8 requests at
once); this release's prompt path is faster still.

| Prompt | NVFP4 | TTFT | INT4-AR | TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 4k tokens | 2,648 | 1.56 s | 3,244 | 1.27 s |
| 8k tokens | 2,597 | 3.17 s | 3,303 | 2.49 s |
| 16k tokens | 2,757 | 5.96 s | 3,246 | 5.06 s |
| 32k tokens | 2,575 | 12.74 s | 3,285 | 9.99 s |
| 64k tokens | 2,511 | 26.12 s | 3,110 | 21.09 s |
| 128k tokens | 2,601 | 50.40 s | 2,982 | 43.96 s |
| 256k tokens | 2,458 | 106.69 s | 2,673 | 98.09 s |

A ~1M-token prompt (998,986 tokens, a needle in a haystack, NVFP4) prefilled in 1,309 s with an earlier build and the
needle was found.

## Choose a quant

One image serves both; `prepare.sh` downloads only the one you choose. Set `QUANT` in `scripts/local.sh` or `.env`,
or pass `./start.sh --quant int4ar` (`./start.sh restart --quant ...` switches a running server).

| `QUANT` | Checkpoint | What it is | Speed here |
| --- | --- | --- | --- |
| `nvfp4` (default) | [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) @ `fc694b54`, ~124 GiB | the full model: all 10 routed experts a token, 6.0B active parameters | prose 66.8 tok/s alone, 221.3 for 8 requests; prefill ~2,500-2,750 tok/s |
| `int4ar` | [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound) @ `1464274`, ~122 GiB | 5 routed experts a token instead of 10, with a healed shared expert, 4.8B active: faster, and about 10% lower general capability on its authors' own harness (46.6-47.6 against the original's 51.8; tool use 91 against 91-92), per its model card | prose 89.3 tok/s alone, 284.9 for 8 requests, 483.6 for 16; prefill ~2,700-3,300 tok/s |

Speeds from the sparkDash runs above ([Performance](#performance)). Both serve the same API, window and drafts; the replies differ between them (different weights), each exact against
its own serial decoding.

## Requirements

- **Two DGX Sparks** (or two GB10 systems with 128 GB unified memory), with nothing else large on their GPUs. Free
  memory each needs at start: ~102 GiB `MemAvailable` for the default 1,048,576-token window (~72 GiB at 262,144;
  ~4 GiB less with `QUANT=int4ar`);
  `start.sh` stops before launching below it (`MEM_NEED_GIB`). Stop other GPU work first.
- **A direct ConnectX-7 link:** a QSFP cable between the CX7 ports and an IPv4 address on each end in one private
  subnet (e.g. `192.0.2.1/24` and `192.0.2.2/24`; `ping` must work), with a RoCE v2 GID (`start.sh` checks). One QSFP
  port of a Spark reaches the GB10 over two PCIe Gen5 x4 links, so it appears as two netdevs and two RoCE devices
  (`enp1s0f0np0` / `enP2p1s0f0np0`, `rocep1s0f0` / `roceP2p1s0f0`). Address both twins on each Spark, in the link's
  subnet or each in its own (NVIDIA's two-Spark playbook gives them different ones): both are then used as NCCL rails,
  as is a second cabled port in the link's subnet. One rail is one x4 (~112 Gb/s of the port's 200).
- **Key-based ssh** from the first Spark (the head, which runs `./start.sh` and the API) to the second (the worker):
  `ssh-copy-id user@<worker>` (after `ssh-keygen -t ed25519` if you have no key); check with
  `ssh -o BatchMode=yes user@<worker> true`.
- Docker with the NVIDIA container runtime, your user in the `docker` group, and `rsync`, on both Sparks.
- **Disk, on each Spark:** ~150 GB: the checkpoint (124 GiB, ~133 GB; INT4-AR ~122 GiB) under `~/.cache/huggingface` and the image under
  Docker's root. `prepare.sh` asks for what the download still needs (files already in the cache, from an earlier
  revision say, do not count; 140 GB, `MIN_FREE_GB`, when it cannot ask the Hub what is missing) and 35 GB under
  Docker's root (`IMAGE_FREE_GB`; the sum when they share a filesystem), and checks the worker for what the copy must
  send. With `WORKER_WEIGHTS=nfs` the worker needs only the image ([Worker weights over NFS](#worker-weights-over-nfs)).
- Optional: the `hf` CLI on the head (faster download) and a Hugging Face token (`~/.cache/huggingface/token` or
  `HF_TOKEN`; the checkpoint is public).

## Quick start

On the head:

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold.git
cd Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold
cp scripts/local.sh.example scripts/local.sh     # set WORKER=user@192.0.2.2 in it (and FABRIC_PEER, see below)
./start.sh
```

`WORKER` is the worker's ssh target. The ranks talk over the route to it: if it is on another network than the link,
set `FABRIC_PEER` to the worker's CX7 address. The worker needs no copy of this repository. The example sets
`WORKER_WEIGHTS=nfs`, the two-Spark setup this recipe is built around: the worker reads the head's checkpoint over
NFS, which needs the head to export its Hugging Face cache once ([Worker weights over NFS](#worker-weights-over-nfs));
`WORKER_WEIGHTS=copy` (the default without `scripts/local.sh`) copies the checkpoint to the worker instead.

The first run sets up both Sparks (see below): the image on each (pulled from GitHub Container Registry, or built on the head with `PULL=0`),
the checkpoint (~124 GiB) downloaded on the head and read by the worker over NFS (or copied to it). Later starts load half of the weights on
each Spark (about two minutes). `start.sh` shows each step and the server's log, runs a smoke test through both ranks, and
prints `Qwen3.8-Flash-Next is now LIVE! on port 8888` with the endpoint.

Any OpenAI client works with `base_url = "http://<head-address>:8888/v1"` and the model `Qwen3.8-Flash-Next`. The
model thinks before it answers (`reasoning_content`), so give replies enough `max_tokens`.

```bash
curl -s http://<head-address>:8888/v1/models
curl -s http://<head-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Write a Python fibonacci function."}],
  "max_tokens": 2000
}'

./start.sh restart                                       # restart both ranks, e.g. after changing a setting
./stop.sh                                                # stop both ranks and free their GPU memory
docker logs -f qwen38-fn-tf                              # rank 0's log (here)
ssh <worker> docker logs -f qwen38-fn-tf                 # rank 1's log (why it exited, if it did)
curl -s http://<head-address>:8888/health                # status, memory, live streams
```

**Logs of earlier runs.** `docker rm` deletes a container's log, so `stop.sh` (and `start.sh`, before it removes a
stopped container left from an earlier run, a crashed one say) first saves each rank's log, stdout and stderr with
timestamps, gzipped, and prints where: `~/.cache/tensorfold-qwen38fn/logs/<date>-<time>-rank0.log.gz` on the head
(`LOG_DIR`) and `~/.cache/tensorfold-qwen38fn/logs/<date>-<time>-rank1.log.gz` on the worker. The newest 10 of each
rank are kept (`LOG_KEEP`; `0` saves none). Read one with `zcat` or `zless`; attach both when you report a crash.

If `start.sh` stops at a check:

| Message | What to do |
| --- | --- |
| `only N GiB memory available` | other GPU work runs on that Spark: stop it (`docker ps`) and restart |
| `no RoCE device or RoCE v2 GID for the link` / `no route from this node to ...` | the route to the worker does not go over the CX7 port: set `FABRIC_PEER` to the worker's CX7 address and check both ends are addressed (`ip -4 addr`) |
| `the worker's .../hub is not writable` | a container left it root-owned: fix its ownership on the worker |
| `... has no kernel set` | the image was built without the engine's compiled kernels: see [the kernel set](#what-startsh-and-scriptsprepares-do) |

## What `start.sh` and `scripts/prepare.sh` do

**`./start.sh`** works in five steps, each shown as it runs:

1. **Setup:** `scripts/prepare.sh`, when the setup is not ready on both Sparks (first run, new patches or kernels,
   another model, revision, image or worker).
2. **Checks:** the arguments (the engine's own parser, in a throwaway container), the link, the previous server
   (stopped on `restart`, or when only one rank is up; a stopped container left from an earlier run is removed, its
   log saved first), the port, free memory (it stops below `MEM_NEED_GIB` on either Spark, see
   [KV pool and memory](#kv-pool-and-memory)).
3. **Launch:** rank 1 on the worker over ssh, then rank 0 and the API here, with the settings from `scripts/config.sh`:
   `tensorfold-native serve <snapshot> --name ... --host ... --port ... --context N --parallel N --kv-dtype ...
   [--vision ...] [--no-drafts] --tp 2 --rank R --master <head's link address> --master-port 29551`, with NCCL on the
   link's RoCE devices.
4. **Loading:** rank 0's log as it comes, and every 15 s the elapsed time and how much is on each Spark's GPU; if a
   rank stops, both ranks' last log lines and why. If a Spark's free memory falls below `MEM_FLOOR_GIB` (10 GiB)
   while the ranks load, it stops both before the machine runs out.
5. **Smoke test:** one chat completion through both ranks, thinking off, then the LIVE message and the endpoint.

A running server is left alone; `./start.sh restart` stops it only after the setup and argument check pass, so a
typo leaves it running (running requests are cut off; `stop.sh` warns). Extra arguments go to `tensorfold-native
serve` on both ranks after the defaults, so they win (`./start.sh restart --max-tokens 16384`); `./start.sh --help`
lists the settings. `FOREGROUND=1 ./start.sh` stays attached to rank 0's log and exits with its exit code (for a
systemd unit), without the progress lines, the memory watch or the smoke test; when either rank ends, it stops the
other one. `DRY_RUN=1 ./start.sh` prints both ranks' `docker run` and the link found, and stops or starts nothing.

**`scripts/prepare.sh`** does the one-time setup, and is safe to re-run (each step skips work already done):

1. Preflight on both Sparks: Docker, the GPU, `rsync`, key-based ssh, the RoCE link, disk space.
2. The image `tensorfold-qwen38fn:zig-db28187` on the head: TensorFold's Zig engine built from source (the commit in
   `TF_REF`, branch `zig-flashnext`, with every `patches/*.patch` applied by `git apply`) with Zig 0.17.0 (checked
   against its sha256), `zig build -Doptimize=fast fatbins install native` into `/opt/tensorfold`, on NVIDIA's
   `nvcr.io/nvidia/pytorch:26.07-py3`. Only the build reaches the image, with TensorFold's license files. It first
   pulls the published image `ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold:zig-db28187-<image hash>`,
   by the digest pinned in `scripts/config.sh` (`IMAGE_TAG` / `IMAGE_DIGEST`, this release's image) while the image hash is this release's (the patches, the kernel set and the Dockerfile); otherwise, or
   with `PULL=0`, it builds.
3. The same image on the worker: pulled, else streamed from the head (`docker save | docker load`), checked identical.
4. The checkpoint, downloaded into `~/.cache/huggingface` on the head at its pinned revision, checked (every shard its
   index names; a ModelOpt checkpoint of `qwen4_exp` for `nvfp4`, GPTQ int4 with its `ple-table/` for `int4ar`), then copied to the worker with `rsync` over ssh and checked
   file by file. Both resume.

**The kernel set.** The engine replays compiled Triton kernels (`aot.json` and `cubins/`) from
`TENSORFOLD_CUDA_KERNELS`, `/opt/tensorfold/share/tensorfold/cuda/sm121` in the image. The image build compiles every
specialization the engine launches, on one GPU and on two ranks, for sm_121 without a GPU or the checkpoint
(TensorFold's `tools/zig/flashnext_aot.py`), byte for byte the kernels the reference run compiled: the checkout sits
at `/tensorfold` and the kernels' Python sources get the modification time they had then (`KERNEL_SOURCE_MTIME`),
both of which Triton writes into each cubin's line table; the build stops if a captured kernel comes out different.

```bash
scripts/prepare.sh             # set up both Sparks without starting the server
scripts/prepare.sh --rebuild   # rebuild the image from scratch
PREPARE=1 ./start.sh restart   # force prepare.sh, then restart; PREPARE=0 skips the check
```

After changing `patches/` or the kernel set, `scripts/publish-image.sh` pushes the new image to GitHub Container
Registry (`latest` and `zig-db28187-<image hash>`, the tag `prepare.sh` looks for) and prints the digest to pin.

## KV pool and memory

Qwen3.8-Flash-Next keeps full attention caches only in its sparse attention layers and fixed-size recurrent states in
its Gated DeltaNet layers; each rank holds half of the attention and DeltaNet heads, so half of their caches. The engine
sizes its KV pool itself when it loads: the `MemAvailable` left after the weights, less `TENSORFOLD_MEMORY_RESERVE_GIB`
(10 GiB) and, with images on, the vision workspace (`TENSORFOLD_VISION_WORKSPACE_MIB`, 2 GiB). The caches then grow
inside that pool as requests need them, so a Spark's used memory rises with the conversations it holds and stops about
10 GiB short of full. A request is admitted when its prompt plus its reply budget (`max_tokens`) fits beside the
others; one that does not fit is refused.

At the defaults, measured on two Sparks (per rank; the pool is the whole server's, as each rank holds half of every
token's cache):

| | NVFP4 | INT4-AR |
| --- | ---: | ---: |
| Requests at once (`PARALLEL`) | 16 | 16 |
| Window per request (`CONTEXT`) | 1,048,576 tokens | 1,048,576 tokens |
| The engine on its GPU after loading | ~69 GiB | ~65 GiB |
| KV pool (sequence memory) | 26.5 GiB | 31.0 GiB |
| KV pool in tokens (FP8, ~6.9 KB a token a rank) | ~3.9M | ~4.6M |
| Full 1M-token conversations at once | 3 | 4 |

The pool of a pair is the smaller of the two Sparks' (both ranks agree on it), so other work on either Spark shrinks
it. On the Spark's unified memory, running out tends to freeze the machine rather than fail an allocation, so
`start.sh` also checks before launching: a rank starts only where `MemAvailable` is at least `MEM_NEED_GIB` (by default
what a rank of the chosen `CONTEXT` holds under a prompt that fills it, measured, plus `MEM_FLOOR_GIB`; `MEM_CHECK=0`
turns the stop into a warning). A prompt's prefill holds extra memory while it runs (about 31 GiB a rank for a million
tokens, freed after). `CONTEXT=262144` serves the native window without YaRN and needs less.

## KV cache: FP8 by default

`KV_DTYPE=fp8` (the default, `--kv-dtype fp8`) stores each cached key and value in FP8 with a scale. It is **lossy**:
on teacher-forced replies its top-1 token agreed with a bf16 cache's 98.8% of the time (5,058 of 5,120), so a free-running
reply can diverge from the bf16 one, and the exactness guarantees of this recipe (drafted equals serial, concurrent
equals solo, the two ranks identical) hold for the FP8 cache against itself, not against bf16. In return the KV pool is
1.8-1.9x larger (measured at 8 requests, 1M window, two ranks: NVFP4 2,015,232 to 3,801,088 rows, INT4-AR 2,457,600 to
4,390,912), so more requests and longer ones fit. Speed is the same (prefill, and the time of a decode round). For the
exact cache: `KV_DTYPE=bf16 ./start.sh restart` (or `KV_DTYPE=bf16` in `scripts/local.sh`). Both ranks must agree, which
`start.sh` guarantees. Needle tests of 32k, 128k and 200k tokens were found by FP8 and bf16 alike.

## Images and video

`VISION=1` (the default) starts TensorFold's vision helper on rank 0: Pillow and PyAV decode the pictures and video
frames, the checkpoint's own 27-layer vision tower (0.84 GiB) turns them into tokens, and the engine takes them as it
takes text. Images and video work with the FP8 KV cache, drafts, and both quants. Send OpenAI-style content parts in a
user message:

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
    {"type": "text", "text": "What is in this picture?"}]}],
  "max_tokens": 2000
}'
```

A video is a `video_url` part:

```bash
VID=$(base64 -w0 clip.mp4)
curl -s http://<spark-address>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": [
    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,'"$VID"'"}},
    {"type": "text", "text": "Describe what happens in this video."}]}],
  "max_tokens": 2000
}'
```

| | Images | Videos |
| --- | --- | --- |
| Formats | JPEG, PNG, WebP | MP4, WebM, MOV, MKV (anything FFmpeg decodes) |
| Per request | up to 50 (`TENSORFOLD_MAX_IMAGES`; all of a chat's turns count) | up to 4 (`MAX_VIDEOS`) |
| Tokens | up to 16,384 for all images (`TENSORFOLD_IMAGE_TOKENS`), at most 4,096 an image (`"detail": "low"`: 256 an image) | 2 frames a second (at most 256 frames), up to 16,384 tokens a request (`TENSORFOLD_VIDEO_TOKENS`) |

By default only data URLs are accepted; `VISION_URLS=1` also lets the server fetch public `https://` URLs. Image and
video prompts are not kept for prefix reuse, so each turn of a chat with pictures processes them again. `VISION=0
./start.sh restart` serves text only and returns the helper's 2 GiB workspace (`TENSORFOLD_VISION_WORKSPACE_MIB`) to the KV pool.
These are the names and meanings of the single-Spark recipe's "Images and video". Timings on this engine on one Spark
(warm): one image (1,217 tokens) 1.0 s to first token, 10 images 2.3 s, a 30-second video (6,875 tokens)
5.2-5.6 s; the first use of a new size is slower.

## Worker weights over NFS

The documented two-Spark setup (`WORKER_WEIGHTS=nfs`, as in `scripts/local.sh.example`): the worker keeps no copy of
the checkpoint, and rank 1 reads the head's Hugging Face cache over NFS, read-only. With `WORKER_WEIGHTS=copy` (the
default without `scripts/local.sh`) the worker keeps its own copy instead (~124 GiB, copied over the link by
`prepare.sh`, and disk for it on the worker).

1. On the head, export the cache to the worker once (this needs root; the worker needs nothing installed):

   ```bash
   sudo apt install nfs-kernel-server
   echo "$HOME/.cache/huggingface <worker CX7 address>(ro,no_subtree_check)" | sudo tee -a /etc/exports
   sudo exportfs -ra
   ```

2. Put `WORKER_WEIGHTS=nfs` in `scripts/local.sh` or `.env`, and run `./start.sh restart`.

`prepare.sh` then creates a read-only docker NFS volume on the worker (`NFS_VOLUME`, default `qwen38fn-hf`, no sudo)
and checks that the worker sees every file of the snapshot as the head has it, instead of copying. The volume mounts
`:<NFS_PATH>` from `NFS_SERVER` with `nfsvers=4.2,ro,nconnect=8,rsize=1048576,wsize=1048576,hard,timeo=600`.
`NFS_PATH` is the exported path as the worker mounts it: by default the head's `HF_CACHE`, as the export above
names it; an NFSv4 export with `fsid=0` is mounted from its root, so set `NFS_PATH=/` (or the exported subpath below
that root, e.g. `NFS_PATH=/huggingface` for a pseudo-root that holds the cache as `huggingface`). `NFS_SERVER` is the
head's address the export allows (default: its address on the link).

## Configuration

Every setting lives in [`scripts/config.sh`](scripts/config.sh). Set one for a single run from the environment
(`PARALLEL=2 ./start.sh restart`), or keep it in `scripts/local.sh` (sourced as bash) or in a `.env` file next to
`start.sh` (plain `KEY=value` lines, read, never run); both files are yours, not the repository's. The first that
sets a value wins: the environment, then `scripts/local.sh`, then `.env`, then the default.

| Variable | Default | Meaning |
| --- | --- | --- |
| `WORKER` / `FABRIC_PEER` | empty | the worker's ssh target (`user@<address>` or `user@<host name>`), and its CX7 address when `WORKER` is on another network |
| `WORKER_HF_CACHE` | the worker's `HF_HOME` | the worker's Hugging Face cache, when it is not its `HF_HOME` (e.g. a shared models folder) |
| `MASTER_PORT` / `MASTER_ADDR` | `29551` / the head's link address | the ranks' rendezvous (`--master-port`, `--master`; keep it on the private link) |
| `QUANT` | `nvfp4` | the checkpoint: `nvfp4` (NVIDIA's NVFP4) or `int4ar` (INT4-AutoRound); also `./start.sh --quant int4ar` ([Choose a quant](#choose-a-quant)) |
| `CONTEXT` | `1048576` | prompt + reply window per request (`--context`); past 262,144 it needs YaRN |
| `TF_FLASHNEXT_YARN` | `4` past 262,144, else `0` | the YaRN factor (Qwen's: 4.0 over `original_max_position_embeddings` 262,144), read by both ranks |
| `PARALLEL` | `16` | requests decoded together in shared rounds (`--parallel`), 1 to 16; each stream's tokens equal its solo run |
| `MAX_TOKENS` | `32768` | the reply budget (reasoning and answer) of a request that sets no `max_tokens` (`--max-tokens`; the engine's own default is 4,096) |
| `THINKING` | `1` | think before answering by default (`--thinking`); `0` (`--no-thinking`) answers directly unless a request asks to think |
| `DRAFTS` | `1` | drafts from the checkpoint's MTP head; `0`: `--no-drafts`. Drafted replies equal serial ones either way |
| `KV_DTYPE` | `fp8` | the KV cache's dtype (`--kv-dtype`): `fp8` (lossy, ~1.8-1.9x the pool) or `bf16` (exact) ([KV cache](#kv-cache-fp8-by-default)) |
| `VISION` / `VISION_URLS` | `1` / `0` | image and video input (`--vision`; `0`: text only, 2 GiB back to the KV pool); `VISION_URLS=1` also fetches public `https://` URLs ([Images and video](#images-and-video)) |
| `TENSORFOLD_MAX_IMAGES` / `MAX_VIDEOS` | `50` / `4` | images and videos a request (`--vision-max-images`, `--vision-max-videos`) |
| `TENSORFOLD_IMAGE_TOKENS` / `TENSORFOLD_VIDEO_TOKENS` | `16384` / `16384` | image and video tokens a request (at most 4,096 an image) |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | `10` | the `MemAvailable` the engine keeps free when it sizes the KV pool; `start.sh` and `prepare.sh` refuse a Spark with less free (`MEM_FLOOR_GIB`, same default) |
| `MEM_NEED_GIB` / `MEM_CHECK` | measured per `CONTEXT` / `1` | the `MemAvailable` a Spark needs before its rank starts ([KV pool and memory](#kv-pool-and-memory)); `MEM_CHECK=0` warns instead of stopping |
| `WORKER_WEIGHTS` | `copy` | `copy`: the worker keeps its own copy of the weights; `nfs`: it reads the head's over NFS ([Worker weights over NFS](#worker-weights-over-nfs)); with `NFS_PATH`, `NFS_SERVER`, `NFS_VOLUME` |
| `SERVED_NAME` / `PORT` / `HOST` | `Qwen3.8-Flash-Next` / `8888` / `0.0.0.0` | the model id in `/v1/models` and replies; where the API listens |
| `PREPARE` / `PULL` | `auto` / `1` | `start.sh` runs `scripts/prepare.sh` when needed (`1` always, `0` never); `prepare.sh` tries the prebuilt image first (`0`: always build locally) |
| `WAIT_TIMEOUT` / `STOP_TIMEOUT` | `1800` / `30` | seconds `start.sh` waits for the server, and `stop.sh` gives it to shut down |
| `LOG_DIR` / `LOG_KEEP` | `~/.cache/tensorfold-qwen38fn/logs` / `10` | where `stop.sh` saves rank 0's log before it removes the container (rank 1's to the same folder on the worker), and how many of each rank's to keep ([Logs of earlier runs](#quick-start)); `0`: none |

Any `TENSORFOLD_*`, `TF_FLASHNEXT_*`, `TF_TP_*` or `TF_CUDA_*` variable is passed to both ranks (export it in
`scripts/local.sh`; `.env` lines are). `TENSORFOLD_API_KEY` (the server's API keys) goes to rank 0 by name only,
never on a command line.

**Revision pin.** The checkpoint is pinned to the revision this recipe is built for (`MODEL_REVISION`): both ranks
serve exactly it from the local cache, and a new upstream commit changes nothing until the pin does. Set it empty to
take the Hub's `main` when first downloaded.

Less common settings are described in `scripts/config.sh` and `scripts/nodes.sh`: `MODEL_ID`, `TF_REPO`, `TF_REF`,
`ZIG_VERSION` / `ZIG_SHA256`, `BASE_IMAGE` (after changing any of these run
`scripts/prepare.sh --rebuild`), `IMAGE`, `CONTAINER_NAME`, `GHCR_IMAGE`, `IMAGE_TAG` / `IMAGE_DIGEST`, `HF_CACHE`
(default `$HF_HOME` or `~/.cache/huggingface`), `KERNEL_CACHE`, `STATE_DIR`, `MIN_FREE_GB`, `IMAGE_FREE_GB`,
`MEM_FLOOR_GIB`, `KERNEL_SOURCE_MTIME`, `NCCL_RAILS` (`1`: one RoCE port even when the cabled port's two PCIe links, or a second port, are
up), `NCCL_CHANNELS` (4), `NCCL_DEBUG`, `RSYNC_OPTS`. `start.sh` also takes `HF_HUB_OFFLINE=0` (let the server reach
Hugging Face; by default it serves from the local cache only).

### Thinking and sampling

The checkpoint's own sampling defaults apply (its `generation_config.json`: temperature 1.0, top_p 0.95, top_k 20).
Per request:

- `temperature`, `top_p`, `top_k`, `min_p` and `seed` override them (`temperature: 0` decodes greedily).
- `reasoning_effort` and `chat_template_kwargs` (`enable_thinking`, `reasoning_effort`) go to the checkpoint's chat
  template; `"chat_template_kwargs": {"enable_thinking": false}` answers without thinking.
- The reasoning comes back in `reasoning_content`, the answer in `content`. A request without `max_tokens` gets
  32,768 tokens for both (`MAX_TOKENS`). An empty `content` means the model thought until `max_tokens`: give more.

These all work on the Flash Next TP=2 path; `"draft": false` gives the serial reference (`tools/exact.py`).

### API notes

- Endpoints of TensorFold's Zig server: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`
  (Anthropic) and `/v1/messages/count_tokens`, `/v1/models`, `/tokenize` and `/detokenize` (also under `/v1/`),
  `/health`, `/stats` and Prometheus `/metrics`.
- Refused with HTTP 400: `logprobs: true` / `top_logprobs` (no token probabilities) and `n` other than 1.
- `"draft": false` in a request serves it without drafts, TensorFold's serial reference.
- Tool calls work (`tools/toolcheck.py`: array arguments whole and streamed, a tool result turn), including
  `tool_choice: "required"` and a named function (Anthropic `tool_choice: {"type": "any"}` too): the reply then always
  opens a tool call.
- Prompt reuse: a conversation's next turn resumes from the kept state of its earlier turns instead of prefilling the
  whole prompt again (`tools/prompt_reuse.py`); image and video prompts are not kept.
- Structured outputs (`response_format`, the `guided_*` fields) are refused with HTTP 400: this engine does not enforce
  them yet.
- Images and video: see [Images and video](#images-and-video).

## Checks

**Outputs.** Drafts only propose: every drafted token is checked against the model's own keyed sample, so drafted
replies equal the engine's serial, one-token-at-a-time decoding (send `"draft": false` for that reference). The
two-rank engine is built to give the same bits on both ranks, and drafted == serial at two ranks; the checks that show
it: drafted == `"draft": false` == concurrent (`tools/exact.py`, also with ~100, 1k and 3k-token prompts), tool calls
(`tools/toolcheck.py`), and needles found at 200k and 1M tokens (`tools/needle.py`) and 26/26 exact at 128k and
256k (`tools/long_context.py`).

The checks in `tools/` talk to the running server (`API_URL`, default `http://127.0.0.1:8888`; or just `PORT`), from
the head or another machine, with Python's standard library only. Each exits 0 when it checked and passed, 1 when it
checked and failed, 2 when it could not run, 3 when it ran but could not verify (printed as UNCHECKED, never PASS).

| Script | What it does |
| --- | --- |
| `tools/needle.py [label] [size]` | hides a passphrase in ~`size` tokens of generated prose (default 200000; `1M` up to 1,048,576) and checks the reply returns it |
| `tools/long_context.py [--sizes 128k,256k,512k,1M] [--out FILE]` | needles at five depths and key = value retrieval over long documents, scored exactly; `--compare A.json B.json` sets two server settings side by side on identical prompts |
| `tools/toolcheck.py` | a tool call with an array argument comes back as a JSON array, streamed the same, and a tool result turn answers from it |
| `tools/prompt_reuse.py [size] [turns]` | a conversation takes more turns, each resuming at least 90% of its prompt from the kept state |
| `tools/exact.py` | the same greedy requests one at a time, with `"draft": false`, and all at once: the replies must be identical |
| `tools/client.py "message"` | one chat request (thinking off unless `--think`), for a quick look |

## Repository layout

```
start.sh      set up (first run) and start both ranks
stop.sh       stop them
scripts/      config.sh (all settings), local.sh.example (this setup's WORKER), prepare.sh (image + checkpoint on both
              Sparks), nodes.sh (ssh and the RoCE link), publish-image.sh (push the image to GHCR),
              banner.sh (start.sh's banner)
tools/        checks against the running server (needle, long context, tool calls, prompt reuse, exactness)
CHANGELOG.md  what changed in each release
CREDITS.md    who and what this builds on
LICENSE       Apache License 2.0
NOTICE        third-party notices (TensorFold's MIT and Apache-2.0 notices, Zig, the checkpoint's licenses)
```

`patches/` holds the Zig engine's Flash Next port (`0001`-`0009`: build and server flags, the CUDA kernels, config
and weights, the Triton replay, the engine, two ranks, serving, the kernel spec, TensorFold's Python vision helper), applied to TensorFold's source in
filename order by `prepare.sh` (`git apply` in the checkout's root); the kernel set is compiled in the image build.

## License

Apache License 2.0, see [`LICENSE`](LICENSE). [`NOTICE`](NOTICE) carries the third-party notices that go with it: the
image is built from TensorFold's source, and TensorFold's code (and any patch's diff context) stays under TensorFold's
licenses (Apache 2.0 from v0.6.0, and the MIT notice of code written before it, both in `NOTICE`; the image carries
TensorFold's license files). The model files are downloaded from Hugging Face and are not part of this repository:

- **The checkpoint** [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) is
  governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
  with the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) as
  additional information (its model card).
- **The INT4-AR checkpoint** [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
  (`QUANT=int4ar`) is under the license on its model card (the Qwen license).
- **The base model** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is under the license on its
  model card.

**Third-party software in the image.** The prebuilt image (and the one `scripts/prepare.sh` builds) is based on
NVIDIA's PyTorch container `nvcr.io/nvidia/pytorch:26.07-py3`, redistributed as a value-added runtime image. The NVIDIA
software in it is governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/)
and the [Product-Specific Terms for NVIDIA AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/),
which the container prints at every start; by pulling or running the image you accept them. The image also contains
huggingface_hub (Apache 2.0), Hugging Face transformers (Apache 2.0) and PyAV (BSD-3-Clause) with its bundled FFmpeg
libraries (LGPL) for the vision helper, and TensorFold's Python vision code (TensorFold's licenses). The Apache License
above covers this repository's own work only.

## Credits

Built on [TensorFold](https://github.com/ashhart/TensorFold) and its Zig engine by Ash Hart
([ashhart](https://github.com/ashhart)) and the [TensorFold contributors](https://github.com/ashhart/TensorFold/graphs/contributors)
(the Flash Next CUDA engine here is ported from TensorFold's Python Flash Next CUDA engine, written by them), branch [`zig-flashnext`](https://github.com/ashhart/TensorFold/tree/zig-flashnext);
the Zig CUDA serving path and the CUDA family registry, authored by Jürgen Schmied
([jschmied](https://github.com/jschmied)) in [TensorFold PR #443](https://github.com/ashhart/TensorFold/pull/443)
(commit [`59e77e8`](https://github.com/ashhart/TensorFold/commit/59e77e8f4b875ce0e863a8c896fc8e424bc539ac));
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team (Alibaba); the NVFP4 checkpoint
[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) by NVIDIA, made with
[NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer); the INT4-AR checkpoint
[`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
by azampatti ([azampatti](https://huggingface.co/azampatti)), who authored its top-5 expert cut and the
shared-expert healing, built on Intel's AutoRound int4 quantization
([`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound)) and on
the hybrid checkpoint and FP8 n-gram table by Saren-Arterius ([Saren-Arterius](https://github.com/Saren-Arterius),
[`Saren/Qwen3.8-Flash-Next-ple-table-fp8`](https://huggingface.co/Saren/Qwen3.8-Flash-Next-ple-table-fp8)); and [Zig](https://ziglang.org) by the Zig
Software Foundation and the Zig contributors. The one-shot RoCE all-gather (`TF_FLASHNEXT_ROCE`) implements the
RoCEnante protocol of [b12x](https://github.com/local-inference-lab/b12x) by local-inference-lab (Apache-2.0); our
implementation is new code written for this engine. Image and video input is adapted from the patches 0008 and 0009 of
MiaAI-Lab's [Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) and uses TensorFold's vision code by Ash Hart and the TensorFold contributors; the FP8 KV cache
format is adapted from MiaAI-Lab's GLM recipe patch 0038-glm-kv-fp8; image decoding and the tower use Hugging Face
[transformers](https://github.com/huggingface/transformers) (Apache-2.0), [PyAV](https://github.com/PyAV-Org/PyAV) / FFmpeg and Pillow. The scripts are MiaAI-Lab's, adapted from its
[GLM-5.3-Flash two-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold). The full
list, including the runtime stack and licenses, is in [`CREDITS.md`](CREDITS.md).
