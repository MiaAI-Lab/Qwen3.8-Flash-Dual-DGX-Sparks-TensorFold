# Credits

This repository is a thin layer of scripts (and, later, patches). Almost everything that makes it work was built by
others. Its own work is licensed under the Apache License 2.0 ([`LICENSE`](LICENSE)); [`NOTICE`](NOTICE) carries the
third-party notices that go with it (TensorFold's MIT and Apache-2.0 notices, Zig, the checkpoint's licenses).

## Model

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** by the
  [Qwen team](https://huggingface.co/Qwen) (Alibaba): the model's design, training and evaluations, its chat template
  and its MTP head. Its license, the
  [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), comes with the
  weights.
- **[NVIDIA](https://huggingface.co/nvidia)**: the checkpoint this recipe serves,
  [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (NVFP4 routed experts,
  the FP8 n-gram table and MTP experts, bf16 elsewhere), quantized by NVIDIA with
  [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (ModelOpt). Governed by the
  [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
  The weights are not part of this repository; `scripts/prepare.sh` downloads them from Hugging Face.
- **azampatti** ([azampatti](https://huggingface.co/azampatti)): the INT4-AR checkpoint served with `QUANT=int4ar`,
  [`azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound):
  its authors cut the routed experts from top-10 to top-5, healed the cut by distilling the shared expert, and
  published the checkpoint. Under the license on its model card (the Qwen license).
- **Intel** ([Intel](https://huggingface.co/Intel)): the AutoRound int4 quantization the INT4-AR checkpoint is built on,
  [`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound).
- **Saren-Arterius** ([Saren-Arterius](https://github.com/Saren-Arterius)): the hybrid checkpoint the INT4-AR
  checkpoint is built from and its FP8 n-gram table,
  [`Saren/Qwen3.8-Flash-Next-ple-table-fp8`](https://huggingface.co/Saren/Qwen3.8-Flash-Next-ple-table-fp8) (in its
  `ple-table/`). The recipe uses their weights only, no code from their repositories.

## Inference engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** and its Zig engine (`tensorfold-native`, branch
  [`zig-flashnext`](https://github.com/ashhart/TensorFold/tree/zig-flashnext)) by Ash Hart
  ([ashhart](https://github.com/ashhart)) and the TensorFold contributors (Apache 2.0 from v0.6.0; code written before
  v0.6.0 keeps its MIT notice): the engine that serves the model, its CUDA runtime and kernels, MTP drafting with
  exact verification and the OpenAI-compatible server. `scripts/prepare.sh` builds it from TensorFold's source at the
  commit pinned in `scripts/config.sh`.
  The Flash Next CUDA engine in this recipe is ported from TensorFold's Python Flash Next CUDA engine, written by
  Ash Hart and the TensorFold contributors ([contributors](https://github.com/ashhart/TensorFold/graphs/contributors));
  their authorship is recorded in TensorFold's history.
- **The Zig CUDA serving path and the CUDA family registry** were authored by Jürgen Schmied
  ([jschmied](https://github.com/jschmied)) in [TensorFold PR #443](https://github.com/ashhart/TensorFold/pull/443)
  (commit [`59e77e8`](https://github.com/ashhart/TensorFold/commit/59e77e8f4b875ce0e863a8c896fc8e424bc539ac)):
  `tensorfold-native` serving CUDA families through one registry, which this recipe's engine runs on.
- TensorFold itself builds on, and credits in its
  [third-party notices](https://github.com/ashhart/TensorFold/blob/zig-flashnext/THIRD_PARTY_NOTICES.md), the
  projects whose code it adapts.
- **Image and video input.** TensorFold's vision code (the Python `vision/` package and the Flash Next CUDA vision
  path) is by Ash Hart and the TensorFold contributors ([contributors](https://github.com/ashhart/TensorFold/graphs/contributors)).
  Wiring it into this engine's server is adapted from the patches 0008 and 0009 of MiaAI-Lab's own single-Spark recipe,
  [Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) (authored by MiaAI-Lab); the settings and their names follow that recipe's "Images and video".
- **FP8 KV cache.** The FP8 KV format (`--kv-dtype fp8`) is adapted from the patch `0038-glm-kv-fp8` of MiaAI-Lab's own
  GLM recipe, [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) (authored by MiaAI-Lab).
- **[b12x](https://github.com/local-inference-lab/b12x)** by local-inference-lab (Apache-2.0): the one-shot RoCE
  all-gather (`TF_FLASHNEXT_ROCE`) implements the RoCEnante protocol of b12x by local-inference-lab; our
  implementation is new code written for this engine. It carries the ranks' small exchanges over both CX7 rails.
- **[Zig](https://ziglang.org)** by the Zig Software Foundation and the Zig contributors (MIT): the language and
  compiler TensorFold's Zig engine is written in and built with (0.17.0).

## Recipe

- **[321sssrt-bit](https://github.com/321sssrt-bit)**: reported, diagnosed and fixed the context-admission bug
  ([issue #1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
  [PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)): the admission change in
  the engine patches, its tests, and `tools/context_boundary.py` are theirs.
- The scripts (`start.sh`, `stop.sh`, `scripts/`) and the checks in `tools/` (`client.py`, `needle.py`,
  `toolcheck.py`, `prompt_reuse.py`; `long_context.py` and `exact.py` are new) are MiaAI-Lab's, adapted from
  MiaAI-Lab's own two-Spark recipe
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (the link and rail detection, the NCCL settings, the image and checkpoint setup, the checks), developed with
  [Claude Code](https://claude.com/claude-code), under the Apache License 2.0.

## Runtime stack

- **[NVIDIA PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)**
  (`nvcr.io/nvidia/pytorch:26.07-py3`), the base of the image, with NVIDIA's CUDA (nvcc builds the engine's kernels),
  NCCL and related libraries. Governed by the NVIDIA Software License Agreement and the Product-Specific Terms for
  NVIDIA AI Products.
- **[NCCL](https://github.com/NVIDIA/nccl)** (BSD-3-Clause) and **[rdma-core](https://github.com/linux-rdma/rdma-core)**
  (libibverbs, GPL-2.0 / BSD-2-Clause): the two ranks' exchanges over the Sparks' ConnectX-7 RoCE link.
- **[Hugging Face transformers](https://github.com/huggingface/transformers)** (Apache 2.0): the Qwen vision tower and
  image processor the vision helper runs (`transformers==5.17.0`, installed in the image).
- **[PyAV](https://github.com/PyAV-Org/PyAV)** (BSD-3-Clause) with **[FFmpeg](https://ffmpeg.org/)** (LGPL): video
  decoding. **[Pillow](https://python-pillow.org/)** (MIT-CMU): image decoding (from the PyTorch container).
- **[Triton](https://github.com/triton-lang/triton)** (MIT): the language the engine's Triton kernels are written in;
  the engine replays their compiled cubins.
- **[Hugging Face Hub](https://huggingface.co/)**: model hosting, the `hf` CLI and `huggingface_hub` (Apache 2.0), and
  the [safetensors](https://github.com/huggingface/safetensors) format (Apache 2.0) the checkpoint ships in.
- **[Docker](https://www.docker.com/)** and the
  **[NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)** (Apache 2.0): running the server
  on the GPU in a container.

## Hardware

- **[NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)** (GB10 Grace Blackwell,
  128 GB unified memory), two of them linked by their ConnectX-7 ports.

If you believe something here is missing or credited wrongly, please open an issue.
