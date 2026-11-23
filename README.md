# mini-inference-stack

<p align="center">
  <b>An educational project exploring how LLM inference works, how to measure its performance, and how GPU optimizations make it faster.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-NVIDIA%20Blackwell%20(sm__120)-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Blackwell Architecture" />
  <img src="https://img.shields.io/badge/CUDA-13.1-green?style=for-the-badge&logo=nvidia" alt="CUDA 13.1" />
  <img src="https://img.shields.io/badge/PyTorch-2.13.0%2Bcu130-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.13.0+cu130" />
  <img src="https://img.shields.io/badge/Environment-uv-blueviolet?style=for-the-badge&logo=uv" alt="uv package manager" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License" />
  <a href="https://shin-x-nihs.github.io/mini-inference-stack/">
    <img src="https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-2ea44f?style=for-the-badge&logo=github&logoColor=white" alt="Live Dashboard" />
  </a>
</p>

---

## Overview

This project studies how LLM inference behaves on a single laptop GPU (`Qwen/Qwen2.5-Coder-1.5B-Instruct`, `bfloat16`, NVIDIA GeForce RTX 5080 Laptop GPU) using closed-loop benchmarks alongside Nsight Systems (`nsys`) and Nsight Compute (`ncu`) profiling.

The codebase has **two parallel tracks** (see [WORKFLOW.md](WORKFLOW.md#track-philosophy) for the branching rules):
* **`vllm-reference/` (Reference Track):** Production vLLM stages that establish reference ceilings and test high-level features in isolation.
* **`custom-serving/` (Custom Track):** From-scratch PyTorch and Triton inference engines built step by step (Stages 0–6) to isolate and solve each GPU bottleneck in order.

The goal is to learn by measuring: build a baseline, profile where time goes, change one part of the stack, and verify what actually improved.

* **[Interactive Benchmark Dashboard ↗](https://shin-x-nihs.github.io/mini-inference-stack/)** (`benchmarks/results/dashboard.html`)
* **[Full Experiment Log & Methodology →](REPORTS.md)** (`REPORTS.md`)
* **[Benchmarking & Profiling Guide →](WORKFLOW.md)** (`WORKFLOW.md`)
* **[Custom Engine Architecture →](custom-serving/README.md)** (`custom-serving/README.md`)

---

## Hardware and Software Environment

All benchmarks and profiler captures use the same machine and shared `config.yaml` defaults:

| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 5080 Laptop GPU (16 GB GDDR7, Blackwell `sm_120`) |
| **Reference Ceilings** | `896 GB/s` GDDR7 bandwidth ([NVIDIA RTX 50 Series Laptops](https://www.nvidia.com/en-us/geforce/laptops/50-series/)) and a frozen `40 TFLOP/s` conservative compute reference in `config.yaml` (actual BF16 peak varies with laptop TGP/clock) |
| **OS** | Windows 11 + WSL2 (Ubuntu 24.04 LTS) |
| **CUDA / PyTorch / vLLM** | CUDA 13.1 (`nvcc 13.1.80`), PyTorch 2.13.0+cu130, vLLM 0.31.0 |
| **Python** | Python 3.12 (managed with `uv`) |
| **Target / Draft Models** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`) / `0.5B-Instruct` (speculative decoding) |
| **Profilers** | NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`) |

---

## Results at a Glance

Single-run closed-loop results (`n = 1` sweep per point; see [REPORTS.md](REPORTS.md) for full sweep tables, profiler breakdowns, first-principles floors, and measurement caveats).

| Stage | Track & Optimization | C=1 TPOT (ms) | C=64 Tput (tok/s) | L=8192 TTFT (ms) |
| :--- | :--- | :---: | :---: | :---: |
| `stage-0-vllm` | vLLM baseline | 10.9 | 1,232 | 674 |
| `stage-1-vllm` | vLLM speculative decoding (`K=5`) | 8.4 | 872 | 1,753 |
| `stage-2-vllm` | vLLM automatic prefix caching* | 12.2 | 2,874 | 198 |
| `stage-0-custom` | Custom naive PyTorch (no KV cache) | 66.8 | — | — |
| `stage-1-custom` | Custom naive KV cache | 26.4 | — | — |
| `stage-2-custom` | Custom Triton FlashAttention-2 | 22.4 | — | — |
| `stage-3-custom` | Custom Triton Split-K decode | 21.9 | — | — |
| `stage-4-custom` | Custom paged KV + continuous batching | 25.7 | 1,260 | 812 |
| `stage-5-custom` | Custom fused RMSNorm / RoPE / SwiGLU | 14.2 | 1,486 | 812 |
| `stage-6-custom` | Custom bucketed CUDA Graphs | 8.4 | 1,771 | 629 |

*\* `stage-2-vllm` runs on identical repeated prompts (`100%` prefix hit after request 1), so its numbers represent a best-case upper bound. Detailed numbers, profiler traces, and limitations live in [REPORTS.md](REPORTS.md).*

---

## Repository Structure

```text
mini-inference-stack/
├── config.yaml                     # Shared model, server, ceiling, and benchmark settings
├── WORKFLOW.md                     # How to run, benchmark, profile, and interpret metrics
├── REPORTS.md                      # Single source of truth for results, budgets, and analysis
├── ui/                             # Chainlit web UI for manual sanity checks
├── vllm-reference/                 # Reference track (vLLM)
│   ├── stage-0-baseline/           # Reference baseline
│   ├── stage-1-spec-dec/           # Speculative decoding (0.5B draft)
│   └── stage-2-prefix-cache/       # Hash-based automatic prefix caching
├── custom-serving/                 # Custom track (from-scratch PyTorch/Triton)
│   ├── README.md                   # Per-stage architecture notes
│   ├── test_parity.py              # Greedy token & prefill logit parity harness
│   ├── stage-0-baseline/           # Naive PyTorch (no KV cache, no SDPA)
│   ├── stage-1-kv-cache/           # Per-layer KV cache
│   ├── stage-2-tiled-attn/         # Triton FlashAttention-2
│   ├── stage-3-split-k/            # Triton Split-K decode attention
│   ├── stage-4-paged-kv/           # Paged KV cache & continuous batching
│   ├── stage-5-fused-ops/          # Fused Triton RMSNorm, RoPE, gate_up + SwiGLU
│   └── stage-6-cuda-graphs/        # Bucketed CUDA Graphs for decode
└── benchmarks/                     # Shared benchmark, profiling, and plotting suite
    ├── bench_closed_loop.sh        # Concurrency (C=1..64) & context (L=128..8192) sweeps
    ├── profile_nsys.sh             # Nsight Systems timeline capture & parser
    ├── profile_ncu.sh              # Standalone Layer 0 Nsight Compute profiler
    ├── plot_dashboard.py           # Generates benchmarks/results/dashboard.html
    ├── results/                    # Committed JSON measurements & dashboard HTML
    └── profiles/                   # Local profiler traces (.nsys-rep, .sqlite, .ncu-rep)
```

---

## Quickstart

See [WORKFLOW.md](WORKFLOW.md) for the complete stage-ID table, runtime overrides, and profiling instructions.

```bash
# 1. Start a server in Terminal 1 (examples: vLLM baseline or Custom Stage 6)
./vllm-reference/stage-0-baseline/serve.sh
./custom-serving/stage-6-cuda-graphs/serve.sh

# 2. Run closed-loop benchmarks in Terminal 2
./benchmarks/bench_closed_loop.sh <stage-id>          # e.g., stage-0-vllm or stage-6-custom

# 3. Profile timeline (start server with NSYS=ON ./<track>/<stage-dir>/serve.sh first)
./benchmarks/profile_nsys.sh <stage-id>

# 4. Profile Layer 0 kernels (stop the server first to free VRAM)
./benchmarks/profile_ncu.sh <stage-id>

# 5. Regenerate the comparison dashboard
uv run --frozen python benchmarks/plot_dashboard.py
```

---

## Optimization Roadmap

### Reference Track (`vllm-reference/`)
* **Stage 0 (`stage-0-vllm`):** Baseline vLLM server establishing reference latency, throughput, and Roofline targets.
* **Stage 1 (`stage-1-vllm`):** Speculative decoding pairing the 1.5B target with a 0.5B draft model (`K = 5`) to measure low- vs. high-concurrency trade-offs.
* **Stage 2 (`stage-2-vllm`):** Automatic prefix caching (`--enable-prefix-caching`) measuring best-case shared-prefix throughput and TTFT reduction.

### Custom Track (`custom-serving/`)
* **Stage 0 (`stage-0-custom`):** Minimal eager PyTorch forward pass with no KV cache and `asyncio.Lock` serialization.
* **Stage 1 (`stage-1-custom`):** Per-layer KV cache to eliminate `O(S^2)` full-sequence decode recomputation.
* **Stage 2 (`stage-2-custom`):** Custom Triton FlashAttention-2 kernel to remove `O(S^2)` HBM attention materialization during prefill.
* **Stage 3 (`stage-3-custom`):** Two-stage Triton Split-K decode attention to fix single-token SM starvation.
* **Stage 4 (`stage-4-custom`):** Paged KV block pool (`_reshape_and_cache`) and iteration-level continuous batching to unlock horizontal concurrency scaling.
* **Stage 5 (`stage-5-custom`):** Fused Triton RMSNorm+residual, in-place RoPE, and combined `gate_up_proj` + SwiGLU kernels to cut kernel launch count.
* **Stage 6 (`stage-6-custom`):** Bucketed `torch.cuda.CUDAGraph` decode runners (`B ∈ {1..64}`) to remove per-kernel host dispatch gaps.

---

## Project Status

| Component | Status | Notes |
| :--- | :---: | :--- |
| **Environment & Harness** | Tested | WSL2 Ubuntu 24.04, CUDA 13.1, `bench_closed_loop.sh`, `profile_nsys.sh`, `profile_ncu.sh`, `plot_dashboard.py` |
| **vLLM Stages 0–2** | Tested | Baseline, speculative decoding, and automatic prefix caching benchmarked & profiled |
| **Custom Stages 0–6** | Tested | Parity-verified (`test_parity.py --stage all`), benchmarked, and profiled (Stage 0 `ncu` deferred to Stage 1) |

---

## AI Assistance

AI coding tools were used for implementation, debugging, and documentation. Design decisions, experiments, and measurements are reviewed against the project's goals.

## License

Released under the [MIT License](LICENSE).
