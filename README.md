# mini-inference-stack

<p align="center">
  <b>An educational project exploring how LLM inference works, how to measure its performance, and how GPU optimizations make it faster.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-NVIDIA%20Blackwell%20(sm__120)-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Blackwell Architecture" />
  <img src="https://img.shields.io/badge/CUDA-13.1-green?style=for-the-badge&logo=nvidia" alt="CUDA 13.1" />
  <img src="https://img.shields.io/badge/PyTorch-2.13.0%2Bcu130-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.13.0+cu130" />
  <img src="https://img.shields.io/badge/Environment-uv-blueviolet?style=for-the-badge" alt="uv package manager" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License" />
  <a href="https://shin-x-nihs.github.io/mini-inference-stack/">
    <img src="https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-2ea44f?style=for-the-badge&logo=github&logoColor=white" alt="Live Dashboard" />
  </a>
</p>

---

## Overview

This project explores how LLM inference works and how GPU optimizations affect its performance, using `Qwen2.5-Coder-1.5B-Instruct` on an NVIDIA GeForce RTX 5080 Laptop GPU.

The codebase is organized into **two parallel folds**:
* **`vllm-reference/` (Reference Track):** Production-grade vLLM implementations used to establish upper-bound performance targets, latency boundaries, and reference profiler traces. **(Philosophy: Feature Isolation — evaluating one high-level optimization at a time).**
* **`custom-serving/` (Custom Track):** Progressive, from-scratch inference implementations (currently through `stage-6-cuda-graphs`) progressing through custom kernels and memory managers, built to demystify GPU performance from first principles. **(Philosophy: Strictly Additive — redoing history by stacking each optimization on top of the last).**

The goal isn't to build a replacement for vLLM. It's to learn by experimenting: measure the baseline, find out where execution time is spent, implement optimizations one step at a time, and compare the results.

Client benchmarks measure response times and throughput. NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`) help investigate what's happening inside the CPU and GPU.

**[Explore the interactive benchmark dashboard ↗](https://shin-x-nihs.github.io/mini-inference-stack/)**.

---

## What I Want to Learn

* **Measure a baseline:** Find out how quickly the model responds and how much output it can generate under different workloads.
* **Understand generation:** Follow the steps of a request from processing the input prompt to generating tokens one by one.
* **Find bottlenecks:** Use profiling tools to understand whether performance is limited by GPU computation, memory access, GPU utilization, or CPU overhead.
* **Try GPU optimizations:** Explore techniques such as KV caching, tiled attention, operator fusion, and speculative decoding, and measure whether they improve performance.
* **Compare fairly:** Keep the model, precision, prompts, token lengths, and hardware settings consistent across experiments. Common workload defaults live in `config.yaml` to keep test conditions consistent across runner scripts.

---

## Hardware and Software Environment

All benchmarks and profiling runs use the same setup:

| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 5080 Laptop GPU (16 GB GDDR7) |
| **Architecture** | NVIDIA Blackwell (`sm_120`) |
| **OS** | Windows 11 + WSL2 (Ubuntu 24.04 LTS) |
| **CUDA** | CUDA 13.1 (`nvcc 13.1.80`) |
| **Python** | Python 3.12 (managed with `uv`) |
| **PyTorch** | PyTorch 2.13.0+cu130 |
| **Reference Engine** | vLLM 0.31.0 |
| **Target Model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`) |
| **Draft Model** | `Qwen/Qwen2.5-Coder-0.5B-Instruct` (`bfloat16`, for speculative decoding) |
| **Profilers** | NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`) |

The 1.5B model is small enough to run, profile, and iterate on quickly on a laptop GPU.

---

## Repository Structure

```text
mini-inference-stack/
├── config.yaml                     # Model, server, and benchmark settings
├── WORKFLOW.md                     # Engineering guide for benchmarks and profiling
├── REPORTS.md                      # Experiments, observations, hypotheses, and conclusions
├── ui/                             # Chainlit web UI for quick sanity checks
├── vllm-reference/                 # Reference track: vLLM reference implementations
│   ├── stage-0-baseline/           # vLLM baseline server (Stage 0)
│   ├── stage-1-spec-dec/           # vLLM speculative decoding server (Stage 1)
│   └── stage-2-prefix-cache/       # vLLM automatic prefix caching server (Stage 2)
├── custom-serving/                 # Custom track: from-scratch inference engines
│   ├── README.md                   # Custom track architecture notes & file inventory
│   ├── test_parity.py              # Shared greedy token & prefill logit parity runner
│   ├── stage-0-baseline/           # Naive PyTorch baseline (Stage 0; no KV cache/SDPA)
│   ├── stage-1-kv-cache/           # Naive PyTorch KV cache engine (Stage 1)
│   ├── stage-2-tiled-attn/         # Custom Triton FlashAttention-2 engine (Stage 2)
│   ├── stage-3-split-k/            # Custom Triton Split-K / FlashDecode engine (Stage 3)
│   ├── stage-4-paged-kv/           # Paged KV cache & continuous batching engine (Stage 4)
│   ├── stage-5-fused-ops/          # Fused Triton RMSNorm/RoPE/SwiGLU engine (Stage 5)
│   └── stage-6-cuda-graphs/        # Bucketed CUDA Graph decode engine (Stage 6)
└── benchmarks/                     # Benchmark and profiling suite
    ├── bench_closed_loop.sh        # Closed-loop concurrency & context sweeps
    ├── profile_nsys.sh             # Nsight Systems timeline profiling
    ├── profile_ncu.sh              # Nsight Compute kernel profiling
    ├── plot_dashboard.py           # Generates the HTML comparison dashboard
    ├── results/                    # Benchmark JSONs and generated dashboards
    └── profiles/                   # Raw profiler traces (.nsys-rep, .sqlite, .txt, .ncu-rep)
```

---

## How Performance Is Evaluated

The project combines two approaches: benchmarking the running system and profiling individual GPU operations.

### 1. Benchmark Results

The dashboard (`benchmarks/results/dashboard.html`) compares performance across four views:

* **Throughput vs. Latency:** How does total output throughput change as more requests run at the same time? How much slower do responses become?
* **Prompt Size vs. Time to First Token:** How does the wait for the first token change as the input prompt gets longer?
* **Context Length vs. Generation Speed:** How does generation speed change as the KV cache grows?
* **Hardware Roofline:** How does the performance of individual GPU kernels compare with the GPU's compute and memory limits?

The main metrics are:

| Metric | What it tells us |
| :--- | :--- |
| **TTFT (Time to First Token)** | How long the client waits for the first generated token (`ms`). |
| **TPOT (Time Per Output Token)** | The average time per generated token after the first (`ms/token`). |
| **ITL (Inter-Token Latency)** | The time between consecutive generated tokens (`ms`, tracked at P50, P90, P99). |
| **Generation Speed** | How quickly one request receives output tokens (`tokens/s/req`). |
| **System Throughput** | How many output tokens the server generates per second across all requests (`tokens/s`). |

*Note: TTFT is measured from the client's perspective. It includes queueing, scheduling, and serving overhead, so it isn't a direct measurement of prefill kernel execution time.*

### 2. GPU Profiling

Benchmarks show *what* changed. Profiling helps explain *why*.

* **Nsight Systems (`nsys`):** Shows when CPU and GPU operations run, how long they take, and where synchronization waits or idle gaps occur.
* **Nsight Compute (`ncu`):** Examines individual GPU kernels, including their execution time, memory bandwidth, compute utilization, occupancy, and register usage.

Together, these tools help connect changes in end-to-end performance to what's happening inside the GPU.

---

## Quickstart

See [WORKFLOW.md](WORKFLOW.md) for full details on running experiments.

### 1. Start the Server

In Terminal 1:

```bash
# Reference Track Stage 0: Baseline
./vllm-reference/stage-0-baseline/serve.sh

# Or Reference Track Stage 1: Speculative Decoding
./vllm-reference/stage-1-spec-dec/serve.sh

# Or Reference Track Stage 2: Automatic Prefix Caching
./vllm-reference/stage-2-prefix-cache/serve.sh

# Custom Track Stage 0: Naive PyTorch Baseline
./custom-serving/stage-0-baseline/serve.sh

# Or Custom Track Stage 1: Naive KV Cache
./custom-serving/stage-1-kv-cache/serve.sh

# Or Custom Track Stage 2: Tiled Attention (Triton FlashAttention-2)
./custom-serving/stage-2-tiled-attn/serve.sh

# Or Custom Track Stage 3: Parallel Decode Attention (Triton Split-K)
./custom-serving/stage-3-split-k/serve.sh

# Or Custom Track Stage 4: Paged KV Cache & Continuous Batching
./custom-serving/stage-4-paged-kv/serve.sh

# Or Custom Track Stage 5: Operator Fusion
./custom-serving/stage-5-fused-ops/serve.sh

# Or Custom Track Stage 6: Bucketed CUDA Graphs for Decode
./custom-serving/stage-6-cuda-graphs/serve.sh
```

### 2. Run Benchmarks

In Terminal 2 (while the server is running):

```bash
# Run both concurrency and context-length sweeps (closed loop)
./benchmarks/bench_closed_loop.sh stage-0-vllm
# Or for Stage 1 / Stage 2 / Custom Stages:
./benchmarks/bench_closed_loop.sh stage-1-vllm
./benchmarks/bench_closed_loop.sh stage-2-vllm
./benchmarks/bench_closed_loop.sh stage-0-custom
./benchmarks/bench_closed_loop.sh stage-1-custom
./benchmarks/bench_closed_loop.sh stage-2-custom
./benchmarks/bench_closed_loop.sh stage-3-custom
./benchmarks/bench_closed_loop.sh stage-4-custom
./benchmarks/bench_closed_loop.sh stage-5-custom
./benchmarks/bench_closed_loop.sh stage-6-custom
```

### 3. Profile GPU Execution

```bash
# Timeline profile (start server with: NSYS=ON ./<track>/<stage>/serve.sh)
./benchmarks/profile_nsys.sh stage-0-vllm
./benchmarks/profile_nsys.sh stage-1-vllm
./benchmarks/profile_nsys.sh stage-2-vllm
./benchmarks/profile_nsys.sh stage-0-custom
./benchmarks/profile_nsys.sh stage-1-custom
./benchmarks/profile_nsys.sh stage-2-custom
./benchmarks/profile_nsys.sh stage-3-custom
./benchmarks/profile_nsys.sh stage-4-custom
./benchmarks/profile_nsys.sh stage-5-custom
./benchmarks/profile_nsys.sh stage-6-custom

# Kernel profile (runs standalone; stop the server first to avoid GPU memory contention.
# Stage 0 custom deferred NCU since nsys already isolated full-sequence decode recomputation.)
./benchmarks/profile_ncu.sh stage-0-vllm
./benchmarks/profile_ncu.sh stage-1-vllm
./benchmarks/profile_ncu.sh stage-2-vllm
./benchmarks/profile_ncu.sh stage-1-custom
./benchmarks/profile_ncu.sh stage-2-custom
./benchmarks/profile_ncu.sh stage-3-custom
./benchmarks/profile_ncu.sh stage-4-custom
./benchmarks/profile_ncu.sh stage-5-custom
./benchmarks/profile_ncu.sh stage-6-custom
```

### 4. Generate the Dashboard

```bash
# Plot all stages found in benchmarks/results/
uv run python benchmarks/plot_dashboard.py

# Compare specific stages
uv run python benchmarks/plot_dashboard.py --stages stage-0-vllm stage-1-vllm stage-2-vllm stage-0-custom stage-1-custom stage-2-custom stage-3-custom stage-4-custom stage-5-custom stage-6-custom -o comparison.html
```

---

## Optimization Roadmap

Each stage explores one main question at a time, checks that the implementation produces correct outputs, and measures its performance:

### Reference Track (vLLM)
1. **vLLM Stage 0: Reference Baseline (`stage-0-vllm`)**  
   Measure baseline serving performance and use profiling tools to understand how the reference implementation runs.
2. **vLLM Stage 1: Speculative Decoding (`stage-1-vllm`)**  
   Pair a 1.5B target model with a smaller 0.5B draft model to test whether speculative decoding improves generation speed. The results show how acceptance rate, drafting overhead, and concurrency affect performance. See [REPORTS.md](REPORTS.md) for the hypothesis and measurements.
3. **vLLM Stage 2: Automatic Prefix Caching (`stage-2-vllm`)**  
   Enable RadixAttention KV-block reuse (`--enable-prefix-caching`) to eliminate redundant prefill compute and queuing tail latency across shared prompts. See [REPORTS.md](REPORTS.md) for concurrency scaling (+133.3% throughput at $C=64$) and bottleneck-shift measurements.

### Custom Track (From Scratch)
4. **Custom Stage 0: Minimal PyTorch Baseline (`stage-0-custom`)**  
   Build a strict first-principles forward pass and generation loop (no KV cache, no SDPA, serialized via `asyncio.Lock`) to establish the unoptimized baseline and quantify full-sequence recomputation and launch overhead. See [custom-serving/README.md](custom-serving/README.md).
5. **Custom Stage 1: KV Caching (`stage-1-custom`)**  
   Cache per-layer Key and Value tensors across decode steps so each new token projects only $M=1$ query/key/value vector ($9.2\times$ throughput gain at $L=2048$), isolating unfused PyTorch kernel launch overhead and attention memory traffic via `nsys` and `ncu`. See [REPORTS.md](REPORTS.md) and [custom-serving/README.md](custom-serving/README.md).
6. **Custom Stage 2: Tiled Attention (FlashAttention-style) (`stage-2-custom`)**  
   Replace unfused PyTorch attention with a custom Triton FlashAttention-2 GQA kernel using SRAM online softmax ($11.8\times$ prefill attention AI jump to $1,597.8\text{ FLOP/B}$, $-89.1\%$ attention DRAM bandwidth, and $-40.6\%$ TTFT at $L=2048$ matching vLLM), while exposing single-token decode SM under-utilization (`grid=(1, 12, 1)`). See [REPORTS.md](REPORTS.md) and [custom-serving/README.md](custom-serving/README.md).
7. **Custom Stage 3: Parallel Decode Attention (Split-K / FlashDecode-style) (`stage-3-custom`)**  
   Partition decode KV sequences across parallel thread blocks with a log-sum-exp combine kernel (`3.26×` faster Layer 0 decode attention, `6.54×` lower total attention kernel time, and up to `+7.3%` context-sweep throughput), exposing `703.33 ms` of CPU launch overhead as the next bottleneck. See [REPORTS.md](REPORTS.md) and [custom-serving/README.md](custom-serving/README.md).
8. **Custom Stage 4: Paged KV Cache & Continuous Batching (`stage-4-custom`)**  
   Replace per-request `torch.cat` KV reallocation and `asyncio.Lock` serialization with a `BlockAllocator` (`KV_BLOCK_SIZE = 16`), Triton `_reshape_and_cache` (`4.61×` faster Layer 0 decode KV write), and iteration-level $M = B$ batched decode (`1,260.39 tok/s` at $C=64$ — `102.3%` of `stage-0-vllm`, `13.60×` over Stage 3 at $C=16$), isolating `669.98 ms` of CPU launch overhead and `3,584` unfused RoPE `CatArrayBatchedCopy` calls for Stage 5. See [REPORTS.md](REPORTS.md) and [custom-serving/README.md](custom-serving/README.md).
9. **Custom Stage 5: Operator Fusion (`stage-5-custom`)**  
   Replace unfused PyTorch elementwise/concat chains with custom Triton `_rms_norm_fwd` (plus residual-add fusion), in-place `_rope_embedding_fwd` (eliminating all `3,584` `CatArrayBatchedCopy` launches), and combined `gate_up_proj` + `_swiglu_fwd` kernels—cutting total kernel launches **3.03×** (`72,933` → `24,101`, `-66.95%`), CPU launch overhead by **`-45.57%`** (`669.98 ms` → `364.68 ms`), and GPU idle time from `53.53%` to `28.97%`, lifting $C=1$ throughput **`+77.7%`** (`38.51` → `68.44 tok/s`, TPOT `25.70` → `14.19 ms`) and $C=64$ throughput to **`1,486.13 tok/s`** (`120.6%` of `stage-0-vllm`), isolating the remaining `364.68 ms` host dispatch overhead (`cuda_graph_launches = 0`) for Stage 6. See [REPORTS.md](REPORTS.md) and [custom-serving/README.md](custom-serving/README.md).
10. **Custom Stage 6: Bucketed CUDA Graphs for Decode (`stage-6-custom`)**  
    Capture fixed-shape decode steps (`q_len == 1`) into bucketed `torch.cuda.CUDAGraph` runners for $B \in \{1, 2, 4, 8, 16, 32, 64\}$ (`DecodeGraphRunner` with static buffers and padded-lane KV masking)—raising `cuda_graph_launches` from `0` to **`63`**, slashing host `cpu_launch_calls` **50.6×** (`24,100` → `476`, `-98.02%`) and `cpu_launch_overhead_ms` by **`-21.78%`** (`364.68 ms` → `285.25 ms`), shrinking GPU idle time **3.89×** (`28.97%` → `9.03%`, `90.97%` active), and cutting `decode_step_avg_ms` by **`-31.32%`** (`12.58 ms` → `8.64 ms`, beating `stage-0-vllm`'s `10.67 ms`). End-to-end throughput surpasses `stage-0-vllm` at **every** concurrency level ($C=1$: **`113.42 tok/s`**, `+65.7%` over Stage 5, `129.2%` of vLLM; $C=64$: **`1,770.61 tok/s`**, `+19.1%` over Stage 5, `143.7%` of vLLM) and **every** context length $L \in [128, 8192]$ (`7.50–8.60 ms` TPOT). See [REPORTS.md](REPORTS.md) and [custom-serving/README.md](custom-serving/README.md).

These are learning experiments, not guaranteed improvements. Each optimization is evaluated by measuring its results and understanding its trade-offs.

---

## Project Status

| Component | Status | Deliverables |
| :--- | :---: | :--- |
| **Environment setup** | Tested | WSL2 Ubuntu 24.04, CUDA 13.1, PyTorch on RTX 5080 |
| **vLLM Stage 0: Reference baseline** | Tested | Server script, benchmark runs, `nsys` and `ncu` profiles |
| **vLLM Stage 1: Speculative decoding** | Tested | Server script, benchmark sweeps, NCU verification profiling |
| **vLLM Stage 2: Automatic prefix caching** | Tested | Server script, benchmark sweeps, `nsys` and `ncu` profiles |
| **Benchmark & profiling suite** | Tested | `bench_closed_loop.sh`, `profile_nsys.sh`, `profile_ncu.sh` (vLLM & custom tracks) |
| **Comparison dashboard** | Implemented | Interactive Plotly HTML report (`benchmarks/plot_dashboard.py`) |
| **Custom Stage 0: Naive PyTorch** | Tested | From-scratch model, naive engine, FastAPI SSE server, parity test, closed-loop sweeps (`C=1..16`, `L=128..2048`), `nsys` profile |
| **Custom Stage 1: Naive KV Cache** | Tested | Per-layer KV cache model & engine, shared `custom-serving/test_parity.py`, closed-loop sweeps (`C=1..16`, `L=128..2048`), `nsys` and `ncu` profiles |
| **Custom Stage 2: Tiled Attention** | Tested | Custom Triton FlashAttention-2 GQA kernel (`_flash_attention_fwd`), closed-loop sweeps (`C=1..16`, `L=128..2048`), `nsys` and `ncu` profiles |
| **Custom Stage 3: Parallel Decode (Split-K)** | Tested | Two-stage Triton Split-K decode + combine kernels, closed-loop sweeps (`C=1..16`, `L=128..2048`), `nsys` and `ncu` profiles |
| **Custom Stage 4: Paged KV & Continuous Batching** | Tested | `BlockAllocator` + Triton `_reshape_and_cache` + continuous batching scheduler, closed-loop sweeps (`C=1..64`, `L=128..8192`), `nsys` and `ncu` profiles |
| **Custom Stage 5: Operator Fusion** | Tested | Fused RMSNorm+residual / RoPE / `gate_up_proj`+SwiGLU Triton kernels (`stage-5-fused-ops/`), closed-loop sweeps (`C=1..64`, `L=128..8192`), `nsys` and `ncu` profiles |
| **Custom Stage 6: CUDA Graphs** | Tested | Bucketed `DecodeGraphRunner` ($B \in \{1..64\}$) + padded-lane KV masking (`stage-6-cuda-graphs/`), closed-loop sweeps (`C=1..64`, `L=128..8192`), `nsys` and `ncu` profiles |

---

## AI Assistance

AI coding tools were used for implementation, debugging, and documentation. Design decisions, experiments, and performance results are reviewed against the project's goals.

---

## License

This project is released under the [MIT License](LICENSE).
