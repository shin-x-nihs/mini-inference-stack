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
* **`vllm-reference/` (Reference Track):** Production-grade vLLM baseline used to establish upper-bound performance targets, latency boundaries, and reference profiler traces.
* **`custom-serving/` (Custom Track):** Progressive, from-scratch inference implementations starting from `stage-0-baseline` (minimal naive PyTorch) through custom kernels and memory managers, built to demystify GPU performance from first principles.

The goal isn't to build a replacement for vLLM. It's to learn by experimenting: measure the baseline, find out where execution time is spent, implement optimizations one step at a time, and compare the results.

Client benchmarks measure response times and throughput. NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`) help investigate what's happening inside the CPU and GPU.

**[Explore the interactive benchmark dashboard ↗](https://shin-x-nihs.github.io/mini-inference-stack/)**.
---

## What I Want to Learn

* **Measure a baseline:** Find out how quickly the model responds and how much output it can generate under different workloads.
* **Understand generation:** Follow a request from processing the input prompt to generating tokens one by one.
* **Find bottlenecks:** Use profiling tools to understand whether performance is limited by GPU computation, memory access, GPU utilization, or CPU overhead.
* **Try GPU optimizations:** Explore techniques such as KV caching, tiled attention, and operator fusion, and measure whether they improve performance.
* **Compare fairly:** Keep the model, precision, prompts, token lengths, and hardware settings consistent across experiments.

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
| **Model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`) |
| **Profilers** | NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`) |

The 1.5B model is small enough to run, profile, and iterate on quickly on a laptop GPU.

---

## Repository Structure

```text
mini-inference-stack/
├── config.yaml                     # Model, server, and benchmark settings
├── WORKFLOW.md                     # Engineering guide for benchmarks and profiling
├── ui/                             # Chainlit web UI for quick sanity checks
├── vllm-reference/                 # Reference track: vLLM baseline implementations
│   └── stage-0-baseline/           # vLLM baseline server (Stage 0)
├── custom-serving/                 # Custom track: from-scratch inference engines
│   └── stage-0-baseline/           # Minimal PyTorch baseline server
└── benchmarks/                     # Benchmark and profiling suite
    ├── bench_closed_loop.sh        # Closed-loop concurrency & context sweeps
    ├── bench_open_loop(temporary).sh # Open-loop arrival rate experiments (future strain)
    ├── profile_nsys.sh             # Nsight Systems timeline profiling
    ├── profile_ncu.sh              # Nsight Compute kernel profiling
    ├── plot_dashboard.py           # Generates the HTML comparison dashboard
    ├── results/                    # Benchmark JSONs and generated dashboards
    └── profiles/                   # Raw profiler traces (.nsys-rep, .ncu-rep)
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
./vllm-reference/stage-0-baseline/serve.sh
```

### 2. Run Benchmarks

In Terminal 2 (while the server is running):

```bash
# Run both concurrency and context-length sweeps (closed loop)
./benchmarks/bench_closed_loop.sh stage-0-vllm
```

> *Note: For open-loop arrival rate experiments (evaluating queue buildup under offered load), see `./benchmarks/bench_open_loop(temporary).sh`.*

### 3. Profile GPU Execution

```bash
# Timeline profile (start server with: NSYS=ON ./vllm-reference/stage-0-baseline/serve.sh)
./benchmarks/profile_nsys.sh stage-0-vllm

# Kernel profile (runs standalone; stop the server first)
./benchmarks/profile_ncu.sh stage-0-vllm
```

### 4. Generate the Dashboard

```bash
# Plot all stages found in benchmarks/results/
uv run python benchmarks/plot_dashboard.py

# Compare specific stages
uv run python benchmarks/plot_dashboard.py --stages stage-0-vllm stage-1-naive -o comparison.html
```

---

## Optimization Roadmap

Each stage explores one main change at a time, checks that the implementation produces correct outputs, and measures its performance:

1. **Stage 0: vLLM Reference Baseline**  
   Measure serving performance and use profiling tools to understand how the reference implementation runs.
2. **Stage 1: Minimal Python/PyTorch Inference (Planned)**  
   Build a simple forward pass and generation loop to understand the model's operations, memory use, and CPU overhead.
3. **Stage 2: KV Caching (Planned)**  
   Reuse attention keys and values from previous tokens instead of recomputing them at every generation step. Measure the effects on memory use and generation speed.
4. **Stage 3: Tiled Prefill Attention (FlashAttention-style) (Planned)**  
   Explore how processing attention in smaller tiles can reduce intermediate data written to GPU memory.
5. **Stage 4: Parallel Decode Attention (Split-K / FlashDecode-style) (Planned)**  
   Explore how splitting attention work across more GPU thread blocks can improve utilization during token generation.
6. **Stage 5: Paged KV Cache (PagedAttention-style) (Planned)**  
   Explore storing the KV cache in blocks to manage GPU memory more flexibly as requests come and go.
7. **Stage 6: Operator Fusion (Planned)**  
   Combine selected operations to reduce unnecessary memory traffic and GPU kernel launches.

These are learning experiments, not guaranteed improvements. Each optimization is evaluated by measuring its results and understanding its trade-offs.

---

## Project Status

| Component | Status | Deliverables |
| :--- | :---: | :--- |
| **Environment setup** | Completed | WSL2 Ubuntu 24.04, CUDA 13.1, PyTorch on RTX 5080 |
| **Stage 0: vLLM baseline** | Completed | Server script, benchmark runs, `nsys` and `ncu` profiles |
| **Benchmark scripts** | Completed | Closed-loop sweeps (`bench_closed_loop.sh`), open-loop prototype |
| **Profiling scripts** | Completed | `profile_nsys.sh` and `profile_ncu.sh` |
| **Comparison dashboard** | Completed | Interactive Plotly HTML report (`benchmarks/plot_dashboard.py`) |
| **Stage 1: Basic PyTorch engine** | Planned | Simple forward pass and generation loop |
| **Stages 2–6: GPU optimizations** | Planned | KV cache, tiled attention, Split-K decode, fused ops |

---

## License

This project is released under the [MIT License](LICENSE).