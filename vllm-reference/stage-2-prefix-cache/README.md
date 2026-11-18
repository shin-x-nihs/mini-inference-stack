# Stage 2: Automatic Prefix Caching (`stage-2-vllm`)

*Server directory: `vllm-reference/stage-2-prefix-cache/` | Benchmark stage join key: `stage-2-vllm`*

---

## 1. Executive Summary & Optimization Hypothesis

### Observation (Stage 0 Baseline & Stage 1 Retrospective)
Profiling the baseline vLLM engine running `Qwen2.5-Coder-1.5B-Instruct` on an NVIDIA GeForce RTX 5080 Laptop GPU revealed that while decode operations dominate single-stream wall time ($C=1$), **the primary system-level scalability bottleneck in production serving is prompt prefill serialization and time-to-first-token (TTFT) explosion**:

1. **Concurrency Scaling TTFT Collapse:** In the Stage 0 baseline, scaling concurrency from $C=1$ to $C=64$ caused mean TTFT to explode by **$12.2\times$** (from 73.3 ms to 896.2 ms), with P99 TTFT reaching **3,889.9 ms (~3.9 seconds)**. Under heavy continuous batching, prefill requests must wait in queue or preempt decode steps, inflating tail latency.
2. **Context Length Penalty:** At long prompt lengths ($L=8192$), prefill wall duration reached **673.7 ms** per request, depressing end-to-end client generation throughput by **42.8%** (from 89.0 tok/s at $L=128$ down to 50.9 tok/s at $L=8192$) despite invariant decode speed (~10 ms/token).
3. **Speculative Decoding Limitations:** While Stage 1's speculative decoding improved single-stream decode TPOT (-23.1% at $C=1$), it incurred drafting overhead that degraded high-concurrency throughput by **-29.2%** at $C=64$ and inflated TTFT by **+160.2%** at $L=8192$ (from 673.7 ms to 1,752.7 ms) due to dual-model prefill initialization.

### Hypothesis
Enabling **Automatic Prefix Caching (APC)** via RadixAttention retains KV cache blocks for processed prompt prefixes in an in-memory radix tree across requests. When subsequent requests share an identical prompt prefix (common in multi-turn conversations, system prompts, few-shot evaluations, and synthetic benchmark sweeps), the engine reuses cached KV blocks directly without re-executing prefill compute GEMMs or re-streaming attention keys and values.

Eliminating redundant prefill computation directly removes queue buildup and execution bubbles, slashing mean and P99 TTFT under high concurrency and long contexts while maximizing aggregate throughput.

---

## 2. Theoretical Speedup Model

For request $i$ with total prompt length $L$ and matching shared prefix length $P \le L$:

$$\text{Effective Prefill Cost}(L, P) = (1 - h) \cdot T_{\text{prefill}}(L) + h \cdot \left[ T_{\text{prefill}}(L - P) + t_{\text{lookup}} \right]$$

Where:
* $h \in [0, 1]$ represents the cache hit indicator ($h=1$ when matching blocks exist in the radix tree).
* $T_{\text{prefill}}(N)$ is the compute and memory latency for prefilling $N$ tokens through all Transformer layers.
* $t_{\text{lookup}}$ is the radix tree prefix lookup overhead ($t_{\text{lookup}} \ll T_{\text{prefill}}(P)$, typically $< 1\text{ ms}$).

### Idealized Bound
In closed-loop benchmarking sweeps and common serving patterns where prompts share a fixed prefix ($P = L = 512$ tokens), prefill compute drops from:

$$T_{\text{prefill}}(512) \approx 112\text{ ms} \longrightarrow t_{\text{lookup}} \approx 2\text{--}5\text{ ms} \quad (\sim 95\%\text{--}98\%\text{ reduction})$$

Under concurrency $C=64$, eliminating $640 \times 512 = 327,680$ redundant prefill tokens prevents GPU execution saturation, allowing compute units to focus continuously on batched decode iterations.

---

## 3. Hardware & Memory Footprint (RTX 5080 Laptop GPU)

* **Architecture:** NVIDIA Blackwell `sm_120`, 16 GB GDDR7 VRAM, ~512 GB/s physical memory bandwidth.
* **Model Parameters:** `Qwen/Qwen2.5-Coder-1.5B-Instruct` (28 layers, 16 query heads, 2 KV heads, head dimension 128, `bfloat16`).
* **Per-Token KV Footprint:**
  $$\text{KV Size per Token} = 2 \times 28 \text{ layers} \times 2 \text{ KV heads} \times 128 \text{ dim} \times 2 \text{ bytes (BF16)} = 28,672 \text{ bytes} \approx 28.0 \text{ KiB}$$
* **Cache Capacity:**
  - A 512-token prompt prefix consumes **$14.34\text{ MiB}$** of VRAM.
  - An 8,192-token prompt prefix consumes **$229.4\text{ MiB}$** of VRAM.
  - With `gpu_memory_utilization: 0.70` allocating $\approx 8.1\text{ GiB}$ for KV cache, prefix retention introduces negligible memory pressure and operates comfortably without block thrashing or OOM risk.

---

## 4. Experiment Setup & Configuration

| Parameter | Value | Source / Flag |
| :--- | :--- | :--- |
| **Stage Join Key** | `stage-2-vllm` | Directory name in `benchmarks/results/` |
| **Model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `model.name` in `config.yaml` |
| **Precision** | `bfloat16` | `model.dtype` in `config.yaml` |
| **Max Model Length** | 9,216 tokens | `model.max_model_len` in `config.yaml` |
| **GPU Memory Utilization**| 0.70 | `server.gpu_memory_utilization` |
| **Prefix Caching** | `true` | `--enable-prefix-caching` |
| **Sampling** | Greedy (`temperature: 0.0`) | `benchmarks.temperature` |

---

## 5. Usage & Operational Workflow

### 1. Launch the Inference Server
```bash
# Standard serving
./vllm-reference/stage-2-prefix-cache/serve.sh

# Profiler-enabled dormant launch (for Nsight Systems)
NSYS=ON ./vllm-reference/stage-2-prefix-cache/serve.sh
```

### 2. Verify Server Readiness
```bash
curl -i http://127.0.0.1:8000/health
# Expected: HTTP/1.1 200 OK
```

### 3. Run Closed-Loop Benchmark Suite
```bash
# Via local wrapper
./vllm-reference/stage-2-prefix-cache/bench.sh

# Or via centralized harness
./benchmarks/bench_closed_loop.sh stage-2-vllm
```
Outputs: `pareto_c{1,2,4,8,16,32,64}.json` and `context_l{128..8192}.json` in `benchmarks/results/stage-2-vllm/`.

### 4. Run Nsight Systems Timeline Profiling
```bash
# Ensure server was launched with NSYS=ON, then run:
./vllm-reference/stage-2-prefix-cache/profile_nsys.sh

# Or via centralized harness
./benchmarks/profile_nsys.sh stage-2-vllm
```
Outputs: `timeline.nsys-rep`, `timeline.sqlite`, and `nsys_metrics.json`.

### 5. Run Nsight Compute Kernel & Roofline Profiling
```bash
# 1. Stop the HTTP server to free all GPU VRAM
pkill -f "vllm serve" || true
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 2. Run standalone NCU profiler
./vllm-reference/stage-2-prefix-cache/profile_ncu.sh

# Or via centralized harness
./benchmarks/profile_ncu.sh stage-2-vllm
```
Outputs: `kernels.ncu-rep` and `ncu_metrics.json`.

### 6. Regenerate Interactive Dashboard
```bash
uv run python benchmarks/plot_dashboard.py --stages stage-0-vllm stage-1-vllm stage-2-vllm
```
Outputs: `benchmarks/results/dashboard.html`.
