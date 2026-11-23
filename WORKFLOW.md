# Engineering Workflow

Operational guide for running servers, checking correctness, benchmarking workloads, profiling GPU execution, and generating comparison plots.

> **Core Principle:** Change one variable at a time, keep `config.yaml` workload defaults fixed across stages, and never run two GPU workloads concurrently on the 16 GB laptop GPU.

---

## Stage Directory & Identifier Table

Server launch scripts live in a stage directory; benchmark and profiler scripts take a `<stage-id>` that names `benchmarks/results/<stage-id>/` and `benchmarks/profiles/<stage-id>/`.

| Track | Server Directory (`<stage-dir>`) | Stage ID (`<stage-id>`) | Focus |
| :--- | :--- | :--- | :--- |
| `vllm-reference` | `vllm-reference/stage-0-baseline` | `stage-0-vllm` | Reference baseline |
| `vllm-reference` | `vllm-reference/stage-1-spec-dec` | `stage-1-vllm` | Speculative decoding (`0.5B` draft, `K=5`) |
| `vllm-reference` | `vllm-reference/stage-2-prefix-cache` | `stage-2-vllm` | Hash-based automatic prefix caching |
| `custom-serving` | `custom-serving/stage-0-baseline` | `stage-0-custom` | Naive PyTorch (no KV cache, no SDPA) |
| `custom-serving` | `custom-serving/stage-1-kv-cache` | `stage-1-custom` | Per-layer KV cache |
| `custom-serving` | `custom-serving/stage-2-tiled-attn` | `stage-2-custom` | Triton FlashAttention-2 |
| `custom-serving` | `custom-serving/stage-3-split-k` | `stage-3-custom` | Triton Split-K decode attention |
| `custom-serving` | `custom-serving/stage-4-paged-kv` | `stage-4-custom` | Paged KV cache & continuous batching |
| `custom-serving` | `custom-serving/stage-5-fused-ops` | `stage-5-custom` | Fused RMSNorm, RoPE, `gate_up` + SwiGLU |
| `custom-serving` | `custom-serving/stage-6-cuda-graphs` | `stage-6-custom` | Bucketed CUDA Graphs for decode |

---

## Workflow at a Glance

| Step | Terminal 1 (Server) | Terminal 2 (Client / Runner) | Main Output |
| :--- | :--- | :--- | :--- |
| **1. Start** | `./<stage-dir>/serve.sh` | — | OpenAI API at `http://127.0.0.1:8000/v1` |
| **2. Check** | Keep running | Parity test + API probe or Chainlit UI (then stop UI) | Correctness check |
| **3. Benchmark** | Keep running | `./benchmarks/bench_closed_loop.sh <stage-id>` | `benchmarks/results/<stage-id>/*.json` |
| **4a. Profile (`nsys`)** | Restart with `NSYS=ON ./<stage-dir>/serve.sh` | `./benchmarks/profile_nsys.sh <stage-id>` | `timeline.*`, `nsys_metrics.json` |
| **4b. Profile (`ncu`)** | **Stop server first** (free VRAM) | `./benchmarks/profile_ncu.sh <stage-id>` | `kernels.ncu-rep`, `ncu_metrics.json` |
| **5. Plot** | — | `uv run --frozen python benchmarks/plot_dashboard.py` | `benchmarks/results/dashboard.html` |
| **6. Record** | — | Update `REPORTS.md` and summary table | Reproducible experiment log |

---

## 1. Configuration & Server Launch

Shared defaults live in root `config.yaml`. Precedence across scripts is: **environment variable > `config.yaml` > script default**.

| Key | Value | Purpose |
| :--- | :--- | :--- |
| `model.name` / `dtype` / `max_model_len` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` / `bfloat16` / `9216` | Fixed target model (`8192` prompt + `64` output + template headroom) |
| `draft_model.name` / `num_speculative_tokens` | `Qwen/Qwen2.5-Coder-0.5B-Instruct` / `5` | Draft model for `stage-1-vllm` |
| `server.host` / `port` / `gpu_memory_utilization` | `0.0.0.0` / `8000` / `0.70` | Bind address (clients use `127.0.0.1`), port, vLLM VRAM target |
| `benchmarks.temperature` / `random_seed` | `0.0` / `42` | Greedy sampling and fixed seed |
| `benchmarks.pareto` | `512` in / `128` out, `C = [1, 2, 4, 8, 16, 32, 64]` | Concurrency scaling sweep |
| `benchmarks.context_sweep` | `L = [128..8192]` in / `64` out, `10` prompts at `C = 1` | Prefill & decode context scaling |
| `benchmarks.profiling` | `nsys`: `1024` in / `64` out; `ncu`: `2048` in | Profiler workloads |
| `hardware_ceilings` | `896.0 GB/s`, `40.0 TFLOP/s` | `896 GB/s` GDDR7 bandwidth ([NVIDIA RTX 50 Series Laptops](https://www.nvidia.com/en-us/geforce/laptops/50-series/)) + frozen conservative `40 TFLOP/s` Roofline baseline (actual BF16 peak varies with TGP/clock) |

### Launch Commands

```bash
# Standard launch (Terminal 1)
./<stage-dir>/serve.sh

# Profiler-ready launch for Nsight Systems (Terminal 1)
NSYS=ON ./<stage-dir>/serve.sh

# Example runtime override without editing config.yaml
VLLM_GPU_MEMORY_UTILIZATION=0.75 VLLM_PORT=8001 ./vllm-reference/stage-0-baseline/serve.sh
```

Wait for `GET http://127.0.0.1:8000/health` → `200 OK` before sending requests.

---

## 2. Check Correctness

1. **Custom track parity gate** (run before starting the server):
   ```bash
   uv run --frozen python custom-serving/test_parity.py --stage all
   ```
2. **Quick API probe** (Terminal 2):
   ```bash
   curl -s http://127.0.0.1:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Qwen/Qwen2.5-Coder-1.5B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}'
   ```
3. **Optional Chainlit UI** (`uv run --frozen chainlit run ui/app.py --port 8501`): sanity checks only. **Stop the UI with `Ctrl+C` before benchmarking or profiling.**

---

## 3. Run Benchmarks

With the server running in Terminal 1:

```bash
# Run both concurrency and context sweeps
./benchmarks/bench_closed_loop.sh <stage-id>

# Or run one sweep at a time
./benchmarks/bench_closed_loop.sh <stage-id> pareto
./benchmarks/bench_closed_loop.sh <stage-id> context
```

* Outputs are saved to `benchmarks/results/<stage-id>/pareto_c<C>.json` and `context_l<L>.json`. Verify `failed: 0` in every JSON before drawing conclusions.
* **Prompt construction note:** `bench_closed_loop.sh` repeats `benchmarks.prompt` to length `L` and sends the same prompt to every request in a sweep. Keep this in mind when interpreting prefix caching (`100%` hit after request 1) and speculative decoding (easier drafting on repeated text).

---

## 4. Profile GPU Behavior

Benchmark first to see **what** changed; profile next to see **why**. Never use profiled runs as headline latency/throughput numbers.

### A. Nsight Systems (`nsys`) — Timeline & Host Dispatch
1. Restart the server in Terminal 1 with `NSYS=ON ./<stage-dir>/serve.sh`.
2. In Terminal 2: `./benchmarks/profile_nsys.sh <stage-id>`.
3. Outputs: `timeline.nsys-rep`, `timeline.sqlite`, `nsys_stats.txt` (in `benchmarks/profiles/<stage-id>/`) and `nsys_metrics.json` (in `benchmarks/results/<stage-id>/`).

**Exact `nsys_metrics.json` field definitions (from `profile_nsys.sh`):**
* `gpu_active_ms` / `gpu_active_pct`: **interval union** of non-overlapping GPU kernel execution inside `[first_kernel_start, last_kernel_end]`. `gpu_idle_ms = wall_time_ms - gpu_active_ms`.
* `cpu_launch_overhead_ms` / `cpu_launch_calls`: sum of host durations of all CUDA runtime API calls whose name contains `Launch` (including `cudaGraphLaunch`) clipped to the active window. Because blocking calls (such as `cudaGraphLaunch` under back-pressure) add to this duration, it measures **Launch API time**, not pure CPU overhead.
* `cuda_graph_launches`: count of runtime calls containing `GraphLaunch`.
* `cuda_sync_wait_ms`: sum of runtime calls containing `Synchronize`.
* `prefill_duration_ms` / `decode_total_ms` / `decode_step_avg_ms`: computed from host **NVTX** ranges (`prefill_step`, `decode_step`) when present (custom stages), falling back to a **GPU kernel timeline split** heuristic when absent (vLLM stages). Check `phase_measurement_source` before comparing across tracks.
* `kernel_time_breakdown`: sums GPU kernel time into `linear`, `attention` (via `attn_compute` NVTX correlation IDs or kernel-name heuristic), and `norm_act_other`.

### B. Nsight Compute (`ncu`) — Layer 0 Kernel Roofline
1. **Stop the server** in Terminal 1 and confirm no GPU compute process remains (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`).
2. In Terminal 2: `./benchmarks/profile_ncu.sh <stage-id>`.
   * Standard vLLM and custom stages profile standalone Layer 0 Prefill (`L = 2048`) and Decode (`M = 1`).
   * `stage-1-vllm` profiles target Layer 0 verification at `M = K + 1 = 6` alongside Prefill.
3. Outputs: `kernels.ncu-rep` (in `benchmarks/profiles/<stage-id>/`) and `ncu_metrics.json` (in `benchmarks/results/<stage-id>/`).

**Exact `ncu_metrics.json` caveats (from `profile_ncu.sh`):**
* **Locked base clock:** `ncu` runs with its default `--clock-control base`, locking clocks below boost. NCU's internal `dram_utilization_pct` / `compute_utilization_pct` (`SOL%`) are measured against that locked clock and therefore read ~2× higher than `% of 896 GB/s` (boost GDDR7 peak).
* **Tensor Core FLOP estimate:** FLOPs estimate tensor instructions as `smsp__inst_executed_pipe_tensor.sum × 4096`. For padded tiles (e.g., `M = 6` padded to `m16` or `M = 1` padded to `BLOCK_M = 16`), this counts padding work and inflates reported Arithmetic Intensity.
* **Longest-GEMM rule:** `profile_ncu.sh` keeps the single longest GEMM/GEMV in Layer 0 (separate `1×` projections in Custom Stages 1–4 vs. merged `2×` `gate_up_proj` in vLLM and Custom Stages 5–6).

---

## 5. Generate Dashboard Plots

Run from repo root:

```bash
# Plot all stages found in benchmarks/results/
uv run --frozen python benchmarks/plot_dashboard.py

# Plot a custom subset
uv run --frozen python benchmarks/plot_dashboard.py --stages stage-0-vllm stage-6-custom -o comparison.html
```

Default output is `benchmarks/results/dashboard.html`, containing four views plus Tables A (`nsys`) and B (`ncu`):
1. **Throughput vs. Latency:** aggregate output throughput vs. TPOT or ITL P50/P90/P99 across `C = 1..64`.
2. **Prompt Length vs. TTFT:** end-to-end time to first token across `L = 128..8192` at `C = 1` (includes client/serving overhead, not pure prefill kernel time).
3. **Context Length vs. Generation Speed:** decode speed across growing KV cache history at `C = 1`.
4. **Hardware Roofline:** attained TFLOP/s vs. Arithmetic Intensity (`FLOP/byte`) against the frozen `896 GB/s` and `40 TFLOP/s` reference ceilings.

---

## 6. Metrics Reference

| Metric | Phase | Unit | Definition |
| :--- | :--- | :---: | :--- |
| **TTFT** | Prefill + queue | `ms` | Request dispatch to first output token |
| **TPOT** | Decode | `ms/token` | Mean time per output token after the first: `(T_last - T_first) / (N - 1)` |
| **ITL** | Decode | `ms` | Time between consecutive output tokens (P50, P90, P99 capture jitter/bursts) |
| **Throughput** | End-to-end | `tokens/s` | Completed output tokens divided by total elapsed time |
| **Arithmetic Intensity** | Kernel | `FLOP/byte` | Floating-point operations divided by DRAM bytes transferred |

---

## Track Philosophy

* **Custom Track (`custom-serving/` — Strictly Additive):** Reconstructs LLM inference bottlenecks in order (`0: Naive PyTorch` → `1: KV Cache` → `2: Tiled Attention` → `3: Split-K Decode` → `4: Paged KV & Continuous Batching` → `5: Operator Fusion` → `6: Bucketed CUDA Graphs`). Copy `stage-(N-1)` into `stage-<N>` as a self-contained snapshot (no cross-stage imports) and implement one targeted upgrade on top.
* **Reference Track (`vllm-reference/` — Feature Isolation):** `stage-0-baseline` already includes FlashAttention, PagedAttention, continuous batching, and CUDA Graphs. Branch each new feature (`stage-1-spec-dec`, `stage-2-prefix-cache`) directly off `stage-0-baseline` so profiling metrics stay isolated (e.g., drafting launch overhead does not pollute prefix-cache lookup metrics). Only combine features if explicitly building a combined max-performance stage (`stage-3-vllm-max`).
