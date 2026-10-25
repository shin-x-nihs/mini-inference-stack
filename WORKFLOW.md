# Engineering Workflow

A shared guide for running the inference server, checking functionality, benchmarking workloads, profiling GPU behavior, and comparing optimization stages across both project tracks:
* **Reference track (`vllm-reference/`):** Production vLLM baseline implementations (`stage-0-baseline`).
* **Custom track (`custom-serving/`):** Progressive from-scratch implementations starting from `stage-0-baseline` through custom kernels and memory layouts.

> **Core Principle:** Keep experiments repeatable—change one variable at a time and record the environment and settings.

---

## Workflow at a Glance

| Step | Terminal 1 (Server) | Terminal 2 (Client / Runner) | What It Does | Main Output |
| :--- | :--- | :--- | :--- | :--- |
| **1. Start** | Start the selected implementation | — | Loads the model and starts the API | API at `127.0.0.1:8000/v1` |
| **2. Check** | Keep the server running | UI or quick API probe | Verifies requests and streaming work | Functional verification |
| **3. Benchmark** | Keep the server running | `./benchmarks/bench_closed_loop.sh <stage-name>` | Measures throughput and latency under standard workloads | Raw JSON in `benchmarks/results/<stage-name>/` |
| **4a. Profile (`nsys`)** | Restart in profiler-ready mode | `./benchmarks/profile_nsys.sh <stage-name>` | Shows where time is spent across CPU, CUDA, and GPU activity | `.nsys-rep`, `.sqlite`, `nsys_stats.txt`, `nsys_metrics.json` |
| **4b. Profile (`ncu`)** | Stop the server if using standalone replay | `./benchmarks/profile_ncu.sh <stage-name>` | Examines selected kernels and hardware behavior | `.ncu-rep` and `ncu_metrics.json` (Roofline data) |
| **5. Plot** | — | `uv run python benchmarks/plot_dashboard.py` | Compares benchmark and profiler results | `benchmarks/results/dashboard.html` |
| **6. Record** | — | Save notes and commit relevant files | Preserves the setup, measurements, and conclusions | Reproducible experiment record |

> **Important:** The UI is for sanity checks, not performance measurement. Stop it before benchmarking or profiling. Commands below show the current vLLM reference setup; use the corresponding launch command for another implementation.

---

## 1. Start the Server

Configuration lives in the root `config.yaml`. The reference server reads these settings and supports environment overrides where implemented. Check the configuration and launch script before each experiment, especially when switching implementations.

### Key Configuration Settings

| Setting | Current Default | Why It Matters |
| :--- | :--- | :--- |
| **Target model** (`model.name`) | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Keeps the reference workload fixed |
| **Draft model** (`draft_model.name`) | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | For speculative decoding experiments; not required for the baseline |
| **Quantized model** (`quantized_model.name`) | `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` | For quantization experiments; not required for the baseline |
| `server.host` | `0.0.0.0` | Interface the server listens on |
| `server.port` | `8000` | API port |
| `server.gpu_memory_utilization` | `0.70` | Serving-engine VRAM allocation target, not exact physical usage |
| `server.enable_prefix_caching` | `false` | Controls prefix caching when supported |
| `benchmarks.input_len` | `1024` tokens | Input workload length |
| `benchmarks.output_len` | `256` tokens | Output workload length |
| `benchmarks.random_seed` | `42` | Makes randomized workload generation more repeatable |

The table above summarizes shared settings; the actual `config.yaml` may contain additional implementation-specific parameters. The current reference workload also uses the settings below where configured:

| Setting | Reference Value | Purpose |
| :--- | :--- | :--- |
| `model.dtype` | `bfloat16` | Model precision |
| `model.max_model_len` | `9216` | Headroom for an 8192-token prompt, 64 output tokens, and chat-template overhead |
| `benchmarks.pareto` | `512` input / `128` output | Fixed workload for concurrency scaling |
| `benchmarks.context_sweep` | `[128, 256, 512, 1024, 2048, 4096, 8192]` input; `64` output | Measures context scaling at concurrency 1 |

> **Note on Repeatability:** A fixed random seed improves repeatability but does not guarantee identical outputs or timings across implementations. Check that requested input and output lengths are actually used.

### Standard Launch

In **Terminal 1**:

```bash
./vllm-reference/stage-0-baseline/serve.sh
```

Wait until the model has loaded and the server reports readiness. The reference API is available at `http://127.0.0.1:8000/v1`.

For another implementation, start its documented server script. Keep the API interface and port consistent where practical so the same client and benchmark harness can be reused.

### Profiler-Ready Launch (Nsight Systems)

To capture serving activity without including model startup, launch the reference server in profiler-ready mode:

```bash
NSYS=ON ./vllm-reference/stage-0-baseline/serve.sh
```

*Use this only if the implementation's launch script supports the `NSYS` option.*

### Runtime Overrides

Where supported by the launch script, override settings without editing `config.yaml`:

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.75 VLLM_PORT=8001 ./vllm-reference/stage-0-baseline/serve.sh
```

Record overrides with the experiment. A server bound to `0.0.0.0` listens on all available interfaces; a client in the same WSL environment can connect through `127.0.0.1`.

---

## 2. Check the Server

Before measuring performance, send a short request to confirm that the server behaves correctly.

### Option A: Web UI

In **Terminal 2**, while the server is running:

```bash
uv run chainlit run ui/app.py --port 8501
```

Open `http://localhost:8501`, submit a short prompt, and verify that:
- The request completes without an API or server error.
- The response is plausible.
- Streaming starts and continues correctly.
- Any displayed metrics populate as expected.

> **Reminder:** Stop the UI with `Ctrl+C` before benchmarking. UI rendering and networking add overhead, so do not use UI timings as performance results.

### Option B: Quick API Probe

For the current reference model, a short request can be sent directly to the API:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen2.5-Coder-1.5B-Instruct","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}'
```

For another engine, use its served model name and compatible request format. A successful response verifies basic functionality, not performance.

---

## 3. Run Benchmarks

Benchmarks require an active server. Run one sweep at a time, stop the UI, avoid competing GPU workloads, and keep the workload consistent across stages. The benchmark scripts read shared settings from `config.yaml`.

### Closed-Loop Sweeps (Standard)

In **Terminal 2**:

```bash
./benchmarks/bench_closed_loop.sh <stage-name>
```

For the current reference baseline, use:

```bash
./benchmarks/bench_closed_loop.sh stage-0-vllm
```

The standard closed-loop script runs two sweeps:

1. **Concurrency sweep:** Uses a 512-token prompt and 128 generated tokens across `C = {1, 2, 4, 8, 16, 32, 64}`. This shows how aggregate throughput and latency change as more requests are active, and helps identify saturation.
2. **Context-length sweep:** Uses `C = 1`, prompt lengths `[128, 256, 512, 1024, 2048, 4096, 8192]`, and 64 generated tokens. This shows how prompt processing and generation change as the initial context grows.

### Run One Sweep at a Time

When iterating on a specific optimization:

```bash
# Concurrency sweep only
./benchmarks/bench_closed_loop.sh <stage-name> pareto

# Context-length sweep only
./benchmarks/bench_closed_loop.sh <stage-name> context
```

Use the stage name to keep outputs separate. Results are stored under `benchmarks/results/<stage-name>/`.

### Open-Loop Arrival Rate Testing (Experimental Strain)

While closed-loop sweeps measure capacity by keeping client concurrency fixed, open-loop benchmarks test behavior under varying offered loads (Poisson/QPS arrival rates):
```bash
./benchmarks/bench_open_loop\(temporary\).sh <stage-name>
```
This strain explores queue buildup, scheduling stalls, and tail-latency degradation when requests arrive independently of response completions.

### Keep Comparisons Fair

Before comparing two implementations, confirm that they use equivalent:
- **Model & Precision:** Model weights, tokenizer, and data type.
- **Workload:** Input and output lengths.
- **Load Profile:** Concurrency levels and/or offered arrival rates.
- **Methodology:** Warm-up and measurement procedures.
- **Engine Features:** Relevant caching, batching, and scheduling settings.

> **Verification:** If an implementation cannot support the same workload or feature, document the difference rather than treating the results as directly equivalent. Check the raw JSON for failed requests, missing observations, unexpected token counts, and other anomalies before drawing conclusions. Repeat noisy measurements when practical.

---

## 4. Profile GPU Behavior

Benchmark first to identify **what** changed; profile next to investigate **why**. Profiling adds overhead, so use unprofiled benchmark runs for the main performance comparison.

### A. Nsight Systems (`nsys`): Where Does Time Go?

Nsight Systems captures the overall timeline: CPU activity, CUDA calls, GPU kernels, synchronization, transfers, and periods with no recorded GPU work. It helps locate where time is spent before you investigate individual kernels.

* **Terminal 1:** Start the reference server in profiler-ready mode:
  ```bash
  NSYS=ON ./vllm-reference/stage-0-baseline/serve.sh
  ```
* **Terminal 2:** Run the profile:
  ```bash
  ./benchmarks/profile_nsys.sh <stage-name>
  ```

For the reference baseline, use `stage-0-vllm` as the stage name. For another engine, use its compatible launch/profile procedure.

**Outputs:**
- `timeline.nsys-rep` (in `benchmarks/profiles/<stage-name>/`) — Full trace for the Nsight Systems GUI.
- `timeline.sqlite` (in `benchmarks/profiles/<stage-name>/`) — SQLite database with CUPTI activity and runtime records.
- `nsys_stats.txt` (in `benchmarks/profiles/<stage-name>/`) — Summary tables for GPU kernels, CUDA API, memory ops, and OSRT.
- `nsys_metrics.json` (in `benchmarks/results/<stage-name>/`) — Timeline and kernel breakdown metrics (visualized in Table A within `dashboard.html`).

**Interpret timeline metrics carefully:**
- **GPU Active Time:** Should represent the union of active intervals if reported as wall-clock activity. Do not add overlapping kernel durations and call the sum GPU busy time.
- **GPU Idle Gaps:** Show periods without recorded GPU work, but do not automatically prove CPU overhead.
- **CUDA API Wait Time:** Means a host thread is waiting; it does not necessarily mean the GPU is idle.
- **Efficiency vs. Activity:** GPU activity does not by itself mean the kernels are efficient or compute-bound.
- **Phase Ranges:** Separate prefill and decode ranges are reported for the request.

---

### B. Nsight Compute (`ncu`): Why Is a Kernel Slow?

Nsight Compute examines selected kernels in more detail, including execution time, memory traffic, compute activity, occupancy indicators, register use, and local-memory spills. It helps investigate a bottleneck identified by benchmarking or the timeline.

* **Terminal 1:** Stop the HTTP server if the profiling script uses standalone replay.
* **Terminal 2:** Run the kernel profiler:
  ```bash
  ./benchmarks/profile_ncu.sh <stage-name>
  ```

For the reference baseline, use `stage-0-vllm`. The current script performs an eager forward pass on selected operations, including Layer 0 and `lm_head`, for a prefill shape and a decode shape. Check the script for the exact shapes and replay procedure before interpreting results.

**Outputs:**
- `kernels.ncu-rep` (in `benchmarks/profiles/<stage-name>/`) — Full trace for the Nsight Compute GUI.
- `ncu_metrics.json` (in `benchmarks/results/<stage-name>/`) — Kernel metrics and Roofline coordinates (visualized in Table B and Plot 4 within `dashboard.html`, showing achieved DRAM, attained compute, and arithmetic intensity).

Useful evidence includes kernel duration, DRAM throughput relative to the supported peak, relevant compute-pipeline activity, active-warps/occupancy indicators, registers per thread, and local-memory spills. Metric names vary by GPU and Nsight Compute version, so validate them against the installed tool.

> **Caution:** Do not treat fixed thresholds such as "high DRAM percentage" or "many registers" as universal verdicts. SM throughput is not the same as Tensor Core utilization, and no single counter proves the bottleneck. Interpret counters alongside the kernel, input shape, precision, and timeline.

---

### Recommended Profiling Order

1. **Benchmark:** Use benchmark results to find a meaningful regression or opportunity.
2. **Timeline (`nsys`):** Use Nsight Systems to locate where time is spent.
3. **Kernel (`ncu`):** Use Nsight Compute on important kernels to investigate their behavior.
4. **Modify:** Make one targeted change.
5. **Verify:** Re-run the unprofiled benchmark under the original conditions.
6. **Document:** Keep the evidence that supports or contradicts your hypothesis.

---

## 5. Generate Dashboard Plots

Generate the interactive Plotly dashboard from benchmark results and available profiler data:

```bash
# Plot all stages found under benchmarks/results/
uv run python benchmarks/plot_dashboard.py

# Plot one stage
uv run python benchmarks/plot_dashboard.py --stages stage-0-vllm

# Compare stages
uv run python benchmarks/plot_dashboard.py --stages stage-0-vllm stage-1-naive -o comparison.html
```

Open the generated HTML file in a browser. By default, the dashboard is saved to `benchmarks/results/dashboard.html`; when using `-o`, check the path supplied to that option.

### The Four Dashboard Views

1. **Throughput vs. Latency:** Shows aggregate output-token throughput against a selected latency measure, such as ITL P50/P90/P99 or mean TPOT, across concurrency levels. This is a trade-off curve; it is not automatically a mathematically calculated Pareto frontier.
2. **Prompt Length vs. TTFT:** Shows how time to first token changes as input length grows. TTFT is a practical, prefill-focused measure, not pure prefill kernel time; it also includes request and serving overhead. FlashAttention does not change dense attention's $O(S^2)$ arithmetic complexity; it reduces memory traffic and avoids materializing the full attention matrix through tiling.
3. **Initial Context Length vs. Generation Speed:** Shows how TPOT or tokens per second per request changes as the KV-cache history grows. Keep the generation window and concurrency fixed. Batch size one is useful for studying single-request behavior, but may not show the full benefit of techniques designed for larger batches.
4. **Hardware Roofline:** Plots attained compute throughput against arithmetic intensity (FLOP/byte) with appropriate memory-bandwidth and compute ceilings. Use consistent metrics and the correct precision-specific compute ceiling. Roofline helps characterize kernel limits but does not explain every system bottleneck.

> **Consistency:** Keep units, workload settings, and axis definitions consistent across stages. Distinguish measured values from derived values.

---

## 6. Record and Compare Experiments

For each meaningful stage, record:

1. **Implementation and version:** Stage name, Git commit, and relevant framework/library versions.
2. **Hardware and environment:** GPU model, driver, CUDA, and profiler versions.
3. **Settings:** Exact `config.yaml` values, environment overrides, and implementation-specific flags.
4. **Workload:** Prompt/output lengths, concurrency or arrival rate, warm-up, repetitions, and any unsupported features.
5. **Artifacts:** Raw benchmark JSON/CSV, plots, profiler summaries, and relevant reports.
6. **Conclusion:** What changed, what improved or regressed, the likely explanation, and the limitations of the result.

Keep raw measurements; plots alone are not enough to reproduce or inspect an experiment. Full `.nsys-rep`, `.sqlite`, `nsys_stats.txt`, and `.ncu-rep` files can be large, so retain them locally or in separate artifact storage if committing them would make the repository unwieldy.

Example commit after reviewing the artifacts:

```bash
git add benchmarks/ config.yaml
git commit -m "feat(bench): record <stage-name> baseline results"
```

> **Value of Negative Results:** Do not discard unexpected or negative results. They can reveal a bottleneck, an invalid assumption, or a limitation of the optimization.

---

## Metrics Reference

Check how the benchmark scripts calculate each field, and keep the definition consistent across implementations.

| Metric | Phase | Unit | Definition | What It Can Help Explain |
| :--- | :--- | :---: | :--- | :--- |
| **TTFT** | Prefill-focused | `ms` | Time from request dispatch to receipt of the first output token | Prompt-processing cost plus queueing and request/serving overhead |
| **TPOT** | Decode-focused | `ms/token` | Average time per generated token after the first, using consistent timing boundaries | Average generation-step cost; does not by itself identify the bottleneck |
| **ITL** | Decode | `ms` | Time between consecutive output-token arrivals | Token timing variation; high percentiles can reveal jitter or stalls |
| **Throughput** | Aggregate or per-request | `tokens/s` | Completed output tokens divided by a defined elapsed interval | Serving capacity under the measured workload |
| **Arithmetic intensity** | Kernel | `FLOP/byte` | Floating-point operations divided by DRAM bytes moved | Position relative to compute and memory-bandwidth ceilings in a Roofline model |

For a response with $N > 1$ generated tokens, a common TPOT definition is:

$$\mathrm{TPOT} = \frac{T_{\mathrm{last}} - T_{\mathrm{first}}}{N - 1}$$

Use consistent timestamp boundaries and verify the benchmark's implementation. TPOT is an average, not a percentile and not end-to-end latency. ITL percentiles describe the distribution of individual token intervals.

> **Diagnostic Caution:** Interpret results as evidence, not automatic diagnoses. High TTFT does not prove a GEMM bottleneck; high TPOT does not prove memory-bandwidth saturation; and higher GPU activity does not necessarily mean more useful work. Compare throughput alongside latency and error rate.

---

## Adding New Optimization Stages

Use the same workflow for each new vLLM configuration or self-made engine:

1. **Keep the reference unchanged:** Create a new implementation or stage directory rather than modifying the reference baseline to test an idea.
2. **Keep workloads identical:** Do not change shared benchmark sizes between stages for a direct comparison.
3. **Isolate the optimization:** Change one kernel or algorithmic mechanism at a time where practical.
4. **Reuse the harness:** Use the shared configuration and benchmark scripts when their assumptions still apply. Document any required adapter or workload difference.
5. **Use a unique stage name:** Pass it to the benchmark and profiling scripts so outputs remain separate.
6. **Validate before measuring:** Confirm functional correctness, then benchmark, plot, and profile as needed.
7. **Record the conclusion:** Include negative results and any differences that limit the comparison.

### Example Stage Execution

```bash
./benchmarks/bench_closed_loop.sh stage-1-naive
./benchmarks/profile_nsys.sh stage-1-naive
./benchmarks/profile_ncu.sh stage-1-naive
uv run python benchmarks/plot_dashboard.py --stages stage-0-vllm stage-1-naive -o comparison.html
```

Keep implementation-specific launch and profiling details near that implementation, while this file remains the common procedure. Update commands and defaults when repository scripts change, and label planned functionality as planned until it is implemented and tested.