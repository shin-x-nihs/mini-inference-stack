# Engineering Workflow

A quick guide to running the server, checking it, benchmarking it, and understanding the results. Keep experiments repeatable: change one relevant variable at a time and record anything that differs.

## Workflow at a glance

| Step | What to do | Main output |
|---|---|---|
| 1. Start | Launch the vLLM reference server | API at `127.0.0.1:8000/v1` |
| 2. Check | Send a short request through the UI or API | Functional sanity check |
| 3. Benchmark | Run closed-loop and open-loop sweeps | Raw JSON results |
| 4. Plot | Generate the results dashboard and PNG exports | `benchmarks/results/dashboard.html` and plots |
| 5. Profile | Use Nsight Systems, then Nsight Compute | Timeline and kernel-level evidence |
| 6. Record | Save settings, environment, and conclusions | Reproducible experiment record |

The UI is for functional checks, not performance measurement. Stop it before benchmarking.

---

## 1. Configuration

Shared settings live in the root `config.yaml`. Keep the configuration and benchmark scripts aligned.

### Models

| Model | Status |
|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Current reference model |
| `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Planned draft model for speculative decoding |
| `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` | Planned quantized-model evaluation |

Only the reference model is needed for the initial baseline.

### Key defaults

| Setting | Default | Notes |
|---|---:|---|
| `server.host` | `0.0.0.0` | Listen on available interfaces |
| `server.port` | `8000` | API port |
| `server.gpu_memory_utilization` | `0.70` | vLLM allocation target, not exact VRAM usage |
| `server.enable_prefix_caching` | `false` | Keeps prefix caching disabled for the baseline |
| `benchmarks.input_len` | `1024` tokens | Default benchmark prompt length |
| `benchmarks.output_len` | `256` tokens | Default generated length |
| `benchmarks.random_seed` | `42` | Helps repeat randomized workloads |

Some dashboard plots use their own fixed workloads to isolate a specific effect. Document those settings with each plot rather than assuming every plot uses the defaults above. Always check that the scripts actually send the intended token lengths.

A server bound to `0.0.0.0` can be reached locally through `127.0.0.1`. If you change the model, context length, or concurrency, check actual GPU memory use.

---

## 2. Start the reference server

In Terminal 1:

```bash
./vllm-reference/stage-0-baseline/serve.sh
```

Wait until the model has loaded and the server reports that it is ready. The API is available at:

```text
http://127.0.0.1:8000/v1
```

For a supported runtime override, for example:

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.75 ./vllm-reference/stage-0-baseline/serve.sh
```

---

## 3. Check the server

In Terminal 2:

```bash
uv run chainlit run ui/app.py --port 8501
```

Open `http://localhost:8501` and send a short prompt. Check that the response is coherent, streaming works, the request completes, and any displayed metrics appear.

Stop the UI with `Ctrl+C` before benchmarking. UI rendering and background requests can affect measurements; a direct API or `curl` request is also fine for a sanity check.

---

## 4. Run benchmarks

Run one sweep at a time and avoid other GPU workloads.

### Closed-loop saturation

```bash
./benchmarks/bench_closed_loop.sh stage-0-vllm
```

The client concurrency sweep is configured in the benchmark script. The planned dashboard sweep is `C = {1, 2, 4, 8, 16, 32, 64}`; add `32` and `64` only if the machine can run them without out-of-memory errors or unstable measurements. Keep the documented sweep, script, and result filenames in sync.

Each client submits another request as soon as its previous request completes. This shows how aggregate throughput and latency change as concurrency rises.

### Open-loop arrival-rate sweep

```bash
./benchmarks/bench_open_loop.sh stage-0-vllm
```

The current configured offered rates are:

`λ = {1, 2, 4, 6, 8}` requests/s

Requests arrive independently of response completion, making this useful for observing queueing and tail latency. Offered request rate is not the same as completed request rate; if the server cannot keep up, queues and latency can grow.

### Before comparing implementations

Keep the model, tokenizer, precision, workload lengths, concurrency or arrival rates, warm-up, measurement procedure, and relevant caching settings equivalent. Record any differences. Different workloads are not directly comparable.

---

## 5. Generate plots

The target results view is an interactive four-plot dashboard at:

```text
benchmarks/results/dashboard.html
```

Export PNG versions for the README or reports. Keep the raw JSON/CSV measurements alongside the plots. Update the plotting command and output paths here to match the implementation when the dashboard is wired into the script.

The dashboard plots use distinct, documented workloads:

- **Throughput vs. Latency:** aggregate output throughput against a selectable latency metric, with fixed input and output lengths.
- **Prompt Size vs. Time to First Token:** prompt length against TTFT at concurrency 1 and one output token. TTFT is a practical prefill-focused proxy, not pure prefill time.
- **Context Length vs. Generation Speed:** initial KV-cache context length against TPOT or per-request generation rate, at concurrency 1 with a fixed generation window.
- **Roofline:** attained kernel performance against arithmetic intensity, with relevant memory-bandwidth and compute ceilings.

Check for failed requests, missing observations, unexpected token counts, and outliers before interpreting the plots.

---

## 6. Metrics

| Metric | Meaning |
|---|---|
| **TTFT** | Time from request dispatch to the first generated token |
| **TPOT** | Average time per output token after the first token |
| **ITL** | Time between consecutive generated tokens |
| **Throughput** | Completed output tokens per second over a defined interval |
| **Tail latency** | A high percentile, such as P95 or P99, of the stated latency measure |
| **E2E latency** | Time from request dispatch until the response completes |

For a response with more than one output token:

\[
\mathrm{TPOT} = \frac{T_{\mathrm{last}} - T_{\mathrm{first}}}{N - 1}
\]

Use consistent timestamp boundaries and confirm how the benchmark reports the metric. TPOT is an average; it is not a percentile and does not show variation between individual token intervals. ITL percentiles describe that variation.

For closed-loop tests, read throughput and latency together as concurrency rises. For open-loop tests, check whether the server sustains the offered rate and how queueing and latency change.

---

## 7. Profile GPU behavior

Use profiling to explain *why* a result changed, not as a replacement for end-to-end benchmarks.

- **Nsight Systems (`nsys`):** inspect the CPU/GPU timeline, kernel launches, transfers, synchronization, and idle gaps.
- **Nsight Compute (`ncu`):** inspect individual kernels, including memory behavior, compute utilization, occupancy, registers, and spills.

Start with a representative workload and a clear question. Use Systems to identify an interesting kernel or gap, then Compute to investigate that kernel. Profile runs can add overhead, so do not compare their timings directly with unprofiled benchmark results.

GPU-busy time is not the same as useful computation. Idle gaps can have several causes, so use the timeline and supporting evidence before attributing them to CPU overhead.

---

## 8. Keep experiment records

For each meaningful run, save:

- Model, precision, serving settings, and workload parameters.
- GPU model, driver, CUDA, framework, and relevant software versions.
- Git commit, warm-up settings, and number of repetitions.
- Concurrency or offered arrival rate, throughput, latency metrics, and error counts.
- Raw results and relevant profiler reports.
- A short conclusion, including limitations or unexpected results.

Keep previous result directories rather than overwriting them. A negative or surprising result can still reveal a bottleneck.

---

## 9. Add new experiments

1. Keep the reference baseline unchanged.
2. Check correctness before measuring performance.
3. Reuse the benchmark harness when its workload and metrics fit.
4. Record implementation-specific settings and limitations.
5. Save results in a new directory.

Add planned components to this workflow once they are implemented and tested. The goal is simple: make every experiment understandable and every comparison fair.
