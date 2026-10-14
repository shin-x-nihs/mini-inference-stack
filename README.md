# mini-inference-stack

<p align="center">
  <b>An educational project exploring LLM inference performance, from a vLLM serving baseline to custom GPU kernels.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-NVIDIA%20Blackwell%20(sm__120)-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="Blackwell Architecture" />
  <img src="https://img.shields.io/badge/CUDA-13.1-green?style=for-the-badge&logo=nvidia" alt="CUDA 13.1" />
  <img src="https://img.shields.io/badge/PyTorch-2.13.0%2Bcu130-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Environment-uv-blueviolet?style=for-the-badge" alt="uv package manager" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License" />
</p>

---

The project starts with `Qwen2.5-Coder-1.5B-Instruct` on an NVIDIA RTX 5080 Laptop GPU. The goal is to understand how inference performance changes as different optimizations are introduced, and to **measure those changes rather than assume them**.

The reference implementation uses **vLLM**. A separate educational implementation will be developed incrementally, with a focus on correctness, GPU execution, memory behavior, and performance analysis.

---

## Goals

* Establish a reproducible vLLM serving baseline.
* Measure latency and throughput under controlled workloads.
* Build a simple inference implementation to understand the execution path.
* Investigate the impact of KV caching, attention kernels, and request scheduling.
* Use GPU profiling to explain performance bottlenecks and optimization results.
* Compare implementations under equivalent conditions, documenting both improvements and regressions.

> The focus is not to reproduce an entire production inference engine. It is to understand the important design decisions and performance trade-offs behind one.

---

## Project Status

**Current phase:** `vLLM reference baseline`

The initial milestone is to get the reference environment running, benchmark a fixed workload, and establish a reliable measurement process.

| Component | Status |
| :--- | :--- |
| **Hardware and software environment** | Verified & documented |
| **vLLM serving baseline** | Initial milestone (In progress) |
| **Benchmark harness** | Planned |
| **Educational inference implementation** | Planned |
| **Custom CUDA/Triton kernels** | Planned |
| **Profiling and performance comparisons** | To be added as experiments are completed |

*Only experiments that have actually been run will be reported as results.*

---

## Hardware and Software

The initial target environment is:

| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 5080 Laptop GPU (16 GB GDDR7) |
| **GPU Architecture** | NVIDIA Blackwell, compute capability 12.0 (`sm_120`) |
| **Operating System** | Windows 11 with WSL2 (Ubuntu 24.04 LTS) |
| **CUDA Toolkit** | CUDA 13.1 (`nvcc 13.1.80`) |
| **Python Runtime** | Python 3.12 managed via `uv` |
| **Deep Learning Framework** | PyTorch 2.13.0+cu130 (Nightly with `sm_120` SASS support) |
| **Reference Engine** | vLLM 0.31.0 |
| **Target Model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| **GPU Profiling** | NVIDIA Nsight Systems (`nsys`) & Nsight Compute (`ncu`) |

The initial model is deliberately small enough to make iterative experiments practical on a laptop GPU. Actual memory use and available KV-cache capacity will be measured rather than inferred from model weight size alone.

---

## Architecture

The project separates the reference serving stack from the educational implementation:

```text
mini-inference-stack/
├── vllm-reference/     # Reference serving and benchmark configuration
├── benchmarks/         # Workloads, measurements, and analysis
├── custom-serving/     # Educational inference implementation
└── docs/               # Experiment notes and technical explanations
```

* The educational implementation will be built incrementally. Its internal structure may evolve as experiments reveal which abstractions are useful.
* The reference implementation will remain separate so that changes to the educational engine do not alter the baseline.

---

## Reference Baseline

The first milestone is to serve the selected model with vLLM through its OpenAI-compatible API.

The baseline will establish:
* Whether the model runs reliably in the target environment.
* Latency and throughput at different request concurrency levels.
* GPU memory consumption under the selected workload.
* The behavior of the serving stack during prefill and autoregressive decoding.
* A reproducible configuration for subsequent comparisons.

*vLLM is a practical reference implementation, not a theoretical performance ceiling. Results will be interpreted in the context of the tested workload and configuration.*

---

## Benchmark Methodology

Performance comparisons are meaningful only when the workload and measurement procedure are controlled.

The benchmark harness will progressively support:
* Fixed prompt and output lengths.
* Controlled concurrency and request counts.
* Warm-up runs followed by repeated measurements.
* Separate prefill-oriented and decode-oriented workloads.
* Recorded configuration, software versions, and raw measurements.

### Metrics

#### 1. Time to First Token (TTFT)
The elapsed time from sending a request to receiving its first generated output token. It captures serving overhead, queueing, and prefill work.

#### 2. Time Per Output Token (TPOT)
The average time per generated token after the first token. For a response containing $N > 1$ output tokens:

$$\text{TPOT} = \frac{T_{\text{end}} - T_{\text{first}}}{N - 1}$$

The exact timing boundaries will be defined consistently in the benchmark harness.

#### 3. Inter-Token Latency (ITL)
The time between consecutive generated tokens. Its distribution helps reveal latency variation that an average can hide.

#### 4. Throughput
The number of tokens generated per second over a defined measurement interval. Output-token throughput and total-token throughput will be distinguished where relevant.

*Latency percentiles, including P50, P90, and P99, will be reported when the number of observations makes them meaningful.*

---

## GPU Profiling

Profiling will be used to explain measured behavior, not just to collect counters.

* **Nsight Systems (`nsys`)**: Used to identify kernel launch overhead, synchronization, CPU-GPU interaction, and gaps in GPU execution.
* **Nsight Compute (`ncu`)**: Used to investigate selected kernels, including memory throughput, occupancy, register usage, and other relevant hardware metrics.

*Profiling results will be interpreted alongside end-to-end measurements. A faster individual kernel does not necessarily produce a faster inference system.*

---

## Optimization Roadmap

The following are planned investigations, not completed features. Their order may change as profiling and correctness testing provide evidence about where optimization is most useful.

1. **Simple Autoregressive Inference**  
   Build a minimal implementation to understand tokenization, model execution, sampling, and the generation loop. Establish correctness before focusing on speed.
2. **KV Caching**  
   Compare generation with and without cached keys and values. Measure how the cost of generating tokens changes as the context grows.
3. **Prefill Attention**  
   Study the memory traffic and intermediate storage of attention. Compare an appropriate PyTorch attention backend with a custom tiled implementation using Triton or CUDA.
4. **Decode Attention**  
   Investigate attention over the growing KV cache, including memory access patterns and opportunities to increase GPU parallelism.
5. **Request Scheduling and KV-Cache Management**  
   Explore batching, paged KV-cache allocation, and the effects of request lengths and concurrency on throughput and latency.
6. **Speculative Decoding and KV Reuse**  
   Investigate draft-model-based speculative decoding and reuse of cached prefixes. Evaluate the end-to-end benefit under workloads where these techniques are applicable.

*Each investigation should include correctness checks, a baseline, measured results, and an explanation of the observed trade-offs.*

---

## Results and Experiment Notes

Results will be added as experiments become reproducible. Each experiment will record:
* Hardware and software versions.
* Model revision, precision, and relevant serving settings.
* Prompt and output lengths, request count, and concurrency.
* Warm-up procedure and number of repetitions.
* Latency and throughput measurements.
* Relevant profiling evidence.
* Interpretation, limitations, and any unexpected results.

Raw data and generated profiling traces will be kept separate from the smaller results and analysis intended for version control. **No performance improvement will be claimed without a corresponding measurement and a clearly defined comparison.**

---

## Reproducibility

The project uses Git to track implementation changes, benchmark configurations, and experiment notes.

Setup and benchmark instructions will be documented alongside the relevant scripts once the reference environment has been validated. Hardware-specific limitations and deviations from the intended setup will be recorded rather than hidden.

---

## License

This project is intended to be released under the [MIT License](LICENSE).
