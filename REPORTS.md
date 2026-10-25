# Engineering Reports & Optimization Log

This document records the main experiments across the `vllm-reference` and `custom-serving` tracks. Each report explains what we observed, what we expected, what we tested, and what we learned.

---

## vllm-reference

### Stage 1: Speculative Decoding (`stage-1-spec-dec`)

#### Motivation

**Observation**  
The Stage 0 baseline suggests that single-request generation is dominated by the cost of running the target model one token at a time:
* **Decode time:** 491.2 ms out of 550.6 ms of wall time (89.2%) for the measured workload.
* **Linear layers:** Account for 96.2% of measured GPU kernel time, compared with 0.7% for attention.
* **Memory traffic:** Nsight Compute reports an arithmetic intensity of 1.001 FLOP/byte for the profiled decode linear kernel. The kernel reaches 320.58 GB/s, or 62.6% of peak DRAM bandwidth.
* **Context length:** Mean TPOT increases from 10.06 to 13.15 ms/token as the initial context grows from 128 to 8,192 tokens at concurrency 1.

Together, these measurements suggest that repeatedly reading model weights is a major cost during single-request decode.

**Hypothesis**  
Speculative decoding may reduce this cost by using a smaller draft model to propose several tokens, then verifying those proposals in a single target-model pass.  
The target model can process multiple tokens in parallel during verification, potentially making better use of GPU compute. Whether this improves overall performance depends on how many draft tokens are accepted and how much time drafting and verification add.

**Expected Results**  
* **Low concurrency:** Generation may become faster if the saved target-model steps outweigh draft-model and verification overhead.
* **High concurrency:** The benefit may shrink or reverse if regular batching already uses the GPU efficiently.
* **Kernel behavior:** Target-model verification should process multiple tokens per pass rather than performing only single-token decode. Profiler data can help explain the change, but end-to-end measurements determine whether it is worthwhile.

*These are predictions to test, not assumed outcomes.*

---

#### Experiment Setup

| Setting | Value |
| :--- | :--- |
| **Target model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| **Draft model** | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| **Precision** | `bfloat16` |
| **Draft length** | 5 speculative tokens |
| **Backend** | vLLM with draft-model speculative decoding |
| **GPU** | NVIDIA RTX 5080 Laptop GPU, 16 GB, `sm_120` |
| **Environment** | CUDA 13.1, WSL2 Ubuntu 24.04 |

**Benchmarks**
* **Concurrency sweep:** 512 input tokens, 128 output tokens; concurrency levels: 1, 2, 4, 8, 16, 32, and 64.
* **Context sweep:** Concurrency 1, input lengths from 128 to 8,192 tokens, 64 output tokens.
* Keep the baseline and speculative runs on the same hardware and workload settings. Record any configuration differences required to enable speculative decoding.
* **Validation:** Check that responses are valid and that streaming behaves as expected. Compare output correctness against the baseline, not just whether the UI displays a response.

---

#### Results

##### 1. Context Sweep ($C=1$, Input 128–8192, Output 64)

| Metric | Baseline | Speculative | Change |
| :--- | ---: | ---: | ---: |
| Mean TPOT (ms/token) | TBD | TBD | TBD |
| Generation speed (tokens/s/request) | TBD | TBD | TBD |
| Mean TTFT (ms) | TBD | TBD | TBD |
| Accepted draft tokens / step | N/A | TBD | TBD |

*Report the context length alongside each result. Include profiler measurements only where the same metric can be meaningfully compared between runs.*

##### 2. Concurrency Sweep (Input 512, Output 128)

| Concurrency | Baseline Throughput | Speculative Throughput | Baseline TPOT | Speculative TPOT |
| :---: | ---: | ---: | ---: | ---: |
| **1** | 92.0 tok/s | TBD | 10.38 ms | TBD |
| **2** | 154.1 tok/s | TBD | 12.12 ms | TBD |
| **4** | 223.8 tok/s | TBD | 15.20 ms | TBD |
| **8** | 388.9 tok/s | TBD | 16.90 ms | TBD |
| **16** | 599.5 tok/s | TBD | 21.40 ms | TBD |
| **32** | 937.2 tok/s | TBD | 28.87 ms | TBD |
| **64** | 1262.5 tok/s | TBD | 44.30 ms | TBD |

*Throughput is aggregate output tokens/s across requests. TPOT is mean time per output token after the first token. Lower TPOT is better; higher throughput is better.*

---

#### Data and Profiles

* **Baseline:** `benchmarks/results/stage-0-vllm/`
* **Speculative decoding:** `benchmarks/results/stage-1-spec-dec/`
* **Comparison dashboard:** `benchmarks/results/dashboard.html`

---

#### Conclusion

*(To be completed after running the benchmarks.)*

* Did speculative decoding improve generation speed? Under which workloads?
* How did acceptance rate and drafting overhead affect the result?
* At what concurrency, if any, did the benefit disappear?
* What did profiling reveal about the target-model execution?
* Is this approach worth pursuing further?

**Next step:** Continue with the next experiment based on the measured results. In parallel, the `custom-serving` track can establish a minimal PyTorch baseline to investigate the model's execution without vLLM's serving optimizations.