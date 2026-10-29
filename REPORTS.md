# Engineering Reports & Optimization Log

This document records the main experiments across the `vllm-reference` and `custom-serving` tracks. Each report explains what we observed, what we expected, what we tested, and what we learned.

---

## vllm-reference

### Stage 1: Speculative Decoding (`stage-1-vllm`)

*Server directory: `vllm-reference/stage-1-spec-dec/` | Benchmark stage: `stage-1-vllm`*

---

### 1. Motivation

#### Observation (Stage 0 Baseline)  
Profiling the baseline vLLM engine running `Qwen2.5-Coder-1.5B-Instruct` on an RTX 5080 Laptop GPU showed that token generation is overwhelmingly dominated by the decode phase, specifically dense linear layers:
* **Decode dominance:** Decode accounts for **77.5%** of total wall time (672.3 ms out of 867.3 ms for 1024 prompt / 64 output tokens at concurrency $C=1$).
* **Linear layer bottleneck:** Matrix multiplications consume **91.9%** (704.9 ms) of all GPU kernel time, whereas attention accounts for only **0.7%** (5.1 ms).
* **Memory-bound execution:** Nsight Compute reports that `Decode Linear GEMV` runs at an Arithmetic Intensity of **1.001 FLOP/Byte**, streaming weights from DRAM at **321.1 GB/s** (62.7% of physical peak) while attaining only **0.32 TFLOP/s** of compute (0.3% of Tensor Core peak).

Streaming entire model weights from VRAM for a single token vector ($M=1$) leaves compute hardware mostly idle.

#### Hypothesis  
Speculative decoding can reduce this per-token weight-streaming cost by having a smaller draft model propose $K$ tokens sequentially, followed by a single forward pass on the target model to verify all $K+1$ token positions in parallel.

Processing multiple tokens in parallel shifts the target model's projection from a vector-matrix multiply ($\text{GEMV}, M=1$) toward a matrix-matrix multiply ($\text{GEMM}, M=K+1$), increasing arithmetic intensity and reusing loaded weights across multiple tokens.

#### Speedup Model  
The per-round latency speedup can be modeled as:

$$S \approx \frac{T_{\text{baseline}}}{T_{\text{spec}}} = \frac{L_{\text{acc}} \cdot t_{\text{target}}}{K \cdot t_{\text{draft}} + t_{\text{verify}}}$$

Where:
* **$L_{\text{acc}}$**: Mean output tokens produced per verification round ($1 \le L_{\text{acc}} \le K + 1$).
* **$t_{\text{target}}$**: Time for one standard decode step on the target model alone.
* **$t_{\text{draft}}$**: Time for one decode step on the draft model.
* **$t_{\text{verify}}$**: Time for the target model to verify all proposals in parallel.

**Idealized Estimate:**  
In a single-stream memory-bound regime ($C=1$), assuming verification takes roughly the same time as single-token decode ($t_{\text{verify}} \approx t_{\text{target}}$) and 100% acceptance ($L_{\text{acc}} = K + 1$):

$$S_{\text{ideal}} \approx \frac{K + 1}{K \cdot c + 1} \quad \text{where } c = \frac{t_{\text{draft}}}{t_{\text{target}}}$$

Estimating $c$ from parameter count ($0.49\text{B} / 1.54\text{B} \approx 0.32 \approx \frac{1}{3}$) with $K = 5$:

$$S_{\text{ideal}} \approx \frac{5 + 1}{5 \cdot \frac{1}{3} + 1} = \frac{6}{2.67} \approx 2.25\times$$

*Note: This is an idealized ceiling assuming zero engine overhead and 100% acceptance. For formal statistical derivations, see Leviathan et al. (ICML 2023) and Chen et al. (2023).*

#### Expected Results:  
* **Low concurrency ($C=1\text{--}2$):** Lower TPOT and higher throughput if the savings from skipped target passes outweigh draft and verification costs.
* **High concurrency ($C \ge 8$):** Neutral or negative speedup. Batching multiple concurrent requests already forms large GEMMs naturally; adding draft iterations wastes compute and bandwidth on rejected proposals.
* **Kernel behavior:** The target model's decode linear kernel should shift from GEMV to GEMM with higher Arithmetic Intensity.

---

### 2. Experiment Setup

| Setting | Value |
| :--- | :--- |
| **Target model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`) |
| **Draft model** | `Qwen/Qwen2.5-Coder-0.5B-Instruct` (`bfloat16`) |
| **Speculative length ($K$)** | 5 draft tokens |
| **Sampling** | Greedy (`temperature: 0.0`) |
| **Prompt** | Standard fixed Python code prompt |
| **Backend** | vLLM 0.31.0 (`--speculative-config '{"method": "draft_model", ...}'`) |
| **Hardware** | NVIDIA RTX 5080 Laptop GPU (16 GB GDDR7, `sm_120`), CUDA 13.1 |

**Validation:**  
Confirmed functional correctness via UI probe and verified zero failed requests across all benchmark JSON runs. Output text remained strictly identical between baseline and speculative runs under greedy verification.

---

### 3. Benchmark Results

#### A. Concurrency Scaling (Input 512, Output 128)

| Concurrency ($C$) | Baseline Throughput | Speculative Throughput | Throughput Change | Baseline TPOT | Speculative TPOT | TPOT Change |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 87.8 tok/s | 106.0 tok/s | <span style="color:#10b981;">**+20.7%**</span> | 10.90 ms | 8.38 ms | <span style="color:#10b981;">**-23.1%**</span> |
| **2** | 169.5 tok/s | 182.2 tok/s | <span style="color:#10b981;">**+7.5%**</span> | 10.98 ms | 9.06 ms | <span style="color:#10b981;">**-17.5%**</span> |
| **4** | 353.7 tok/s | 275.9 tok/s | <span style="color:#ef4444;">**-22.0%**</span> | 10.15 ms | 11.28 ms | <span style="color:#ef4444;">**+11.1%**</span> |
| **8** | 561.9 tok/s | 448.3 tok/s | <span style="color:#ef4444;">**-20.2%**</span> | 12.27 ms | 14.27 ms | <span style="color:#ef4444;">**+16.3%**</span> |
| **16** | 831.2 tok/s | 622.9 tok/s | <span style="color:#ef4444;">**-25.1%**</span> | 16.08 ms | 20.43 ms | <span style="color:#ef4444;">**+27.1%**</span> |
| **32** | 967.0 tok/s | 755.5 tok/s | <span style="color:#ef4444;">**-21.9%**</span> | 27.86 ms | 33.77 ms | <span style="color:#ef4444;">**+21.2%**</span> |
| **64** | 1232.3 tok/s | 872.5 tok/s | <span style="color:#ef4444;">**-29.2%**</span> | 45.15 ms | 59.06 ms | <span style="color:#ef4444;">**+30.8%**</span> |


#### B. Context Scaling Endpoints ($C=1$, Output 64)

| Metric | Baseline ($L=128$) | Speculative ($L=128$) | Baseline ($L=8192$) | Speculative ($L=8192$) |
| :--- | :---: | :---: | :---: | :---: |
| **Output Throughput** | 89.04 tok/s | 71.99 tok/s <span style="color:#ef4444;">**(-19.1%)**</span> | 50.95 tok/s | 24.62 tok/s <span style="color:#ef4444;">**(-51.7%)**</span> |
| **Mean TPOT** | 10.67 ms | 12.12 ms <span style="color:#ef4444;">**(+13.6%)**</span> | 9.24 ms | 13.43 ms <span style="color:#ef4444;">**(+45.3%)**</span> |
| **Mean TTFT** | 46.34 ms | 125.28 ms <span style="color:#ef4444;">**(+170.3%)**</span> | 673.66 ms | 1752.66 ms <span style="color:#ef4444;">**(+160.2%)**</span> |

#### C. Acceptance Dynamics

* **Acceptance rate ($\alpha$):** ~98.2% on the structured coding prompt under greedy decoding.
* **Mean accepted length ($L_{\text{acc}}$):** ~5.91 tokens emitted per verification pass (out of $K+1 = 6$).

#### D. The TPOT vs. ITL Trade-off

At concurrency $C=1$, mean TPOT dropped by <span style="color:#10b981;">**23.1%**</span> (10.90 ms $\to$ 8.38 ms), but Inter-Token Latency (ITL P50) surged from **10.91 ms to 42.86 ms** (and ITL P99 jumped from 14.5 ms to 69.1 ms).

This is a structural property of speculative decoding:
* In autoregressive decoding, the engine streams 1 token per forward pass ($\sim 10.9\text{ ms}$ cadence).
* In speculative decoding, unverified tokens cannot be streamed over HTTP SSE. The client waits for the entire draft + verify round ($\sim 43\text{ ms}$ silence), after which the engine flushes all $\sim 5\text{--}6$ accepted tokens at once in a burst.
* The ratio $\frac{\text{ITL P50}}{\text{TPOT}} = \frac{42.86\text{ ms}}{8.38\text{ ms}} \approx 5.11$ directly reflects the mean tokens produced per round. Aggregate speed is higher, but token delivery is chunked rather than steady.

---

### 4. Profiler & Hardware Analysis

Profiling traces (`timeline.nsys-rep` and `kernels.ncu-rep`) explain both why the kernel optimization worked and why system-level gains remained limited:

#### 1. Kernel Shift (Nsight Compute)
The hypothesis regarding the decode linear layer was confirmed on hardware:
* **Stage 0 Baseline:** Ran `Decode Linear GEMV` with $\text{AI} = \mathbf{1.001\text{ FLOP/Byte}}$, achieving $0.32\text{ TFLOP/s}$ at $321.1\text{ GB/s}$ DRAM throughput.
* **Stage 1 Speculative:** Ran `Decode Linear GEMM` with $\text{AI} = \mathbf{15.473\text{ FLOP/Byte}}$ (<span style="color:#10b981;">**$15.5\times$ increase**</span>) and attained performance jumped to $\mathbf{5.037\text{ TFLOP/s}}$ (<span style="color:#10b981;">**$15.7\times$ increase**</span>) at $325.6\text{ GB/s}$ DRAM throughput.
* Total decode duration on the target model decreased from **672.3 ms down to 556.5 ms** (average decode step dropped from 10.67 ms to 8.83 ms).

#### 2. Host Orchestration Overhead (Nsight Systems)
While the GPU executed target verification more efficiently, server-level orchestration introduced significant CPU overhead:
* **CPU launch overhead surged:** Rose from **28.90 ms** (1,636 calls, 63 CUDA graphs) to **277.00 ms** (19,843 calls, 13 CUDA graphs).
* **GPU idle gaps widened:** GPU idle time increased from **11.6% to 35.3%** of total wall time.
* Alternating between the draft model, target model, KV-cache managers, and rejection logic prevented full CUDA graph coverage and caused CPU launch serialization.

#### 3. The Model Size Ratio ($1.5\text{B} \text{ vs. } 0.5\text{B}$)
The draft model ($0.49\text{B}$) is roughly one-third the size of the target model ($1.54\text{B}$). Because the draft model is also memory-bandwidth bound, running $K=5$ sequential draft steps consumes $5 \times \frac{1}{3} \approx 1.67\times$ the time of an ordinary target decode step. 

Even with a 98% acceptance rate, draft-model execution and engine management consume most of the latency budget, capping real-world gains at $+20.7\%$ rather than the $2.25\times$ ideal ceiling.

---

### 5. Conclusion

1. **The kernel optimization worked:** Verifying batches of draft tokens shifted the decode linear layer from GEMV to GEMM, increasing Arithmetic Intensity from $1.0\text{ FLOP/B}$ to $15.5\text{ FLOP/B}$ and reducing target decode time from 672 ms to 556 ms.
2. **End-to-end gains are constrained by model ratio:** With a $1.5\text{B} / 0.5\text{B}$ pair, the draft model is not small enough relative to the target model. Drafting sequential tokens remains a major fraction of total round time.
3. **CPU launch overhead matters:** Speculative decoding increased launch overhead by nearly $10\times$ (28.9 ms $\to$ 277.0 ms), leaving the GPU idle for 35.3% of the run.
4. **Concurrency crossover:** The technique provides latency and throughput improvements only at low load ($C=1\text{--}2$). At $C \ge 4$, continuous batching naturally forms compute-dense GEMMs, making speculative drafting an active bottleneck (-20% to -29% throughput).

**Next Steps:**
* **Custom Track:** Transition to `custom-serving/stage-0-baseline/` to build a minimal PyTorch inference engine from scratch, measuring baseline execution without vLLM's internal optimizers and memory managers.
* **Future Speculative Reference:** Re-evaluate speculative decoding using a much larger size disparity (e.g., pairing a $14\text{B}\text{-AWQ}$ target model with the $0.5\text{B}$ draft model), where $t_{\text{draft}} \ll t_{\text{target}}$ and draft cost is negligible compared to target verification.

---

### 6. Data Sources and Reproducibility

| Reported Metric | Source Artifact | Counter / Field |
| :--- | :--- | :--- |
| **Throughput & TPOT** | `benchmarks/results/{stage}/pareto_c*.json` | `output_throughput`, `mean_tpot_ms` |
| **Context Latencies** | `benchmarks/results/{stage}/context_l*.json` | `mean_ttft_ms`, `mean_tpot_ms` |
| **Timeline Breakdown** | `benchmarks/profiles/{stage}/nsys_metrics.json` | `wall_time_ms`, `decode_total_ms`, `cpu_launch_overhead_ms` |
| **Kernel Roofline** | `benchmarks/profiles/{stage}/ncu_metrics.json` | `achieved_dram_gbps`, `attained_tflops`, `arithmetic_intensity_flop_byte` |
| **Interactive Plots** | `benchmarks/results/dashboard.html` | Generated comparison dashboard |

*All percentage deltas are calculated directly from raw benchmark JSON outputs. Profiler counters represent activity during sampled kernel windows and serve to support engineering hypotheses rather than absolute proof of single-variable causation.*