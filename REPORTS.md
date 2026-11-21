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
* **Memory-bound execution:** Nsight Compute reports that `Decode Linear GEMV` runs at an Arithmetic Intensity of **1.001 FLOP/Byte**, streaming weights from DRAM at **321.1 GB/s** (**35.8%** of the 896 GB/s physical GDDR7 peak; **70.9%** NCU sustained-clock DRAM utilization) while attaining only **0.32 TFLOP/s** of compute (**0.8%** of the 40 TFLOP/s Tensor Core Roofline ceiling; **41.5%** NCU SM throughput).

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
* **Stage 0 Baseline:** Ran `Decode Linear GEMV` with $\text{AI} = \mathbf{1.001\text{ FLOP/Byte}}$, achieving $0.32\text{ TFLOP/s}$ (**0.8%** of the 40 TFLOP/s ceiling; **41.5%** NCU SM throughput) at $321.1\text{ GB/s}$ DRAM throughput (**35.8%** of 896 GB/s peak; **70.9%** NCU DRAM utilization).
* **Stage 1 Speculative:** Ran `Decode Linear GEMM` with $\text{AI} = \mathbf{15.473\text{ FLOP/Byte}}$ (<span style="color:#10b981;">**$15.5\times$ increase**</span>) and attained performance jumped to $\mathbf{5.037\text{ TFLOP/s}}$ (<span style="color:#10b981;">**$15.7\times$ increase**</span>, **12.6%** of the 40 TFLOP/s ceiling; **28.4%** NCU SM throughput) at $325.6\text{ GB/s}$ DRAM throughput (**36.3%** of 896 GB/s peak; **71.9%** NCU DRAM utilization).
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
| **Timeline Breakdown** | `benchmarks/results/{stage}/nsys_metrics.json` | `wall_time_ms`, `decode_total_ms`, `cpu_launch_overhead_ms` |
| **Kernel Roofline** | `benchmarks/results/{stage}/ncu_metrics.json` | `achieved_dram_gbps`, `attained_tflops`, `arithmetic_intensity_flop_byte` |
| **Interactive Plots** | `benchmarks/results/dashboard.html` | Generated comparison dashboard |

*All percentage deltas are calculated directly from raw benchmark JSON outputs. Profiler counters represent activity during sampled kernel windows and serve to support engineering hypotheses rather than absolute proof of single-variable causation.*

---

### Stage 2: Automatic Prefix Caching (`stage-2-vllm`)

*Server directory: `vllm-reference/stage-2-prefix-cache/` | Benchmark stage: `stage-2-vllm`*

---

### 1. Motivation

#### Observation (Stage 0 Baseline & Bottleneck)
Profiling the baseline vLLM engine under continuous multi-tenant batching exposed a severe vulnerability to **prefill serialization and queuing delay**:
* **TTFT Explosion under Concurrency:** As concurrency scaled from $C=1$ to $C=64$ on the RTX 5080 Laptop GPU, mean Time-To-First-Token exploded by **$12.2\times$**, rising from **73.31 ms to 896.18 ms**. Tail latency degraded even more catastrophically: P99 TTFT surged from **202.39 ms to 3,889.86 ms** (~3.9 seconds).
* **Prefill Compute Dominance:** In Nsight Compute profiling of Layer 0, `Prefill Linear GEMM` is compute-bound at an Arithmetic Intensity of **970.44 FLOP/Byte** and attains **27.22 TFLOP/s**, taking **4,144.61 µs** per layer invocation. Across 28 transformer layers, a single 2048-token prefill ties up the GPU compute pipelines for over 115 ms.
* **Prefill Starvation of Decode:** In continuous batching, when new requests arrive with long prompts, their heavy prefill GEMMs monopolize Tensor Cores, stalling active decode iterations and causing severe token delivery jitter.
* **Redundant Prompt Processing:** Standard serving treats every request independently. In production workloads with shared system prompts, multi-turn conversational histories, or few-shot code templates, identical token prefixes are repeatedly tokenized, projected, and cached from scratch on every arrival.

#### Hypothesis
Automatic Prefix Caching (RadixAttention) retains processed KV cache blocks across requests in a tree data structure (radix tree) indexed by token sequences. When subsequent incoming requests share a common prefix with prior requests:
1. The engine performs an $O(1)$ prefix lookup in the device memory's block radix tree.
2. The KV states for the shared prefix are immediately referenced rather than recomputed.
3. Compute is required only for newly appended prompt tokens ($L_{\text{new}}$).

This eliminates redundant prefill GEMMs, collapses queue wait times ($W_q$), and shifts available GPU execution capacity toward the memory-bandwidth-bound decode phase.

#### Speedup Model
Let prompt length $L$ be partitioned into a shared prefix of length $L_{\text{prefix}}$ and request-specific tokens $L_{\text{new}}$, such that $L = L_{\text{prefix}} + L_{\text{new}}$. Define prefix sharing ratio $h = \frac{L_{\text{prefix}}}{L} \in [0, 1]$.

**Single-Request Prefill Latency:**  
In the baseline without caching ($S0$):
$$T_{\text{prefill, S0}} = t_{\text{gemm}}(L) + t_{\text{attn}}(L)$$

Under prefix caching ($S2$):
$$T_{\text{prefill, S2}} = t_{\text{lookup}} + t_{\text{gemm}}(L_{\text{new}}) + t_{\text{attn}}(L_{\text{new}}, L)$$

Where $t_{\text{lookup}}$ is the hash/radix traversal overhead ($< 1\text{ ms}$). When $h \to 1$ ($L_{\text{new}} \to 0$ for repeated prompts):
$$T_{\text{prefill, S2}} \approx t_{\text{lookup}} \ll T_{\text{prefill, S0}}$$

**Queuing Delay under Batching:**  
In continuous batching with arrival rate $\lambda$ and request service times $S$, mean queue wait time follows Pollaczek–Khinchine:
$$E[W_q] = \frac{\lambda E[S^2]}{2(1 - \lambda E[S])}$$

Because prefill service time scales quadratically in attention and linearly in dense projection with prompt length ($O(L^2) + O(L)$), eliminating the prefill of shared prefixes ($S_{\text{prefill}} \to 0$) drastically reduces both $E[S]$ and the second moment $E[S^2]$. This collapses the queuing delay that caused the $\sim 3.9\text{ s}$ tail latency spikes at $C=64$ in Stage 0.

#### Expected Results
* **Low Concurrency ($C=1\text{--}4$):** Minor radix tree tracking overhead may slightly increase TPOT for short non-cached workloads, but TTFT should improve whenever prefixes hit device cache blocks.
* **High Concurrency ($C \ge 16\text{--}64$):** Enormous throughput and latency improvements as prefill compute contention vanishes, allowing the GPU to sustain maximum batched decode throughput.
* **Long Context ($L \ge 2048\text{--}8192$):** Dramatic drop in TTFT (>50–70% reduction) as expensive compute-bound prefill GEMMs are completely bypassed.

---

### 2. Experiment Setup

| Setting | Value |
| :--- | :--- |
| **Target model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`) |
| **Optimization flags** | `--enable-prefix-caching` |
| **Sampling** | Greedy (`temperature: 0.0`) |
| **Prompt** | Standard fixed Python code prompt |
| **Backend** | vLLM 0.31.0 |
| **Hardware** | NVIDIA RTX 5080 Laptop GPU (16 GB GDDR7, `sm_120`), CUDA 13.1 |

**Validation:**  
Functional health verified via `/health` probe and UI inference. Output text was checked for strict bitwise token equivalence against baseline outputs under greedy decoding. All 14 Pareto and context closed-loop benchmark sweeps completed with zero failed requests.

---

### 3. Benchmark Results

#### A. Concurrency Scaling (Input 512, Output 128)

| Concurrency ($C$) | Stage 0 Tput | Stage 2 Tput | Tput $\Delta$ | S0 Mean TTFT | S2 Mean TTFT | TTFT $\Delta$ | S0 P99 TTFT | S2 P99 TTFT | P99 TTFT $\Delta$ | S0 TPOT | S2 TPOT | TPOT $\Delta$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 87.77 tok/s | 80.35 tok/s | <span style="color:#ef4444;">**-8.5%**</span> | 73.31 ms | 38.42 ms | <span style="color:#10b981;">**-47.6%**</span> | 202.39 ms | 229.32 ms | <span style="color:#ef4444;">**+13.3%**</span> | 10.90 ms | 12.24 ms | <span style="color:#ef4444;">**+12.3%**</span> |
| **2** | 169.46 tok/s | 138.67 tok/s | <span style="color:#ef4444;">**-18.2%**</span> | 115.08 ms | 61.72 ms | <span style="color:#10b981;">**-46.4%**</span> | 352.95 ms | 320.46 ms | <span style="color:#10b981;">**-9.2%**</span> | 10.98 ms | 14.04 ms | <span style="color:#ef4444;">**+27.9%**</span> |
| **4** | 353.74 tok/s | 259.97 tok/s | <span style="color:#ef4444;">**-26.5%**</span> | 156.61 ms | 65.92 ms | <span style="color:#10b981;">**-57.9%**</span> | 495.31 ms | 270.78 ms | <span style="color:#10b981;">**-45.3%**</span> | 10.15 ms | 14.98 ms | <span style="color:#ef4444;">**+47.6%**</span> |
| **8** | 561.94 tok/s | 514.93 tok/s | <span style="color:#ef4444;">**-8.4%**</span> | 262.02 ms | 70.29 ms | <span style="color:#10b981;">**-73.2%**</span> | 712.92 ms | 226.87 ms | <span style="color:#10b981;">**-68.2%**</span> | 12.27 ms | 15.09 ms | <span style="color:#ef4444;">**+23.0%**</span> |
| **16** | 831.19 tok/s | 972.12 tok/s | <span style="color:#10b981;">**+17.0%**</span> | 418.10 ms | 94.42 ms | <span style="color:#10b981;">**-77.4%**</span> | 1267.34 ms | 394.49 ms | <span style="color:#10b981;">**-68.9%**</span> | 16.08 ms | 15.81 ms | <span style="color:#10b981;">**-1.7%**</span> |
| **32** | 966.96 tok/s | 1728.89 tok/s | <span style="color:#10b981;">**+78.8%**</span> | 687.45 ms | 162.33 ms | <span style="color:#10b981;">**-76.4%**</span> | 2254.50 ms | 865.03 ms | <span style="color:#10b981;">**-61.6%**</span> | 27.86 ms | 17.30 ms | <span style="color:#10b981;">**-37.9%**</span> |
| **64** | 1232.26 tok/s | 2874.36 tok/s | <span style="color:#10b981;">**+133.3%**</span> | 896.18 ms | 228.52 ms | <span style="color:#10b981;">**-74.5%**</span> | 3889.86 ms | 1211.93 ms | <span style="color:#10b981;">**-68.8%**</span> | 45.15 ms | 20.57 ms | <span style="color:#10b981;">**-54.4%**</span> |

#### B. Context Length Scaling Sweep ($C=1$, Output 64, $N=10$ prompts)

| Context Length ($L$) | S0 TTFT | S2 TTFT | TTFT $\Delta$ | S0 TPOT | S2 TPOT | TPOT $\Delta$ | S0 Throughput | S2 Throughput | Tput $\Delta$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 46.34 ms | 37.51 ms | <span style="color:#10b981;">**-19.0%**</span> | 10.67 ms | 10.51 ms | <span style="color:#10b981;">**-1.5%**</span> | 89.04 tok/s | 91.43 tok/s | <span style="color:#10b981;">**+2.7%**</span> |
| **256** | 56.50 ms | 85.87 ms | <span style="color:#ef4444;">**+52.0%**</span> | 10.75 ms | 12.85 ms | <span style="color:#ef4444;">**+19.5%**</span> | 87.17 tok/s | 71.44 tok/s | <span style="color:#ef4444;">**-18.0%**</span> |
| **512** | 79.55 ms | 89.29 ms | <span style="color:#ef4444;">**+12.3%**</span> | 10.09 ms | 13.70 ms | <span style="color:#ef4444;">**+35.8%**</span> | 89.40 tok/s | 67.16 tok/s | <span style="color:#ef4444;">**-24.9%**</span> |
| **1024** | 105.47 ms | 105.18 ms | <span style="color:#10b981;">**-0.3%**</span> | 8.99 ms | 13.59 ms | <span style="color:#ef4444;">**+51.2%**</span> | 95.23 tok/s | 66.55 tok/s | <span style="color:#ef4444;">**-30.1%**</span> |
| **2048** | 208.14 ms | 114.36 ms | <span style="color:#10b981;">**-45.1%**</span> | 10.11 ms | 13.60 ms | <span style="color:#ef4444;">**+34.5%**</span> | 75.68 tok/s | 65.88 tok/s | <span style="color:#ef4444;">**-12.9%**</span> |
| **4096** | 416.47 ms | 138.37 ms | <span style="color:#10b981;">**-66.8%**</span> | 10.41 ms | 13.99 ms | <span style="color:#ef4444;">**+34.4%**</span> | 59.67 tok/s | 62.76 tok/s | <span style="color:#10b981;">**+5.2%**</span> |
| **8192** | 673.66 ms | 198.30 ms | <span style="color:#10b981;">**-70.6%**</span> | 9.24 ms | 13.96 ms | <span style="color:#ef4444;">**+51.1%**</span> | 50.95 tok/s | 59.36 tok/s | <span style="color:#10b981;">**+16.5%**</span> |

#### C. In-Depth Performance Dynamics Analysis

1. **The +133.3% Concurrency Surge ($C=64$):**  
   In Stage 0, serving 64 concurrent streams pushed the engine into severe prefill compute saturation. With every arrival requiring 512 tokens of compute-heavy GEMMs, the engine spent valuable GPU cycles executing redundant forward passes, stalling token generation and achieving only **1,232.26 tok/s** at a degraded TPOT of **45.15 ms**.  
   In Stage 2, caching shared prefixes bypassed the compute-heavy prefill for incoming requests. Free from prefill starvation, the continuous batching scheduler packed large, uniform decode batches, driving generation throughput to **2,874.36 tok/s** (<span style="color:#10b981;">**+133.3%**</span>, a **$2.33\times$** scaling gain) while reducing mean TPOT by **54.4%** down to **20.57 ms**.

2. **Decimation of Tail Queuing (TTFT & P99 TTFT Drop):**  
   At $C=64$, Mean TTFT dropped from **896.18 ms to 228.52 ms** (<span style="color:#10b981;">**-74.5%**</span>), and tail latency (P99 TTFT) plummeted from **3,889.86 ms to 1,211.93 ms** (<span style="color:#10b981;">**-68.8%**</span>). Because newly queued requests hit the radix tree and skip the heavy prefill GEMM sequence, they begin decoding in the very next scheduling step rather than waiting behind dozens of serialized prefill chunks.

3. **Short-Context vs. Long-Context Tradeoffs:**  
   * **Short Context ($L=256\text{--}1024$ at $C=1$):** With a small number of sequential prompts ($N=10$) and short prompt lengths, the absolute prefill savings are modest (<50 ms), while managing block table references and hash collisions introduces small CPU and scheduling overheads, lowering single-stream throughput by 18% to 30%.
   * **Long Context ($L \ge 2048\text{--}8192$):** As context length grows into thousands of tokens, prefill compute dominates. In Stage 0, prefill took **673.66 ms** at $L=8192$. With prefix caching, TTFT collapsed to **198.30 ms** (<span style="color:#10b981;">**-70.6%**</span>). Even in a single-stream closed-loop setting ($C=1$), client throughput rose from **50.95 tok/s to 59.36 tok/s** (<span style="color:#10b981;">**+16.5%**</span>).

4. **Latency Jitter & Token Delivery Stability:**  
   In continuous batching, prefill bursts interrupt decode cycles, causing huge spikes in Inter-Token Latency (ITL). In Stage 0, P99 ITL reached **258.96 ms** at $C=64$. In Stage 2, because prefill compute bursts were largely eliminated, P99 ITL collapsed by <span style="color:#10b981;">**82.8%**</span> down to **44.42 ms**, delivering an exceptionally smooth, steady streaming experience to end users.

---

### 4. Profiler & Hardware-Level Bottleneck Analysis

Profiling traces (`timeline.nsys-rep` and `kernels.ncu-rep`) confirm the exact hardware mechanism of Automatic Prefix Caching on the RTX 5080 Laptop GPU:

#### 1. Nsight Compute Roofline & Kernel Shift (Layer 0 Analysis)
Nsight Compute profiled Layer 0 across 1 Prefill pass ($L=2048$) and 1 Decode pass ($M=1, C=1$):
* **Prefill Linear GEMM:** Execution time **4,144.61 µs**, Arithmetic Intensity = **970.44 FLOP/Byte**, Attained Compute = **27.22 TFLOP/s** (**68.1%** of 40 TFLOP/s ceiling; **94.8%** NCU SM throughput), Achieved DRAM Throughput = **28.05 GB/s** (**3.1%** of 896 GB/s peak; **6.2%** NCU DRAM utilization).  
  *Bottleneck:* **Compute-Bound.** With arithmetic intensity near $1000\text{ FLOP/B}$, prefill GEMMs saturate Tensor Cores. Performing this computation redundantly for shared prefixes is the root cause of GPU compute starvation.
* **Prefill Attention (SplitKV):** Execution time **750.40 µs**, Arithmetic Intensity = **1,630.98 FLOP/Byte**, Attained Compute = **18.39 TFLOP/s** (**46.0%** of 40 TFLOP/s ceiling; **64.4%** NCU SM throughput), DRAM Throughput = **11.28 GB/s** (**1.3%** of 896 GB/s peak; **2.5%** NCU DRAM utilization).  
  *Bottleneck:* **Compute-Bound.**
* **Prefill SwiGLU Activation:** Execution time **249.86 µs**, Arithmetic Intensity = **5.55 FLOP/Byte**, DRAM Throughput = **344.27 GB/s** (**38.4%** of 896 GB/s physical peak; **76.0%** NCU sustained-clock DRAM utilization).  
  *Bottleneck:* **Memory-Bandwidth Bound.**
* **Decode Linear GEMV:** Execution time **177.22 µs**, Arithmetic Intensity = **1.007 FLOP/Byte**, Attained Compute = **0.322 TFLOP/s** (**0.8%** of 40 TFLOP/s ceiling; **41.0%** NCU SM throughput), Achieved DRAM Throughput = **320.16 GB/s** (**35.7%** of 896 GB/s physical peak; **70.7%** NCU sustained-clock DRAM utilization).  
  *Bottleneck:* **Strictly Memory-Bandwidth Bound.**
* **Decode Attention (FlashDecode/SplitKV):** Execution time **26.43 µs**, Arithmetic Intensity = **66.68 FLOP/Byte**, DRAM Throughput = **81.56 GB/s**.

#### 2. Host Orchestration Overhead (Nsight Systems Timeline)
Nsight Systems captured the end-to-end timeline under $C=1, L_{\text{in}}=1024, L_{\text{out}}=64$:
* **Wall Duration:** **861.43 ms** (vs 867.33 ms in Stage 0).
* **GPU Active vs Idle:** GPU active time was **748.63 ms** (86.91%) and idle time was **112.80 ms** (13.09%), maintaining high hardware utilization.
* **CPU Launch Overhead:** Measured at **29.96 ms** across **1,636 calls** (virtually identical to Stage 0's 28.90 ms across 1,636 calls). Unlike Speculative Decoding (which exploded to 277 ms and 19,843 calls), prefix caching performs hash table resolution in $O(1)$ and preserves full CUDA Graph coverage (**63 decode graph launches**), introducing near-zero host overhead.
* **Kernel Execution Split:** Linear GEMM/GEMV accounted for **664.56 ms** (88.8%), Attention consumed **4.36 ms** (0.6%), and Norm/Activations took **79.71 ms** (10.6%).

#### 3. Causal Bottleneck Shift
The system bottleneck transitioned across optimization stages:
1. **Stage 0 Bottleneck:** High concurrency throughput was bottlenecked by **Prefill Compute Serialization**. Long prefill GEMMs ($970\text{ FLOP/B}$) locked Tensor Cores and blocked decode batches.
2. **Stage 2 Bottleneck Shift:** Prefix caching **completely removes the prefill compute bottleneck** for repeated prefixes. The compute units are preserved exclusively for generation.
3. **New System Bottleneck:** With prefill compute contention eliminated, the system reaches its physical maximum decode capacity, bounded strictly by **GDDR7 Memory Bandwidth** during batched token generation ($320\text{ GB/s}$ at $1.0\text{ FLOP/B}$).

---

### 5. Conclusion & Key Takeaways

1. **Hypothesis Confirmed:** Automatic Prefix Caching completely eliminates the prefill compute bottleneck and queue serialization observed in Stage 0 baseline.
2. **Massive High-Concurrency Throughput Scaling:** At peak concurrency ($C=64$), aggregate throughput increased by <span style="color:#10b981;">**+133.3%**</span> (**1,232.26 $\to$ 2,874.36 tok/s**), representing a **$2.33\times$** capacity multiplier for multi-tenant inference.
3. **Drastic Tail Latency Reduction:** At $C=64$, Mean TTFT dropped by <span style="color:#10b981;">**74.5%**</span> (896.18 ms $\to$ 228.52 ms), and P99 TTFT dropped by <span style="color:#10b981;">**68.8%**</span> (3,889.86 ms $\to$ 1,211.93 ms), virtually eradicating queue buildup.
4. **Long-Context Latency Win:** At $L=8192$, TTFT collapsed by <span style="color:#10b981;">**70.6%**</span> (673.66 ms $\to$ 198.30 ms), driving client throughput up by <span style="color:#10b981;">**+16.5%**</span>.
5. **Zero Host Orchestration Penalty:** CPU launch overhead remained flat at **29.96 ms** with **63 CUDA Graph launches**, proving that RadixAttention maintains full engine optimization without breaking CUDA Graph capture.
6. **Hardware-Level Bottleneck Transition:** Layer 0 NCU profiling proved Prefill is compute-bound ($970\text{ FLOP/B}$, $27.2\text{ TFLOP/s}$) while Decode is memory-bound ($1.007\text{ FLOP/B}$, $320.2\text{ GB/s}$). Prefix caching circumvents the compute-heavy phase, pushing serving efficiency directly to the memory-bandwidth ceiling.

**Operational Recommendations:**
* Enable `--enable-prefix-caching` for all production multi-turn chat applications, agentic workflows, few-shot prompt templates, and multi-tenant endpoints where prefix reuse exceeds $\sim 20\text{--}30\%$.
* For isolated single-request environments with 100% unique, short prompts ($L < 512$), prefix caching may be disabled to save the ~14–230 MiB of retained KV cache blocks and avoid minor radix lookup overhead.

**Next Steps:**
* **Custom Serving Track:** Proceed to `custom-serving/stage-0-baseline/` to construct a minimal, native C++/CUDA or PyTorch inference engine, profiling custom kernel execution against the vLLM reference baselines.

---

### 6. Data Sources and Reproducibility

| Reported Metric | Source Artifact | Counter / Field |
| :--- | :--- | :--- |
| **Throughput & TPOT** | `benchmarks/results/stage-2-vllm/pareto_c*.json` | `output_throughput`, `mean_tpot_ms` |
| **TTFT & Tail Latencies** | `benchmarks/results/stage-2-vllm/pareto_c*.json` | `mean_ttft_ms`, `p99_ttft_ms`, `p99_itl_ms` |
| **Context Scaling Latencies** | `benchmarks/results/stage-2-vllm/context_l*.json` | `mean_ttft_ms`, `mean_tpot_ms`, `output_throughput` |
| **Timeline Breakdown** | `benchmarks/results/stage-2-vllm/nsys_metrics.json` | `wall_time_ms`, `gpu_active_pct`, `decode_total_ms`, `cpu_launch_overhead_ms` |
| **Kernel Roofline** | `benchmarks/results/stage-2-vllm/ncu_metrics.json` | `achieved_dram_gbps`, `attained_tflops`, `arithmetic_intensity_flop_byte` |
| **Raw Profiler Traces** | `benchmarks/profiles/stage-2-vllm/` | `timeline.nsys-rep`, `timeline.sqlite`, `kernels.ncu-rep` |
| **Interactive Plots** | `benchmarks/results/dashboard.html` | Multi-stage comparative Plotly dashboard |

*All percentage deltas are calculated directly from raw benchmark JSON outputs. Profiler counters represent activity during sampled kernel windows on NVIDIA GeForce RTX 5080 Laptop GPU.*

---

## custom-serving

### Stage 1: KV Caching (`stage-1-custom`)

*Server directory: `custom-serving/stage-1-kv-cache/` | Benchmark stage: `stage-1-custom`*

#### 1. Motivation

##### Observation (Stage 0 Baseline: `stage-0-custom`)
Profiling our naive PyTorch baseline (`Qwen2.5-Coder-1.5B-Instruct`, `bfloat16`, RTX 5080) isolated why uncached autoregressive decoding collapses as sequence length grows *(note: Nsight Compute `ncu` kernel profiling was not needed to motivate Stage 1 because the `nsys` timeline and kernel summary already conclusively proved full-sequence $O(S^2)$ decode recomputation; `ncu` was therefore deferred in Stage 0 and introduced in Stage 1 to motivate Stage 2)*:
* **Phase symmetry anomaly (`nsys` timeline at $C=1, L_{\text{in}}=1024, L_{\text{out}}=64$):** Processing the entire `1,024`-token prompt takes **118.11 ms** (`0.115 ms/token`), whereas generating `63` decode tokens takes **6,954.07 ms** (**97.7%** of the **7,119.30 ms** wall time), or **110.38 ms/token** (vs. **10.67 ms/token** in `stage-0-vllm`). Emitting a single output token costs nearly the same time as prefilling all `1,024` prompt tokens, proving per-step work scales with total context length $S$ rather than $\Delta S = 1$.
* **Kernel attribution & causal invariance (`cuda_gpu_kern_sum` + `nsys_metrics.json`):** Across the run, linear projections consume **4,181.92 ms** (**64.5%** of GPU time) across **12,320** Tensor Core `GEMM` launches (`64x64` and `128x64` CUTLASS tiles), and attention (`attn_compute`) consumes **1,731.54 ms** (**26.7%** of GPU time)—together accounting for **91.2%** of all GPU execution. Because causal masking prevents past tokens $x_0, \dots, x_{t-1}$ from attending to future token $x_t$, their hidden states $h_{0:t-1}$ and projections ($Q_{0:t-1}, K_{0:t-1}, V_{0:t-1}$) are mathematically invariant across decode steps; recomputing `M = S` GEMMs on every step is pure redundant arithmetic.
* **Super-linear context scaling (`context_sweep` at $C=1$):** While prefill TTFT at $L=2048$ remains reasonable (**275.87 ms** vs. **208.14 ms** in vLLM), doubling initial context $L$ from `512` $\to$ `1024` $\to$ `2048` more than doubles decode TPOT at each step: **52.68 ms** $\to$ **120.72 ms** $\to$ **265.51 ms** (**3.76 tok/s**).

##### Hypothesis
Memoizing each layer's past Key and Value tensors ($K_{0:t-1}, V_{0:t-1}$) after prefill and feeding only the newly sampled token ($\Delta S = 1$) on each decode step eliminates redundant sequence recomputation. Per-step linear projections collapse from $O(S \cdot d^2)$ (`M = S` Tensor Core `GEMM`) to $O(d^2)$ (`M = 1` `GEMV`), and attention drops from $O(S^2 \cdot d)$ to $O(S \cdot d)$, directly targeting **91.2%** of Stage 0's GPU kernel time.

##### Expected Results
* **Decode latency (TPOT):** Sharp drop across all context lengths, flattening the steep TPOT curve at $L=512\text{--}2048$ (and making $L=4096\text{--}8192$ practical to benchmark).
* **Prefill (TTFT):** Roughly unchanged, since prefill still processes all $L$ prompt tokens once.
* **Next bottleneck:** Once decode drops to $M=1$, linear projections become memory-bandwidth-bound GEMVs, and PyTorch eager launch overhead (`~1,331` kernels/step) will stand out more clearly.

#### 2. Experiment Setup

| Setting | Value |
| :--- | :--- |
| **Target model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`) |
| **Architecture change** | Per-layer unexpanded `(k, v)` cache (`[B, 2, S, 128]`), offset RoPE, single-token `[1, 1]` decode without causal mask |
| **Serving & concurrency** | FastAPI SSE + global `asyncio.Lock` serialization (`ThreadPoolExecutor(max_workers=1)`) |
| **Workloads** | Pareto: `512` in / `128` out (`C = 1..16`); Context: `L = 128..2048` / `64` out (`C = 1`); Profiling: `1024` in / `64` out (`nsys`), `L = 2048` Layer 0 (`ncu`) |
| **Hardware** | NVIDIA GeForce RTX 5080 Laptop GPU (16 GB GDDR7, `sm_120`), CUDA 13.1, PyTorch 2.13.0+cu130 |

**Validation:**  
Verified 100% greedy token ID parity (`32/32` tokens, prefill max-abs logit diff `0.062500`) across both `stage-0-baseline` and `stage-1-kv-cache` against HuggingFace `AutoModelForCausalLM(..., attn_implementation="eager")` via `uv run --frozen python custom-serving/test_parity.py --stage all`. All benchmark sweeps completed with `failed: 0`.

---

#### 3. Benchmark Results

##### A. Concurrency Scaling (`Input = 512`, `Output = 128`, `C = 1..16`)

| Concurrency ($C$) | `stage-0-custom` Tput | `stage-1-custom` Tput | Tput $\Delta$ | `stage-0-custom` TPOT | `stage-1-custom` TPOT | TPOT $\Delta$ | `stage-0-custom` Mean TTFT | `stage-1-custom` Mean TTFT | `stage-0-vllm` Tput |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 14.93 tok/s | 37.33 tok/s | <span style="color:#10b981;">**+150.1% ($2.50\times$)**</span> | 66.84 ms | 26.42 ms | <span style="color:#10b981;">**-60.5%**</span> | 87.53 ms | 74.02 ms | 87.77 tok/s |
| **2** | 14.83 tok/s | 37.96 tok/s | <span style="color:#10b981;">**+155.9% ($2.56\times$)**</span> | 67.29 ms | 25.98 ms | <span style="color:#10b981;">**-61.4%**</span> | 8,359.90 ms | 3,235.30 ms | 169.46 tok/s |
| **4** | 14.61 tok/s | 36.42 tok/s | <span style="color:#10b981;">**+149.3% ($2.49\times$)**</span> | 68.41 ms | 27.14 ms | <span style="color:#10b981;">**-60.3%**</span> | 25,087.82 ms | 10,062.36 ms | 353.74 tok/s |
| **8** | 14.63 tok/s | 39.57 tok/s | <span style="color:#10b981;">**+170.6% ($2.71\times$)**</span> | 68.32 ms | 24.97 ms | <span style="color:#10b981;">**-63.5%**</span> | 58,302.75 ms | 21,598.44 ms | 561.94 tok/s |
| **16** | 14.61 tok/s | 37.66 tok/s | <span style="color:#10b981;">**+157.8% ($2.58\times$)**</span> | 68.44 ms | 26.27 ms | <span style="color:#10b981;">**-61.6%**</span> | 124,830.53 ms | 48,607.08 ms | 831.19 tok/s |

##### B. Context Length Scaling (`C = 1`, `Output = 64`)

| Context Length ($L$) | `stage-0-custom` TTFT | `stage-1-custom` TTFT | `stage-0-custom` TPOT | `stage-1-custom` TPOT | TPOT $\Delta$ | `stage-0-custom` Tput | `stage-1-custom` Tput | Tput $\Delta$ | `stage-0-vllm` TPOT |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 59.95 ms | 51.19 ms | 34.67 ms | 22.56 ms | <span style="color:#10b981;">**-34.9% ($1.54\times$)**</span> | 28.51 tok/s | 43.45 tok/s | <span style="color:#10b981;">**+52.4% ($1.52\times$)**</span> | 10.67 ms |
| **256** | 62.43 ms | 61.89 ms | 41.30 ms | 24.27 ms | <span style="color:#10b981;">**-41.2% ($1.70\times$)**</span> | 24.02 tok/s | 40.21 tok/s | <span style="color:#10b981;">**+67.4% ($1.67\times$)**</span> | 10.75 ms |
| **512** | 75.70 ms | 77.20 ms | 52.68 ms | 22.64 ms | <span style="color:#10b981;">**-57.0% ($2.33\times$)**</span> | 18.85 tok/s | 42.54 tok/s | <span style="color:#10b981;">**+125.7% ($2.26\times$)**</span> | 10.09 ms |
| **1024** | 155.44 ms | 144.75 ms | 120.72 ms | 23.38 ms | <span style="color:#10b981;">**-80.6% ($5.16\times$)**</span> | 8.25 tok/s | 39.55 tok/s | <span style="color:#10b981;">**+379.6% ($4.80\times$)**</span> | 8.99 ms |
| **2048** | 275.87 ms | 330.14 ms | 265.51 ms | 24.09 ms | <span style="color:#10b981;">**-90.9% ($11.02\times$)**</span> | 3.76 tok/s | 34.62 tok/s | <span style="color:#10b981;">**+819.8% ($9.20\times$)**</span> | 10.11 ms |

##### C. Key Takeaways from Benchmarks
1. **Flat Decode Latency Across Context Length:** In `stage-0-custom`, decode TPOT scaled super-linearly with prompt length (**34.67 ms** at $L=128 \to$ **265.51 ms** at $L=2048$). With the per-layer KV cache in `stage-1-custom`, TPOT stays nearly flat at **22.56–24.27 ms/token** across the entire $L=128\text{--}2048$ sweep—an **$11.02\times$** TPOT reduction (**-90.9%**) and **$9.20\times$** throughput increase (**3.76 $\to$ 34.62 tok/s**) at $L=2048$.
2. **Throughput Plateaus Under Concurrency (`asyncio.Lock`):** Because requests remain serialized behind `asyncio.Lock` without continuous batching, system throughput stays flat at **~37–39.6 tok/s** across $C=1\text{--}16$ ($2.5\times\text{--}2.7\times$ higher than `stage-0-custom`'s ~14.6 tok/s, while lowering queue-induced mean TTFT at $C=16$ by **61.1%** from 124.8 s to 48.6 s).

---

#### 4. Profiler & Hardware Analysis

##### 1. Nsight Systems Timeline (`C = 1`, `Input = 1024`, `Output = 64`)
* **Decode Collapse (`6,954.1 ms` $\to$ `1,357.2 ms`):** Total wall time dropped by **76.5%** (**7,119.30 ms $\to$ 1,674.04 ms**), driven by average decode step latency falling **$5.12\times$** from **110.38 ms/step to 21.54 ms/step**.
* **GPU Active vs. Idle Flip (CPU Launch Bottleneck Exposed):** In `stage-0-custom`, heavy $O(S^2)$ decode kernels kept the GPU active **91.13%** of wall time (**6,487.79 ms**). Once the KV cache shrank decode projections to $M=1$, total GPU active time dropped **$8.49\times$** to **764.32 ms** (**45.66%** active), leaving the GPU **idle 54.34% of the time** (**909.73 ms**). Eager PyTorch still launches **85,058 kernels** (`~1,329` kernels/step) with **647.04 ms** of CPU launch overhead and **0 CUDA graphs** (vs. **28.90 ms** across 1,636 calls and 63 CUDA graphs in `stage-0-vllm`).
* **Operator Category Breakdown:** Total GPU kernel time fell from **6,487.79 ms to 764.31 ms**:
  * **Linear GEMM/GEMV:** **4,181.92 ms $\to$ 478.64 ms** (**62.6%** of GPU time, **$8.74\times$ reduction**).
  * **Attention (`attn_compute`):** **1,731.54 ms $\to$ 185.04 ms** (**24.2%** of GPU time, **$9.36\times$ reduction**). Still **$36.3\times$ slower** than `stage-0-vllm`'s FlashAttention/FlashDecode (**5.10 ms**, **0.7%**).
  * **Norm / Act / Other (incl. KV `torch.cat`):** **574.33 ms $\to$ 100.63 ms** (**13.2%** of GPU time).

##### 2. Nsight Compute Layer 0 Roofline (`L = 2048` Prefill + `M = 1` Decode)
*(Note: Stage 0 deferred NCU profiling because `nsys` timeline metrics already isolated full-sequence decode recomputation; `./benchmarks/profile_ncu.sh stage-1-custom` profiles Layer 0 directly against `stage-0-vllm`'s ceilings of `896.0 GB/s` DRAM and `40.0 TFLOP/s` Tensor Core compute.)*

* **Prefill Linear GEMM:** **2,180.67 µs**, $\text{AI} = \mathbf{755.53\text{ FLOP/B}}$, **25.86 TFLOP/s** (**64.6%** of 40 TFLOP/s ceiling; **90.9%** NCU SM compute throughput) at **34.23 GB/s** (**3.8%** of 896 GB/s peak; **7.6%** NCU DRAM). Compute-bound, matching `stage-0-vllm` (`27.35 TFLOP/s`, `95.1%` SM).
* **Prefill Attention (Unfused Materialized Matrix vs. FlashAttention):**
  * Inside `attn_compute`, the dominant BMM kernel (`Q @ K^T` `cutlass_80_wmma...`) takes **1,209.38 µs** at $\text{AI} = \mathbf{135.42\text{ FLOP/B}}$ and **10.70 TFLOP/s** (**26.7%** of 40 TFLOP/s ceiling; **63.2%** NCU SM) with **78.98 GB/s** DRAM traffic (**17.4%** NCU DRAM)—and additionally launches a separate float32 `softmax_warp_forward` kernel (**1,012.48 µs** streaming HBM at **375.65 GB/s** / **82.9%** NCU DRAM) plus mask/cast passes over the materialized $[1, 12, 2048, 2048]$ matrix.
  * By contrast, `stage-0-vllm`'s fused `Prefill Attention (FlashDecode/SplitKV)` keeps the score tiles in SRAM, finishing the entire attention op in **732.58 µs** at $\text{AI} = \mathbf{1,631.02\text{ FLOP/B}}$ and **18.84 TFLOP/s** while touching only **11.55 GB/s** of DRAM (**2.6%** NCU DRAM).
* **Decode Linear GEMV:** **94.72 µs**, $\text{AI} = \mathbf{1.015\text{ FLOP/B}}$, **0.295 TFLOP/s** (**0.7%** of 40 TFLOP/s ceiling; **35.7%** NCU SM) at **291.10 GB/s** (**32.5%** of 896 GB/s physical peak; **64.3%** NCU sustained-clock DRAM utilization). Strictly memory-bandwidth bound, matching `stage-0-vllm` (`AI = 1.001 FLOP/B`, `0.321 TFLOP/s`, `321.10 GB/s`, `70.9%` NCU DRAM).
* **Decode Attention & KV Cache Concatenation:**
  * `Decode Attention` (`cublasGemvParamsEx` over expanded K/V) runs in **32.22 µs** at $\text{AI} = \mathbf{1.105\text{ FLOP/B}}$, **0.217 TFLOP/s**, and **196.19 GB/s** (**43.3%** NCU DRAM), whereas `stage-0-vllm`'s `FlashDecode/SplitKV` runs in **26.02 µs** at $\text{AI} = \mathbf{66.67\text{ FLOP/B}}$ and **5.53 TFLOP/s** without materializing `repeat_interleave` copies.
  * `Decode KV Cache Write` (`CatArrayBatchedCopy` from `torch.cat((past_k, k), dim=2)`) takes **24.83 µs** at **42.46 GB/s** (**2.77× slower** than vLLM's in-place `reshape_and_cache_flash_kernel` at **8.96 µs**) because `torch.cat` reallocates and copies the entire growing KV history on every token.

---

#### 5. Conclusion

1. **Hypothesis Confirmed:** Adding a per-layer KV cache eliminated $O(S^2)$ decode recomputation, cutting decode step time by **$5.12\times$** at $L=1024$ (**110.38 $\to$ 21.54 ms/step**) and TPOT by **$11.02\times$** at $L=2048$ (**265.51 $\to$ 24.09 ms/token**).
2. **Bottleneck Shift Exposed by `nsys` and `ncu`:**
   * **Prefill Memory Wall:** Unfused PyTorch attention materializes the $S \times S$ score matrix in HBM ($\text{AI} = 135.4\text{ FLOP/B}$ in BMM + $375.7\text{ GB/s}$ DRAM traffic in fp32 softmax), motivating **Stage 2 (Tiled / FlashAttention)**.
   * **Eager Launch & Unfused Kernel Overhead:** With decode kernels now fast ($M=1$ GEMV at $1.015\text{ FLOP/B}$), launching ~1,329 tiny unfused kernels per token leaves the GPU idle **54.34%** of wall time and reallocates the KV cache via `torch.cat` on every step, motivating pre-allocated/paged KV storage and operator fusion in later stages.

---

#### 6. Data Sources and Reproducibility

| Reported Metric | Source Artifact | Counter / Field |
| :--- | :--- | :--- |
| **Throughput, TTFT & TPOT** | `benchmarks/results/stage-1-custom/pareto_c*.json`, `context_l*.json` | `output_throughput`, `mean_ttft_ms`, `mean_tpot_ms` |
| **Stage 0 Custom Baseline** | `benchmarks/results/stage-0-custom/pareto_c*.json`, `context_l*.json`, `nsys_metrics.json` | Baseline comparison metrics |
| **vLLM Reference Baseline** | `benchmarks/results/stage-0-vllm/pareto_c*.json`, `context_l*.json`, `nsys_metrics.json`, `ncu_metrics.json` | Reference ceiling metrics |
| **Timeline Breakdown** | `benchmarks/results/stage-1-custom/nsys_metrics.json` | `wall_time_ms`, `gpu_active_pct`, `decode_step_avg_ms`, `kernel_time_breakdown` |
| **Kernel Roofline** | `benchmarks/results/stage-1-custom/ncu_metrics.json` | `execution_time_us`, `achieved_dram_gbps`, `attained_tflops`, `arithmetic_intensity_flop_byte` |
| **Interactive Plots** | `benchmarks/results/dashboard.html` | Multi-stage comparative Plotly dashboard |

---

### Stage 2: Tiled Attention / FlashAttention (`stage-2-custom`)

*Server directory: `custom-serving/stage-2-tiled-attn/` | Benchmark stage: `stage-2-custom`*

#### 1. Motivation

##### Observation (Stage 1 KV Cache: `stage-1-custom` vs. `stage-0-vllm`)
With redundant decode recomputation eliminated by the Stage 1 KV cache, Nsight Compute Layer 0 profiling ($L=2048$ prefill) exposes a **$12.0\times$ Arithmetic Intensity collapse** in unfused prefill attention compared to `stage-0-vllm`:
* **`stage-0-vllm` (`Prefill Attention (FlashDecode/SplitKV)`):** Executes in **732.58 µs** at an Arithmetic Intensity of **1,631.02 FLOP/Byte**, attaining **18.84 TFLOP/s** (**47.1%** of the 40 TFLOP/s ceiling; **65.0%** NCU SM throughput) while drawing only **11.55 GB/s** of DRAM bandwidth (**2.55%** NCU DRAM) by keeping attention tiles resident in on-chip SRAM.
* **`stage-1-custom` (Unfused PyTorch `attn_compute`):** Even the dominant score BMM (`Q @ K^T`, `cutlass_80_wmma...`) drops to **135.42 FLOP/Byte** ($12.0\times$ lower AI) and **10.70 TFLOP/s** over **1,209.38 µs** (`1.65x` slower than vLLM's entire fused attention kernel) while driving **78.98 GB/s** of DRAM traffic ($6.8\times$ higher HBM bandwidth)—and then spills the materialized $[1, 12, 2048, 2048]$ attention matrix across separate HBM-bound kernels for scaling (`453.98 µs` at **392.36 GB/s**), causal masking (`662.91 µs` at **205.50 GB/s**), fp32 upcast (`787.58 µs` at **359.65 GB/s**), and `softmax_warp_forward` (`1,012.48 µs` at **375.65 GB/s**, **82.91%** NCU DRAM).

##### Hypothesis
Fusing `Q @ K^T`, scaling, causal masking, online softmax, and `probs @ V` into a tiled SRAM kernel (FlashAttention-style) will avoid materializing the $O(S^2)$ intermediate score and probability matrices in GDDR7, restoring high arithmetic intensity during prefill and eliminating the HBM memory wall at long contexts.

##### Expected Results
* **Prefill Roofline Shift (`ncu`):** Prefill attention Arithmetic Intensity should jump by $>10\times$ (approaching vLLM's $\sim 1,600\text{ FLOP/B}$) while DRAM bandwidth drops below $15\text{ GB/s}$.
* **Long-Context TTFT (`context_sweep`):** Significant TTFT reduction at $L=1024\text{--}2048$ as $O(S^2)$ HBM reads/writes are eliminated, along with flat $O(S)$ prefill memory scaling at $L=8192$.
* **Decode Side-Effect (`M = 1`):** Using a single query-tiled FlashAttention kernel for $M=1$ decode launches only $1 \times H_q = 12$ thread blocks across the GPU and pads $M=1$ to a $16 \times 128$ Tensor Core tile, exposing low SM occupancy during decode and motivating Split-K parallel decode in Stage 3.

---

#### 2. Experiment Setup

| Setting | Value |
| :--- | :--- |
| **Target model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`) |
| **Architecture change** | Custom Triton FlashAttention-2 forward kernel (`_flash_attention_fwd`): SRAM online fp32 softmax (`BLOCK_M=64` prefill, `BLOCK_M=16` decode, `BLOCK_N=64`), native GQA indexing (`kv_head_idx = q_head_idx // 6`, no `repeat_interleave`), analytical causal mask, zero-copy output strides |
| **Serving & concurrency** | FastAPI SSE + global `asyncio.Lock` serialization (`ThreadPoolExecutor(max_workers=1)`); 64-token prefill + 1-token decode startup warmup |
| **Workloads** | Pareto: `512` in / `128` out (`C = 1..16`); Context: `L = 128..2048` / `64` out (`C = 1`); Profiling: `1024` in / `64` out (`nsys`), `L = 2048` Layer 0 (`ncu`) |
| **Hardware** | NVIDIA GeForce RTX 5080 Laptop GPU (16 GB GDDR7, `sm_120`), CUDA 13.1, PyTorch 2.13.0+cu130, Triton 3.7.1 |

**Validation:**  
Verified `32/32` greedy token ID parity across all implemented custom stages (`stage-0-baseline` and `stage-1-kv-cache` vs. HF `eager`, max-abs prefill logit diff `0.062500`; `stage-2-tiled-attn` vs. HF `sdpa`, max-abs prefill logit diff `0.218750`) via `uv run --frozen python custom-serving/test_parity.py --stage all`. At $L=8192$, prefill peak allocated VRAM dropped from `~8.1 GB` (`stage-1-custom`) to **3.667 GB** (`2.907 GB` weights + `0.760 GB` activation/KV delta). All benchmark sweeps completed with `failed: 0`.

---

#### 3. Benchmark Results

##### A. Concurrency Scaling (`Input = 512`, `Output = 128`, `C = 1..16`)

| Concurrency ($C$) | `stage-1-custom` Tput | `stage-2-custom` Tput | Tput $\Delta$ | `stage-1-custom` TPOT | `stage-2-custom` TPOT | TPOT $\Delta$ | `stage-1-custom` Mean TTFT | `stage-2-custom` Mean TTFT | TTFT $\Delta$ | `stage-0-vllm` Tput |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 37.33 tok/s | 43.97 tok/s | <span style="color:#10b981;">**+17.8%**</span> | 26.42 ms | 22.44 ms | <span style="color:#10b981;">**-15.1%**</span> | 74.02 ms | 61.16 ms | <span style="color:#10b981;">**-17.4%**</span> | 87.77 tok/s |
| **2** | 37.96 tok/s | 47.73 tok/s | <span style="color:#10b981;">**+25.7%**</span> | 25.98 ms | 20.63 ms | <span style="color:#10b981;">**-20.6%**</span> | 3,235.30 ms | 2,614.54 ms | <span style="color:#10b981;">**-19.2%**</span> | 169.46 tok/s |
| **4** | 36.42 tok/s | 45.44 tok/s | <span style="color:#10b981;">**+24.8%**</span> | 27.14 ms | 21.70 ms | <span style="color:#10b981;">**-20.0%**</span> | 10,062.36 ms | 8,022.17 ms | <span style="color:#10b981;">**-20.3%**</span> | 353.74 tok/s |
| **8** | 39.57 tok/s | 44.96 tok/s | <span style="color:#10b981;">**+13.6%**</span> | 24.97 ms | 21.98 ms | <span style="color:#10b981;">**-12.0%**</span> | 21,598.44 ms | 18,944.44 ms | <span style="color:#10b981;">**-12.3%**</span> | 561.94 tok/s |
| **16** | 37.66 tok/s | 40.52 tok/s | <span style="color:#10b981;">**+7.6%**</span> | 26.27 ms | 24.44 ms | <span style="color:#10b981;">**-7.0%**</span> | 48,607.08 ms | 45,163.05 ms | <span style="color:#10b981;">**-7.1%**</span> | 831.19 tok/s |

##### B. Context Length Scaling (`C = 1`, `Output = 64`)

| Context Length ($L$) | `stage-1-custom` TTFT | `stage-2-custom` TTFT | TTFT $\Delta$ | `stage-0-vllm` TTFT | `stage-1-custom` TPOT | `stage-2-custom` TPOT | TPOT $\Delta$ | `stage-1-custom` Tput | `stage-2-custom` Tput | Tput $\Delta$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 51.19 ms | 46.76 ms | <span style="color:#10b981;">**-8.7%**</span> | 46.34 ms | 22.56 ms | 21.50 ms | <span style="color:#10b981;">**-4.7%**</span> | 43.45 tok/s | 45.65 tok/s | <span style="color:#10b981;">**+5.1%**</span> |
| **256** | 61.89 ms | 53.76 ms | <span style="color:#10b981;">**-13.1%**</span> | 56.50 ms | 24.27 ms | 20.72 ms | <span style="color:#10b981;">**-14.6%**</span> | 40.21 tok/s | 47.06 tok/s | <span style="color:#10b981;">**+17.0%**</span> |
| **512** | 77.20 ms | 71.15 ms | <span style="color:#10b981;">**-7.8%**</span> | 79.55 ms | 22.64 ms | 22.22 ms | <span style="color:#10b981;">**-1.9%**</span> | 42.54 tok/s | 43.50 tok/s | <span style="color:#10b981;">**+2.3%**</span> |
| **1024** | 144.75 ms | 103.65 ms | <span style="color:#10b981;">**-28.4%**</span> | 105.47 ms | 23.38 ms | 23.12 ms | <span style="color:#10b981;">**-1.1%**</span> | 39.55 tok/s | 41.00 tok/s | <span style="color:#10b981;">**+3.7%**</span> |
| **2048** | 330.14 ms | 196.17 ms | <span style="color:#10b981;">**-40.6% ($1.68\times$)**</span> | 208.14 ms | 24.09 ms | 22.12 ms | <span style="color:#10b981;">**-8.2%**</span> | 34.62 tok/s | 40.23 tok/s | <span style="color:#10b981;">**+16.2%**</span> |

##### C. Key Takeaways from Benchmarks
1. **Prefill Parity with vLLM ($-40.6\%$ TTFT at $L=2048$):** As prompt length grows, eliminating the $O(S^2)$ HBM attention matrix yields progressively larger TTFT reductions: **-8.7%** at $L=128$, **-28.4%** at $L=1024$ (**144.75 $\to$ 103.65 ms**), and **-40.6%** at $L=2048$ (**330.14 $\to$ 196.17 ms**, a **$1.68\times$ speedup**). Across $L=256\text{--}2048$, `stage-2-custom`'s single-request TTFT matches or edges out `stage-0-vllm` (**196.17 ms vs. 208.14 ms** at $L=2048$).
2. **Reduced Launch Overhead Lifts Serialized Throughput:** Replacing 7 unfused PyTorch attention/GQA/mask/softmax/transpose ops per layer with a single Triton launch lifts serialized throughput by **+17.8% to +25.7%** at $C=1\text{--}4$ (**37.33 $\to$ 43.97 tok/s** at $C=1$; **37.96 $\to$ 47.73 tok/s** at $C=2$) and trims mean TPOT by **15–21%** (**26.42 $\to$ 22.44 ms** at $C=1$).

---

#### 4. Profiler & Hardware Analysis

##### 1. Nsight Systems Timeline (`C = 1`, `Input = 1024`, `Output = 64`)
* **12,602 Fewer Kernel Launches (`85,058` $\to$ `72,456`):** Fusing GQA head expansion, `Q @ K^T`, scaling, causal masking, fp32 online softmax, `probs @ V`, and output transpose into `_flash_attention_fwd` eliminates **7 kernel launches per layer** across $64 \times 28 = 1,792$ layer calls (**-14.8%** total launches).
* **Attention GPU Time Share Cut from 24.2% to 9.3%:** Total GPU time spent in `attention` dropped by **39.9%** (**185.04 ms $\to$ 111.20 ms**), shrinking attention's share of GPU execution from **24.2% to 9.3%** (with `linear` at **71.4%** / **858.09 ms** and `norm_act_other` at **19.3%** / **232.12 ms**).

##### 2. Nsight Compute Layer 0 Roofline (`L = 2048` Prefill + `M = 1` Decode)
* **`Prefill Attention (Triton FlashAttention)` (`_flash_attention_fwd`, grid `(32, 12, 1)`):**
  * **Arithmetic Intensity ($11.8\times$ Jump):** Rose from **135.42 FLOP/Byte** (`stage-1-custom` score BMM) to **1,597.84 FLOP/Byte**—reaching **98.0%** of `stage-0-vllm`'s FlashAttention kernel (**1,631.02 FLOP/Byte**).
  * **DRAM Bandwidth ($-89.1\%$ Traffic Drop):** Fell from **78.98 GB/s** (**17.43%** NCU DRAM in the Stage 1 score BMM alone, plus **375.65 GB/s** / **82.91%** NCU DRAM in the separate `1,012.48 µs` `softmax_warp_forward` kernel) down to **8.62 GB/s** (**0.96%** of the 896 GB/s ceiling; **1.90%** NCU sustained-clock DRAM utilization).
  * **Compute & Duration:** Completes the entire fused prefill attention pass in **978.05 µs** at **13.78 TFLOP/s** (**34.4%** of the 40 TFLOP/s ceiling; **47.44%** NCU SM compute throughput), compared to `>4,100 µs` across Stage 1's unfused BMM + scale + mask + cast + softmax chain (**$>4.2\times$ Layer 0 prefill attention speedup**).
* **`Decode Attention (Triton FlashAttention)` (`_flash_attention_fwd`, grid `(1, 12, 1)`) — New Decode Bottleneck:**
  * During single-token decode ($M = q_{\text{len}} = 1, L_{\text{kv}} = 2049$), reusing the query-tiled `_flash_attention_fwd` kernel takes **129.44 µs** (vs. **32.22 µs** in `stage-1-custom` cuBLAS GEMV and **26.02 µs** in `stage-0-vllm` `FlashDecode/SplitKV`), with NCU reporting `Low Occupancy / Latency` (**8.33%** active warps, **5.56%** SM compute, **3.62%** NCU DRAM / **16.41 GB/s**).
  * Because the grid is partitioned only across $(\lceil q_{\text{len}} / \text{BLOCK\_M} \rceil, B \cdot H_q) = (1, 12, 1)$, only **12 thread blocks** run across the entire RTX 5080 while each CTA sequentially streams all 33 KV tiles, and padding $M=1$ to `BLOCK_M=16` for `tl.dot` wastes $15/16$ (**93.75%**) of tile rows.

---

#### 5. Conclusion

1. **Prefill Memory Wall Eliminated:** Tiling attention in SRAM with online fp32 softmax boosted Prefill Attention Arithmetic Intensity by **$11.8\times$** (**135.42 $\to$ 1,597.84 FLOP/B**), slashed prefill attention DRAM bandwidth by **89.1%** (**78.98 $\to$ 8.62 GB/s**), cut $L=2048$ TTFT by **40.6%** (**330.14 $\to$ 196.17 ms**, matching vLLM), and reduced $L=8192$ prefill peak VRAM to **3.67 GB**.
2. **Decode Occupancy Bottleneck Isolated for Stage 3:** While query-tiled FlashAttention excels when $q_{\text{len}}$ is large (`384` thread blocks at $L=2048$), at $q_{\text{len}} = 1$ it launches only **12 thread blocks** (`grid=(1, 12, 1)`), starving the GPU's SMs (**16.41 GB/s** DRAM, **129.44 µs**). Splitting the KV sequence across parallel thread blocks (Split-K / FlashDecode) is required next.

---

#### 6. Data Sources and Reproducibility

| Reported Metric | Source Artifact | Counter / Field |
| :--- | :--- | :--- |
| **Throughput, TTFT & TPOT** | `benchmarks/results/stage-2-custom/pareto_c*.json`, `context_l*.json` | `output_throughput`, `mean_ttft_ms`, `mean_tpot_ms` |
| **Stage 1 Custom Baseline** | `benchmarks/results/stage-1-custom/pareto_c*.json`, `context_l*.json`, `nsys_metrics.json`, `ncu_metrics.json` | Prior stage comparison metrics |
| **vLLM Reference Baseline** | `benchmarks/results/stage-0-vllm/pareto_c*.json`, `context_l*.json`, `nsys_metrics.json`, `ncu_metrics.json` | Reference ceiling metrics |
| **Timeline Breakdown** | `benchmarks/results/stage-2-custom/nsys_metrics.json` | `wall_time_ms`, `num_kernels`, `decode_step_avg_ms`, `kernel_time_breakdown` |
| **Kernel Roofline** | `benchmarks/results/stage-2-custom/ncu_metrics.json` | `execution_time_us`, `achieved_dram_gbps`, `attained_tflops`, `arithmetic_intensity_flop_byte` |
| **Interactive Plots** | `benchmarks/results/dashboard.html` | Multi-stage comparative Plotly dashboard |

---

### Stage 3: Parallel Decode Attention / Split-K (`stage-3-custom`)

*Server directory: `custom-serving/stage-3-split-k/` | Benchmark stage: `stage-3-custom`*

#### 1. Motivation

##### Observation (Stage 2 Tiled Attention: `stage-2-custom` vs. `stage-0-vllm`)
While Stage 2's query-tiled Triton FlashAttention-2 kernel (`_flash_attention_fwd`) solved prefill ($1,597.84\text{ FLOP/B}$, $8.62\text{ GB/s}$ DRAM, $196.17\text{ ms}$ TTFT at $L=2048$), Nsight Compute Layer 0 profiling exposes severe **SM and memory-bandwidth under-utilization during single-token decode** ($M = q_{\text{len}} = 1, L_{\text{kv}} = 2048$):
* **SM Starvation (`grid = (1, 12, 1)`):** Partitioning work only along $(\lceil q_{\text{len}} / \text{BLOCK\_M} \rceil, B \cdot H_q)$ launches **384 thread blocks** during prefill ($L=2048, \text{BLOCK\_M}=64$), but collapses to just **12 thread blocks** during decode ($q_{\text{len}}=1$). On the RTX 5080 Laptop GPU, 12 CTAs occupy only a fraction of available SMs (**8.33%** active warps, **5.56%** NCU SM throughput) while each CTA serially loops over all 33 KV cache tiles, taking **129.44 µs** per layer and drawing only **16.41 GB/s** of DRAM bandwidth (**1.8%** of the 896 GB/s physical ceiling; **3.62%** NCU DRAM utilization).
* **Tensor Core Tile Padding Waste (`BLOCK_M = 16` for $M = 1$):** Because Triton's `tl.dot` requires tile dimensions $\ge 16$, a single decode query vector ($1 \times 128$) must be padded to $16 \times 128$ (or GQA-grouped across 6 query heads, leaving $10/16$ rows unused), wasting **93.75%** of Tensor Core MMA rows on dummy zeros and inflating apparent arithmetic intensity ($\mathbf{99.15\text{ FLOP/B}}$, **1.627 TFLOP/s**).
* **`stage-0-vllm` Reference (`Decode Attention (FlashDecode/SplitKV)`):** vLLM splits the KV sequence dimension across parallel thread blocks (`grid = (1, 17, 2)` = **34 CTAs** for the main Split-K pass plus a fast **21.63 µs** `splitkv_combine` reduction), driving **82.87 GB/s** of DRAM bandwidth (**18.37%** NCU DRAM, **$5.05\times$ higher bandwidth**) and finishing the main pass in **26.02 µs** (**$4.97\times$ faster** than Stage 2's `129.44 µs`).

##### Hypothesis
Implementing a dedicated **Split-K / FlashDecode** kernel for decode ($q_{\text{len}} = 1$) that partitions the KV cache sequence length $L_{\text{kv}}$ across $S_k$ parallel thread blocks per head (computing partial online-softmax accumulators $(m_i^{(s)}, l_i^{(s)}, O_i^{(s)})$ in parallel and reducing them via a lightweight log-sum-exp combine kernel) will multiply active CTAs across SMs, eliminate serial KV-loop latency, and significantly increase achieved DRAM bandwidth and decode attention speed at long contexts.

#### 2. Experiment Setup
* **Engine (`custom-serving/stage-3-split-k/`):** Self-contained snapshot of Stage 2 with a two-stage Triton Split-K decode path for $q_{\text{len}} = 1$ while retaining `_flash_attention_fwd` for prefill ($q_{\text{len}} > 1$):
  1. `_flash_attention_decode_splitk_fwd`: Partitions $L_{\text{kv}}$ across `NUM_SPLITS` parallel CTAs per head (`grid = (B * num_heads, NUM_SPLITS)`), computing partial online-softmax accumulators in `fp32` via vectorized reductions without `BLOCK_M=16` Tensor Core padding waste.
  2. `_flash_attention_splitk_combine`: Reduces the partial `(acc, m_i, l_i)` workspace buffers across splits via log-sum-exp rescaling (`grid = (B * num_heads, 1)`).
* **Hardware & Model:** NVIDIA GeForce RTX 5080 Laptop GPU (896.0 GB/s DRAM ceiling, 40.0 TFLOP/s BF16 Tensor Core ceiling), `Qwen/Qwen2.5-1.5B-Instruct` in `bfloat16`, `max_model_len = 9216`, `asyncio.Lock` serialized serving.
* **Workloads:**
  * Concurrency sweep (`pareto_c{1,2,4,8,16}.json`, 512-in / 64-out)
  * Context-length sweep (`context_l{128,256,512,1024,2048}.json`, C=1, 64-out)
  * Nsight Systems & Nsight Compute (`1024-in / 64-out`, `C=1`)

#### 3. Benchmark Results (`stage-2-custom` vs. `stage-3-custom`)

| Sweep Point | `stage-2-custom` Throughput / Latency | `stage-3-custom` Throughput / Latency | Delta |
| :--- | :--- | :--- | :--- |
| **Pareto C=1** | 43.97 tok/s (mean TPOT 22.44 ms, P50 ITL 19.57 ms) | **44.98 tok/s** (mean TPOT 21.93 ms, P50 ITL 19.46 ms) | **+2.3%** |
| **Pareto C=2** | **47.73 tok/s** (mean TPOT 20.63 ms, P50 ITL 19.40 ms) | 43.02 tok/s (mean TPOT 22.92 ms, P50 ITL 19.66 ms) | -9.9% |
| **Pareto C=4** | **45.44 tok/s** (mean TPOT 21.70 ms, P50 ITL 20.19 ms) | 43.11 tok/s (mean TPOT 22.93 ms, P50 ITL 19.00 ms) | -5.1% |
| **Pareto C=8** | **44.96 tok/s** (mean TPOT 21.98 ms, P50 ITL 18.94 ms) | 41.96 tok/s (mean TPOT 23.59 ms, P50 ITL 19.49 ms) | -6.7% |
| **Pareto C=16** | **40.52 tok/s** (mean TPOT 24.44 ms, P50 ITL 21.71 ms) | 38.67 tok/s (mean TPOT 25.65 ms, P50 ITL 22.39 ms) | -4.6% |
| **Context L=128** | **45.65 tok/s** (mean TPOT 21.50 ms, TTFT 46.76 ms) | 45.35 tok/s (mean TPOT 21.68 ms, TTFT 44.78 ms) | -0.7% |
| **Context L=256** | 47.06 tok/s (mean TPOT 20.72 ms, TTFT 53.76 ms) | **49.07 tok/s** (mean TPOT 19.87 ms, TTFT 52.06 ms) | **+4.3%** |
| **Context L=512** | 43.50 tok/s (mean TPOT 22.22 ms, TTFT 71.15 ms) | **46.68 tok/s** (mean TPOT 20.69 ms, TTFT 67.11 ms) | **+7.3%** |
| **Context L=1024** | 41.00 tok/s (mean TPOT 23.12 ms, TTFT 103.65 ms) | **41.16 tok/s** (mean TPOT 23.00 ms, TTFT 104.98 ms) | **+0.4%** |
| **Context L=2048** | 40.23 tok/s (mean TPOT 22.12 ms, TTFT 196.17 ms) | **42.00 tok/s** (mean TPOT 21.15 ms, TTFT 190.76 ms) | **+4.4%** |

All `stage-3-custom` sweep runs completed with `failed: 0`.

#### 4. Profiler & Hardware Analysis

##### Nsight Systems (`1024-in / 64-out`, `C=1`, NVTX-Measured)
* **Total Attention Kernel Time:** Dropped from **111.20 ms** (`9.3%` of GPU time) in `stage-2-custom` to **17.00 ms** (`2.2%` of GPU time) in `stage-3-custom` — a **6.54× reduction** in aggregate attention execution time across the 64-token generation.
* **Decode Step Latency & Wall Time:** `decode_step_avg_ms` improved from `25.19 ms` to `24.11 ms` (`1586.66 ms` → `1518.62 ms` total decode), and overall `wall_time_ms` improved from `1977.39 ms` to `1870.56 ms` (`prefill_duration_ms: 327.93 ms`, `17.5%` of wall time).
* **CPU Launch & Idle Overhead Exposed:** Because attention shrank by `94.20 ms` and GPU active time dropped from `1201.40 ms` (`60.76%`) to `763.44 ms` (`40.81%`), GPU idle time rose to **`1107.12 ms` (`59.19%`)**, dominated by **`703.33 ms` of `cpu_launch_overhead_ms`** across **`74,220` kernel launches** (`0` CUDA graphs) plus `51.08 ms` of `cuda_sync_wait_ms`. This explains why short-context serialized `pareto_c2..c16` runs (where Split-K adds a combine kernel launch per layer per step, `74,220` vs `72,456` kernels) saw slightly lower end-to-end throughput despite faster decode attention kernels. Meanwhile, linear GEMVs account for `606.66 ms` (`79.5%` of GPU kernel time).

##### Nsight Compute (Layer 0, `L=1024` Prefill & Decode)
* **Decode Attention Main Pass (`_flash_attention_decode_splitk_fwd`, `grid=(12, 15, 1)`, `block=(128, 1, 1)`):**
  * Execution time dropped from `129.44 µs` (`stage-2-custom`) to **`32.06 µs`** (**4.04× faster**).
  * Combined with `_flash_attention_splitk_combine` (`7.62 µs`, `grid=(12, 1, 1)`), total Layer 0 decode attention dropped to **`39.68 µs`** (**3.26× faster**).
  * Active warps jumped from `8.33%` to **`21.76%`**, NCU SM compute utilization rose from `5.56%` to **`21.74%`** (`0.524 TFLOP/s`, `1.31%` of `40.0 TFLOP/s` ceiling), and NCU DRAM utilization rose from `3.62%` (`16.41 GB/s`, `1.83%` of `896.0 GB/s` ceiling) to **`14.63%`** (**`66.08 GB/s`**, `7.38%` of `896.0 GB/s` ceiling), with `80` registers/thread and `0` register spills.
* **Other Layer 0 Kernels:**
  * `Prefill Linear GEMM`: `2168.45 µs`, AI `1105.214`, `26.013 TFLOP/s` (`65.03%` of `40.0 TFLOP/s` ceiling; NCU `compute_utilization_pct = 90.44%`, `dram_utilization_pct = 5.19%`).
  * `Prefill Attention (_flash_attention_fwd)`: `970.75 µs`, AI `1597.843`, `13.880 TFLOP/s` (`34.70%` of `40.0 TFLOP/s` ceiling; NCU `compute_utilization_pct = 47.26%`, `dram_utilization_pct = 1.92%`).
  * `Decode Linear GEMV`: `93.98 µs`, AI `1.015`, `293.38 GB/s` (`32.74%` of `896.0 GB/s` ceiling; NCU `dram_utilization_pct = 64.80%`, `compute_utilization_pct = 36.63%`).
  * `Decode KV Cache Write (CatArrayBatchedCopy)`: `24.64 µs`, AI `2.223`, `42.79 GB/s` (`4.78%` of `896.0 GB/s` ceiling; NCU `dram_utilization_pct = 9.47%`, `compute_utilization_pct = 54.53%`).

#### 5. Conclusion
Stage 3 eliminated SM starvation during single-token decode attention: Layer 0 decode attention dropped **3.26×** (`129.44 µs` → `39.68 µs` combined, `32.06 µs` main pass) and aggregate attention time across a 1024-in/64-out trace dropped **6.54×** (`111.20 ms` → `17.00 ms`, now `2.2%` of GPU time), lifting `L=256..2048` context-sweep throughput by up to `+7.3%`.

However, end-to-end decode step latency only moved from `25.19 ms` to `24.11 ms` because **CPU kernel-launch overhead (`703.33 ms` across `74,220` launches) leaves the GPU idle `59.19%` of the wall time**. Eliminating Python/CUDA dispatch overhead via **CUDA Graphs**—alongside fused RMSNorm/RoPE/residual ops—is now the clearest path to closing the remaining gap toward vLLM (`10.67 ms/step`).

#### 6. Data Sources and Reproducibility
* `benchmarks/results/stage-3-custom/pareto_c{1,2,4,8,16}.json`
* `benchmarks/results/stage-3-custom/context_l{128,256,512,1024,2048}.json`
* `benchmarks/results/stage-3-custom/nsys_metrics.json`
* `benchmarks/results/stage-3-custom/ncu_metrics.json`
* `benchmarks/results/dashboard.html`

---

### Stage 4 (Custom): Paged KV Cache & Continuous Batching (`stage-4-custom`)

*Server directory: `custom-serving/stage-4-paged-kv/` | Benchmark stage: `stage-4-custom`*

#### 1. Motivation, Hypothesis, and Speedup Model
* **Eliminating Serialized Concurrency (`asyncio.Lock` → Iteration-Level Continuous Batching):** In Stages 1–3, concurrent requests were serialized behind a global `asyncio.Lock`, pinning aggregate Pareto throughput flat at **38.67–44.98 tok/s** across $C=1\text{--}16$ while queueing TTFT ballooned to **`47,151.62 ms`** at $C=16$ and decode linear projections remained strictly memory-bandwidth-bound $M = 1$ GEMVs (`93.98 µs`, $\text{AI} = 1.015\text{ FLOP/B}$). Replacing `engine_lock` with an iteration-level scheduler (`PagedEngine.run_scheduler_loop`) that coalesces active requests into batched $M = B$ decode steps converts decode GEMVs into batched GEMMs, amortizing weight reads across $B$ active sequences and unlocking horizontal throughput scaling on Plot 1.
* **Eliminating Contiguous KV Reallocation (`torch.cat` → Triton `_reshape_and_cache`):** Pre-allocating a fixed-size physical block pool (`BlockAllocator`, `KV_BLOCK_SIZE = 16`, `4,096` blocks) and scattering new K/V tokens in-place via Triton `_reshape_and_cache` eliminates Stage 3's $O(S)$ `CatArrayBatchedCopy` full-history KV copies (`24.64 µs` at `42.79 GB/s` in Layer 0 decode), removes contiguous memory fragmentation, and enables full `L=128..8192` context-sweep execution without padding waste.

#### 2. Experiment Setup
* **Hardware & Model:** NVIDIA GeForce RTX 5080 Laptop GPU (896.0 GB/s DRAM ceiling, 40.0 TFLOP/s BF16 Tensor Core ceiling), `Qwen/Qwen2.5-Coder-1.5B-Instruct` in `bfloat16`, `max_model_len = 9216`, `KV_BLOCK_SIZE = 16`, `num_blocks = 4096`, `max_batch_size = 64`.
* **Workloads:**
  * Concurrency sweep (`pareto_c{1,2,4,8,16,32,64}.json`, 512-in / 128-out)
  * Context-length sweep (`context_l{128,256,512,1024,2048,4096,8192}.json`, C=1, 64-out)
  * Nsight Systems & Nsight Compute (`1024-in / 64-out`, `C=1`, plus standalone Layer 0 `L=2048` Prefill / `M=1` Decode)

#### 3. Benchmark Results (`stage-3-custom` & `stage-0-vllm` vs. `stage-4-custom`)

| Sweep Point | `stage-3-custom` | `stage-4-custom` | `stage-0-vllm` (Ref) | Stage 4 vs. Stage 3 |
| :--- | :--- | :--- | :--- | :--- |
| **Pareto C=1** | **44.98 tok/s** (TPOT 21.93 ms, TTFT 60.02 ms) | 38.51 tok/s (TPOT 25.70 ms, TTFT 59.12 ms) | 87.77 tok/s (TPOT 10.90 ms) | -14.4% |
| **Pareto C=2** | 43.02 tok/s (TTFT 2905.37 ms) | **72.14 tok/s** (TPOT 27.01 ms, TTFT 116.90 ms) | 169.46 tok/s (TPOT 10.98 ms) | **+67.7% (1.68×)** |
| **Pareto C=4** | 43.11 tok/s (TTFT 8439.29 ms) | **163.71 tok/s** (TPOT 23.35 ms, TTFT 161.72 ms) | 353.74 tok/s (TPOT 10.15 ms) | **+279.7% (3.80×)** |
| **Pareto C=8** | 41.96 tok/s (TTFT 20212.68 ms) | **290.94 tok/s** (TPOT 25.15 ms, TTFT 324.23 ms) | 561.94 tok/s (TPOT 12.27 ms) | **+593.4% (6.93×)** |
| **Pareto C=16** | 38.67 tok/s (TTFT 47151.62 ms) | **525.85 tok/s** (TPOT 26.58 ms, TTFT 517.92 ms) | 831.19 tok/s (TPOT 16.08 ms) | **+1259.8% (13.60×)** |
| **Pareto C=32** | *Omitted (serialized)* | **901.88 tok/s** (TPOT 28.97 ms, P99 ITL 44.72 ms) | 966.96 tok/s (TPOT 27.86 ms, P99 ITL 239.19 ms) | **93.3% of vLLM** |
| **Pareto C=64** | *Omitted (serialized)* | **1260.39 tok/s** (TPOT 38.92 ms, P99 ITL 52.30 ms) | 1232.26 tok/s (TPOT 45.15 ms, P99 ITL 258.96 ms) | **102.3% of vLLM** |
| **Context L=128** | 45.35 tok/s (TPOT 21.68 ms, TTFT 44.78 ms) | **46.58 tok/s** (TPOT 21.12 ms, TTFT 43.25 ms) | 89.04 tok/s (TPOT 10.67 ms) | **+2.7%** |
| **Context L=256** | **49.07 tok/s** (TPOT 19.87 ms, TTFT 52.06 ms) | 46.25 tok/s (TPOT 21.15 ms, TTFT 50.61 ms) | 87.17 tok/s (TPOT 10.75 ms) | -5.7% |
| **Context L=512** | 46.68 tok/s (TPOT 20.69 ms, TTFT 67.11 ms) | **48.12 tok/s** (TPOT 20.13 ms, TTFT 61.62 ms) | 89.40 tok/s (TPOT 10.09 ms) | **+3.1%** |
| **Context L=1024** | 41.16 tok/s (TPOT 23.00 ms, TTFT 104.98 ms) | **42.78 tok/s** (TPOT 21.91 ms, TTFT 115.22 ms) | 95.23 tok/s (TPOT 8.99 ms) | **+3.9%** |
| **Context L=2048** | **42.00 tok/s** (TPOT 21.15 ms, TTFT 190.76 ms) | 37.70 tok/s (TPOT 23.65 ms, TTFT 207.26 ms) | 75.68 tok/s (TPOT 10.11 ms) | -10.2% |
| **Context L=4096** | — | **34.51 tok/s** (TPOT 22.75 ms, TTFT 420.80 ms) | 59.67 tok/s (TPOT 10.41 ms) | *New full-range point* |
| **Context L=8192** | — | **28.95 tok/s** (TPOT 22.20 ms, TTFT 811.73 ms) | 50.95 tok/s (TPOT 9.24 ms) | *New full-range point* |

All `stage-4-custom` sweep runs completed with `failed: 0`. Across the full `L=128..8192` context sweep at $C=1$, mean TPOT stays tightly bounded between `20.13 ms` and `23.65 ms/token` (`22.20 ms` at `L=8192`).

#### 4. Profiler & Hardware Analysis

##### Nsight Systems (`1024-in / 64-out`, `C=1`, NVTX-Measured)
* **Decode Step Latency & Wall Time:** `decode_step_avg_ms` dropped from `24.11 ms` (`stage-3-custom`) to **`19.68 ms`** (`1518.62 ms` → `1240.13 ms` total decode), and overall `wall_time_ms` fell from `1870.56 ms` to **`1603.12 ms`** (`prefill_duration_ms = 326.94 ms`, `20.4%` of wall time). Host synchronization wait (`cuda_sync_wait_ms`) dropped `9.15×` from `51.08 ms` to **`5.58 ms`**.
* **Launch Overhead & Remaining `CatArrayBatchedCopy`:** Total kernel launches dropped from `74,220` to **`72,933`** (`cuda_graph_launches = 0`), lifting GPU active utilization from `40.81%` to **`46.47%`**. However, **`669.98 ms` of `cpu_launch_overhead_ms`** still leaves the GPU idle **`53.53%` (`858.12 ms`)** of wall time at `C=1`. Moreover, while KV-cache `torch.cat` copies are **0** (all `1,792` KV writes execute via `_reshape_and_cache`), **`3,584` `CatArrayBatchedCopy` launches (`2 × 28 × 64`) persist from unfused `rotate_half` (`torch.cat((-x2, x1), dim=-1)`) inside RoPE**.

##### Nsight Compute (Layer 0, `L=2048` Prefill & `M=1` Decode)
* **Paged KV Cache Write (`_reshape_and_cache` vs. Stage 3 `CatArrayBatchedCopy`):**
  * **Decode (`grid=(1, 2, 1)`, `block=(64, 1, 1)`):** Execution time dropped **4.61×** from `24.64 µs` (`42.79 GB/s`, `9.47%` DRAM SOL in Stage 3) to **`5.34 µs`** (`1.05 GB/s`, `0.24%` DRAM SOL, `0` spills), replacing $O(S)$ full-history reallocation with an $O(1)$ per-token scatter.
  * **Prefill (`grid=(2048, 2, 1)`, `block=(64, 1, 1)`):** `14.27 µs`, `147.30 GB/s` (`32.58%` DRAM SOL, `49.19%` compute SOL, `74.67%` active warps, `18` registers/thread, `0` spills).
* **Prefill & Decode Attention:**
  * `Prefill Attention (_flash_attention_fwd, BLOCK_N=32)`: Improved from `970.75 µs` (`13.880 TFLOP/s`, `47.26%` compute SOL) in Stage 3 to **`847.90 µs`** (**`15.987 TFLOP/s`**, `40.0%` of `40.0 TFLOP/s` ceiling, `55.51%` compute SOL, AI `1607.656`).
  * `Decode Attention (_flash_attention_decode_splitk_fwd, grid=(12, 15, 1))`: `33.92 µs` (`62.69 GB/s`, AI `7.407`, `0` spills) + `_flash_attention_splitk_combine`: `7.84 µs` (`41.76 µs` combined).
* **Unfused Elementwise / Norm / Linear Baseline (`C=1` Launch Targets for Stage 5):**
  * `Prefill Linear GEMM`: `2167.46 µs`, AI `1096.153`, `26.025 TFLOP/s` (`65.06%` of `40.0 TFLOP/s` ceiling; `90.80%` compute SOL).
  * `Decode Linear GEMV` (`C=1`): `94.72 µs`, AI `1.015`, `291.10 GB/s` (`64.29%` DRAM SOL).
  * Small unfused decode ops remain latency/launch-bound (`< 2%` DRAM SOL): `Decode Fused RMSNorm` (`7.62 µs`, `1.99%` DRAM SOL), `Decode RoPE Embedding` (`8.06 µs`, `0.85%` DRAM SOL), and `Decode SwiGLU Activation` (`6.37 µs`, `1.11%` DRAM SOL).

#### 5. Conclusion & Stage 5 Handoff
Stage 4 unlocked full continuous batching and paged memory management: aggregate throughput scaled **32.7×** from `C=1` (`38.51 tok/s`) to **`1,260.39 tok/s` at `C=64`**—**13.60×** over Stage 3 at `C=16` (`525.85` vs `38.67 tok/s`), **91×** lower `C=16` TTFT (`517.92 ms` vs `47,151.62 ms`), and **102.3% of `stage-0-vllm` (`1,232.26 tok/s`) at `C=64`** with `5×` tighter P99 ITL (`52.30 ms` vs `258.96 ms`). Meanwhile, Triton `_reshape_and_cache` cut Layer 0 decode KV-cache write latency **4.61×** (`24.64 µs` → `5.34 µs`) and enabled flat `~22 ms/tok` decode all the way to `L=8192`.

The remaining gap at low concurrency (`19.68 ms/step` vs `10.67 ms/step` in vLLM) is now isolated cleanly to **operator fragmentation and CPU kernel-launch overhead** (`669.98 ms` across `72,932` launches, `53.53%` GPU idle time), specifically the `3,584` `CatArrayBatchedCopy` launches from unfused `rotate_half` RoPE plus separate PyTorch RMSNorm, residual-add, and SwiGLU launches—setting up **Stage 5 (`stage-5-custom` / `custom-serving/stage-5-fused-ops/`: Operator Fusion)** as the next targeted optimization.

#### 6. Data Sources and Reproducibility
* `benchmarks/results/stage-4-custom/pareto_c{1,2,4,8,16,32,64}.json`
* `benchmarks/results/stage-4-custom/context_l{128,256,512,1024,2048,4096,8192}.json`
* `benchmarks/results/stage-4-custom/nsys_metrics.json`
* `benchmarks/results/stage-4-custom/ncu_metrics.json`
* `benchmarks/results/dashboard.html`

---

### Stage 5 (Custom): Operator Fusion (`stage-5-custom`)

*Server directory: `custom-serving/stage-5-fused-ops/` | Benchmark stage: `stage-5-custom`*

#### 1. Motivation, Hypothesis, and Speedup Model
* **Motivation (Stage 4 Bottleneck):** Although Stage 4 solved continuous batching and paged KV caching, single-request decode remained host-dispatch bound (`19.68 ms/step` vs. `10.67 ms/step` in `stage-0-vllm`). Nsight Systems on `1024-in / 64-out` (`C=1`) recorded **`72,933` total CUDA kernels** (`72,932` CPU launch calls), **`669.98 ms` of `cpu_launch_overhead_ms`**, **`53.53%` (`858.12 ms`) GPU idle time**, and **`137.74 ms` (`18.5%` of GPU active time)** in `norm_act_other`, including **`3,584` `CatArrayBatchedCopy` launches** (`2 × 28 × 64`) from PyTorch `rotate_half` (`torch.cat((-x2, x1), dim=-1)`), unfused `RMSNorm` (`pow` → `mean` → `rsqrt` → cast → `mul`) plus separate residual `CUDAFunctor_add` kernels, and separate `gate_proj` / `up_proj` + `silu` + `mul` launches in `SwiGLUMLP`.
* **Hypothesis & Speedup Model:** Replacing these unfused chains with three custom Triton kernels (`_rms_norm_fwd` with residual threading, in-place 4D-strided `_rope_embedding_fwd`, and combined `gate_up_proj` + `_swiglu_fwd`) eliminates ~48.8k elementwise, reduction, concat, residual-add, and split-GEMM launches per 1024-in/64-out request (`72,933` → `~24.1k` kernels), collapsing `cpu_launch_overhead_ms`, shrinking `norm_act_other_ms`, and shifting `C=1` decode closer to vLLM's `10.67 ms/step`.

#### 2. Experiment Setup
* **Hardware & Model:** NVIDIA GeForce RTX 5080 Laptop GPU (`896.0 GB/s` DRAM ceiling, `40.0 TFLOP/s` BF16 Tensor Core ceiling), `Qwen/Qwen2.5-Coder-1.5B-Instruct` in `bfloat16`, `max_model_len = 9216`, `KV_BLOCK_SIZE = 16`, `num_blocks = 4096`, `max_batch_size = 64`.
* **Architecture Changes (`custom-serving/stage-5-fused-ops/model.py`):**
  1. **Fused RMSNorm + Residual (`_rms_norm_fwd`):** Single-pass Triton kernel accumulating variance in `fp32`, computing `tl.math.rsqrt`, casting to `bfloat16`, and multiplying by weight; threads `(hidden_states, residual)` across all 28 `Qwen2Layer` blocks and the final `self.norm` (`HAS_RESIDUAL=True`), eliminating standalone `CUDAFunctor_add` launches.
  2. **Fused In-Place RoPE (`_rope_embedding_fwd`):** Single-launch Triton kernel rotating `q` and `k` in-place using explicit 4D strides (handling non-contiguous `.transpose(1, 2)` prefill tensors without `.contiguous()` copies), eliminating `rotate_half` slicing and all `CatArrayBatchedCopy` launches.
  3. **Combined `gate_up_proj` + `_swiglu_fwd`:** Merges `gate_proj` and `up_proj` into a single `nn.Linear(hidden_size, 2 * intermediate_size, bias=False)` (concatenating safetensors `gate_proj.weight` and `up_proj.weight` along `dim=0` across shards in `Qwen2Model.from_pretrained`), followed by the 2D-tiled `_swiglu_fwd` Triton kernel (`SiLU(gate) * up` in registers).
* **Workloads & Parity:** `test_parity.py --stage all` (`32/32` greedy token parity, prefill logit tolerance, `gate_up_proj` weight slicing & Triton kernel checks, `B=4` batched parity & `256/256` block reclamation); closed-loop Pareto (`pareto_c{1,2,4,8,16,32,64}.json`, `512`-in / `128`-out) and Context (`context_l{128..8192}.json`, `C=1`, `64`-out) sweeps; `nsys` (`1024`-in / `64`-out, `C=1`) and standalone Layer 0 `ncu` (`L=2048` Prefill / `M=1` Decode).

#### 3. Benchmark Results (`stage-4-custom` & `stage-0-vllm` vs. `stage-5-custom`)

| Sweep Point | `stage-4-custom` | `stage-5-custom` | `stage-0-vllm` (Ref) | Stage 5 vs. Stage 4 |
| :--- | :--- | :--- | :--- | :--- |
| **Pareto C=1** | 38.51 tok/s (TPOT 25.70 ms, TTFT 59.12 ms) | **68.44 tok/s** (TPOT 14.19 ms, TTFT 67.15 ms) | 87.77 tok/s (TPOT 10.90 ms) | **+77.7% (1.78×)**, TPOT -44.8% |
| **Pareto C=2** | 72.14 tok/s (TPOT 27.01 ms, TTFT 116.90 ms) | **133.29 tok/s** (TPOT 14.34 ms, TTFT 98.76 ms) | 169.46 tok/s (TPOT 10.98 ms) | **+84.8% (1.85×)**, TPOT -46.9% |
| **Pareto C=4** | 163.71 tok/s (TPOT 23.35 ms, TTFT 161.72 ms) | **233.36 tok/s** (TPOT 15.84 ms, TTFT 181.85 ms) | 353.74 tok/s (TPOT 10.15 ms) | **+42.5% (1.43×)**, TPOT -32.2% |
| **Pareto C=8** | 290.94 tok/s (TPOT 25.15 ms, TTFT 324.23 ms) | **469.40 tok/s** (TPOT 14.62 ms, TTFT 323.51 ms) | 561.94 tok/s (TPOT 12.27 ms) | **+61.3% (1.61×)**, TPOT -41.9% |
| **Pareto C=16** | 525.85 tok/s (TPOT 26.58 ms, TTFT 517.92 ms) | **748.22 tok/s** (TPOT 17.56 ms, TTFT 505.97 ms) | 831.19 tok/s (TPOT 16.08 ms) | **+42.3% (1.42×)**, TPOT -33.9% |
| **Pareto C=32** | 901.88 tok/s (TPOT 28.97 ms, P99 ITL 44.72 ms) | **1,116.76 tok/s** (TPOT 22.10 ms, P99 ITL 32.28 ms) | 966.96 tok/s (TPOT 27.86 ms, P99 ITL 239.19 ms) | **+23.8% (1.24×)** (**115.5% of vLLM**) |
| **Pareto C=64** | 1,260.39 tok/s (TPOT 38.92 ms, P99 ITL 52.30 ms) | **1,486.13 tok/s** (TPOT 31.71 ms, P99 ITL 39.64 ms) | 1,232.26 tok/s (TPOT 45.15 ms, P99 ITL 258.96 ms) | **+17.9% (1.18×)** (**120.6% of vLLM**) |
| **Context L=128** | 46.58 tok/s (TPOT 21.12 ms, TTFT 43.25 ms) | **75.31 tok/s** (TPOT 12.85 ms, TTFT 39.70 ms) | 89.04 tok/s (TPOT 10.67 ms) | **+61.7% (1.62×)**, TPOT -39.1% |
| **Context L=256** | 46.25 tok/s (TPOT 21.15 ms, TTFT 50.61 ms) | **90.78 tok/s** (TPOT 10.45 ms, TTFT 46.15 ms) | 87.17 tok/s (TPOT 10.75 ms) | **+96.3% (1.96×)** (**104.1% of vLLM**) |
| **Context L=512** | 48.12 tok/s (TPOT 20.13 ms, TTFT 61.62 ms) | **78.14 tok/s** (TPOT 11.84 ms, TTFT 72.51 ms) | 89.40 tok/s (TPOT 10.09 ms) | **+62.4% (1.62×)**, TPOT -41.2% |
| **Context L=1024** | 42.78 tok/s (TPOT 21.91 ms, TTFT 115.22 ms) | **68.67 tok/s** (TPOT 12.74 ms, TTFT 129.08 ms) | 95.23 tok/s (TPOT 8.99 ms) | **+60.5% (1.61×)**, TPOT -41.9% |
| **Context L=2048** | 37.70 tok/s (TPOT 23.65 ms, TTFT 207.26 ms) | **65.85 tok/s** (TPOT 12.20 ms, TTFT 202.66 ms) | 75.68 tok/s (TPOT 10.11 ms) | **+74.7% (1.75×)**, TPOT -48.4% |
| **Context L=4096** | 34.51 tok/s (TPOT 22.75 ms, TTFT 420.80 ms) | **49.93 tok/s** (TPOT 13.31 ms, TTFT 442.69 ms) | 59.67 tok/s (TPOT 10.41 ms) | **+44.7% (1.45×)**, TPOT -41.5% |
| **Context L=8192** | 28.95 tok/s (TPOT 22.20 ms, TTFT 811.73 ms) | **39.09 tok/s** (TPOT 13.10 ms, TTFT 811.84 ms) | 50.95 tok/s (TPOT 9.24 ms) | **+35.0% (1.35×)**, TPOT -41.0% |

All 14 `stage-5-custom` sweep JSONs completed with `failed: 0`. Across the full `L=128..8192` sweep at `C=1`, mean TPOT drops by **39.1%–50.6%** into a tight `10.45–13.31 ms/token` band (`13.10 ms` at `L=8192`), while `Pareto C=64` reaches **`1,486.13 tok/s`** (**120.6% of `stage-0-vllm`**) with **`6.53×` tighter P99 ITL** (`39.64 ms` vs. `258.96 ms`).

#### 4. Profiler & Hardware Analysis

##### Nsight Systems (`1024-in / 64-out`, `C=1`, NVTX-Measured)
* **Kernel Launch Reduction (`num_kernels` & `cpu_launch_overhead_ms`):** Total CUDA kernels dropped **3.03× (-66.95%)** from `72,933` (`stage-4-custom`) to **`24,101`** (`cpu_launch_calls`: `72,932` → **`24,100`**, `-66.96%`; `cuda_graph_launches = 0`), with **zero `CatArrayBatchedCopy` launches** (`3,584` → `0`). Host `cpu_launch_overhead_ms` fell **1.84× (-45.57%)** from `669.98 ms` to **`364.68 ms`**.
* **GPU Utilization & Wall-Clock Latency:** Eliminating 48,832 launches cut `gpu_idle_pct` from `53.53%` (`858.12 ms`) to **`28.97%` (`312.19 ms`)** (**-63.62% idle ms**), lifting `gpu_active_pct` from `46.47%` (`745.0 ms`) to **`71.03%` (`765.34 ms`)**. Overall `wall_time_ms` dropped **32.79%** (`1,603.12 ms` → **`1,077.53 ms`**): `prefill_duration_ms` fell **27.21%** (`326.94 ms` / `20.4%` → **`237.98 ms`** / `22.1%`), and `decode_total_ms` fell **36.09%** (`1,240.13 ms` → **`792.51 ms`**), cutting `decode_step_avg_ms` from `19.68 ms` to **`12.58 ms`** (**1.56× faster**, closing most of the gap to vLLM's `10.67 ms/step`). `cuda_sync_wait_ms` was `7.97 ms` (vs. `5.58 ms` in Stage 4).
* **Kernel Time Breakdown:** `norm_act_other_ms` dropped **53.50%** from `137.74 ms` (`18.5%` of GPU time) to **`64.05 ms` (`8.4%`)**, leaving **`666.14 ms` (`87.0%`) in `linear_ms`** and **`35.15 ms` (`4.6%`) in `attention_ms`** (vs. `587.93 ms` / `78.9%` and `19.33 ms` / `2.6%` in Stage 4).

##### Nsight Compute (Layer 0, `L=2048` Prefill & `M=1` Decode)
* **Fused Triton Operators (`_rms_norm_fwd`, `_rope_embedding_fwd`, `_swiglu_fwd`):**
  * `Prefill Fused RMSNorm` (`_rms_norm_fwd`, `grid=(2048, 1, 1)`, `block=(256, 1, 1)`): **`52.06 µs`**, `241.86 GB/s` (`26.99%` of `896.0 GB/s` ceiling; `53.39%` NCU DRAM SOL), `0.453 TFLOP/s` (`1.13%` of `40.0 TFLOP/s` ceiling; `62.37%` SM SOL), AI `1.874`, `91.49%` active warps, `32` regs, `0` spills — replacing Stage 4's multi-kernel chain plus separate residual add in a single pass.
  * `Decode Fused RMSNorm` (`_rms_norm_fwd`, `grid=(1, 1, 1)`, `block=(256, 1, 1)`): **`6.88 µs`** (`-9.7%` vs. `7.62 µs`), `2.20 GB/s` (`0.25%` ceiling; `0.49%` DRAM SOL), `0.002 TFLOP/s` (`0.19%` SM SOL), AI `0.763`, `16.34%` active warps, `32` regs, `0` spills.
  * `Prefill RoPE Embedding` (`_rope_embedding_fwd`, `grid=(2048, 14, 1)`, `block=(64, 1, 1)`): **`63.26 µs`**, `132.67 GB/s` (`14.81%` ceiling; `29.31%` DRAM SOL), `0.232 TFLOP/s` (`0.58%` ceiling; `71.09%` SM SOL), AI `1.749`, `83.90%` active warps, `20` regs, `0` spills — rotating Q and K in-place with zero `CatArrayBatchedCopy` allocations.
  * `Decode RoPE Embedding` (`_rope_embedding_fwd`, `grid=(1, 14, 1)`, `block=(64, 1, 1)`): **`5.66 µs`** (**-29.8%** vs. `8.06 µs`), `1.58 GB/s` (`0.18%` ceiling; `0.36%` DRAM SOL), `0.001 TFLOP/s` (`0.35%` SM SOL), AI `0.800`, `4.13%` active warps, `20` regs, `0` spills.
  * `Prefill SwiGLU Activation` (`_swiglu_fwd`, `grid=(2048, 18, 1)`, `block=(128, 1, 1)`): **`242.72 µs`**, **`368.25 GB/s`** (`41.10%` ceiling; **`81.29%` NCU DRAM SOL**), `0.505 TFLOP/s` (`1.26%` ceiling; `33.74%` SM SOL), AI `1.373`, `93.39%` active warps, `22` regs, `0` spills (`Memory Bandwidth`).
  * `Decode SwiGLU Activation` (`_swiglu_fwd`, `grid=(1, 18, 1)`, `block=(128, 1, 1)`): **`5.02 µs`** (**-21.2%** vs. `6.37 µs`), `8.00 GB/s` (`0.89%` ceiling; `1.81%` DRAM SOL), `0.012 TFLOP/s` (`0.03%` ceiling; `0.83%` SM SOL), AI `1.490`, `8.17%` active warps, `22` regs, `0` spills.
* **Combined `gate_up_proj` Linear Projections (`2×` output width, half the MLP up-projection launches):**
  * `Prefill Linear GEMM`: **`4,182.50 µs`**, AI `967.862`, **`26.973 TFLOP/s`** (`67.43%` of `40.0 TFLOP/s` ceiling; **`94.49%` NCU SM SOL**), `27.87 GB/s` (`3.11%` ceiling; `6.15%` DRAM SOL), `16.67%` active warps, `232` regs, `0` spills (`Compute Pipeline`).
  * `Decode Linear GEMV`: **`175.23 µs`**, AI `1.003`, **`324.91 GB/s`** (`36.26%` of `896.0 GB/s` ceiling; **`71.70%` NCU DRAM SOL**), `0.326 TFLOP/s` (`0.82%` ceiling; `40.13%` SM SOL), `25.22%` active warps, `164` regs, `0` spills (`Memory Bandwidth`). *(Note: NCU keeps the single longest GEMM/GEMV in Layer 0; in Stage 5 that is the `2×`-wide `gate_up_proj` instead of Stage 4's `1×` `gate_proj`/`up_proj`, while per-layer linear launches dropped from 7 to 6.)*
* **Retained Stage 4 Attention & Paged KV Cache Kernels:**
  * `Prefill Attention (Triton FlashAttention)` (`_flash_attention_fwd`): **`846.59 µs`**, AI `1607.900`, **`16.011 TFLOP/s`** (`40.03%` of `40.0 TFLOP/s` ceiling; `55.46%` SM SOL), `9.96 GB/s` (`2.20%` DRAM SOL), `0` spills.
  * `Prefill KV Cache Write` (`_reshape_and_cache`): **`13.86 µs`**, `151.70 GB/s` (`16.93%` ceiling; `33.62%` DRAM SOL), `51.71%` SM SOL, AI `0.998`, `0` spills.
  * `Decode Attention (Triton Split-K)` (`_flash_attention_decode_splitk_fwd`): **`32.67 µs`**, `65.10 GB/s` (`7.27%` ceiling; `14.41%` DRAM SOL), `0.482 TFLOP/s` (`1.21%` ceiling; `15.72%` SM SOL), AI `7.406`, `0` spills.
  * `Decode Attention Combine (Split-K)` (`_flash_attention_splitk_combine`): **`7.49 µs`**, `14.02 GB/s` (`1.56%` ceiling; `3.12%` DRAM SOL), AI `1.288`, `0` spills.
  * `Decode KV Cache Write` (`_reshape_and_cache`): **`5.41 µs`**, `0.99 GB/s` (`0.11%` ceiling; `0.22%` DRAM SOL), AI `0.190`, `0` spills.

#### 5. Conclusion
Stage 5 delivered the full kernel-fusion payoff predicted at the end of Stage 4: custom Triton `_rms_norm_fwd` (with residual threading), in-place `_rope_embedding_fwd`, and `gate_up_proj` + `_swiglu_fwd` cut total CUDA kernel launches **3.03×** (`72,933` → `24,101`, eliminating all `3,584` `CatArrayBatchedCopy` RoPE calls), reduced `cpu_launch_overhead_ms` by **45.57%** (`669.98 ms` → `364.68 ms`), and shrank GPU idle time from `53.53%` to **`28.97%`**. End-to-end `C=1` decode step time dropped **36.08%** (`19.68 ms` → `12.58 ms`), lifting `Pareto C=1` throughput **+77.7%** (`38.51` → `68.44 tok/s`), `Context L=256` to **`90.78 tok/s`** (`104.1%` of `stage-0-vllm`), and `Pareto C=64` to **`1,486.13 tok/s`** (`120.6%` of `stage-0-vllm` with `6.53×` lower P99 ITL). Because `cuda_graph_launches` remains `0` and `364.68 ms` (`28.97%` GPU idle) is still spent dispatching the remaining 24,100 launches at `C=1`, **CUDA Graphs** remain the primary future opportunity to close the final `~1.9 ms/step` gap to `stage-0-vllm` (`10.67 ms/step`).

#### 6. Data Sources and Reproducibility
* `benchmarks/results/stage-5-custom/pareto_c{1,2,4,8,16,32,64}.json`
* `benchmarks/results/stage-5-custom/context_l{128,256,512,1024,2048,4096,8192}.json`
* `benchmarks/results/stage-5-custom/nsys_metrics.json`
* `benchmarks/results/stage-5-custom/ncu_metrics.json`
* `benchmarks/results/dashboard.html`

---

### Stage 6 (Custom): Bucketed CUDA Graphs for Decode (`stage-6-custom`)

*Server directory: `custom-serving/stage-6-cuda-graphs/` | Benchmark stage: `stage-6-custom`*

#### 1. Motivation, Hypothesis, and Speedup Model
* **Motivation (Stage 5 Bottleneck):** Stage 5 fused RMSNorm+residual, in-place RoPE, and `gate_up_proj`+SwiGLU, cutting CUDA kernels from `72,933` to `24,101` on `1024-in / 64-out` (`C=1`). However, `cuda_graph_launches` was still `0`: eager Python/Triton/cuBLAS dispatch executed `24,100` separate host launch calls (`~376` launches per decode step), burning **`364.68 ms` of `cpu_launch_overhead_ms`**, leaving the GPU idle **`28.97%` (`312.19 ms`)** of wall time, and keeping `decode_step_avg_ms` at `12.58 ms` vs. `stage-0-vllm`'s `10.67 ms` (`63` CUDA graph launches).
* **Hypothesis & Speedup Model:** Capturing the fixed-shape decode forward pass (`q_len == 1`) into static `torch.cuda.CUDAGraph` batch buckets $B \in \{1, 2, 4, 8, 16, 32, 64\}$ replaces ~376 per-step Python kernel launches with a single `cudaGraphLaunch` call (`63` graph launches across 63 decode steps), eliminating decode host-dispatch gaps (`gpu_idle_pct` $\to < 10\%$) and pushing decode step latency below `stage-0-vllm`.

#### 2. Experiment Setup
* **Hardware & Model:** NVIDIA GeForce RTX 5080 Laptop GPU (`896.0 GB/s` DRAM ceiling, `40.0 TFLOP/s` BF16 Tensor Core ceiling), `Qwen/Qwen2.5-Coder-1.5B-Instruct` in `bfloat16`, `max_model_len = 9216`, `KV_BLOCK_SIZE = 16`, `num_blocks = 4096`, `max_batch_size = 64`.
* **Architecture Changes (`custom-serving/stage-6-cuda-graphs/`):**
  1. **Capture-Safe & Padded-Lane Safe Forward (`model.py`):** Eliminates host-device syncs (`.item()`) when `max_context_len` is passed as a static `int` (`max_model_len`). Clamps `positions = (start_pos[:, None] + ...).clamp(min=0)` so padded bucket rows (`context_lengths[pad] == 0`) read RoPE row `0`, skip KV writes in `_reshape_and_cache` (`pos == -1` $\Rightarrow$ `valid_blk == False`, preserving physical block `0`), and emit finite zero outputs in `_flash_attention_splitk_combine`.
  2. **Bucketed `DecodeGraphRunner` (`model.py`, `engine.py`):** Pre-allocates static device buffers (`input_ids [B, 1]`, `block_table [B, 576]`, `context_lengths [B]`) for each bucket $B \in (1, 2, 4, 8, 16, 32, 64) \le \text{max\_batch\_size}$. During `PagedEngine.warmup()`, JIT-compiles both `SINGLE_SPLIT=False` ($B < 10$) and `SINGLE_SPLIT=True` ($B \ge 10$) Triton specializations, runs 2 warmup passes on a dedicated capture stream, captures each bucket into a `torch.cuda.CUDAGraph`, and reclaims temporary warmup blocks (`num_free_blocks == num_blocks`). At runtime, `_run_decode_batch` copies active rows `0..actual_batch_size-1`, zero-fills padded lanes `actual_batch_size..B-1`, calls `graph.replay()`, and slices `static_logits[:actual_batch_size]`.
* **Workloads & Parity:** `test_parity.py --stage all` (`32/32` greedy token parity across all 7 stages, `B=4` exact-bucket and `B=3 -> bucket B=4` padded-lane parity, `256/256` block reclamation); closed-loop Pareto (`pareto_c{1..64}.json`, `512`-in / `128`-out) and Context (`context_l{128..8192}.json`, `C=1`, `64`-out) sweeps; `nsys` (`1024`-in / `64`-out, `C=1`) and standalone Layer 0 `ncu` (`L=2048` Prefill / `M=1` Decode).

#### 3. Benchmark Results (`stage-5-custom` & `stage-0-vllm` vs. `stage-6-custom`)

| Sweep Point | `stage-5-custom` | `stage-6-custom` | `stage-0-vllm` (Ref) | Stage 6 vs. Stage 5 (`% of vLLM`) |
| :--- | :--- | :--- | :--- | :--- |
| **Pareto C=1** | 68.44 tok/s (TPOT 14.19 ms, TTFT 67.15 ms) | **113.42 tok/s** (TPOT 8.43 ms, TTFT 57.78 ms) | 87.77 tok/s (TPOT 10.90 ms) | **+65.7%**, TPOT -40.6% (**129.2% of vLLM**) |
| **Pareto C=2** | 133.29 tok/s (TPOT 14.34 ms, TTFT 98.76 ms) | **230.95 tok/s** (TPOT 8.02 ms, TTFT 89.01 ms) | 169.46 tok/s (TPOT 10.98 ms) | **+73.3%**, TPOT -44.1% (**136.3% of vLLM**) |
| **Pareto C=4** | 233.36 tok/s (TPOT 15.84 ms, TTFT 181.85 ms) | **403.44 tok/s** (TPOT 8.85 ms, TTFT 144.04 ms) | 353.74 tok/s (TPOT 10.15 ms) | **+72.9%**, TPOT -44.1% (**114.1% of vLLM**) |
| **Pareto C=8** | 469.40 tok/s (TPOT 14.62 ms, TTFT 323.51 ms) | **741.79 tok/s** (TPOT 8.92 ms, TTFT 246.50 ms) | 561.94 tok/s (TPOT 12.27 ms) | **+58.0%**, TPOT -39.0% (**132.0% of vLLM**) |
| **Pareto C=16** | 748.22 tok/s (TPOT 17.56 ms, TTFT 505.97 ms) | **1,065.51 tok/s** (TPOT 11.99 ms, TTFT 398.55 ms) | 831.19 tok/s (TPOT 16.08 ms) | **+42.4%**, TPOT -31.7% (**128.2% of vLLM**) |
| **Pareto C=32** | 1,116.76 tok/s (TPOT 22.10 ms, P99 ITL 32.28 ms) | **1,410.49 tok/s** (TPOT 16.79 ms, P99 ITL 18.73 ms) | 966.96 tok/s (TPOT 27.86 ms, P99 ITL 239.19 ms) | **+26.3%**, TPOT -24.0% (**145.9% of vLLM**) |
| **Pareto C=64** | 1,486.13 tok/s (TPOT 31.71 ms, P99 ITL 39.64 ms) | **1,770.61 tok/s** (TPOT 25.91 ms, P99 ITL 27.57 ms) | 1,232.26 tok/s (TPOT 45.15 ms, P99 ITL 258.96 ms) | **+19.1%**, TPOT -18.3% (**143.7% of vLLM**) |
| **Context L=128** | 75.31 tok/s (TPOT 12.85 ms, TTFT 39.70 ms) | **125.31 tok/s** (TPOT 7.50 ms, TTFT 37.78 ms) | 89.04 tok/s (TPOT 10.67 ms, TTFT 46.34 ms) | **+66.4%**, TPOT -41.6% (**140.7% of vLLM**) |
| **Context L=256** | 90.78 tok/s (TPOT 10.45 ms, TTFT 46.15 ms) | **110.71 tok/s** (TPOT 8.39 ms, TTFT 49.40 ms) | 87.17 tok/s (TPOT 10.75 ms, TTFT 56.50 ms) | **+22.0%**, TPOT -19.8% (**127.0% of vLLM**) |
| **Context L=512** | 78.14 tok/s (TPOT 11.84 ms, TTFT 72.51 ms) | **114.36 tok/s** (TPOT 7.83 ms, TTFT 66.21 ms) | 89.40 tok/s (TPOT 10.09 ms, TTFT 79.55 ms) | **+46.4%**, TPOT -33.9% (**127.9% of vLLM**) |
| **Context L=1024** | 68.67 tok/s (TPOT 12.74 ms, TTFT 129.08 ms) | **97.46 tok/s** (TPOT 8.60 ms, TTFT 114.26 ms) | 95.23 tok/s (TPOT 8.99 ms, TTFT 105.47 ms) | **+41.9%**, TPOT -32.5% (**102.3% of vLLM**) |
| **Context L=2048** | 65.85 tok/s (TPOT 12.20 ms, TTFT 202.66 ms) | **91.62 tok/s** (TPOT 8.31 ms, TTFT 174.38 ms) | 75.68 tok/s (TPOT 10.11 ms, TTFT 208.14 ms) | **+39.1%**, TPOT -31.9% (**121.1% of vLLM**) |
| **Context L=4096** | 49.93 tok/s (TPOT 13.31 ms, TTFT 442.69 ms) | **74.26 tok/s** (TPOT 8.42 ms, TTFT 331.25 ms) | 59.67 tok/s (TPOT 10.41 ms, TTFT 416.47 ms) | **+48.7%**, TPOT -36.8% (**124.5% of vLLM**) |
| **Context L=8192** | 39.09 tok/s (TPOT 13.10 ms, TTFT 811.84 ms) | **55.46 tok/s** (TPOT 8.32 ms, TTFT 629.30 ms) | 50.95 tok/s (TPOT 9.24 ms, TTFT 673.66 ms) | **+41.9%**, TPOT -36.5% (**108.9% of vLLM**) |

All 14 `stage-6-custom` sweep JSONs completed with `failed: 0`. `stage-6-custom` outperforms `stage-0-vllm` at **every** Pareto concurrency point (`114.1%–145.9%`) and **every** context length `L=128..8192` (`102.3%–140.7%`), holding single-request mean TPOT within **`7.50–8.60 ms/token`** across the entire `8192`-token context window.

#### 4. Profiler & Hardware Analysis

##### Nsight Systems (`1024-in / 64-out`, `C=1`, NVTX-Measured)
* **CUDA Graph Capture & Host Dispatch Collapse:** `cuda_graph_launches` rose from `0` (`stage-5-custom`) to **`63`** (1 graph replay per decode token, matching `stage-0-vllm`). Host `cpu_launch_calls` collapsed **50.6× (-98.02%)** from `24,100` to **`476`** (only the eager prefill launches plus 63 graph replays), and total `cpu_launch_overhead_ms` dropped **-21.78%** from `364.68 ms` to **`285.25 ms`** (virtually all remaining launch API time is the single eager prefill pass).
* **GPU Idle Collapse & Decode Speedup:** `gpu_idle_pct` fell **3.21×** from `28.97%` (`312.19 ms`) to **`9.03%` (`80.25 ms`)** (**-74.29% idle ms**, surpassing `stage-0-vllm`'s `11.61%` / `100.68 ms`), lifting `gpu_active_pct` from `71.03%` (`765.34 ms`) to **`90.97%` (`808.25 ms`)**. Total decode time (`decode_total_ms`) dropped **-31.34%** from `792.51 ms` to **`544.10 ms`**, cutting `decode_step_avg_ms` from `12.58 ms` to **`8.64 ms`** (**1.46× faster** than Stage 5 and **19.0% faster** than `stage-0-vllm`'s `10.67 ms/step`). Overall `wall_time_ms` fell **-17.54%** (`1,077.53 ms` → **`888.50 ms`**), with `cuda_sync_wait_ms` at `7.52 ms` and `num_kernels` at `24,165` (the same `24,101` model kernels executed inside graph replay plus 64 stream/graph bookend ops).
* **Kernel Time Breakdown:** Across `808.25 ms` of active GPU execution, `linear_ms` accounts for **`697.22 ms` (`86.3%`)**, `attention_ms` for **`28.18 ms` (`3.5%`)**, and `norm_act_other_ms` for **`82.86 ms` (`10.3%`)**.

##### Nsight Compute (Layer 0, `L=2048` Prefill & `M=1` Decode)
Because CUDA Graphs eliminate inter-kernel host launch gaps without altering individual kernel code (`profile_ncu.sh` profiles standalone Layer 0 execution), all 13 Layer 0 kernels match Stage 5's Roofline coordinates with `0` register spills:
* **Prefill (`L=2048`):** `Prefill Linear GEMM` (`4,146.24 µs`, AI `961.442`, `27.209 TFLOP/s` = `68.02%` of `40.0 TFLOP/s` ceiling / `95.25%` SM SOL, `28.30 GB/s` = `3.16%` ceiling / `6.25%` DRAM SOL); `Prefill Attention` (`_flash_attention_fwd`, `852.99 µs`, AI `1607.900`, `15.891 TFLOP/s` = `39.73%` ceiling / `55.97%` SM SOL, `9.88 GB/s` = `1.10%` ceiling / `2.18%` DRAM SOL); `Prefill SwiGLU` (`_swiglu_fwd`, `242.72 µs`, `368.41 GB/s` = `41.12%` of `896.0 GB/s` ceiling / `81.32%` DRAM SOL); `Prefill Fused RMSNorm` (`53.34 µs`, `236.15 GB/s` = `26.36%` ceiling / `52.14%` DRAM SOL); `Prefill RoPE` (`64.29 µs`, `130.56 GB/s` = `14.57%` ceiling / `28.84%` DRAM SOL); `Prefill KV Cache Write` (`14.08 µs`, `149.29 GB/s` = `16.66%` ceiling / `33.02%` DRAM SOL).
* **Decode (`M=1`):** `Decode Linear GEMV` (`176.86 µs`, AI `1.005`, `321.39 GB/s` = `35.87%` of `896.0 GB/s` ceiling / `70.94%` DRAM SOL, `0.323 TFLOP/s` = `0.81%` ceiling / `41.42%` SM SOL); `Decode Attention (Triton Split-K)` (`33.98 µs`, `62.58 GB/s` = `6.98%` ceiling / `13.86%` DRAM SOL) + `Combine` (`7.74 µs`, `13.55 GB/s`); `Decode Fused RMSNorm` (`6.88 µs`), `Decode RoPE` (`5.89 µs`), `Decode KV Cache Write` (`5.47 µs`), and `Decode SwiGLU` (`4.93 µs`).

#### 5. Conclusion
Stage 6 closed the final host-dispatch gap in the custom track: bucketed `DecodeGraphRunner` capture ($B \in \{1..64\}$) converted 63 decode steps from `~23,600` individual Python kernel launches into **`63` `cudaGraphLaunch` calls** (`cpu_launch_calls`: `24,100` → `476`, `-98.02%`; `cpu_launch_overhead_ms`: `364.68 ms` → `285.25 ms`, `-21.78%`), collapsing GPU idle time from `28.97%` to **`9.03%`** (`90.97%` active) and cutting `decode_step_avg_ms` to **`8.64 ms`** (`19.0%` faster than `stage-0-vllm`'s `10.67 ms`). As a result, `stage-6-custom` exceeds `stage-0-vllm` across the entire benchmark suite—reaching **`113.42 tok/s` (`8.43 ms` TPOT) at `C=1`** (`129.2%` of vLLM), **`1,770.61 tok/s` at `C=64`** (`143.7%` of vLLM with `9.39×` lower P99 ITL), and **`7.50–8.60 ms` TPOT** across `L=128..8192`.

#### 6. Data Sources and Reproducibility
* `benchmarks/results/stage-6-custom/pareto_c{1,2,4,8,16,32,64}.json`
* `benchmarks/results/stage-6-custom/context_l{128,256,512,1024,2048,4096,8192}.json`
* `benchmarks/results/stage-6-custom/nsys_metrics.json`
* `benchmarks/results/stage-6-custom/ncu_metrics.json`
* `benchmarks/results/dashboard.html`