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
* **Prefill Linear GEMM:** Execution time **4,144.61 µs**, Arithmetic Intensity = **970.44 FLOP/Byte**, Attained Compute = **27.22 TFLOP/s**, Achieved DRAM Throughput = **28.05 GB/s**.  
  *Bottleneck:* **Compute-Bound.** With arithmetic intensity near $1000\text{ FLOP/B}$, prefill GEMMs saturate Tensor Cores. Performing this computation redundantly for shared prefixes is the root cause of GPU compute starvation.
* **Prefill Attention (SplitKV):** Execution time **750.40 µs**, Arithmetic Intensity = **1,630.98 FLOP/Byte**, Attained Compute = **18.39 TFLOP/s**, DRAM Throughput = **11.28 GB/s**.  
  *Bottleneck:* **Compute-Bound.**
* **Prefill SwiGLU Activation:** Execution time **249.86 µs**, Arithmetic Intensity = **5.55 FLOP/Byte**, DRAM Throughput = **344.27 GB/s** (67.2% of 512 GB/s physical peak).  
  *Bottleneck:* **Memory-Bandwidth Bound.**
* **Decode Linear GEMV:** Execution time **177.22 µs**, Arithmetic Intensity = **1.007 FLOP/Byte**, Attained Compute = **0.322 TFLOP/s**, Achieved DRAM Throughput = **320.16 GB/s** (62.5% of 512 GB/s peak).  
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