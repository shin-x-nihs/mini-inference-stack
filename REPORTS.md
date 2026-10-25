# Engineering Reports & Optimization Log

This document records the main experiments across the `vllm-reference` and `custom-serving` tracks. Each report explains what we observed, what we expected, what we tested, and what we learned.

---

## vllm-reference

### Stage 1: Speculative Decoding (`stage-1-vllm`)

*Server directory: `vllm-reference/stage-1-spec-dec/` | Benchmark stage: `stage-1-vllm`*

#### Motivation

**Observation**  
The Stage 0 baseline profiling shows that single-request generation spends a substantial fraction of its time running the target model one token at a time:
* **Decode time:** 672.3 ms out of 867.3 ms of wall time (77.5%) for the standardized workload (1024 prompt tokens, 64 output tokens, $C=1$).
* **Linear layers:** Account for 91.9% of measured GPU kernel time (linear: 704.88 ms; norm/act/other: 56.66 ms; attention: 5.10 ms).
* **Memory traffic:** Nsight Compute reports an arithmetic intensity of 1.001 FLOP/byte for the profiled decode linear kernel, reaching 321.10 GB/s (62.7% of the 512 GB/s peak DRAM bandwidth).
* **Context length:** Mean TPOT ranges from 10.67 ms ($L=128$) to 9.24 ms ($L=8192$) under the standardized fixed-prompt workload.

*(Note on profiling baseline: An earlier exploratory run reported 491.2 ms decode / 550.6 ms wall time. The updated figures reflect the standardized benchmark run using the fixed prompt and greedy temperature from `config.yaml`.)*

The low arithmetic intensity and high DRAM bandwidth utilization support the hypothesis that weight memory traffic is an important cost during single-token generation. However, profiling alone does not prove weight traffic is the sole bottleneck; launch overheads, scheduling, and cache effects also contribute.

**Hypothesis**  
Speculative decoding may reduce per-token decode latency by using a lightweight draft model to propose $K$ tokens sequentially, and then verifying all $K$ tokens in a single parallel forward pass on the target model.

Because the target model processes multiple tokens in parallel during verification, it can achieve higher arithmetic intensity than a single-token decode step. The overall latency benefit can be modeled using the explanatory approximation:

$$S \approx \frac{L_{\text{acc}} \cdot t_{\text{decode}}}{t_{\text{draft,total}} + t_{\text{verify}}} = \frac{L_{\text{acc}} \cdot t_{\text{decode}}}{K \cdot t_{\text{draft}} + t_{\text{verify}}}$$

Where:
* **$L_{\text{acc}}$**: Mean number of output tokens produced per verification round, including the bonus token where applicable ($1 \le L_{\text{acc}} \le K + 1$).
* **$t_{\text{decode}}$**: Baseline time to generate one token using the target model alone.
* **$t_{\text{draft,total}}$ ($K \cdot t_{\text{draft}}$)**: Total time spent generating the $K$ draft tokens sequentially.
* **$t_{\text{verify}}$**: Time for the target model to verify the draft proposals in parallel.

*This formula is a simplified model: actual speedup depends on the engine's implementation, batching efficiency, draft-model overhead, and serving pipeline dynamics.*

**Model Asymmetry (Hypothesis & Future Experiment)**  
* **Small model pair (1.5B target vs. 0.5B draft):** Based on parameter count and memory footprint, the per-token draft cost is estimated at $t_{\text{draft}} \approx 0.35 \cdot t_{\text{decode}}$. With $K = 5$, the drafting denominator ($5 \times 0.35 + t_{\text{verify}} / t_{\text{decode}}$) is non-trivial. The draft overhead consumes much of the savings, limiting potential speedup even when acceptance is high.
* **Larger target model (Planned Experiment):** In a setup pairing a heavier target model (e.g., `Qwen2.5-Coder-14B-Instruct-AWQ`) with the 0.5B draft model, $t_{\text{draft}} \ll t_{\text{decode}}$. In theory, the drafting overhead becomes a much smaller fraction of the target's verification time, making speculative decoding more favorable. This remains a hypothesis to test in a future experiment.

**Expected Results**  
* **Low concurrency:** Generation speed and latency may improve if the saved sequential target passes exceed the drafting and verification overhead.
* **High concurrency:** The throughput benefit may diminish or invert as higher concurrency saturates GPU compute via standard batching, making speculative drafting an extra computational overhead.
* **Kernel behavior:** Target verification executes with batch size $M = K + 1$, shifting execution characteristics relative to single-token decode.

---

#### Benchmark Controls and Limitations

Early tests with synthetic random inputs showed poor speculative-decoding performance because arbitrary token distributions disrupt language patterns, reducing draft acceptance. This motivated testing with a fixed code prompt, which better reflects the intended coding workload.

To improve repeatability:
1. **Shared configuration:** Common benchmark settings—including the fixed Python code prompt (`benchmarks.prompt`) and greedy sampling (`benchmarks.temperature: 0.0`)—are stored in `config.yaml` where supported.
2. **Consistent input lengths:** The benchmark runner tokenizes and adjusts the input to the exact requested token counts.
3. **Greedy sampling:** Greedy evaluation (`temperature: 0.0`) is configured across runner scripts and server generation settings (`--override-generation-config '{"temperature": 0.0}'`) where supported.
4. **Fair comparisons:** The baseline and speculative stages are run on identical hardware with the same workload lengths and concurrency levels. Differences in engine flags are documented.

*Limitation:* A fixed prompt improves repeatability, but it does not represent all serving workloads. Acceptance rate and latency depend on the prompt distribution, model pair, and sampling temperature.

---

#### Experiment Setup

| Setting | Value |
| :--- | :--- |
| **Target model** | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| **Draft model** | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |
| **Precision** | `bfloat16` |
| **Draft length ($K$)** | 5 speculative tokens |
| **Sampling** | Greedy (`temperature: 0.0`) |
| **Backend** | vLLM 0.31.0 with draft-model speculative decoding |
| **GPU** | NVIDIA RTX 5080 Laptop GPU, 16 GB, `sm_120` |
| **Environment** | CUDA 13.1, WSL2 Ubuntu 24.04 |

**Functional Validation Checklist**
Before recording performance data:
- [x] Confirmed the HTTP server starts and responds to `/v1/chat/completions`.
- [x] Verified token streaming and response completion via UI and curl probe.
- [x] Verified that requested input lengths (512 prompt) and output lengths (128 / 64) were produced.
- [x] Inspected raw benchmark JSONs for zero failed requests (`failed_requests: 0`).
- [x] Checked output consistency and confirmed greedy speculative verification produced coherent completions.

---

#### Results

##### 1. Concurrency Sweep (Input 512, Output 128)

*Measured results from `benchmarks/results/stage-0-vllm/pareto_c*.json` and `stage-1-vllm/pareto_c*.json`:*

| Concurrency ($C$) | Baseline Throughput | Speculative Throughput | Throughput Change | Baseline TPOT | Speculative TPOT | TPOT Change |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 87.77 tok/s | 105.97 tok/s | **+20.7%** | 10.90 ms | 8.38 ms | **-23.1%** |
| **2** | 169.46 tok/s | 182.17 tok/s | **+7.5%** | 10.98 ms | 9.06 ms | **-17.5%** |
| **4** | 353.74 tok/s | 275.85 tok/s | **-22.0%** | 10.15 ms | 11.28 ms | **+11.1%** |
| **8** | 561.94 tok/s | 448.35 tok/s | **-20.2%** | 12.27 ms | 14.27 ms | **+16.3%** |
| **16** | 831.19 tok/s | 622.88 tok/s | **-25.1%** | 16.08 ms | 20.43 ms | **+27.1%** |
| **32** | 966.96 tok/s | 755.49 tok/s | **-21.9%** | 27.86 ms | 33.77 ms | **+21.2%** |
| **64** | 1232.26 tok/s | 872.47 tok/s | **-29.2%** | 45.15 ms | 59.06 ms | **+30.8%** |

*Throughput is aggregate output tokens per second across all requests. TPOT is mean time per output token after the first token. Lower TPOT and higher throughput represent better performance.*

##### 2. Context Sweep Endpoints ($C=1$, Output 64)

*Endpoints of the context scaling sweep (lengths 128 to 8192; see `dashboard.html` for full intermediate points):*

| Metric | Baseline ($L=128$) | Speculative ($L=128$) | Baseline ($L=8192$) | Speculative ($L=8192$) |
| :--- | :---: | :---: | :---: | :---: |
| **Output Throughput** | 89.04 tok/s | 71.99 tok/s (-19.1%) | 50.95 tok/s | 24.62 tok/s (-51.7%) |
| **Mean TPOT** | 10.67 ms | 12.12 ms (+13.6%) | 9.24 ms | 13.43 ms (+45.3%) |
| **Mean TTFT** | 46.34 ms | 125.28 ms (+170.3%) | 673.66 ms | 1752.66 ms (+160.2%) |

##### 3. Acceptance Metrics

* Acceptance rate: ~98.2% across verified draft tokens on the fixed code prompt under greedy sampling (`temperature: 0.0`).
* Mean accepted sequence length ($L_{\text{acc}}$): ~5.91 tokens produced per verification step (including the target model's final evaluated token, out of a maximum $K+1 = 6$).

*Metrics definition:*
* **Acceptance rate:** Proportion of proposed draft tokens verified and accepted by the target model.
* **Mean acceptance length ($L_{\text{acc}}$):** Average total tokens accepted or produced per verification round.

---

#### Data and Profiles

* **Baseline artifacts:** `benchmarks/results/stage-0-vllm/`, `benchmarks/profiles/stage-0-vllm/`
* **Speculative decoding artifacts:** `benchmarks/results/stage-1-vllm/`, `benchmarks/profiles/stage-1-vllm/`
* **Interactive dashboard:** `benchmarks/results/dashboard.html`

---

#### Conclusion

1. **Modest benefit at low concurrency:** At concurrency $C=1$, speculative decoding produced a measurable generation-speed improvement: output throughput rose by +20.7% (87.77 to 105.97 tok/s) and TPOT decreased by -23.1% (10.90 to 8.38 ms). However, at concurrency $C=1$ in the context sweep, initial request overhead (TTFT) and longer contexts showed higher latency.
2. **Throughput trade-off at higher concurrency:** Above $C=2$, speculative decoding throughput fell behind the baseline (e.g., -20.2% at $C=8$ and -29.2% at $C=64$). When multiple requests are batched together, standard target-model decoding already achieves higher GPU compute utilization, making the extra draft-model forward passes a source of compute and memory contention.
3. **Acceptance rate is not enough:** Even with a high draft acceptance rate (~98.2%), overall speedup on the 1.5B/0.5B pair remained bounded. The time required for sequential drafting ($K \cdot t_{\text{draft}}$) and target verification ($t_{\text{verify}}$) limits the net savings when the target model itself is small.
4. **Workload matters:** Results obtained with a deterministic coding prompt under greedy sampling differ from tests with arbitrary or synthetic prompts. Workload composition and sampling temperature directly govern speculative efficiency.

**Next Steps:**
- Establish the minimal PyTorch baseline in `custom-serving/stage-0-baseline/` (`stage-0-custom`) to begin analyzing execution without vLLM's internal optimizers and memory managers.
- Plan a future experiment evaluating whether a larger target model (such as `Qwen2.5-Coder-14B-Instruct-AWQ`) makes speculative decoding with a 0.5B draft model significantly more advantageous within the 16 GB VRAM budget.
