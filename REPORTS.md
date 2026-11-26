# Engineering Reports & Optimization Log

Every benchmark and profiler run across `vllm-reference` and `custom-serving`. All measured values come from `benchmarks/results/<stage>/*.json`; all bounds come from first-principles hardware math.

---

## Summary Table

Single-run closed-loop results (`n = 1` sweep per point) on `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`), NVIDIA GeForce RTX 5080 Laptop GPU (16 GB GDDR7). Physical single-token decode floor is **3.45 ms/tok** (`3.09 GB ÷ 896 GB/s`).

| Stage | Track | C=1 TPOT (ms) | Gap to 3.45 ms Floor | C=1 Tput (tok/s) | C=64 Tput (tok/s) | L=2048 TTFT (ms) | L=8192 TTFT (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `stage-0-vllm` | vLLM (Baseline) | 10.9 | 3.2× | 87.8 | 1,232 | 208 | 674 |
| `stage-1-vllm` | vLLM (Spec Dec, K=5) | 8.4 | 2.4× | 106 | 872 | 412 | 1,753 |
| `stage-2-vllm` | vLLM (Prefix Cache)* | 12.2 | 3.5× | 80.4 | 2,874 | 114 | 198 |
| `stage-0-custom` | Custom (Naive PyTorch) | 66.8 | 19.4× | 14.9 | — | 276 | — |
| `stage-1-custom` | Custom (KV Cache) | 26.4 | 7.7× | 37.3 | — | 330 | — |
| `stage-2-custom` | Custom (Tiled Attn) | 22.4 | 6.5× | 44.0 | — | 196 | — |
| `stage-3-custom` | Custom (Split-K) | 21.9 | 6.4× | 45.0 | — | 191 | — |
| `stage-4-custom` | Custom (Paged KV + Batch) | 25.7 | 7.4× | 38.5 | 1,260 | 207 | 812 |
| `stage-5-custom` | Custom (Fused Ops) | 14.2 | 4.1× | 68.4 | 1,486 | 203 | 812 |
| `stage-6-custom` | Custom (CUDA Graphs) | 8.4 | 2.4× | 113 | 1,771 | 174 | 629 |

*\* `stage-2-vllm` runs on identical repeated prompts (`h ≈ 1` after request 1) — an upper bound, not mixed traffic.*

---

## Methodology, Theoretical Bounds & Known Artifacts

### 1. Hardware & Reference Ceilings
* **Platform:** NVIDIA GeForce RTX 5080 Laptop GPU (16 GB GDDR7, 256-bit bus, 7,680 CUDA cores / 60 SMs, Blackwell `sm_120`), Windows 11 + WSL2 Ubuntu 24.04, CUDA 13.1, PyTorch 2.13.0+cu130, vLLM 0.31.0.
* **Reference ceilings (`config.yaml`):** **896 GB/s** GDDR7 bandwidth ([NVIDIA RTX 50 Series Laptops](https://www.nvidia.com/en-us/geforce/laptops/50-series/)) and a frozen **40 TFLOP/s** conservative compute reference (NVIDIA lists 1,334 FP4-sparse AI TOPS rather than a fixed BF16 figure; actual BF16 dense peak varies with laptop TGP/clock, roughly 46–70 TFLOP/s across 1,500–2,287 MHz per the [RTX Blackwell Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)). `40 TFLOP/s` is kept frozen solely as a fixed Roofline baseline; unprofiled boost-clock prefill at `L = 8192` reaches ~40–42 TFLOP/s end-to-end.
* **NCU locked base clock:** `profile_ncu.sh` uses NCU's default `--clock-control base`, locking clocks below boost. NCU's internal `SOL%` counters (`DRAM SOL%`, `SM SOL%`) therefore read ~2× higher than `% of 896 GB/s`, and Layer 0 prefill GEMM sits at ~27 TFLOP/s (~95% SM SOL). Both `% of reference` and `NCU SOL%` are reported to match `dashboard.html`.

### 2. First-Principles Hardware Bounds (`Qwen2.5-Coder-1.5B-Instruct`, `bfloat16`)
* **Model weight bytes:** 28 layers, `d = 1536`, `d_mlp = 8960`, 12 Q / 2 KV heads (GQA 6:1), `head_dim = 128`, tied embeddings → **1.54B params ≈ 3.09 GB (2.88 GiB)**.
* **Decode floor (`C = 1`):** Streaming all weights once per token sets a hard floor of `3.09 GB ÷ 896 GB/s = 3.45 ms/tok` (`~290 tok/s`).
* **KV cache traffic:** `28 layers × 2 (K,V) × 2 KV heads × 128 × 2 B = 28 KiB/tok` → `14 MiB` at `L = 512`, `56 MiB` at `L = 2048`, `224 MiB` / `235 MB` at `L = 8192` (`7.6%` of weight bytes). With KV caching, `C = 1` TPOT should stay nearly flat across `L = 128..8192`.
* **Prefill attention vs. linear FLOPs:** Causal attention costs `~86k · L` FLOP/tok vs. `~2.62 GFLOP/tok` for linear layers (`ratio ≈ L / 30,500`): **1.7%** at `L = 512`, **6.7%** at `L = 2048`, **27%** at `L = 8192`. Below a few thousand tokens, prefill compute is mostly linear in `L`.

### 3. Workload & Measurement Artifacts
* **Identical repeated prompts:** `bench_closed_loop.sh` repeats `benchmarks.prompt` to length `L` and sends the exact same prompt to every request. This makes `stage-2-vllm` (`+133%` at `C = 64`) a **100% cache-hit upper bound (`h ≈ 1`)** and inflates `stage-1-vllm` speculative acceptance (`~98.2%`).
* **Tensor Core tile-padding inflation (`ncu`):** `profile_ncu.sh` estimates tensor FLOPs as `inst_pipe_tensor × 4096`. Padded tiles inflate reported Arithmetic Intensity above physical limits — notably `stage-1-vllm` `M = 6` verification padded to `m16` (`AI = 15.5` vs. physical max `6 FLOP/B`) and `stage-2-custom` `M = 1` decode padded to `BLOCK_M = 16` (`AI = 99.2 FLOP/B`).
* **`cpu_launch_overhead_ms` includes blocking calls:** `profile_nsys.sh` sums host duration of all CUDA runtime `*Launch*` calls (including `cudaGraphLaunch`). Under back-pressure, blocking `cudaGraphLaunch` calls inflate this counter (`stage-6-custom`: `285 ms` across 476 calls ≈ `4.4 ms` per graph launch, not eager prefill dispatch).
* **Single runs & ±8–10% laptop variance:** Each sweep point ran once (`10` prompts for context; `max(20, 10·C)` for Pareto) across separate dates (vLLM Oct 2026; custom Nov 2026) on a thermally constrained laptop GPU. Serialized stages (`stage-1..3-custom`) still spread `16–17%` across `C`, and `stage-0-vllm` context TPOT varies non-monotonically (`10.7 → 9.0 → 10.4 → 9.2 ms`). Deltas under ~10% sit inside noise.
* **Nominal vs. actual prompt lengths:** Chat-template framing and re-tokenization shift actual input tokens: `128 → 155`, `256 → 280`, `512 → 531`, `1024 → 1,033`, `2048 → 2,037`, `4096 → 4,045`, `8192 → 8,061`.
* **Cross-track profiler asymmetries:** Custom `nsys` phase timers use host **NVTX** ranges; vLLM stages fall back to a **GPU kernel timeline split** heuristic. Compare tracks via client benchmark TPOT/TTFT. In `ncu`, `profile_ncu.sh` records the single longest Layer 0 GEMM/GEMV (`1×` `gate_proj`/`up_proj` in Custom Stages 1–4 vs. `2×` merged `gate_up_proj` in vLLM and Custom Stages 5–6).

---

## Track 1: `vllm-reference`

### Stage 0: Reference Baseline (`stage-0-vllm`)
*Directory: `vllm-reference/stage-0-baseline/` | Data: `benchmarks/results/stage-0-vllm/`*

Production baseline (`vLLM 0.31.0`, `bfloat16`, `--no-enable-prefix-caching`, CUDA Graphs + continuous batching enabled).

#### Measured Results
* **Concurrency (`L_in = 512`, `L_out = 128`, `failed = 0`):**

| C | Tput (tok/s) | TPOT (ms) | TTFT (ms) | P99 TTFT (ms) | P50 ITL (ms) | P99 ITL (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 87.8 | 10.9 | 73.3 | 202 | 10.9 | 14.5 |
| **2** | 169 | 11.0 | 115 | 353 | 10.3 | 15.1 |
| **4** | 354 | 10.2 | 157 | 495 | 9.0 | 16.3 |
| **8** | 562 | 12.3 | 262 | 713 | 10.5 | 32.1 |
| **16** | 831 | 16.1 | 418 | 1,267 | 13.2 | 143 |
| **32** | 967 | 27.9 | 687 | 2,255 | 17.7 | 239 |
| **64** | 1,232 | 45.2 | 896 | 3,890 | 20.2 | 259 |

* **Context (`C = 1`, `L_out = 64`, `failed = 0`):**

| Nominal L (Actual) | Tput (tok/s) | TPOT (ms) | TTFT (ms) |
| :---: | :---: | :---: | :---: |
| **128** (155) | 89.0 | 10.7 | 46.3 |
| **256** (280) | 87.2 | 10.8 | 56.5 |
| **512** (531) | 89.4 | 10.1 | 79.6 |
| **1024** (1,033) | 95.2 | 9.0 | 105 |
| **2048** (2,037) | 75.7 | 10.1 | 208 |
| **4096** (4,045) | 59.7 | 10.4 | 416 |
| **8192** (8,061) | 51.0 | 9.2 | 674 |

#### Baseline Profile (`nsys` & `ncu`)
* **Timeline (`nsys`, `1024-in / 64-out`, `C = 1`):** `867 ms` wall time (`88.4%` GPU active, `11.6%` idle), `21,041` kernels, `1,636` launch calls (`28.9 ms` Launch API time), `63` CUDA graph launches. Decode takes `672 ms` (`77.5%`, `10.7 ms/step` via kernel-split heuristic) vs. `195 ms` prefill. GPU time split: `91.9%` linear (`705 ms`), `0.7%` attention (`5.1 ms`), `7.4%` norm/act/other (`56.7 ms`).
* **Layer 0 (`ncu`, `L = 2048` prefill / `M = 1` decode, `0` spills):**
  * `Prefill Linear GEMM`: `4,125 µs`, `AI = 969 FLOP/B`, `27.3 TFLOP/s` (`68.4%` ref; `95.0%` SM SOL), `28.2 GB/s` (`6.2%` DRAM SOL).
  * `Prefill Attention`: `733 µs`, `AI = 1,631 FLOP/B`, `18.8 TFLOP/s` (`47.1%` ref; `65.0%` SM SOL), `11.6 GB/s` (`2.5%` DRAM SOL).
  * `Decode Linear GEMV`: `178 µs`, `AI = 1.00 FLOP/B`, `0.32 TFLOP/s` (`0.8%` ref; `41.5%` SM SOL), `321 GB/s` (`35.8%` of 896 GB/s; `70.9%` DRAM SOL).
  * `Decode Attention (SplitKV + Combine)`: `26.0 µs` (`82.9 GB/s`) + `21.6 µs` (`5.9 GB/s`).
* **Two bottlenecks motivating isolated vLLM stages:**
  1. `C = 1` decode sits at `10.9 ms` (**3.2×** above the 3.45 ms floor), streaming 3.09 GB/token at `AI = 1.0 FLOP/B` → **Stage 1: Speculative Decoding**.
  2. `C = 64` prefill contention drives `P99 TTFT = 3.89 s` and `P99 ITL = 259 ms` → **Stage 2: Automatic Prefix Caching**.

---

### Stage 1: Speculative Decoding (`stage-1-vllm`)
*Directory: `vllm-reference/stage-1-spec-dec/` | Data: `benchmarks/results/stage-1-vllm/`*

#### 1. Bottleneck (from Prior Profiler)
In `stage-0-vllm` (`C = 1`), decode consumes `77.5%` of `nsys` wall time (`672 / 867 ms`), with `91.9%` of GPU time in `M = 1` linear GEMVs (`AI = 1.00 FLOP/B`, `321 GB/s`, `0.32 TFLOP/s`).

#### 2. Theoretical Bound
* **Decode floor (`C = 1`):** `stage-0-vllm` runs at `10.9 ms/tok` (`3.2×` above the `3.45 ms` floor).
* **Speculative speedup ceiling:** With `K` draft tokens verified in parallel (`M = K + 1`):
  $$S \approx \frac{L_{\text{acc}} \cdot t_{\text{target}}}{K \cdot t_{\text{draft}} + t_{\text{verify}}}$$
  Assuming `t_verify ≈ t_target`, `100%` acceptance (`L_acc = K + 1 = 6`), and weight-bandwidth scaling `c = t_draft / t_target ≈ 0.49B / 1.54B ≈ 1/3`, the idealized ceiling is `6 / (5/3 + 1) ≈ 2.25×`. At `C ≥ 8`, continuous batching already forms `M ≥ 8` GEMMs, removing the unused compute slack that speculation exploits.

#### 3. Change
Enabled draft-model speculative decoding in `vllm-reference/stage-1-spec-dec/serve.sh` (`Qwen/Qwen2.5-Coder-0.5B-Instruct`, `bfloat16`, `K = 5`, greedy `temperature = 0.0`). Greedy outputs matched baseline; `failed = 0`.

#### 4. Measured Results

* **Concurrency (`L_in = 512`, `L_out = 128`):**

| C | S0 Tput | S1 Tput | Tput Δ | S0 TPOT (ms) | S1 TPOT (ms) | TPOT Δ | S1 P50 ITL (ms) | S1 P99 ITL (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 87.8 | 106 | +21% | 10.9 | 8.4 | −23% | 42.9 | 69.1 |
| **2** | 169 | 182 | +7.5% | 11.0 | 9.1 | −18% | 46.1 | 76.5 |
| **4** | 354 | 276 | −22% | 10.2 | 11.3 | +11% | 52.1 | 85.0 |
| **8** | 562 | 448 | −20% | 12.3 | 14.3 | +16% | 58.1 | 279 |
| **16** | 831 | 623 | −25% | 16.1 | 20.4 | +27% | 72.6 | 336 |
| **32** | 967 | 755 | −22% | 27.9 | 33.8 | +21% | 101 | 361 |
| **64** | 1,232 | 872 | −29% | 45.2 | 59.1 | +31% | 371 | 393 |

* **Context (`C = 1`, `L_out = 64`):**

| L | S0 Tput | S1 Tput | Tput Δ | S0 TPOT (ms) | S1 TPOT (ms) | S0 TTFT (ms) | S1 TTFT (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 89.0 | 72.0 | −19% | 10.7 | 12.1 | 46.3 | 125 |
| **256** | 87.2 | 99.2 | +14% | 10.8 | 8.0 | 56.5 | 142 |
| **512** | 89.4 | 93.0 | +4.0% | 10.1 | 8.2 | 79.6 | 174 |
| **1024** | 95.2 | 79.0 | −17% | 9.0 | 8.7 | 105 | 262 |
| **2048** | 75.7 | 59.3 | −22% | 10.1 | 10.6 | 208 | 412 |
| **4096** | 59.7 | 45.9 | −23% | 10.4 | 10.2 | 416 | 755 |
| **8192** | 51.0 | 24.6 | −52% | 9.2 | 13.4 | 674 | 1,753 |

* **Acceptance & burstiness:** `~98.2%` (`L_acc ≈ 5.91 / 6`, optimistic on repeated prompts). At `C = 1`, TPOT dropped `10.9 → 8.4 ms`, while P50 ITL rose `10.9 → 42.9 ms` (`42.9 / 8.4 ≈ 5.1×`) because tokens flush in verified bursts.

#### 5. Gap Analysis & Next Move
* **Why `1.30×` TPOT speedup fell short of the `2.25×` ceiling:** Using `P50 ITL ≈ 42.9 ms` as round time and `t_verify ≈ 10.9 ms`, each draft step took `(42.9 − 10.9) / 5 ≈ 6.4 ms`, so **`c ≈ 0.59`** — nearly double the `0.33` parameter ratio. The `0.5B` draft has `24` layers vs. `28` in the `1.5B` target (`0.86` ratio), making it launch/latency-bound rather than weight-bound. Plugging `c = 0.59` into the model gives `5.91 × 10.9 / 42.9 ≈ 1.50×`, close to the measured `1.30×` (`8.4 ms`, still `2.4×` above the 3.45 ms floor).
* **Profiler (`ncu` & `nsys`):**
  * `ncu` (`Decode Linear GEMM`, `M = 6`): took **`175 µs`** at **`326 GB/s`** (`36.3%` of 896 GB/s; `71.9%` DRAM SOL) — virtually identical to `stage-0-vllm`'s `M = 1` GEMV (`178 µs`, `321 GB/s`), confirming 6-token weight reuse. *(Artifact: `ncu` reports `AI = 15.5 FLOP/B` and `5.04 TFLOP/s` from `m16` tile padding; physical max is `6 FLOP/B`, useful compute `~1.9 TFLOP/s`.)*
  * `nsys`: `decode_total_ms` dropped `672 → 557 ms`, but alternating draft/target work broke full CUDA Graph capture (`cuda_graph_launches`: `63 → 13`, `cpu_launch_calls`: `1,636 → 19,843`, Launch API time: `28.9 → 277 ms`, GPU idle: `11.6% → 35.3%`).
* **Benchmark discrepancy:** TTFT rose `1.8×–2.7×` across `L` (`674 → 1,753 ms` at `L = 8192`) from dual-model prefill and initial-round overhead, dragging short-output (`64`-tok) context throughput down. The `L = 128` TPOT spike (`12.1 ms` vs. `8.0–8.2 ms` at `L = 256–512`) is likely single-run variance on the first context sweep point.
* **Next move:** At `C ≥ 4`, speculation regresses `20–29%`. Branch **Stage 2: Automatic Prefix Caching (`stage-2-vllm`)** off `stage-0-vllm` to test high-concurrency prefill contention in isolation.
* **Sources:** `benchmarks/results/stage-1-vllm/{pareto_c*,context_l*,nsys_metrics,ncu_metrics}.json`.

---

### Stage 2: Automatic Prefix Caching (`stage-2-vllm`)
*Directory: `vllm-reference/stage-2-prefix-cache/` | Data: `benchmarks/results/stage-2-vllm/`*

#### 1. Bottleneck (from Prior Profiler)
In `stage-0-vllm`, scaling `C = 1 → 64` drove mean TTFT `73.3 → 896 ms` (`12.2×`), P99 TTFT `202 → 3,890 ms`, and P99 ITL `14.5 → 259 ms`, as repeated `4,125 µs`/layer prefill GEMMs (`95.0%` SM SOL) stalled active decode batches.

#### 2. Theoretical Bound
* Under closed-loop concurrency `C`, Little's Law links throughput to request time: `Throughput ≈ C × L_out / (TTFT + (L_out − 1) × TPOT)`.
* With prefix reuse `h = L_prefix / L`, skipping cached tokens reduces prefill to `t_lookup + t_prefill((1 − h)L)`. Because `bench_closed_loop.sh` sends identical prompts (`h ≈ 1` after request 1), prefill compute approaches zero — an **upper bound** on prefix caching.
* Retained KV memory from model geometry (`28 KiB/tok`): `14 MiB` at `L = 512` up to `224 MiB` at `L = 8192` per cached sequence.

#### 3. Change
Enabled vLLM's hash-based Automatic Prefix Caching (`--enable-prefix-caching`) in `vllm-reference/stage-2-prefix-cache/serve.sh` off `stage-0-vllm`. vLLM indexes KV blocks on the CPU via parent-chained block hashes (`O(#blocks)` lookup, distinct from SGLang's RadixAttention tree). Greedy outputs matched baseline; `failed = 0`.

#### 4. Measured Results

* **Concurrency (`L_in = 512`, `L_out = 128`, identical prompts `h ≈ 1`):**

| C | S0 Tput | S2 Tput | Tput Δ | S0 TTFT (ms) | S2 TTFT (ms) | TTFT Δ | S0 P99 TTFT | S2 P99 TTFT | S0 TPOT | S2 TPOT | S2 P99 ITL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 87.8 | 80.4 | −8.5% | 73.3 | 38.4 | −48% | 202 | 229 | 10.9 | 12.2 | 15.2 |
| **2** | 169 | 139 | −18% | 115 | 61.7 | −46% | 353 | 320 | 11.0 | 14.0 | 15.8 |
| **4** | 354 | 260 | −27% | 157 | 65.9 | −58% | 495 | 271 | 10.2 | 15.0 | 19.1 |
| **8** | 562 | 515 | −8.4% | 262 | 70.3 | −73% | 713 | 227 | 12.3 | 15.1 | 17.6 |
| **16** | 831 | 972 | +17% | 418 | 94.4 | −77% | 1,267 | 394 | 16.1 | 15.8 | 20.3 |
| **32** | 967 | 1,729 | +79% | 687 | 162 | −76% | 2,255 | 865 | 27.9 | 17.3 | 26.5 |
| **64** | 1,232 | 2,874 | +133% | 896 | 229 | −75% | 3,890 | 1,212 | 45.2 | 20.6 | 44.4 |

* **Context (`C = 1`, `L_out = 64`, `10` identical prompts):**

| L | S0 TTFT (ms) | S2 TTFT (ms) | TTFT Δ | S0 TPOT (ms) | S2 TPOT (ms) | TPOT Δ | S0 Tput | S2 Tput | Tput Δ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 46.3 | 37.5 | −19% | 10.7 | 10.5 | −1.5% | 89.0 | 91.4 | +2.7% |
| **256** | 56.5 | 85.9 | +52% | 10.8 | 12.9 | +20% | 87.2 | 71.4 | −18% |
| **512** | 79.6 | 89.3 | +12% | 10.1 | 13.7 | +36% | 89.4 | 67.2 | −25% |
| **1024** | 105 | 105 | −0.3% | 9.0 | 13.6 | +51% | 95.2 | 66.6 | −30% |
| **2048** | 208 | 114 | −45% | 10.1 | 13.6 | +35% | 75.7 | 65.9 | −13% |
| **4096** | 416 | 138 | −67% | 10.4 | 14.0 | +34% | 59.7 | 62.8 | +5.2% |
| **8192** | 674 | 198 | −71% | 9.2 | 14.0 | +51% | 51.0 | 59.4 | +17% |

#### 5. Gap Analysis & Next Move
* **High concurrency (`h ≈ 1` upper bound):** At `C = 64`, throughput rose **`+133%` (`1,232 → 2,874 tok/s`)**, mean TTFT dropped `75%` (`896 → 229 ms`), P99 TTFT dropped `69%` (`3,890 → 1,212 ms`), and P99 ITL dropped `83%` (`259 → 44.4 ms`). Freed from prefill stalls, `C = 64` batches run near `M = 64` linear GEMMs (`AI ≈ 60`; note `M = 1` NCU counters do not describe `C = 64` decode).
* **Profiler (`nsys` & `ncu`, `C = 1`, single uncached request):** `nsys` wall time (`861 ms`), launch calls (`1,636`), Launch API time (`30.0 ms`), CUDA Graphs (`63`), and GPU active share (`86.9%`) matched `stage-0-vllm`, as did all `ncu` Layer 0 kernels (within ~2%).
* **Open discrepancies:**
  1. `nsys` `decode_step_avg_ms` fell `10.7 → 8.7 ms` (`−19%`), yet benchmark TPOT *rose* `+12%` at `C = 1` (`10.9 → 12.2 ms`, `3.5×` above floor) and `+34–51%` across `L = 512..8192`. `stage-2-vllm` ran 19 days after `stage-0-vllm` (2026-11-14 vs. 2026-10-25); thermal state or host-side cache bookkeeping outside GPU kernels may contribute, but no trace isolates the cause.
  2. TTFT regressed at `L = 256` (`+52%`) and `L = 512` (`+12%`) despite `h ≈ 1`, before dropping `45–71%` at `L ≥ 2048`. Needs repeat runs across `h ∈ {0, 0.5, 0.9}` to separate variance from lookup overhead.
* **Next move:** Move to `custom-serving/stage-0-baseline/` to build an unoptimized PyTorch engine from scratch and solve each hardware bottleneck additively.
* **Sources:** `benchmarks/results/stage-2-vllm/{pareto_c*,context_l*,nsys_metrics,ncu_metrics}.json`.

---

## Track 2: `custom-serving` (Strictly Additive)

### Stage 0: Naive PyTorch Baseline (`stage-0-custom`)
*Directory: `custom-serving/stage-0-baseline/` | Data: `benchmarks/results/stage-0-custom/`*

Eager PyTorch baseline with no KV cache, explicit `(Q @ K^T)` attention (`repeat_interleave` + causal mask), and `asyncio.Lock` serialization. Verified `32/32` greedy token parity vs. HF `eager` (`max-abs prefill logit diff = 0.0625`).

#### Measured Results
* **Concurrency (`L_in = 512`, `L_out = 128`, `C = 1..16`, `failed = 0`; `C = 32, 64` omitted due to serialized runtime):**

| C | Tput (tok/s) | TPOT (ms) | TTFT (ms) | P50 ITL (ms) | P99 ITL (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 14.9 | 66.8 | 87.5 | 58.4 | 109 |
| **2** | 14.8 | 67.3 | 8,360 | 58.0 | 110 |
| **4** | 14.6 | 68.4 | 25,088 | 59.1 | 110 |
| **8** | 14.6 | 68.3 | 58,303 | 58.5 | 110 |
| **16** | 14.6 | 68.4 | 124,831 | 58.6 | 109 |

* **Context (`C = 1`, `L_out = 64`, `L = 128..2048`, `failed = 0`; `L = 4096, 8192` omitted due to `O(S^2)` attention cost):**

| L | Tput (tok/s) | TPOT (ms) | TTFT (ms) |
| :---: | :---: | :---: | :---: |
| **128** | 28.5 | 34.7 | 60.0 |
| **256** | 24.0 | 41.3 | 62.4 |
| **512** | 18.9 | 52.7 | 75.7 |
| **1024** | 8.3 | 121 | 155 |
| **2048** | 3.8 | 266 | 276 |

#### Baseline Profile (`nsys`; `ncu` deferred to Stage 1)
* **Timeline (`1024-in / 64-out`, `C = 1`):** `7,119 ms` wall time (`91.1%` GPU active, `85,248` kernels, `0` CUDA graphs). Prefill takes `118 ms`; `63` decode tokens take **`6,954 ms` (`97.7%` of wall time, `110 ms/step`)**.
* **Kernel split:** `64.5%` linear (`4,182 ms`), `26.7%` attention (`1,732 ms`), `8.9%` norm/act/other (`574 ms`).
* **Forced next move:** Generating 1 token costs nearly as much as prefilling 1,024 tokens because every decode step recomputes the full `[1, L + t]` sequence. `nsys` isolates this cleanly without needing `ncu` → **Stage 1: Naive KV Cache**.

---

### Stage 1: Naive KV Cache (`stage-1-custom`)
*Directory: `custom-serving/stage-1-kv-cache/` | Data: `benchmarks/results/stage-1-custom/`*

#### 1. Bottleneck (from Prior Profiler)
In `stage-0-custom`, decoding 1 token at `L = 1024` takes `110 ms` (vs. `118 ms` to prefill all 1,024 tokens), `91.2%` of GPU time is redundant linear + attention recomputation, and `L = 2048` TPOT reaches `266 ms`.

#### 2. Theoretical Bound
* **Decode floor (`C = 1`):** `3.45 ms/tok` (Stage 0 sits at `66.8 ms`, `19.4×` above floor).
* **Complexity reduction:** Caching past `(K, V)` drops per-step linear work from `O(S·d²)` to `O(d²)` (`M = 1` GEMV) and attention from `O(S²·d)` to `O(S·d)`. Because KV bytes at `L = 2048` are only `56 MiB` (`~1.8%` of weight bytes), decode TPOT should flatten across `L = 128..2048`.

#### 3. Change
Added per-layer unexpanded `(k, v)` cache tensors (`[B, 2, S, 128]`), offset RoPE, `torch.cat` append, and single-token decode without a causal mask. Verified `32/32` greedy token parity vs. HF `eager` (`max-abs prefill logit diff = 0.0625`); `failed = 0`.

#### 4. Measured Results

* **Concurrency (`L_in = 512`, `L_out = 128`, serialized):**

| C | S0 Tput | S1 Tput | Tput Δ | S0 TPOT (ms) | S1 TPOT (ms) | TPOT Δ | S1 TTFT (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 14.9 | 37.3 | +150% (2.5×) | 66.8 | 26.4 | −61% | 74.0 |
| **2** | 14.8 | 38.0 | +156% (2.6×) | 67.3 | 26.0 | −61% | 3,235 |
| **4** | 14.6 | 36.4 | +149% (2.5×) | 68.4 | 27.1 | −60% | 10,062 |
| **8** | 14.6 | 39.6 | +171% (2.7×) | 68.3 | 25.0 | −64% | 21,598 |
| **16** | 14.6 | 37.7 | +158% (2.6×) | 68.4 | 26.3 | −62% | 48,607 |

* **Context (`C = 1`, `L_out = 64`):**

| L | S0 TTFT (ms) | S1 TTFT (ms) | S0 TPOT (ms) | S1 TPOT (ms) | TPOT Δ | S0 Tput | S1 Tput | Tput Δ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 60.0 | 51.2 | 34.7 | 22.6 | −35% | 28.5 | 43.5 | +52% |
| **256** | 62.4 | 61.9 | 41.3 | 24.3 | −41% | 24.0 | 40.2 | +67% |
| **512** | 75.7 | 77.2 | 52.7 | 22.6 | −57% | 18.9 | 42.5 | +126% |
| **1024** | 155 | 145 | 121 | 23.4 | −81% | 8.3 | 39.6 | +380% |
| **2048** | 276 | 330 | 266 | 24.1 | −91% (11.0×) | 3.8 | 34.6 | +820% (9.2×) |

#### 5. Gap Analysis & Next Move
* **Gap to floor:** At `L = 2048`, TPOT dropped `11.0×` (`266 → 24.1 ms`) and stayed flat (`22.6–24.3 ms`) across `L = 128..2048`. At Pareto `C = 1`, TPOT is `26.4 ms` (**7.7×** above the `3.45 ms` floor).
* **Timeline (`nsys`, `1024-in / 64-out`, `C = 1`):** Wall time dropped `76%` (`7,119 → 1,674 ms`) and `decode_step_avg_ms` fell `5.1×` (`110 → 21.5 ms`). With GPU active time down `8.5×` (`6,488 → 764 ms`), `85,058` eager launches (`647 ms` Launch API time) flipped GPU idle from `8.9%` to **`54.3%` (`910 ms`)**. Attention dropped `9.4×` (`1,732 → 185 ms`) but stayed `36×` slower than `stage-0-vllm` (`5.1 ms`).
* **Layer 0 (`ncu`, `L = 2048` prefill / `M = 1` decode):**
  * `Prefill Linear GEMM`: `2,181 µs`, `AI = 756 FLOP/B`, `25.9 TFLOP/s` (`64.6%` ref; `90.9%` SM SOL; `1×` separate `gate_proj`/`up_proj`).
  * `Prefill Attention` (unfused): score BMM takes `1,209 µs` (`AI = 135 FLOP/B`, `79.0 GB/s`, `17.4%` DRAM SOL), followed by separate HBM passes over `[1, 12, 2048, 2048]` for scale, mask, fp32 cast, and `softmax_warp_forward` (`1,012 µs` at `376 GB/s`, `82.9%` DRAM SOL) — over `4,100 µs` total vs. `733 µs` (`1,631 FLOP/B`, `11.6 GB/s`) in `stage-0-vllm`.
  * `Decode Linear GEMV`: `94.7 µs`, `AI = 1.01 FLOP/B`, `291 GB/s` (`32.5%` of 896 GB/s; `64.3%` DRAM SOL).
  * `Decode KV Write` (`torch.cat` / `CatArrayBatchedCopy`): `24.8 µs` at `42.5 GB/s` (`2.8×` slower than vLLM's in-place `9.0 µs`).
* **Next move:** Unfused prefill attention materializes the `S × S` matrix in HBM (`>4,100 µs` vs. `733 µs` in vLLM, pushing `L = 2048` TTFT to `330 ms` vs. `208 ms`) → **Stage 2: Tiled FlashAttention-2 (`stage-2-custom`)**.
* **Sources:** `benchmarks/results/stage-1-custom/{pareto_c*,context_l*,nsys_metrics,ncu_metrics}.json`.

---

### Stage 2: Tiled Attention / FlashAttention (`stage-2-custom`)
*Directory: `custom-serving/stage-2-tiled-attn/` | Data: `benchmarks/results/stage-2-custom/`*

#### 1. Bottleneck (from Prior Profiler)
In `stage-1-custom` (`L = 2048`), unfused prefill attention materializes `[1, 12, 2048, 2048]` in GDDR7 (`1,209 µs` BMM at `135 FLOP/B` + `1,012 µs` softmax at `376 GB/s` + scale/mask/cast > `4,100 µs`), driving `L = 2048` TTFT to `330 ms` vs. `208 ms` in `stage-0-vllm`.

#### 2. Theoretical Bound
* From first principles (`L / 30,500`), attention is `<0.5%` of prefill FLOPs at `L = 128` and `6.7%` at `L = 2048`. Yet Stage 1's unfused HBM traffic makes attention take `~1.9×` longer than the `2,181 µs` linear GEMM (`~116 ms` across 28 layers, `~35%` of `330 ms` TTFT).
* Keeping tiles in SRAM eliminates the `S × S` HBM materialization (`~95 MB` in Stage 1's BMM alone), bounding `L = 2048` prefill attention near `stage-0-vllm`'s `733 µs` (`1,631 FLOP/B`, `11.6 GB/s`) and `L = 2048` TTFT near `~200–210 ms`, with minimal effect at `L = 128`.

#### 3. Change
Replaced unfused PyTorch attention with a custom Triton FlashAttention-2 forward kernel (`_flash_attention_fwd`: SRAM online `fp32` softmax, `BLOCK_M = 64` prefill / `16` decode, `BLOCK_N = 64`, native GQA head indexing, analytical causal mask, zero-copy output strides). Verified `32/32` greedy token parity vs. HF `sdpa` (`max-abs prefill logit diff = 0.219`); `failed = 0`.

#### 4. Measured Results

* **Concurrency (`L_in = 512`, `L_out = 128`, serialized):**

| C | S1 Tput | S2 Tput | Tput Δ | S1 TPOT (ms) | S2 TPOT (ms) | TPOT Δ | S1 TTFT (ms) | S2 TTFT (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 37.3 | 44.0 | +18% | 26.4 | 22.4 | −15% | 74.0 | 61.2 |
| **2** | 38.0 | 47.7 | +26% | 26.0 | 20.6 | −21% | 3,235 | 2,615 |
| **4** | 36.4 | 45.4 | +25% | 27.1 | 21.7 | −20% | 10,062 | 8,022 |
| **8** | 39.6 | 45.0 | +14% | 25.0 | 22.0 | −12% | 21,598 | 18,944 |
| **16** | 37.7 | 40.5 | +7.6% | 26.3 | 24.4 | −7.0% | 48,607 | 45,163 |

* **Context (`C = 1`, `L_out = 64`):**

| L | S1 TTFT (ms) | S2 TTFT (ms) | TTFT Δ | `stage-0-vllm` TTFT | S1 TPOT (ms) | S2 TPOT (ms) | S1 Tput | S2 Tput |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 51.2 | 46.8 | −8.7% | 46.3 | 22.6 | 21.5 | 43.5 | 45.7 |
| **256** | 61.9 | 53.8 | −13% | 56.5 | 24.3 | 20.7 | 40.2 | 47.1 |
| **512** | 77.2 | 71.2 | −7.8% | 79.6 | 22.6 | 22.2 | 42.5 | 43.5 |
| **1024** | 145 | 104 | −28% | 105 | 23.4 | 23.1 | 39.6 | 41.0 |
| **2048** | 330 | 196 | −41% | 208 | 24.1 | 22.1 | 34.6 | 40.2 |

#### 5. Gap Analysis & Next Move
* **Prefill closed the gap to vLLM:** TTFT dropped `−8.7%` at `L = 128` and `−41%` at `L = 2048` (`330 → 196 ms`, matching `stage-0-vllm`'s `208 ms`). In `ncu` (`L = 2048`), `_flash_attention_fwd` ran in **`978 µs`** at **`1,598 FLOP/B`** (`13.8 TFLOP/s`, `34.4%` ref / `47.4%` SM SOL) and **`8.6 GB/s`** (`1.9%` DRAM SOL), cutting DRAM bytes from `~95 MB` (Stage 1 BMM alone) to **`~8.4 MB`** (`−91%`).
* **Decode gap (`22.4 ms`, `6.5×` above floor) & `nsys` contradiction:** Removing 7 unfused ops/layer cut `nsys` kernels by `12,602` (`85,058 → 72,456`) and attention time by `40%` (`185 → 111 ms`). Attention's share fell `24.2% → 9.3%`, partly because unchanged `linear_ms` jumped `479 → 858 ms` in the same trace (likely laptop thermal throttling). That also caused a sign contradiction: `nsys` `decode_step_avg_ms` *regressed* `+17%` (`21.5 → 25.2 ms`) while unprofiled `C = 1` TPOT *improved* `−15%` (`26.4 → 22.4 ms`).
* **Forced next move (`ncu` `M = 1` decode):** Reusing query-tiled `_flash_attention_fwd` at `q_len = 1` launched only `grid = (1, 12, 1)` = **12 CTAs** across 60 SMs, taking **`129 µs`** (`8.3%` active warps, `5.6%` SM SOL, `16.4 GB/s` / `3.6%` DRAM SOL) — `4×` slower than Stage 1's cuBLAS GEMV (`32.2 µs`) and `5×` slower than vLLM's SplitKV (`26.0 µs`), while padding `M = 1` to `BLOCK_M = 16` wasted `15/16` tile rows (`AI = 99.2` artifact). This SM starvation forced **Stage 3: Split-K Parallel Decode (`stage-3-custom`)**.
* **Sources:** `benchmarks/results/stage-2-custom/{pareto_c*,context_l*,nsys_metrics,ncu_metrics}.json`.

---

### Stage 3: Parallel Decode Attention / Split-K (`stage-3-custom`)
*Directory: `custom-serving/stage-3-split-k/` | Data: `benchmarks/results/stage-3-custom/`*

#### 1. Bottleneck (from Prior Profiler)
In `stage-2-custom`, `ncu` showed single-token decode attention launched only 12 CTAs (`grid = (1, 12, 1)`), starving SMs (`8.3%` active warps, `16.4 GB/s`, `129 µs` vs. `26.0 + 21.6 = 47.6 µs` in `stage-0-vllm`).

#### 2. Theoretical Bound
* **Amdahl ceiling at `C = 1`:** Cutting decode attention from `129 µs` to `stage-0-vllm`'s `47.6 µs` saves `28 × 81 µs ≈ 2.3 ms/step` at `L = 2048` (`~10%` of Stage 2's `22.1 ms` TPOT), and less at `L = 512` (`~5%`). Adding a combine kernel adds `28` launches/step (`+1,764` across 63 decode steps), partially offsetting the gain while host launch overhead dominates.
* **Decode floor (`C = 1`):** `3.45 ms/tok` (Stage 2 sits at `22.4 ms`, `6.5×` above floor).

#### 3. Change
Added a two-stage Triton Split-K decode path (`q_len = 1`) on top of Stage 2 (`Qwen/Qwen2.5-Coder-1.5B-Instruct`, `bfloat16`): `_flash_attention_decode_splitk_fwd` (up to 32 parallel CTAs per head along `L_kv`, 1D reductions without `BLOCK_M = 16` padding) + `_flash_attention_splitk_combine` (log-sum-exp merge). Prefill keeps `_flash_attention_fwd`. Verified `32/32` greedy token parity; `failed = 0`.

#### 4. Measured Results

* **Concurrency (`L_in = 512`, `L_out = 128`, serialized):**

| C | S2 Tput | S3 Tput | Tput Δ | S2 TPOT (ms) | S3 TPOT (ms) | S2 P50 ITL (ms) | S3 P50 ITL (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 44.0 | 45.0 | +2.3% | 22.4 | 21.9 | 19.6 | 19.5 |
| **2** | 47.7 | 43.0 | −9.9% | 20.6 | 22.9 | 19.4 | 19.7 |
| **4** | 45.4 | 43.1 | −5.1% | 21.7 | 22.9 | 20.2 | 19.0 |
| **8** | 45.0 | 42.0 | −6.7% | 22.0 | 23.6 | 18.9 | 19.5 |
| **16** | 40.5 | 38.7 | −4.6% | 24.4 | 25.7 | 21.7 | 22.4 |

* **Context (`C = 1`, `L_out = 64`):**

| L | S2 Tput | S3 Tput | Tput Δ | S2 TPOT (ms) | S3 TPOT (ms) | S2 TTFT (ms) | S3 TTFT (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 45.7 | 45.4 | −0.7% | 21.5 | 21.7 | 46.8 | 44.8 |
| **256** | 47.1 | 49.1 | +4.3% | 20.7 | 19.9 | 53.8 | 52.1 |
| **512** | 43.5 | 46.7 | +7.3% | 22.2 | 20.7 | 71.2 | 67.1 |
| **1024** | 41.0 | 41.2 | +0.4% | 23.1 | 23.0 | 104 | 105 |
| **2048** | 40.2 | 42.0 | +4.4% | 22.1 | 21.2 | 196 | 191 |

#### 5. Gap Analysis & Next Move
* **Kernel occupancy restored (`ncu`, Layer 0, `L = 2048` prefill / `M = 1` decode):** `_flash_attention_decode_splitk_fwd` dropped `4.0×` (`129 → 32.1 µs`); with combine (`7.6 µs`), combined decode attention hit **`39.7 µs` (`3.26×` faster)**, beating `stage-0-vllm`'s `47.6 µs`. Active warps jumped `8.3% → 21.8%`, SM SOL `5.6% → 21.7%`, and DRAM bandwidth `4.0×` (`16.4 → 66.1 GB/s`, `7.4%` of 896 GB/s / `14.6%` DRAM SOL), `0` spills. Unchanged Layer 0 kernels matched Stage 2 (`Prefill GEMM`: `2,168 µs`; `Prefill Attn`: `971 µs`; `Decode GEMV`: `94.0 µs`; `Decode KV Write`: `24.6 µs`).
* **End-to-end bounded by Amdahl & noise:** All end-to-end deltas (`−9.9%` to `+7.3%`) fall inside the ~±10% noise floor (`C = 1` TPOT `21.9 ms`, still `6.4×` above floor). In `nsys`, aggregate attention GPU time fell **`6.5×` (`111 → 17.0 ms`, `2.2%` of GPU time)**, but total launches rose `72,456 → 74,220`, leaving **`59.2%` (`1,107 ms`) GPU idle** and `703 ms` Launch API time.
* **Forced next move:** With attention down to `2.2%` of GPU time, two system limits dominate: `asyncio.Lock` caps throughput flat at `~40 tok/s` (queueing `C = 16` TTFT to `47.2 s`) while `torch.cat` copies the full KV history every step (`24.6 µs`) → **Stage 4: Paged KV Cache & Continuous Batching (`stage-4-custom`)**.
* **Sources:** `benchmarks/results/stage-3-custom/{pareto_c*,context_l*,nsys_metrics,ncu_metrics}.json`.

---

### Stage 4: Paged KV Cache & Continuous Batching (`stage-4-custom`)
*Directory: `custom-serving/stage-4-paged-kv/` | Data: `benchmarks/results/stage-4-custom/`*

#### 1. Bottleneck (from Prior Profiler)
In `stage-3-custom`, `asyncio.Lock` pins throughput flat (`38.7–45.0 tok/s` across `C = 1..16`) while `C = 16` TTFT reaches `47.2 s`. Contiguous `torch.cat` KV append copies the full history on every step (`24.6 µs` in Layer 0) and blocks practical `L = 4096..8192` sweeps.

#### 2. Theoretical Bound
* **Batched decode weight amortization:** Coalescing `B` active sequences into `M = B` decode steps amortizes the `3.09 GB` weight read across `B` tokens, shifting decode linear AI from `~1` toward `~B` (`~64 FLOP/B` at `B = 64`), removing the flat concurrency ceiling.
* **KV write complexity:** Replacing `O(S)` `torch.cat` with `O(1)` in-place block scatter bounds Layer 0 decode KV write near `stage-0-vllm`'s `9.0 µs`.
* **Decode floor (`C = 1`):** `3.45 ms/tok` (Stage 3 sits at `21.9 ms`, `6.4×` above floor; batching does not reduce single-request launch overhead).

#### 3. Change
Bundles **five** coordinated changes to move from serialized contiguous serving to paged continuous batching:
1. `BlockAllocator` (`KV_BLOCK_SIZE = 16`, `4,096` blocks) + non-contiguous physical KV pool.
2. Triton `_reshape_and_cache` kernel for `O(1)` in-place KV scatter.
3. Paged `block_table` gather in `_flash_attention_fwd` (with a bundled **`BLOCK_N = 64 → 32` retune**) and `_flash_attention_decode_splitk_fwd`.
4. Iteration-level continuous batching scheduler (`PagedEngine.run_scheduler_loop`, `max_batch_size = 64`) with equal-length prefill grouping and batched `.tolist()` sampling.
5. Lock-free FastAPI SSE server with block reclamation on disconnect.

Verified `32/32` greedy token parity + batched parity; `failed = 0` across all 14 sweeps.

#### 4. Measured Results

* **Concurrency (`L_in = 512`, `L_out = 128`):**

| C | S3 Tput | S4 Tput | `stage-0-vllm` Tput | S4 vs. S3 | S4 TPOT (ms) | S4 TTFT (ms) | S4 P99 ITL (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 45.0 | 38.5 | 87.8 | −14% | 25.7 | 59.1 | 45.5 |
| **2** | 43.0 | 72.1 | 169 | +68% (1.7×) | 27.0 | 117 | 43.8 |
| **4** | 43.1 | 164 | 354 | +280% (3.8×) | 23.4 | 162 | 44.3 |
| **8** | 42.0 | 291 | 562 | +593% (6.9×) | 25.2 | 324 | 49.4 |
| **16** | 38.7 | 526 | 831 | +1,260% (13.6×) | 26.6 | 518 | 46.8 |
| **32** | — | 902 | 967 | 93% of vLLM | 29.0 | 860 | 44.7 |
| **64** | — | 1,260 | 1,232 | 102% of vLLM | 38.9 | 1,553 | 52.3 |

* **Context (`C = 1`, `L_out = 64`):**

| L | S3 Tput | S4 Tput | `stage-0-vllm` Tput | S4 TPOT (ms) | S4 TTFT (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 45.4 | 46.6 | 89.0 | 21.1 | 43.3 |
| **256** | 49.1 | 46.3 | 87.2 | 21.2 | 50.6 |
| **512** | 46.7 | 48.1 | 89.4 | 20.1 | 61.6 |
| **1024** | 41.2 | 42.8 | 95.2 | 21.9 | 115 |
| **2048** | 42.0 | 37.7 | 75.7 | 23.7 | 207 |
| **4096** | — | 34.5 | 59.7 | 22.8 | 421 |
| **8192** | — | 29.0 | 51.0 | 22.2 | 812 |

#### 5. Gap Analysis & Next Move
* **Concurrency & KV scaling:** Throughput scaled `32.7×` from `C = 1` to `C = 64` (**`1,260 tok/s`**, matching `stage-0-vllm`'s `1,232 tok/s` with `5×` tighter P99 ITL: `52.3` vs. `259 ms`). `C = 16` TTFT dropped `91×` (`47.2 s → 518 ms`), and `L = 128..8192` TPOT stayed flat (`20.1–23.7 ms`).
* **Layer 0 (`ncu`, `L = 2048` prefill / `M = 1` decode):** `Decode KV Write` (`_reshape_and_cache`) dropped **`4.6×`** (`24.6 → 5.3 µs`, `0` spills); `Prefill KV Write` ran in `14.3 µs` (`147 GB/s`). `Prefill Attention` dropped `971 → 848 µs` (`16.0 TFLOP/s`, `55.5%` SM SOL) from the bundled `BLOCK_N = 64 → 32` retune. Unfused decode elementwise ops stayed launch-bound (`<2%` DRAM SOL): RMSNorm `7.6 µs`, RoPE `8.1 µs`, SwiGLU `6.4 µs`.
* **`C = 1` gap (`25.7 ms`, `7.4×` above floor) & `nsys` contradiction:** In `nsys`, `decode_step_avg_ms` dropped `−18%` (`24.1 → 19.7 ms`) and `cuda_sync_wait_ms` fell `9.2×` (`51.1 → 5.6 ms`), yet unprofiled Pareto `C = 1` TPOT *regressed* `+17%` (`21.9 → 25.7 ms`) while Context `L = 128..1024` TPOT stayed `20.1–21.9 ms` — pointing to `asyncio`/scheduler overhead outside the NVTX range plus single-run variance.
* **Forced next move (`nsys`):** Total launches remained **`72,933`** (`0` CUDA graphs, `670 ms` Launch API time, **`53.5%` / `858 ms` GPU idle**). Even without KV `torch.cat`, **`3,584` `CatArrayBatchedCopy` launches** (`2 × 28 × 64`) persisted from PyTorch `rotate_half` in RoPE, alongside unfused RMSNorm, residual add, and split `gate_proj`/`up_proj` + SwiGLU (`norm_act_other = 138 ms`, `18.5%` of GPU time) → **Stage 5: Operator Fusion (`stage-5-custom`)**.
* **Sources:** `benchmarks/results/stage-4-custom/{pareto_c*,context_l*,nsys_metrics,ncu_metrics}.json`.

---

### Stage 5: Operator Fusion (`stage-5-custom`)
*Directory: `custom-serving/stage-5-fused-ops/` | Data: `benchmarks/results/stage-5-custom/`*

#### 1. Bottleneck (from Prior Profiler)
In `stage-4-custom` (`C = 1`), `nsys` recorded `72,933` launches (`~1,140`/step), `670 ms` Launch API time, `53.5%` (`858 ms`) GPU idle, and `138 ms` in `norm_act_other` — driven by `3,584` `CatArrayBatchedCopy` calls from `rotate_half` plus unfused RMSNorm, residual add, and split `gate_proj`/`up_proj` + SwiGLU chains.

#### 2. Theoretical Bound
* **Decode floor (`C = 1`):** `3.45 ms/tok` (Stage 4 sits at `25.7 ms`, `7.4×` above floor; GPU active time in `nsys` is only `745 ms` across 64 steps, or `~11.6 ms/step` average active GPU time).
* **Upper bound from Stage 4's `nsys` idle + `norm_act_other`:** Eliminating elementwise/concat/split-MLP fragmentation attacks Stage 4's `858 ms` GPU idle gap and `138 ms` `norm_act_other` bucket, with the remaining eager launch floor bounded by `stage-0-vllm`'s fused kernel count (`~21k` kernels across 64 steps).

#### 3. Change
Bundles **four** operator upgrades in `custom-serving/stage-5-fused-ops/model.py` (scheduler unchanged):
1. Triton `_rms_norm_fwd` with residual-add fusion threaded across all 28 layers and final norm.
2. Triton in-place 4D-strided `_rope_embedding_fwd` (eliminating `rotate_half` and all `CatArrayBatchedCopy` calls).
3. Merged `gate_up_proj` linear layer (cutting per-layer linear launches from 7 to 6).
4. Triton 2D-tiled `_swiglu_fwd` (`SiLU(gate) * up` in registers).

Verified `32/32` greedy token parity + `B = 4` batched parity + `256/256` block reclamation; `failed = 0`.

#### 4. Measured Results

* **Concurrency (`L_in = 512`, `L_out = 128`):**

| C | S4 Tput | S5 Tput | `stage-0-vllm` Tput | S5 vs. S4 | S4 TPOT (ms) | S5 TPOT (ms) | S5 P99 ITL (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 38.5 | 68.4 | 87.8 | +78% (1.8×) | 25.7 | 14.2 | 28.7 |
| **2** | 72.1 | 133 | 169 | +85% (1.9×) | 27.0 | 14.3 | 23.1 |
| **4** | 164 | 233 | 354 | +43% (1.4×) | 23.4 | 15.8 | 30.1 |
| **8** | 291 | 469 | 562 | +61% (1.6×) | 25.2 | 14.6 | 22.1 |
| **16** | 526 | 748 | 831 | +42% (1.4×) | 26.6 | 17.6 | 26.7 |
| **32** | 902 | 1,117 | 967 | +24% (116% vLLM) | 29.0 | 22.1 | 32.3 |
| **64** | 1,260 | 1,486 | 1,232 | +18% (121% vLLM) | 38.9 | 31.7 | 39.6 |

* **Context (`C = 1`, `L_out = 64`):**

| L | S4 Tput | S5 Tput | `stage-0-vllm` Tput | S4 TPOT (ms) | S5 TPOT (ms) | S5 TTFT (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 46.6 | 75.3 | 89.0 | 21.1 | 12.9 | 39.7 |
| **256** | 46.3 | 90.8 | 87.2 | 21.2 | 10.5 | 46.2 |
| **512** | 48.1 | 78.1 | 89.4 | 20.1 | 11.8 | 72.5 |
| **1024** | 42.8 | 68.7 | 95.2 | 21.9 | 12.7 | 129 |
| **2048** | 37.7 | 65.9 | 75.7 | 23.7 | 12.2 | 203 |
| **4096** | 34.5 | 49.9 | 59.7 | 22.8 | 13.3 | 443 |
| **8192** | 29.0 | 39.1 | 51.0 | 22.2 | 13.1 | 812 |

#### 5. Gap Analysis & Next Move
* **Gap to floor:** `C = 1` TPOT dropped `45%` (`25.7 → 14.2 ms`, **`4.1×` above the `3.45 ms` floor**), `L = 128..8192` TPOT held at `10.5–13.3 ms`, and `C = 64` reached `1,486 tok/s` (`121%` of `stage-0-vllm`, P99 ITL `39.6` vs. `259 ms`).
* **Timeline (`nsys`, `1024-in / 64-out`, `C = 1`):** Total kernels dropped **`3.03×` (`72,933 → 24,101`, `0` `CatArrayBatchedCopy`)**, Launch API time fell `46%` (`670 → 365 ms`), GPU idle dropped `53.5% → 29.0%` (`858 → 312 ms`), `norm_act_other` fell `54%` (`138 → 64.1 ms`), and `decode_step_avg_ms` fell `36%` (`19.7 → 12.6 ms`).
* **Layer 0 (`ncu`, `L = 2048` prefill / `M = 1` decode, `0` spills):**
  * Fused prefill: `_rms_norm_fwd` `52.1 µs` (`242 GB/s`, `53.4%` DRAM SOL), `_rope_embedding_fwd` `63.3 µs` (`133 GB/s`), `_swiglu_fwd` `243 µs` (`368 GB/s`, `41.1%` of 896 GB/s / `81.3%` DRAM SOL).
  * Fused decode: `_rms_norm_fwd` `6.9 µs`, `_rope_embedding_fwd` `5.7 µs`, `_swiglu_fwd` `5.0 µs` (`<2%` DRAM SOL, latency-bound).
  * Merged `gate_up_proj` is now the longest Layer 0 linear op (`Prefill GEMM`: `4,183 µs`, `27.0 TFLOP/s`, `94.5%` SM SOL; `Decode GEMV`: `175 µs`, `325 GB/s`, `36.3%` of 896 GB/s / `71.7%` DRAM SOL — matching `stage-0-vllm`'s `178 µs` / `321 GB/s`).
* **Forced next move (`nsys`):** With kernel count near `stage-0-vllm` (`24.1k` vs. `21.0k`) and Layer 0 kernels matching vLLM, `24,100` eager host launches (`~376`/decode step, `cuda_graph_launches = 0`) still leave **`29.0%` (`312 ms`) GPU idle** → **Stage 6: Bucketed CUDA Graphs for Decode (`stage-6-custom`)**.
* **Sources:** `benchmarks/results/stage-5-custom/{pareto_c*,context_l*,nsys_metrics,ncu_metrics}.json`.

---

### Stage 6: Bucketed CUDA Graphs for Decode (`stage-6-custom`)
*Directory: `custom-serving/stage-6-cuda-graphs/` | Data: `benchmarks/results/stage-6-custom/`*

#### 1. Bottleneck (from Prior Profiler)
In `stage-5-custom` (`C = 1`), `nsys` showed `24,100` individual host launch calls (`~376`/decode step, `cuda_graph_launches = 0`), leaving **`29.0%` (`312 ms`) GPU idle** and keeping `C = 1` TPOT at `14.2 ms` vs. `10.9 ms` in `stage-0-vllm` (`63` CUDA graphs).

#### 2. Theoretical Bound
* **Host-gap ceiling (`C = 1`):** Eliminating all `312 ms` of Stage 5's GPU idle across `63` decode steps frees at most `~4.95 ms/step`, putting a dispatch-only floor of `~7.6 ms/step` on `nsys` decode (`12.6 − 4.95 ms`) and `~9.2 ms` on Pareto TPOT (`14.2 − 4.95 ms`).
* **Physical weight-streaming floor (`C = 1`):** `3.45 ms/tok`.
* **Kernel invariance:** CUDA Graphs replay the same kernels without inter-kernel host gaps, so standalone Layer 0 `ncu` kernel times should stay unchanged.

#### 3. Change
Added `DecodeGraphRunner` (`custom-serving/stage-6-cuda-graphs/model.py`, `engine.py`): captures fixed-shape decode (`q_len = 1`) into static `torch.cuda.CUDAGraph` batch buckets `B ∈ {1, 2, 4, 8, 16, 32, 64}` with pre-allocated device buffers and padded-lane safety (`context_lengths[pad] == 0` clamps RoPE indices, skips KV writes in `_reshape_and_cache`, and emits finite zeros in Split-K combine). Prefill remains eager. Verified `32/32` greedy token parity + exact (`B = 4`) and padded (`B = 3 → 4`) parity + `256/256` block reclamation; `failed = 0`.

#### 4. Measured Results

* **Concurrency (`L_in = 512`, `L_out = 128`):**

| C | S5 Tput | S6 Tput | `stage-0-vllm` Tput | S6 vs. S5 | S6 / vLLM | S6 TPOT (ms) | S6 P99 ITL (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 68.4 | 113 | 87.8 | +66% | 129% | 8.4 | 11.9 |
| **2** | 133 | 231 | 169 | +73% | 136% | 8.0 | 9.3 |
| **4** | 233 | 403 | 354 | +73% | 114% | 8.9 | 11.1 |
| **8** | 469 | 742 | 562 | +58% | 132% | 8.9 | 10.9 |
| **16** | 748 | 1,066 | 831 | +42% | 128% | 12.0 | 14.4 |
| **32** | 1,117 | 1,410 | 967 | +26% | 146% | 16.8 | 18.7 |
| **64** | 1,486 | 1,771 | 1,232 | +19% | 144% | 25.9 | 27.6 |

* **Context (`C = 1`, `L_out = 64`):**

| L | S5 Tput | S6 Tput | `stage-0-vllm` Tput | S6 / vLLM | S6 TPOT (ms) | S6 TTFT (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **128** | 75.3 | 125 | 89.0 | 141% | 7.5 | 37.8 |
| **256** | 90.8 | 111 | 87.2 | 127% | 8.4 | 49.4 |
| **512** | 78.1 | 114 | 89.4 | 128% | 7.8 | 66.2 |
| **1024** | 68.7 | 97.5 | 95.2 | 102% | 8.6 | 114 |
| **2048** | 65.9 | 91.6 | 75.7 | 121% | 8.3 | 174 |
| **4096** | 49.9 | 74.3 | 59.7 | 125% | 8.4 | 331 |
| **8192** | 39.1 | 55.5 | 51.0 | 109% | 8.3 | 629 |

#### 5. Gap Analysis & Next Move
* **Measured vs. theoretical bounds:** `C = 1` TPOT dropped `41%` (`14.2 → 8.4 ms`, **`2.4×` above the `3.45 ms` floor**), context TPOT held flat at `7.5–8.6 ms` across `L = 128..8192` (matching the `28 KiB/tok` KV bound), and `C = 64` reached `1,771 tok/s` (`+19%` over Stage 5; `144%` of `stage-0-vllm`, with `9.4×` lower P99 ITL: `27.6` vs. `259 ms`).
* **Comparison caveat vs. `stage-0-vllm`:** `stage-6-custom` matches or exceeds `stage-0-vllm` across all measured points (`+2%` to `+46%`, single runs), with `L = 1024` (`102%`) and `L = 8192` (`109%`) inside the ~±10% noise floor. The single-process custom server avoids vLLM's multi-process API-server/detokenizer overhead; `stage-2-vllm` (`2,874 tok/s` at `C = 64`) is higher only due to `h ≈ 1` prompt caching.
* **Timeline (`nsys`, `1024-in / 64-out`, `C = 1`):**
  * `cuda_graph_launches` rose `0 → 63`, `cpu_launch_calls` fell **`50.6×` (`24,100 → 476`, `−98.0%`)**, and GPU idle dropped **`3.21×` by share (`29.0% → 9.0%`) / `3.89×` by time (`312 → 80.3 ms`)**. `decode_step_avg_ms` fell `31%` (`12.6 → 8.6 ms`, within `1.0 ms` of the `7.6 ms` zero-idle bound).
  * **Artifact note (`cpu_launch_overhead_ms`):** Launch API time only dropped `365 → 285 ms` (`−22%`) across `476` calls (`~600 µs/call`). Since eager launches cost `~15 µs` (Stage 5: `365 ms / 24,100`), the `~413` prefill launches explain only `~6 ms`; the remaining **`~280 ms` is host-thread blocking inside the 63 `cudaGraphLaunch` calls** (`~4.4 ms`/call).
  * **Variance note:** Unchanged eager `prefill_duration_ms` shifted `238 → 317 ms` (`+33%`) between Stage 5 and Stage 6 `nsys` captures.
* **Layer 0 (`ncu`, `0` spills):** Unchanged within ~2% of Stage 5 (`Prefill GEMM`: `4,146 µs`, `27.2 TFLOP/s`, `95.2%` SM SOL; `Prefill Attn`: `853 µs`, `1,608 FLOP/B`; `Decode GEMV`: `177 µs`, `321 GB/s`, `35.9%` of 896 GB/s / `70.9%` DRAM SOL; `Decode Attn Split-K`: `34.0 + 7.7 µs`).
* **Where the remaining `2.4×` gap (`8.4` vs. `3.45 ms`) lives:** `nsys` shows `91.0%` GPU active time (`808 ms` active vs. `80.3 ms` idle), with `86.3%` of GPU time in linear GEMVs (`321 GB/s` base-clock / `~367 GB/s` effective end-to-end vs. `896 GB/s` bus peak). Closing the remaining gap requires moving fewer bytes per token (**INT4/FP8/AWQ weight quantization**), fusing `qkv_proj`, or tuning GEMV kernels for laptop-TGP bandwidth.
* **Sources:** `benchmarks/results/stage-6-custom/{pareto_c*,context_l*,nsys_metrics,ncu_metrics}.json`.
