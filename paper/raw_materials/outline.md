# Paper Outline (`outline.md`)

## Metadata & Target Format
- **Title:** *From Naive PyTorch to Surpassing vLLM: A Profile-Guided Empirical Study of LLM Inference Optimization on Consumer Blackwell GPUs*
- **Author:** `shin-x-nihs`
- **Format:** 2-column compact systems paper (ACM/IEEE-style compact layout), **5 pages** (target 4–6 pages excluding references).
- **Framing:** Honest empirical/educational systems study—not claiming to replace production vLLM (which supports multi-GPU tensor parallelism, quantization, chunked prefill, LoRA, etc.), but demonstrating how profile-guided bottleneck elimination from first principles on a single consumer Blackwell GPU (`RTX 5080 Laptop`, `sm_120`) systematically transitions across six distinct hardware/software bottlenecks to surpass baseline vLLM (`stage-0-vllm`) across all concurrency and context-length regimes.

---

## Section-by-Section Structure & Page Budget (Total: 5.0 Pages excl. References)

### Title, Abstract & Section 1: Introduction (~0.85 pages; Page 1)
- **Abstract (~150 words):** Problem (LLM serving performance is governed by shifting bottlenecks across memory, compute, occupancy, scheduling, and host dispatch), methodology (dual-track empirical study on `Qwen2.5-Coder-1.5B-Instruct` bf16 on RTX 5080 Blackwell `sm_120` using `nsys` + `ncu`), headline quantitative results (Stages 0→6 achieve `7.60×` $C=1$ throughput, `72.93×` $C=16$ throughput, `31.95×` $L=2048$ TPOT reduction, and surpass `stage-0-vllm` at every concurrency [`113.42` vs `87.77 tok/s` at $C=1$; `1,770.61` vs `1,232.26 tok/s` at $C=64$] and context length $L \in [128, 8192]$), plus serving-level trade-offs (`stage-1-vllm` speculative decoding vs `stage-2-vllm` prefix caching).
- **1. Introduction:**
  - Motivation: Modern LLM inference engines bundle many optimizations simultaneously, obscuring individual causal effects and bottleneck handoffs—especially on bandwidth- and thermally-constrained consumer Blackwell laptop GPUs.
  - Dual-track study design: `custom-serving/` (strictly additive Stages 0→6) paired with `vllm-reference/` (feature-isolated Stages 0–2).
  - Summary of key contributions (3 crisp bullet points): (1) Reproducible 7-stage additive PyTorch/Triton/CUDA-Graph stack with 100% greedy parity; (2) Full-stack `nsys` + `ncu` roofline bottleneck attribution showing how each optimization exposes the next; (3) Empirical boundary analysis of serving-level features (speculative decoding vs. prefix caching).

### Section 2: Experimental Methodology & System Architecture (~0.75 pages; Page 1–2)
- **2.1 Hardware, Model & Workload Protocol:** RTX 5080 Laptop (`sm_120`, 16 GB GDDR7, `896 GB/s` / `40 TFLOP/s` ceilings), CUDA 13.1, PyTorch 2.13, Triton 3.7.1, vLLM 0.31.0; `Qwen2.5-Coder-1.5B-Instruct` bf16 (+ 0.5B draft). Closed-loop Pareto (`512/128`, $C \in [1..64]$) and Context (`L \in [128..8192]`, `64` out, $C=1$) sweeps via `vllm bench serve`; `nsys` (`1024/64`, $C=1$) and standalone Layer 0 `ncu` (`L=2048` prefill + $M=1$ decode).
- **2.2 Correctness & Parity Discipline:** `test_parity.py` (`32/32` greedy token ID parity against HF `eager`/`sdpa`, batched & padded-lane parity, zero block leaks, `failed = 0`).
- **2.3 Seven-Stage Additive Architecture Overview:**
  - **Visual (TikZ Figure 1, ~0.25 page, spanning 2 columns):** *7-Stage Additive Progression & Bottleneck Transition Diagram* (`Stage 0 Naive` → `Stage 1 KV Cache` → `Stage 2 Triton FA-2` → `Stage 3 Split-K` → `Stage 4 Paged KV + Continuous Batching` → `Stage 5 Fused Ops` → `Stage 6 Bucketed CUDA Graphs`), annotating the exact bottleneck eliminated and the next bottleneck exposed at each arrow.

### Section 3: Profile-Guided Custom Engine Progression (Stages 0 → 6) (~1.90 pages; Pages 2–4)
Structured around the three macro-regimes of bottleneck elimination, backed by exact `nsys`, `ncu`, and benchmark numbers:
- **3.1 Algorithmic & Memory-Hierarchy Bottlenecks (Stages 0 → 2):**
  - *Stage 0→1 (KV Cache):* Eliminating $O(S^2)$ decode recomputation (`110.38 → 21.54 ms/step`, `11.02×` $L=2048$ TPOT drop), exposing `54.34%` GPU idle and `12.0×` prefill attention AI collapse (`135.42` vs `1,631.02 FLOP/B`).
  - *Stage 1→2 (Triton FlashAttention-2):* SRAM online fp32 softmax + native GQA indexing boosts prefill attention AI `11.8×` (`1,597.84 FLOP/B`), cuts DRAM traffic `-89.1%` (`78.98 → 8.62 GB/s`), cuts $L=2048$ TTFT `-40.6%` (`330.14 → 196.17 ms`, matching vLLM), and slashes $L=8192$ VRAM to `3.67 GB`—while exposing `grid=(1, 12, 1)` decode SM starvation (`129.44 µs`, `8.33%` active warps).
- **3.2 Occupancy, Paged Memory & Continuous Batching (Stages 3 → 4):**
  - *Stage 2→3 (Triton Split-K Decode):* Partitioning decode KV across parallel CTAs cuts Layer 0 decode attention `3.26×` (`129.44 → 39.68 µs`) and total `nsys` attention time `6.54×` (`111.20 → 17.00 ms`), exposing `59.19%` GPU idle (`703.33 ms` host launch) and `asyncio.Lock` serialization.
  - *Stage 3→4 (Paged KV + Continuous Batching):* `KV_BLOCK_SIZE=16` pool + Triton `_reshape_and_cache` (`4.61×` faster decode KV write, `24.64 → 5.34 µs`) + iteration-level $M=B$ batching scales throughput `32.7×` to `1,260.39 tok/s` at $C=64$ (`102.3%` of `stage-0-vllm`, `13.60×` over Stage 3 at $C=16$, `91×` lower $C=16$ TTFT), while isolating `72,933` eager launches (`3,584` `CatArrayBatchedCopy` RoPE calls).
- **3.3 Host-Dispatch Elimination: Fusion & Bucketed CUDA Graphs (Stages 5 → 6):**
  - *Stage 4→5 (Triton Operator Fusion):* `_rms_norm_fwd` (with residual threading), in-place 4D-strided `_rope_embedding_fwd`, and `gate_up_proj`+`_swiglu_fwd` cut launches `3.03×` (`72,933 → 24,101`), CPU launch overhead `-45.57%` (`669.98 → 364.68 ms`), and GPU idle `53.53% → 28.97%`, lifting $C=1$ `+77.7%` (`68.44 tok/s`) and $C=64$ to `1,486.13 tok/s` (`120.6%` of vLLM).
  - *Stage 5→6 (Bucketed CUDA Graphs):* Capture-safe padded-lane decode buckets $B \in \{1..64\}$ raise `cuda_graph_launches` `0 → 63`, slash `cpu_launch_calls` `50.6×` (`24,100 → 476`), drop GPU idle `3.21×` (`28.97% → 9.03%`), and cut `decode_step_avg_ms` to `8.64 ms` (`19.0%` faster than `stage-0-vllm`'s `10.67 ms`).
- **Figures & Tables in Section 3:**
  - **Table 1 (~0.30 page, 2-column):** *Master Cross-Stage Summary* (all 10 stages: `C=1/16/64` throughput & TPOT, `L=2048/8192` TTFT & TPOT, `nsys` launches/idle/decode-step).
  - **Figure 2 (`fig1_pareto.pdf`, ~0.30 page):** *Throughput vs. Latency (TPOT & P99 ITL) across Concurrency $C=1..64$*, highlighting Stage 0→6 progression surpassing `stage-0-vllm` and Stage 2 vLLM prefix-cache upper bound.
  - **Figure 3 (`fig2_context_scaling.pdf`, ~0.28 page):** *Context-Length Scaling ($L=128..8192$)*—dual-panel TTFT (showing FlashAttention & Prefix Caching) and TPOT (showing Stage 0 blowup vs. flat `7.50–8.60 ms` in Stage 6).
  - **Figure 4 (`fig3_nsys_bottleneck.pdf`, ~0.28 page):** *Nsight Systems Wall-Time & Host-Dispatch Decomposition (`C=1, 1024/64`)*—stacked GPU active (Linear/Attn/Other) vs. GPU idle + overlaid `cpu_launch_calls` across Stages 0→6 vs. `stage-0-vllm`.
  - **Figure 5 (`fig4_roofline.pdf`, ~0.28 page):** *RTX 5080 Hardware Roofline (Layer 0 `ncu`)*—plotting Prefill/Decode Attention, Linear GEMM/GEMV, and Speculative GEMM (`1.0 → 15.5 FLOP/B`) against `896 GB/s` and `40 TFLOP/s` ceilings.

### Section 4: Serving-Level Optimizations & Trade-offs on vLLM (~0.75 pages; Page 4)
- **4.1 Speculative Decoding Limits (`stage-1-vllm`):** Quantifies why a `15.5×` kernel AI jump (`1.001 → 15.473 FLOP/B`, `98.2%` acceptance) yields only `+20.7%` at $C=1$ and degrades `-20%` to `-29%` at $C \ge 4$: draft-to-target ratio (`0.49B/1.54B`), `9.58×` host launch surge (`28.9 → 277.0 ms`, `35.3%` GPU idle), and bursty SSE delivery (`P50 ITL` `10.91 → 42.86 ms`).
- **4.2 Automatic Prefix Caching Gains (`stage-2-vllm`):** Quantifies RadixAttention eliminating prefill compute serialization at $C=64$ (`+133.3%` throughput to `2,874.36 tok/s`, `-74.5%` mean TTFT, `-82.8%` P99 ITL) and $L=8192$ (`-70.6%` TTFT) with zero host launch penalty (`29.96 ms`, `63` CUDA graphs), alongside minor low-concurrency bookkeeping overhead (`-8.5%` at $C=1$).
- **Table 2 (~0.18 page, 1-column):** *Serving-Level Feature Trade-offs vs. `stage-0-vllm`* (`C=1`, `C=64`, `L=8192`, `nsys` launch overhead & GPU idle).

### Section 5: Discussion, Why Stage 6 Surpasses Baseline vLLM & Limitations (~0.50 pages; Page 5)
- **Why `stage-6-custom` beats `stage-0-vllm`:** Lower per-step Python/scheduler overhead on a specialized single-model engine (`8.64 ms` vs `10.67 ms` decode step; `9.03%` vs `11.61%` GPU idle), tighter P99 ITL (`27.57` vs `258.96 ms` at $C=64$) because equal-length prefill batching avoids vLLM's heavier chunked/mixed-batch scheduling jitter on this uniform benchmark, while matching vLLM's fused Layer 0 kernel roofline (`27.21` vs `27.35 TFLOP/s` prefill GEMM; `321.39` vs `321.10 GB/s` decode GEMV).
- **Honest Limitations & Threats to Validity:** Single-model/single-GPU scope (no TP/PP, no quantization, no chunked prefill or prefix caching in custom track yet), fixed homogeneous prompt workload (favorable to prefix caching and equal-length prefill grouping), WSL2 + consumer laptop thermal/clock dynamics, and estimated tensor FLOPs in NCU roofline.

### Section 6: Conclusion & Future Work (~0.25 pages; Page 5)
- Synthesizes the central lesson: **profile-guided, single-variable iteration** reveals non-obvious bottleneck inversions (e.g., Stage 1 KV cache flipping GPU idle from `8.9%` to `54.3%`; Stage 2 FA-2 regressing decode attention `4×` until Stage 3 Split-K).
- Future directions: Prefix caching in custom track, `14B-AWQ` + `0.5B` high-ratio speculative decoding, FP8/INT4 quantized kernels on Blackwell `sm_120`.

### References (Page 6, excluded from 5-page body budget)
- Core systems citations: FlashAttention-2 (Dao 2023), FlashDecoding (Dao et al. 2023), vLLM / PagedAttention (Kwon et al. 2023), Orca Continuous Batching (Yu et al. 2022), SGLang / RadixAttention (Zheng et al. 2024), Speculative Decoding (Leviathan et al. 2023; Chen et al. 2023), Roofline Model (Williams et al. 2009), Qwen2.5-Coder (Hui et al. 2024), Triton (Tillet et al. 2019).

---

## Asset Mapping Checklist
| Asset File | Target Section | Type | Data Sources (`benchmarks/results/`) |
| :--- | :--- | :--- | :--- |
| TikZ Diagram (inline) | Sec. 2.3 | Architecture / Bottleneck Flow | `custom-serving/README.md`, `REPORTS.md` |
| `fig1_pareto.pdf` | Sec. 3 / 4 | Concurrency Pareto (Tput vs TPOT & P99 ITL) | `*/pareto_c*.json` |
| `fig2_context_scaling.pdf` | Sec. 3 / 4 | Dual-Panel Context Scaling (TTFT & TPOT) | `*/context_l*.json` |
| `fig3_nsys_bottleneck.pdf` | Sec. 3.3 | Timeline Active/Idle Breakdown + Launch Calls | `*/nsys_metrics.json` |
| `fig4_roofline.pdf` | Sec. 3.1 / 4.1 | RTX 5080 Hardware Roofline (Layer 0) | `*/ncu_metrics.json`, `config.yaml` |
| Table 1 (inline LaTeX) | Sec. 3 | Master 10-Stage Metrics Summary | `experimental_log.md` (Tables 1A–1C) |
| Table 2 (inline LaTeX) | Sec. 4 | vLLM Feature Isolation Trade-offs | `experimental_log.md` (Tables 1A–1D) |
