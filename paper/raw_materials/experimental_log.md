# Verified Experimental Log (`experimental_log.md`)

All numerical values below were extracted directly from `benchmarks/results/<stage>/*.json` across all 10 stages (`failed = 0` for every sweep JSON).

---

## 1. Master Cross-Stage Metrics Tables (Verified from JSON)

### Table 1A: Concurrency Sweep (`pareto_c*.json`, Input = 512, Output = 128, Greedy)

| Stage | C=1 Tput (tok/s) | C=1 TPOT (ms) | C=1 P50/P99 ITL (ms) | C=16 Tput (tok/s) | C=16 TPOT (ms) | C=16 P50/P99 ITL (ms) | C=64 Tput (tok/s) | C=64 TPOT (ms) | C=64 P50/P99 ITL (ms) | C=64 TTFT / P99 TTFT (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stage-0-vllm` | 87.77 | 10.90 | 10.91 / 14.52 | 831.19 | 16.08 | 13.17 / 142.73 | 1,232.26 | 45.15 | 20.23 / 258.96 | 896.18 / 3,889.86 |
| `stage-1-vllm` | 105.97 | 8.38 | 42.86 / 69.09 | 622.88 | 20.43 | 72.63 / 336.04 | 872.47 | 59.06 | 371.32 / 392.65 | 1,770.88 / 6,809.41 |
| `stage-2-vllm` | 80.35 | 12.24 | 13.00 / 15.19 | 972.12 | 15.81 | 15.72 / 20.27 | 2,874.36 | 20.57 | 19.63 / 44.42 | 228.52 / 1,211.93 |
| `stage-0-custom` | 14.93 | 66.84 | 58.41 / 109.40 | 14.61 | 68.44 | 58.63 / 109.48 | — | — | — | — |
| `stage-1-custom` | 37.33 | 26.42 | 23.83 / 57.11 | 37.66 | 26.27 | 22.32 / 69.00 | — | — | — | — |
| `stage-2-custom` | 43.97 | 22.44 | 19.57 / 49.11 | 40.52 | 24.44 | 21.71 / 54.62 | — | — | — | — |
| `stage-3-custom` | 44.98 | 21.93 | 19.46 / 52.82 | 38.67 | 25.65 | 22.39 / 63.70 | — | — | — | — |
| `stage-4-custom` | 38.51 | 25.70 | 25.03 / 45.47 | 525.85 | 26.58 | 23.29 / 46.84 | 1,260.39 | 38.92 | 27.22 / 52.30 | 1,553.41 / 3,949.23 |
| `stage-5-custom` | 68.44 | 14.19 | 12.53 / 28.68 | 748.22 | 17.56 | 15.12 / 26.67 | 1,486.13 | 31.71 | 19.74 / 39.64 | 1,480.26 / 3,304.07 |
| `stage-6-custom` | **113.42** | **8.43** | **8.01 / 11.88** | **1,065.51** | **11.99** | **10.38 / 14.35** | **1,770.61** | **25.91** | **16.78 / 27.57** | 1,332.06 / 3,308.24 |

*Note on serialized stages (`stage-0-custom`..`stage-3-custom`):* Mean TTFT at $C=16$ due to `asyncio.Lock` queueing was `124,830.53 ms` (S0), `48,607.08 ms` (S1), `45,163.05 ms` (S2), and `47,151.62 ms` (S3), dropping to `517.92 ms` (S4), `505.97 ms` (S5), and `398.55 ms` (S6) once continuous batching was introduced. Intermediate concurrency points (`C=2, 4, 8, 32`) are preserved in the raw JSONs; notably at `C=32`, `stage-6-custom` reaches `1,410.49 tok/s` (`145.9%` of `stage-0-vllm`'s `966.96 tok/s`, P99 ITL `18.73 ms` vs `239.19 ms`).

---

### Table 1B: Context-Length Sweep (`context_l*.json`, $C = 1$, Output = 64, $N = 10$ prompts)

| Stage | L=128 TTFT (ms) | L=128 TPOT (ms) | L=128 Tput (tok/s) | L=2048 TTFT (ms) | L=2048 TPOT (ms) | L=2048 Tput (tok/s) | L=8192 TTFT (ms) | L=8192 TPOT (ms) | L=8192 Tput (tok/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stage-0-vllm` | 46.34 | 10.67 | 89.04 | 208.14 | 10.11 | 75.68 | 673.66 | 9.24 | 50.95 |
| `stage-1-vllm` | 125.28 | 12.12 | 71.99 | 411.56 | 10.58 | 59.34 | 1,752.66 | 13.43 | 24.62 |
| `stage-2-vllm` | 37.51 | 10.51 | 91.43 | 114.36 | 13.60 | 65.88 | **198.30** | 13.96 | **59.36** |
| `stage-0-custom` | 59.95 | 34.67 | 28.51 | 275.87 | 265.51 | 3.76 | — | — | — |
| `stage-1-custom` | 51.19 | 22.56 | 43.45 | 330.14 | 24.09 | 34.62 | — | — | — |
| `stage-2-custom` | 46.76 | 21.50 | 45.65 | 196.17 | 22.12 | 40.23 | — | — | — |
| `stage-3-custom` | 44.78 | 21.68 | 45.35 | 190.76 | 21.15 | 42.00 | — | — | — |
| `stage-4-custom` | 43.25 | 21.12 | 46.58 | 207.26 | 23.65 | 37.70 | 811.73 | 22.20 | 28.95 |
| `stage-5-custom` | 39.70 | 12.85 | 75.31 | 202.66 | 12.20 | 65.85 | 811.84 | 13.10 | 39.09 |
| `stage-6-custom` | **37.78** | **7.50** | **125.31** | **174.38** | **8.31** | **91.62** | 629.30 | **8.32** | 55.46 |

---

### Table 1C: Nsight Systems Timeline Metrics (`nsys_metrics.json`, $C = 1$, Input = 1024, Output = 64)

| Stage | Wall Time (ms) | Total Kernels | CPU Launch Calls | CPU Launch Overhead (ms) | CUDA Graph Launches | GPU Active / Idle (%) | Prefill (ms) | Decode Total (ms) | Decode Step Avg (ms) | Linear / Attn / Other GPU (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `stage-0-vllm` | 867.33 | 21,041 | 1,636 | 28.90 | 63 | 88.39 / 11.61 | 195.05 | 672.28 | 10.67 | 704.88 / 5.10 / 56.66 |
| `stage-1-vllm` | 776.26 | 23,484 | 19,843 | 277.00 | 13 | 64.73 / 35.27 | 219.74 | 556.51 | 8.83 | 450.88 / 3.63 / 47.96 |
| `stage-2-vllm` | 861.43 | 21,041 | 1,636 | 29.96 | 63 | 86.91 / 13.09 | 313.85 | 547.58 | 8.69 | 664.56 / 4.36 / 79.71 |
| `stage-0-custom` | 7,119.30 | 85,248 | 85,248 | 938.43 | 0 | 91.13 / 8.87 | 118.11 | 6,954.07 | 110.38 | 4,181.92 / 1,731.54 / 574.33 |
| `stage-1-custom` | 1,674.04 | 85,058 | 85,058 | 647.04 | 0 | 45.66 / 54.34 | 288.33 | 1,357.22 | 21.54 | 478.64 / 185.04 / 100.63 |
| `stage-2-custom` | 1,977.39 | 72,456 | 72,455 | 708.83 | 0 | 60.76 / 39.24 | 353.35 | 1,586.66 | 25.19 | 858.09 / 111.20 / 232.12 |
| `stage-3-custom` | 1,870.56 | 74,220 | 74,220 | 703.33 | 0 | 40.81 / 59.19 | 327.93 | 1,518.62 | 24.11 | 606.66 / 17.00 / 139.78 |
| `stage-4-custom` | 1,603.12 | 72,933 | 72,932 | 669.98 | 0 | 46.47 / 53.53 | 326.94 | 1,240.13 | 19.68 | 587.93 / 19.33 / 137.74 |
| `stage-5-custom` | 1,077.53 | 24,101 | 24,100 | 364.68 | 0 | 71.03 / 28.97 | 237.98 | 792.51 | 12.58 | 666.14 / 35.15 / 64.05 |
| `stage-6-custom` | 888.50 | 24,165 | **476** | 285.25 | **63** | **90.97 / 9.03** | 317.09 | **544.10** | **8.64** | 697.22 / 28.18 / 82.86 |

*Phase measurement source:* `NVTX` for all custom stages; `GPU Kernel Timeline Split` for all vLLM stages.

---

### Table 1D: Nsight Compute Layer 0 Key Kernel Metrics (`ncu_metrics.json`, Prefill $L=2048$ & Decode $M=1$ [or $M=6$ for `stage-1-vllm`])

| Stage | Prefill Attn Dur (µs) / DRAM (GB/s) / AI (F/B) / TFLOP/s | Prefill GEMM Dur (µs) / DRAM (GB/s) / AI (F/B) / TFLOP/s | Decode Attn Main (+Combine) (µs) / DRAM (GB/s) / AI (F/B) | Decode Linear Dur (µs) / DRAM (GB/s) / AI (F/B) / TFLOP/s | Decode KV Write Dur (µs) / DRAM (GB/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `stage-0-vllm` | 732.58 / 11.55 / 1,631.02 / 18.84 | 4,125.15 / 28.21 / 969.48 / 27.35 | 26.02 (+21.63) / 82.87 / 66.67 | 177.70 / 321.10 / 1.001 / 0.321 | 8.96 / 2.54 |
| `stage-1-vllm` | 739.65 / 11.45 / 1,629.25 / 18.66 | 4,184.22 / 28.06 / 960.81 / 26.96 | 205.25 (—) / 10.69 / 392.96 | 174.91 / 325.56 / **15.473** / **5.037** | 7.97 / 3.47 |
| `stage-2-vllm` | 750.40 / 11.28 / 1,630.98 / 18.39 | 4,144.61 / 28.05 / 970.44 / 27.22 | 26.43 (+22.30) / 81.56 / 66.68 | 177.22 / 320.16 / 1.007 / 0.322 | 9.86 / 2.31 |
| `stage-0-custom` | *Deferred (nsys isolated $O(S^2)$ recomputation)* | — | — | — | — |
| `stage-1-custom` | 1,209.38* / 78.98 / 135.42 / 10.70 | 2,180.67 / 34.23 / 755.53 / 25.86 | 32.22 (—) / 196.19 / 1.105 | 94.72 / 291.10 / 1.015 / 0.295 | 24.83 / 42.46 |
| `stage-2-custom` | 978.05 / 8.62 / 1,597.84 / 13.78 | 2,199.55 / 33.85 / 757.27 / 25.64 | 129.44 (—) / 16.41 / 99.15 | 97.31 / 283.35 / 1.015 / 0.287 | 26.05 / 40.48 |
| `stage-3-custom` | 970.75 / 8.69 / 1,597.84 / 13.88 | 2,168.45 / 23.54 / 1,105.21 / 26.01 | 32.06 (+7.62) / 66.08 / 7.93 | 93.98 / 293.38 / 1.015 / 0.298 | 24.64 / 42.79 |
| `stage-4-custom` | 847.90 / 9.94 / 1,607.66 / 15.99 | 2,167.46 / 23.74 / 1,096.15 / 26.03 | 33.92 (+7.84) / 62.69 / 7.41 | 94.72 / 291.10 / 1.015 / 0.295 | **5.34** / 1.05 |
| `stage-5-custom` | 846.59 / 9.96 / 1,607.90 / 16.01 | 4,182.50† / 27.87 / 967.86 / 26.97 | 32.67 (+7.49) / 65.10 / 7.41 | 175.23† / 324.91 / 1.003 / 0.326 | 5.41 / 0.99 |
| `stage-6-custom` | 852.99 / 9.88 / 1,607.90 / 15.89 | 4,146.24† / 28.30 / 961.44 / 27.21 | 33.98 (+7.74) / 62.58 / 7.41 | 176.86† / 321.39 / 1.005 / 0.323 | 5.47 / 1.03 |

*\*Stage 1 Prefill Attn reports only the score BMM (`Q @ K^T`); unfused scaling (`453.98 µs`), causal mask (`662.91 µs`), fp32 cast (`787.58 µs`), and `softmax_warp_forward` (`1,012.48 µs` at `375.65 GB/s`) bring total unfused prefill attention to `>4,100 µs`.*
*†Stages 5–6 merge `gate_proj` and `up_proj` into a `2×`-wide `gate_up_proj` (matching vLLM's fused MLP width); NCU reports the single longest GEMM/GEMV in Layer 0, so duration doubles while per-layer linear launches drop from 7 to 6.*

---

## 2. Stage-by-Stage Hypotheses, Findings & Root-Cause Bottleneck Transitions (`REPORTS.md`)

### Reference Track (`vllm-reference/`)

1. **`stage-0-vllm` (Reference Baseline):**
   - **Profile:** At $C=1$ (`1024`-in/`64`-out), decode dominates wall time (`672.28 ms`, `77.5%`), linear layers consume `91.9%` (`704.88 ms`) of GPU time while attention is `0.7%` (`5.10 ms`). `Decode Linear GEMV` is strictly memory-bandwidth bound (`AI = 1.001 FLOP/B`, `321.10 GB/s`, `0.321 TFLOP/s`), whereas `Prefill Linear GEMM` is compute-bound (`AI = 969.48 FLOP/B`, `27.35 TFLOP/s`, `95.1%` SM SOL).
   - **High-Concurrency Bottleneck:** Scaling $C=1 \to 64$ raises throughput to `1,232.26 tok/s`, but redundant 512-token prefills cause mean TTFT to explode `12.2×` (`73.31 → 896.18 ms`), P99 TTFT to reach `3,889.86 ms`, and P99 ITL to spike to `258.96 ms` due to prefill interruptions of decode batches.

2. **`stage-1-vllm` (Speculative Decoding: 0.5B Draft, $K=5$):**
   - **Hypothesis:** Verify $K+1=6$ drafted tokens in parallel to shift target decode linear from GEMV ($M=1$) to GEMM ($M=6$), ideally yielding up to $S_{\text{ideal}} \approx 2.25\times$ speedup.
   - **Findings:** Kernel-level hypothesis succeeded—`Decode Linear` AI jumped **`15.5×`** (`1.001 → 15.473 FLOP/B`) and attained compute jumped **`15.7×`** (`0.321 → 5.037 TFLOP/s`), with `~98.2%` acceptance (`L_acc ≈ 5.91`). However, end-to-end speedup capped at **`+20.7%`** at $C=1$ (`105.97` vs `87.77 tok/s`, TPOT `-23.1%` to `8.38 ms`) and `+7.5%` at $C=2$, then **regressed `-20.2%` to `-29.2%` at $C \ge 4$** (`872.47 tok/s` at $C=64$).
   - **Root Causes:** (a) Poor draft-to-target ratio (`0.49B / 1.54B ≈ 1/3`), making $K=5$ sequential memory-bound draft steps cost $\sim 1.67\times$ a target decode step; (b) host orchestration overhead surged **`9.58×`** (`28.90 → 277.00 ms`, `1,636 → 19,843` calls, `63 → 13` CUDA graphs), widening GPU idle from `11.61%` to `35.27%`; (c) bursty SSE flushing inflated P50 ITL `3.93×` (`10.91 → 42.86 ms` at $C=1$).

3. **`stage-2-vllm` (Automatic Prefix Caching / RadixAttention):**
   - **Hypothesis:** Reuse shared-prefix KV blocks in $O(1)$ via radix tree lookup to bypass compute-bound prefill GEMMs and collapse multi-tenant queuing ($E[W_q]$).
   - **Findings:** At $C=64$, aggregate throughput surged **`+133.3%` (`2.33×`)** (`1,232.26 → 2,874.36 tok/s`), mean TTFT fell `-74.5%` (`896.18 → 228.52 ms`), P99 TTFT fell `-68.8%` (`3,889.86 → 1,211.93 ms`), TPOT fell `-54.4%` (`45.15 → 20.57 ms`), and P99 ITL collapsed `-82.8%` (`258.96 → 44.42 ms`). At $L=8192$ ($C=1$), TTFT dropped `-70.6%` (`673.66 → 198.30 ms`), lifting throughput `+16.5%`.
   - **Trade-off & Bottleneck Shift:** Zero host dispatch penalty (`29.96 ms`, `1,636` calls, `63` CUDA graphs), though short/low-concurrency runs ($C=1..8$) paid a minor `-8.4%` to `-26.5%` throughput tax from radix bookkeeping. Overall system bottleneck shifted cleanly from **prefill compute serialization** to the **GDDR7 decode memory-bandwidth ceiling**.

---

### Custom Track (`custom-serving/`, Strictly Additive Stages 0 → 6)

4. **`stage-0-custom` (Naive PyTorch Baseline):**
   - **Findings:** Without KV caching, decode recomputes the entire sequence on every token (`6,954.07 ms` decode vs `118.11 ms` prefill = `97.7%` of `7,119.30 ms` wall time; `110.38 ms/step`). Doubling context $L=512 \to 1024 \to 2048$ more than doubles TPOT (`52.68 → 120.72 → 265.51 ms`, `3.76 tok/s`). Linear GEMMs (`64.5%`) + attention (`26.7%`) consume `91.2%` of GPU time, keeping GPU active `91.13%` on redundant work.
   - **Next Bottleneck:** $O(S^2)$ full-sequence decode recomputation.

5. **`stage-1-custom` (Naive KV Cache):**
   - **Hypothesis:** Memoize per-layer `(K, V)` tensors so decode projects only $M=1$ token ($O(S)$ per step).
   - **Findings:** Decode step time dropped **`5.12×`** (`110.38 → 21.54 ms`); $L=2048$ TPOT fell **`11.02×` (`-90.9%`)** (`265.51 → 24.09 ms`), lifting $L=2048$ throughput **`9.20×`** (`3.76 → 34.62 tok/s`) and flattening TPOT (`22.56–24.27 ms`) across $L=128..2048$.
   - **Bottleneck Transition:** GPU active time plummeted `8.49×` (`6,487.79 → 764.32 ms`), flipping GPU idle from `8.87%` to **`54.34%`** (`85,058` eager launches, `647.04 ms` CPU launch overhead). NCU exposed a **`12.0×` prefill attention AI collapse** (`135.42` vs vLLM `1,631.02 FLOP/B`, plus `1,012.48 µs` fp32 softmax at `375.65 GB/s`).

6. **`stage-2-custom` (Tiled Attention / Triton FlashAttention-2):**
   - **Hypothesis:** Tile attention in SRAM with online fp32 softmax and native GQA indexing to eliminate $O(S^2)$ HBM score materialization.
   - **Findings:** Prefill attention AI jumped **`11.8×`** (`135.42 → 1,597.84 FLOP/B`, `98.0%` of vLLM), DRAM bandwidth dropped **`-89.1%`** (`78.98 → 8.62 GB/s`), and Layer 0 prefill attention sped up **`>4.2×`** (`>4,100 µs → 978.05 µs`). $L=2048$ TTFT dropped **`-40.6%`** (`330.14 → 196.17 ms`, matching vLLM's `208.14 ms`), $L=8192$ prefill peak VRAM dropped from `~8.1 GB` to `3.667 GB`, and `nsys` kernel launches dropped by `12,602` (`85,058 → 72,456`).
   - **Bottleneck Transition:** Query-tiled FlashAttention at $q_{\text{len}}=1$ launched only `grid=(1, 12, 1)` = **12 CTAs**, starving SMs (`8.33%` active warps, `16.41 GB/s` DRAM, `129.44 µs` Layer 0 decode attention vs `32.22 µs` in Stage 1) and wasting `93.75%` of `BLOCK_M=16` rows on padding.

7. **`stage-3-custom` (Parallel Decode Attention / Triton Split-K):**
   - **Hypothesis:** Partition decode `L_kv` across parallel CTAs (`grid=(12, 15, 1)`) + log-sum-exp combine kernel to eliminate SM starvation.
   - **Findings:** Layer 0 decode attention dropped **`3.26×`** (`129.44 → 39.68 µs` combined; `32.06 µs` main pass = `4.04×`), active warps jumped `8.33% → 21.76%`, DRAM bandwidth jumped `4.03×` (`16.41 → 66.08 GB/s`), and total `nsys` attention time dropped **`6.54×`** (`111.20 → 17.00 ms`, `2.2%` of GPU time), lifting context-sweep throughput up to `+7.3%`.
   - **Bottleneck Transition:** End-to-end $C=1$ decode step barely budged (`25.19 → 24.11 ms`) and short-context serialized `pareto_c2..16` slightly regressed (`-4.6%` to `-9.9%`) because adding the combine kernel raised launches to `74,220` (`703.33 ms` CPU launch overhead, **`59.19%` GPU idle**), while `asyncio.Lock` pinned throughput flat at `~39–45 tok/s` and `torch.cat` KV writes cost `24.64 µs`.

8. **`stage-4-custom` (Paged KV Cache & Continuous Batching):**
   - **Hypothesis:** Replace `asyncio.Lock` and `torch.cat` KV reallocation with a `KV_BLOCK_SIZE=16` physical block pool, Triton `_reshape_and_cache`, and iteration-level $M=B$ continuous batching.
   - **Findings:** Aggregate throughput scaled **`32.7×`** from $C=1$ (`38.51 tok/s`) to **`1,260.39 tok/s` at $C=64$** (`102.3%` of `stage-0-vllm`, `5×` tighter P99 ITL `52.30` vs `258.96 ms`), jumping **`13.60×`** over Stage 3 at $C=16$ (`525.85` vs `38.67 tok/s`) and cutting $C=16$ TTFT **`91×`** (`47,151.62 → 517.92 ms`). Triton `_reshape_and_cache` cut Layer 0 decode KV write **`4.61×`** (`24.64 → 5.34 µs`), enabling full `L=128..8192` sweeps (`22.20 ms` TPOT at $L=8192$).
   - **Bottleneck Transition:** Low-concurrency decode remained host-dispatch bound (`19.68 ms/step`, `72,933` launches, `669.98 ms` CPU launch overhead, `53.53%` GPU idle), with `3,584` `CatArrayBatchedCopy` launches persisting from unfused `rotate_half` RoPE plus unfused RMSNorm/residual/SwiGLU chains.

9. **`stage-5-custom` (Operator Fusion):**
   - **Hypothesis:** Fuse RMSNorm+residual (`_rms_norm_fwd`), in-place 4D-strided RoPE (`_rope_embedding_fwd`), and `gate_up_proj` + `_swiglu_fwd` to slash kernel count and host launch latency.
   - **Findings:** Total CUDA kernels dropped **`3.03×` (`-66.95%`)** (`72,933 → 24,101`, `3,584 → 0` `CatArrayBatchedCopy`), `cpu_launch_overhead_ms` fell **`-45.57%`** (`669.98 → 364.68 ms`), `norm_act_other_ms` fell `-53.50%` (`137.74 → 64.05 ms`), and GPU idle dropped `53.53% → 28.97%`. `decode_step_avg_ms` fell `-36.08%` (`19.68 → 12.58 ms`), lifting $C=1$ throughput **`+77.7%`** (`38.51 → 68.44 tok/s`, TPOT `25.70 → 14.19 ms`) and $C=64$ to **`1,486.13 tok/s`** (`120.6%` of `stage-0-vllm`, `6.53×` tighter P99 ITL).
   - **Bottleneck Transition:** With `cuda_graph_launches = 0`, eager dispatch of the remaining `24,100` launches (`~376/step`) still burned `364.68 ms` (`28.97%` GPU idle), leaving a `1.91 ms/step` gap to `stage-0-vllm` (`10.67 ms`).

10. **`stage-6-custom` (Bucketed CUDA Graphs for Decode):**
    - **Hypothesis:** Capture fixed-shape decode (`q_len == 1`) into static `torch.cuda.CUDAGraph` buckets $B \in \{1, 2, 4, 8, 16, 32, 64\}$ with padded-lane masking to eliminate decode host dispatch.
    - **Findings:** `cuda_graph_launches` rose `0 → 63`, slashing `cpu_launch_calls` **`50.6×` (`-98.02%`)** (`24,100 → 476`), cutting `cpu_launch_overhead_ms` `-21.78%` (`364.68 → 285.25 ms`), collapsing GPU idle **`3.21×`** (`28.97% → 9.03%`, `90.97%` active—beating vLLM's `11.61%`), and cutting `decode_step_avg_ms` **`-31.32%`** (`12.58 → 8.64 ms`, `19.0%` faster than `stage-0-vllm`'s `10.67 ms`).
    - **Headline Result:** `stage-6-custom` surpasses `stage-0-vllm` at **every** Pareto concurrency (`C=1`: **`113.42 tok/s`**, `129.2%` of vLLM, `8.43 ms` TPOT; `C=64`: **`1,770.61 tok/s`**, `143.7%` of vLLM, `25.91 ms` TPOT, `9.39×` tighter P99 ITL `27.57` vs `258.96 ms`) and **every** context length $L \in [128, 8192]$ (`7.50–8.60 ms` TPOT, `102.3%–140.7%` of vLLM throughput). Across Stages 0→6, $C=1$ throughput improved **`7.60×`**, $C=16$ throughput improved **`72.93×`**, and $L=2048$ TPOT improved **`31.95×`**.
