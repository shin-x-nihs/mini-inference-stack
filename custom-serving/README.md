# Custom Serving Architecture Guide

From-scratch PyTorch/CUDA inference engines built **strictly additively**: each stage copies `stage-(N-1)` and implements one architectural upgrade on top of it.

All stages serve `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`) via an OpenAI-compatible `POST /v1/chat/completions` SSE endpoint on port `8000` (configured by root `config.yaml`), emit `prefill_step` / `decode_step` / `attn_compute` NVTX ranges for `benchmarks/profile_nsys.sh`, and support standalone Layer 0 kernel profiling via `benchmarks/profile_ncu.sh <stage-id>` (Stage 0 deferred NCU profiling since `nsys` timeline metrics already isolated full-sequence decode recomputation). Shared parity verification lives at `custom-serving/test_parity.py` (`--stage stage-0-baseline|stage-1-kv-cache|stage-2-tiled-attn|stage-3-split-k|all`), loading each stage's `model.py` and `engine.py` in isolation to verify single-step prefill logits and 100% greedy token ID parity against HuggingFace `AutoModelForCausalLM` (`eager` for Stages 0–1; `sdpa` for Stage 2+ fp32-accumulating kernels).

---

## Stage 0: Naive PyTorch Baseline (`stage-0-baseline/`)

Establishes an unoptimized lower bound with no KV cache, no `F.scaled_dot_product_attention`, and no `transformers.generation` wrappers:

- **`model.py`**: Loads dimensions dynamically from `config.json` and weights from local `.safetensors` shards (`RMSNorm`, float32-cached `RotaryEmbedding` with `rope_theta=1000000.0`, `SwiGLUMLP`, tied `lm_head`). In `NaiveAttention`, expands `2` KV heads to `12` Q heads via `repeat_interleave`, computes `(Q @ K^T) * scale` with an upper-triangular `-inf` causal mask and float32 softmax without a KV cache, and supports `last_token_only=True` in `Qwen2Model.forward` to project only the final hidden state through `lm_head`.
- **`engine.py`**: Runs a naive autoregressive loop (`NaiveEngine`) that concatenates each sampled token and passes the entire `[1, L + t]` sequence through `Qwen2Model` on every step, forcing $O(S^2)$ attention recomputation across prefill and decode. Each forward + sampling + `.item()` sync is wrapped in `prefill_step` or `decode_step` NVTX ranges.
- **`server.py`**: Exposes `/health`, `/v1/models`, and `/v1/chat/completions` (streaming SSE and non-streaming) in FastAPI. Serializes concurrent requests with a global `asyncio.Lock` and pins all CUDA/NVTX execution to a `ThreadPoolExecutor(max_workers=1)` for strict CUDA thread affinity.
- **`serve.sh`**: Launches single-worker `uvicorn` with `config.yaml` defaults, `CUSTOM_*` env overrides, and `NSYS=ON` support.

---

## Stage 1: Naive KV Cache (`stage-1-kv-cache/`)

Eliminates redundant decode sequence recomputation while remaining pure eager PyTorch (no SDPA/FlashAttention):

- **`model.py`**: Caches per-layer unexpanded `(k, v)` tensors (`list[tuple[torch.Tensor, torch.Tensor]]` of shape `[B, 2, S_total, 128]`), applies `RotaryEmbedding` with a `position_offset` to newly arrived tokens before `torch.cat` and GQA `repeat_interleave` (`2 -> 12` heads), skips causal mask allocation/application during single-token `[1, 1]` decode, and wraps attention compute in an `attn_compute` NVTX range.
- **`engine.py`**: Prefills `[1, L]` once inside `prefill_step` to initialize `past_key_values`, then feeds only the newly sampled `[1, 1]` token tensor on each `decode_step` ($O(S)$ per-step attention instead of $O(S^2)$) and releases KV cache references on exit.
- **`server.py` & `serve.sh`**: Self-contained Stage 1 FastAPI/uvicorn serving snapshot (`stage1-gpu` executor thread) that closes the engine generator in `event_generator()`'s `finally` block so cancelled streams release KV cache memory immediately.

---

## Stage 2: Tiled Attention (`stage-2-tiled-attn/`)

Replaces eager PyTorch `(Q @ K^T)` attention with a custom Triton FlashAttention-2 kernel (`_flash_attention_fwd`):

- **`model.py`**: Tiles $Q$ (`BLOCK_M=64` prefill, `BLOCK_M=16` decode) and $K, V$ (`BLOCK_N=64`) in SRAM with online float32 softmax (`m_i`, `l_i`, `acc`) to eliminate the $O(S^2)$ HBM attention matrix materialization. Indexes GQA KV heads directly inside the Triton kernel (`kv_head_idx = head_idx // num_kv_groups`) without `repeat_interleave`, and writes directly to a zero-copy `[B, S_q, H_q, D]` output buffer reshaped to `[B, S_q, hidden_size]` without post-attention transpose copies.
- **`engine.py`**: Warms up with a 64-token dummy prefill followed by a 1-token decode step outside NVTX ranges so both Triton kernel specializations (`BLOCK_M=64` and `BLOCK_M=16`) JIT-compile before `/health` serves traffic.
- **`server.py` & `serve.sh`**: Self-contained Stage 2 FastAPI/uvicorn serving snapshot (`stage2-gpu` executor thread).

---

## Stage 3: Parallel Decode Attention (`stage-3-split-k/`)

Eliminates single-token decode SM starvation (`S_q = 1`) by partitioning the KV sequence reduction across parallel thread blocks while keeping Stage 2's tiled FlashAttention-2 kernel (`_flash_attention_fwd`) for multi-token prefill (`S_q > 1`):

- **`model.py`**: During decode (`S_q = 1`), launches a 2-phase Triton Split-K pipeline over grid `(NUM_SPLITS, B * H_q)`. Phase 1 (`_flash_attention_decode_splitk_fwd`) partitions `L_kv` across `NUM_SPLITS` parallel CTAs (`BLOCK_N=64`), performing vectorized 1D dot-product and value reductions along `HEAD_DIM=128` without wasting `BLOCK_M` rows on padding and writing partial fp32 `(mid_o, mid_m, mid_l)` statistics (`-inf` for out-of-range splits). Phase 2 (`_flash_attention_splitk_combine`) merges the partial statistics across active splits via exact log-sum-exp rescaling into the zero-copy `[B, 1, H_q, D]` output buffer.
- **`engine.py`**: Warms up with a 64-token dummy prefill followed by a 1-token dummy decode pass outside NVTX ranges so all 3 Triton kernels (`_flash_attention_fwd`, `_flash_attention_decode_splitk_fwd`, `_flash_attention_splitk_combine`) JIT-compile before `/health` serves traffic.
- **`server.py` & `serve.sh`**: Self-contained Stage 3 FastAPI/uvicorn serving snapshot (`stage3-gpu` executor thread).

---

## Optimization Roadmap

- **Stage 0: Naive PyTorch (`stage-0-baseline/`)** — Full-sequence recomputation, explicit `(Q @ K^T)` attention, serialized `asyncio.Lock` serving.
- **Stage 1: KV Caching (`stage-1-kv-cache/`)** — Cache per-layer K and V tensors across decode steps to eliminate redundant sequence recomputation.
- **Stage 2: Tiled Attention (`stage-2-tiled-attn/`)** — Tile prefill attention in SRAM to avoid materializing the full $S \times S$ attention matrix in HBM.
- **Stage 3: Parallel Decode Attention (`stage-3-split-k/`)** — Split long-context KV reductions across thread blocks to improve SM occupancy during decode.
- **Stage 4: Paged KV Cache (PagedAttention-style)** — Store KV cache in fixed-size non-contiguous blocks to eliminate memory fragmentation and support batching.
- **Stage 5: Operator Fusion** — Fuse elementwise, normalization, and rotary kernels to reduce kernel-launch overhead and intermediate DRAM traffic.
