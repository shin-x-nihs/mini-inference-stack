# Custom Serving Architecture Guide

From-scratch PyTorch/CUDA inference engines built **strictly additively**: each stage copies `stage-(N-1)` and implements one architectural upgrade on top of it.

All stages serve `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`) via an OpenAI-compatible `POST /v1/chat/completions` SSE endpoint on port `8000` (configured by root `config.yaml`), emit `prefill_step` / `decode_step` / `attn_compute` NVTX ranges for `benchmarks/profile_nsys.sh`, and support standalone Layer 0 kernel profiling via `benchmarks/profile_ncu.sh <stage-id>` (Stage 0 deferred NCU profiling since `nsys` timeline metrics already isolated full-sequence decode recomputation). Shared parity verification lives at `custom-serving/test_parity.py` (`--stage stage-0-baseline|stage-1-kv-cache|stage-2-tiled-attn|stage-3-split-k|stage-4-paged-kv|stage-5-fused-ops|stage-6-cuda-graphs|all`), loading each stage's `model.py` and `engine.py` in isolation to verify single-step prefill logits and 100% greedy token ID parity against HuggingFace `AutoModelForCausalLM` (`eager` for Stages 0–1; `sdpa` for Stage 2+ fp32-accumulating kernels).

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

- **`model.py`**: During decode (`S_q = 1`), launches a 2-phase Triton Split-K pipeline over grid `(B * H_q, num_splits)` (`num_splits` dynamically chosen up to `MAX_SPLITK_SPLITS=32`). Phase 1 (`_flash_attention_decode_splitk_fwd`) partitions `L_kv` across `num_splits` parallel CTAs (`BLOCK_N=64`), performing vectorized 1D dot-product and value reductions along `HEAD_DIM=128` without wasting `BLOCK_M` rows on padding and writing partial fp32 `(partial_acc, partial_m, partial_l)` statistics (`-inf` / `0.0` for empty splits). Phase 2 (`_flash_attention_splitk_combine`) merges the partial statistics across active splits (`BLOCK_S=32`) via exact log-sum-exp rescaling into the zero-copy `[B, 1, H_q, D]` output buffer.
- **`engine.py`**: Warms up with a 64-token dummy prefill followed by a 1-token dummy decode pass outside NVTX ranges so all 3 Triton kernels (`_flash_attention_fwd`, `_flash_attention_decode_splitk_fwd`, `_flash_attention_splitk_combine`) JIT-compile before `/health` serves traffic.
- **`server.py` & `serve.sh`**: Self-contained Stage 3 FastAPI/uvicorn serving snapshot (`stage3-gpu` executor thread).

---

## Stage 4: Paged KV Cache & Continuous Batching (`stage-4-paged-kv/`)

Replaces per-request contiguous `torch.cat` KV tensors and global `asyncio.Lock` request serialization with a physical block pool (`KV_BLOCK_SIZE = 16`) and an iteration-level continuous batching scheduler:

- **`model.py`**: Introduces a Triton `_reshape_and_cache` kernel that scatters newly projected K and V tokens directly into non-contiguous physical pages (`[2, num_blocks, 16, num_kv_heads, head_dim]`) via a 0-padded `int32` `block_table` and per-sequence `context_lengths`, eliminating `torch.cat` (`CatArrayBatchedCopy`) full-history reallocations. Upgrades `_flash_attention_fwd` (prefill) and `_flash_attention_decode_splitk_fwd` (decode) to gather KV tiles indirectly through `block_table` with per-sequence causal bounds from `context_lengths` and per-row RoPE offsets during batched decode.
- **`engine.py`**: Implements `BlockAllocator` (free-block deque + double-free guard) and `PagedEngine` (`NaiveEngine` alias) with an iteration-level continuous batching scheduler (`run_scheduler_loop`). Groups equal-length waiting prompts for zero-padding prefill admission, coalesces all active sequences into batched $M = B$ decode steps (shifting decode linear projections from memory-bound $M = 1$ GEMVs to higher-intensity batched GEMMs), performs a single batch `.tolist()` D2H sampling transfer per step, and supports recomputation preemption when free blocks are exhausted.
- **`server.py` & `serve.sh`**: Lock-free async FastAPI/uvicorn serving snapshot (`stage4-gpu` executor thread, `CUSTOM_NUM_KV_BLOCKS=4096`, `CUSTOM_MAX_BATCH_SIZE=64`) that streams concurrent SSE requests directly through the background scheduler loop and reclaims KV blocks immediately on client disconnect.

---

## Stage 5: Operator Fusion (`stage-5-fused-ops/`)

Replaces unfused PyTorch elementwise, reduction, and concatenation chains with custom Triton kernels and combined `gate_up_proj` MLP weights while preserving Stage 4's paged KV cache, Split-K decode attention, and continuous batching scheduler:

- **`model.py`**: Introduces three custom Triton kernels alongside Stage 4's `_reshape_and_cache`, `_flash_attention_fwd`, and Split-K decode pipeline:
  1. `_rms_norm_fwd`: Computes variance and inverse RMS (`tl.math.rsqrt`) in fp32, casts to input dtype, and applies the scale weight in a single pass (replacing `pow`/`mean`/`rsqrt`), with optional `HAS_RESIDUAL` residual-add fusion threaded across all 28 `Qwen2Layer` blocks and the final `self.norm`.
  2. `_rope_embedding_fwd`: Applies rotary embeddings in-place to `q` and `k` in one kernel launch using explicit 4D strides (handling non-contiguous `.transpose(1, 2)` prefill tensors without `.contiguous()` copies), completely eliminating `rotate_half` slicing and `torch.cat` (`CatArrayBatchedCopy`).
  3. `SwiGLUMLP` (`gate_up_proj` + `_swiglu_fwd`): Combines `gate_proj` and `up_proj` into a single `nn.Linear(hidden_size, 2 * intermediate_size, bias=False)`—loaded in `Qwen2Model.from_pretrained` by concatenating safetensors `gate_proj.weight` and `up_proj.weight` along `dim=0` across shards—followed by the 2D-tiled `_swiglu_fwd` Triton kernel (`SiLU(gate) * up` in registers).
- **`engine.py`**: Self-contained copy of Stage 4's `BlockAllocator` and `PagedEngine` (`NaiveEngine` alias) continuous-batching scheduler (`run_scheduler_loop`), warming up `L=64`, `L=192` (Split-K), and `B=2` batched shapes outside NVTX ranges so all fused Triton + attention specializations JIT-compile before `/health` serves traffic.
- **`server.py` & `serve.sh`**: Self-contained Stage 5 FastAPI/uvicorn serving snapshot (`stage5-gpu` executor thread, `"Custom Serving - Stage 5: Operator Fusion"`, `CUSTOM_NUM_KV_BLOCKS=4096`, `CUSTOM_MAX_BATCH_SIZE=64`, `NSYS=ON` support).

---

## Stage 6: Bucketed CUDA Graphs for Decode (`stage-6-cuda-graphs/`)

Captures the fixed-shape decode forward pass (`q_len == 1`) into static `torch.cuda.CUDAGraph` batch buckets while keeping variable-length prefill eager:

- **`model.py`**: Makes `Qwen2Model.forward` capture-safe (zero host-device syncs or dynamic shapes when `max_context_len` is passed as a static Python `int`) and padded-lane safe (`positions = (start_pos[:, None] + ...).clamp(min=0)` so padded rows with `context_lengths[pad] == 0` read RoPE row `0`, skip KV cache writes in `_reshape_and_cache` via `pos == -1` / `valid_blk == False`, and emit finite zero attention in `_flash_attention_splitk_combine`). Adds `DecodeGraphRunner` managing `GRAPH_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)` (`<= max_batch_size`) with pre-allocated static device buffers (`input_ids [B, 1]`, `block_table [B, max_blocks_per_seq]`, `context_lengths [B]`), `capture_all()` (warming up both `SINGLE_SPLIT=False` and `SINGLE_SPLIT=True` Triton specializations before 2-pass stream warmup and `torch.cuda.graph()` capture), and `replay()` (copying active rows `0..actual_batch_size-1`, zero-filling padded rows `actual_batch_size..B-1`, replaying in one `cudaGraphLaunch`, and returning `static_logits[:actual_batch_size]`).
- **`engine.py`**: Initializes `DecodeGraphRunner` in `PagedEngine`, runs eager prefill/decode warmups (`L=64`, `L=192`, `L=8` + `B=2`, and `B=16`) followed by `graph_runner.capture_all()` using a temporary KV block (asserting `num_free_blocks == num_blocks` after reclamation), and routes `_run_decode_batch` through `graph_runner.replay(...)` inside the `decode_step` NVTX range.
- **`server.py` & `serve.sh`**: Self-contained Stage 6 FastAPI/uvicorn serving snapshot (`stage6-gpu` executor thread, `"Custom Serving - Stage 6: CUDA Graphs"`, `CUSTOM_NUM_KV_BLOCKS=4096`, `CUSTOM_MAX_BATCH_SIZE=64`, `NSYS=ON` support).

---

## Optimization Roadmap

- **Stage 0: Naive PyTorch (`stage-0-baseline/`)** — Full-sequence recomputation, explicit `(Q @ K^T)` attention, serialized `asyncio.Lock` serving.
- **Stage 1: KV Caching (`stage-1-kv-cache/`)** — Cache per-layer K and V tensors across decode steps to eliminate redundant sequence recomputation.
- **Stage 2: Tiled Attention (`stage-2-tiled-attn/`)** — Tile prefill attention in SRAM to avoid materializing the full $S \times S$ attention matrix in HBM.
- **Stage 3: Parallel Decode Attention (`stage-3-split-k/`)** — Split long-context KV reductions across thread blocks to improve SM occupancy during decode.
- **Stage 4: Paged KV Cache & Continuous Batching (`stage-4-paged-kv/`)** — Store KV cache in fixed-size non-contiguous blocks and schedule iteration-level continuous batches to eliminate memory fragmentation and unlock batched decode GEMMs (`1,260.39 tok/s` at $C=64$, `4.61×` faster Layer 0 decode KV write).
- **Stage 5: Operator Fusion (`stage-5-fused-ops/`)** — Fuse RMSNorm + residual (`_rms_norm_fwd`), in-place RoPE (`_rope_embedding_fwd`, eliminating all `3,584` `CatArrayBatchedCopy` launches), and `gate_up_proj` + `_swiglu_fwd`, cutting `nsys` kernel launches **3.03×** (`72,933` → `24,101`), `cpu_launch_overhead_ms` by **-45.57%** (`669.98 ms` → `364.68 ms`), and GPU idle time from `53.53%` to `28.97%` while lifting $C=1$ throughput **+77.7%** (`38.51` → `68.44 tok/s`) and $C=64$ throughput to `1,486.13 tok/s` (`120.6%` of `stage-0-vllm`).
- **Stage 6: Bucketed CUDA Graphs for Decode (`stage-6-cuda-graphs/`)** — Capture decode batches $B \in \{1, 2, 4, 8, 16, 32, 64\}$ into static `torch.cuda.CUDAGraph` runners (`cuda_graph_launches`: `0` → `63`), cutting CPU launch API calls **50.6×** (`24,100` → `476`, `-98.02%`), reducing `cpu_launch_overhead_ms` by **-21.78%** (`364.68 ms` → `285.25 ms`), shrinking GPU idle time **3.89×** (`28.97%` → `9.03%`), cutting `decode_step_avg_ms` by **-31.32%** (`12.58 ms` → `8.64 ms`, beating `stage-0-vllm`'s `10.67 ms`), and surpassing `stage-0-vllm` across **all** Pareto concurrency levels (`113.42 tok/s` at $C=1$, **+65.7%** over Stage 5 / `129.2%` of vLLM; `1,770.61 tok/s` at $C=64$, **+19.1%** over Stage 5 / `143.7%` of vLLM) and **all** context lengths `L=128..8192` (`7.50–8.60 ms` TPOT).
