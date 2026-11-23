# Custom Serving Architecture Guide

From-scratch PyTorch and Triton inference engines built additively: each stage is a self-contained copy of `stage-(N-1)` with one targeted architectural upgrade on top (see [WORKFLOW.md](../WORKFLOW.md#track-philosophy)).

All stages share the same serving and profiling contract:
* Serve `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`) via an OpenAI-compatible streaming `POST /v1/chat/completions` endpoint on port `8000` (configured in root `config.yaml`).
* Emit `prefill_step`, `decode_step`, and `attn_compute` NVTX ranges when launched with `NSYS=ON`.
* Support standalone Layer 0 kernel profiling via `benchmarks/profile_ncu.sh <stage-id>`.
* Pass greedy token and prefill logit checks in `custom-serving/test_parity.py` (`--stage all`) against HuggingFace `AutoModelForCausalLM` (`eager` for Stages 0–1; `sdpa` for Stages 2–6).

For benchmark results, profiler breakdowns, and conclusions, see [REPORTS.md](../REPORTS.md).

---

## Stage 0: Naive PyTorch Baseline (`stage-0-baseline/`)

Unoptimized baseline with no KV cache, no `F.scaled_dot_product_attention`, and no `transformers.generation` helpers:
* **`model.py`**: Loads dimensions from `config.json` and weights from `.safetensors` shards (`RMSNorm`, float32-cached `RotaryEmbedding` with `rope_theta = 1000000.0`, `SwiGLUMLP`, tied `lm_head`). `NaiveAttention` expands 2 KV heads to 12 Q heads with `repeat_interleave` and computes `(Q @ K^T) * scale` with an upper-triangular `-inf` mask and float32 softmax. `Qwen2Model.forward` supports `last_token_only=True` to project only the final hidden state through `lm_head`.
* **`engine.py`**: `NaiveEngine` appends each sampled token and re-runs the full `[1, L + t]` sequence on every decode step, wrapping forward + sampling + `.item()` in `prefill_step` / `decode_step` NVTX ranges.
* **`server.py`**: FastAPI server (`/health`, `/v1/models`, `/v1/chat/completions`) serializing requests with a global `asyncio.Lock` on a single-thread `ThreadPoolExecutor(max_workers=1)` for CUDA thread affinity.
* **`serve.sh`**: Launches single-worker `uvicorn` with `config.yaml` defaults, `CUSTOM_*` overrides, and `NSYS=ON` support.

---

## Stage 1: Naive KV Cache (`stage-1-kv-cache/`)

Eliminates full-sequence decode recomputation while remaining pure eager PyTorch:
* **`model.py`**: Caches per-layer unexpanded `(k, v)` tensors (`[B, 2, S_total, 128]`), applies `RotaryEmbedding` with `position_offset` to new tokens before `torch.cat` and GQA `repeat_interleave`, skips causal mask allocation during single-token `[1, 1]` decode, and wraps attention math in `attn_compute`.
* **`engine.py`**: Prefills `[1, L]` once to build `past_key_values`, then feeds `[1, 1]` per decode step and releases cache references on exit.
* **`server.py` & `serve.sh`**: Self-contained snapshot (`stage1-gpu` worker) that closes the generator in `finally` so cancelled SSE streams free KV tensors immediately.

---

## Stage 2: Tiled Attention (`stage-2-tiled-attn/`)

Replaces eager PyTorch `(Q @ K^T)` attention with a custom Triton FlashAttention-2 forward kernel (`_flash_attention_fwd`):
* **`model.py`**: Tiles `Q` (`BLOCK_M = 64` prefill, `BLOCK_M = 16` decode) and `K, V` (`BLOCK_N = 64`) in SRAM with online float32 softmax (`m_i`, `l_i`, `acc`) so the `S × S` score matrix is never materialized in HBM. Indexes GQA KV heads directly inside the kernel (`kv_head_idx = head_idx // num_kv_groups`) without `repeat_interleave`, and writes directly into a `[B, S_q, H_q, D]` buffer reshaped to `[B, S_q, hidden_size]` without post-attention transposes.
* **`engine.py`**: Runs a 64-token prefill and 1-token decode warmup outside NVTX ranges so both Triton specializations JIT-compile before `/health` returns `200 OK`.
* **`server.py` & `serve.sh`**: Self-contained Stage 2 snapshot (`stage2-gpu` worker).

---

## Stage 3: Parallel Decode Attention (`stage-3-split-k/`)

Partitions single-token decode (`S_q = 1`) along the KV sequence length across parallel thread blocks while keeping `_flash_attention_fwd` for multi-token prefill (`S_q > 1`):
* **`model.py`**: Adds a two-stage Triton Split-K decode pipeline over `grid = (B * H_q, num_splits)` (`num_splits` chosen dynamically up to `MAX_SPLITK_SPLITS = 32`):
  1. `_flash_attention_decode_splitk_fwd`: Partitions `L_kv` across parallel CTAs (`BLOCK_N = 64`), computing 1D dot-product and value reductions along `HEAD_DIM = 128` without `BLOCK_M = 16` padding waste and writing partial fp32 `(partial_acc, partial_m, partial_l)` stats.
  2. `_flash_attention_splitk_combine`: Reduces partial statistics across active splits (`BLOCK_S = 32`) via log-sum-exp rescaling into `[B, 1, H_q, D]`.
* **`engine.py`**: Warms up 64-token prefill and 1-token decode outside NVTX ranges so all three Triton kernels JIT-compile at startup.
* **`server.py` & `serve.sh`**: Self-contained Stage 3 snapshot (`stage3-gpu` worker).

---

## Stage 4: Paged KV Cache & Continuous Batching (`stage-4-paged-kv/`)

Replaces contiguous `torch.cat` KV buffers and `asyncio.Lock` serialization with a physical KV block pool (`KV_BLOCK_SIZE = 16`) and an iteration-level scheduler:
* **`model.py`**: Adds Triton `_reshape_and_cache` to scatter new K/V tokens in-place into `[2, num_blocks, 16, num_kv_heads, head_dim]` using a 0-padded `int32` `block_table` and per-sequence `context_lengths`. Updates `_flash_attention_fwd` (with `BLOCK_N = 32`) and `_flash_attention_decode_splitk_fwd` to gather KV tiles via `block_table` with per-row RoPE offsets during batched decode.
* **`engine.py`**: Implements `BlockAllocator` (free-block deque + double-free check) and `PagedEngine` (`NaiveEngine` alias) with `run_scheduler_loop`. Groups equal-length waiting prompts for zero-padding prefill admission, batches all active sequences into `M = B` decode steps, performs one batched `.tolist()` D2H sampling copy per step, and supports recomputation preemption when free blocks run out.
* **`server.py` & `serve.sh`**: Lock-free async FastAPI snapshot (`stage4-gpu` worker, `CUSTOM_NUM_KV_BLOCKS = 4096`, `CUSTOM_MAX_BATCH_SIZE = 64`) that streams concurrent SSE requests through the scheduler loop and frees blocks on disconnect.

---

## Stage 5: Operator Fusion (`stage-5-fused-ops/`)

Replaces unfused PyTorch elementwise, reduction, concat, and split-MLP chains with custom Triton kernels and merged MLP weights while preserving Stage 4's paged KV cache and scheduler:
* **`model.py`**: Adds three custom Triton kernels plus a linear weight merge:
  1. `_rms_norm_fwd`: Single-pass fp32 variance + `tl.math.rsqrt` + bf16 cast/scale with optional `HAS_RESIDUAL` residual-add fusion threaded through all 28 layers and final norm.
  2. `_rope_embedding_fwd`: In-place Q/K rotary embedding using explicit 4D strides (handling non-contiguous `.transpose(1, 2)` prefill tensors without copies), removing `rotate_half` slicing and `torch.cat`.
  3. `SwiGLUMLP` (`gate_up_proj` + `_swiglu_fwd`): Concatenates safetensors `gate_proj.weight` and `up_proj.weight` along `dim = 0` at load time into one `nn.Linear(hidden_size, 2 * intermediate_size, bias=False)`, followed by a 2D-tiled `_swiglu_fwd` kernel (`SiLU(gate) * up` in registers).
* **`engine.py`**: Self-contained copy of Stage 4's `BlockAllocator` and `PagedEngine`, warming up `L = 64`, `L = 192` (Split-K), and `B = 2` batched shapes before `/health` goes live.
* **`server.py` & `serve.sh`**: Self-contained Stage 5 snapshot (`stage5-gpu` worker).

---

## Stage 6: Bucketed CUDA Graphs for Decode (`stage-6-cuda-graphs/`)

Captures fixed-shape decode (`q_len == 1`) into static `torch.cuda.CUDAGraph` batch buckets while keeping variable-length prefill eager:
* **`model.py`**: Makes `Qwen2Model.forward` capture-safe (zero host syncs when `max_context_len` is passed as a Python `int`) and padded-lane safe (`context_lengths[pad] == 0` clamps RoPE indices to row `0`, skips KV writes in `_reshape_and_cache` via `pos == -1` / `valid_blk == False`, and outputs finite zeros in `_flash_attention_splitk_combine`). Adds `DecodeGraphRunner` managing `GRAPH_BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)` (`<= max_batch_size`) with static device buffers (`input_ids`, `block_table`, `context_lengths`), `capture_all()` (warming both `SINGLE_SPLIT = False` and `True` Triton specializations before stream warmup and capture), and `replay()` (copying active rows, zero-filling padded lanes, replaying in one `cudaGraphLaunch`, and slicing `static_logits[:actual_batch_size]`).
* **`engine.py`**: Instantiates `DecodeGraphRunner` in `PagedEngine`, warms eager prefill/decode paths, captures all buckets using a temporary KV block (asserting full reclamation afterward), and routes `_run_decode_batch` through `graph_runner.replay(...)` inside `decode_step`.
* **`server.py` & `serve.sh`**: Self-contained Stage 6 snapshot (`stage6-gpu` worker).
