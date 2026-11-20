# Custom Serving Architecture Guide

From-scratch PyTorch/CUDA inference engines built **strictly additively**: each stage copies `stage-(N-1)` and implements one architectural upgrade on top of it.

All stages serve `Qwen/Qwen2.5-Coder-1.5B-Instruct` (`bfloat16`) via an OpenAI-compatible `POST /v1/chat/completions` SSE endpoint on port `8000` (configured by root `config.yaml`) and emit `prefill_step` / `decode_step` NVTX ranges for `benchmarks/profile_nsys.sh`.

---

## Stage 0: Naive PyTorch Baseline (`stage-0-baseline/`)

Establishes an unoptimized lower bound with no KV cache, no `F.scaled_dot_product_attention`, and no `transformers.generation` wrappers:

- **`model.py`**: Loads dimensions dynamically from `config.json` and weights from local `.safetensors` shards (`RMSNorm`, float32-cached `RotaryEmbedding` with `rope_theta=1000000.0`, `SwiGLUMLP`, tied `lm_head`). In `NaiveAttention`, expands `2` KV heads to `12` Q heads via `repeat_interleave`, computes `(Q @ K^T) * scale` with an upper-triangular `-inf` causal mask and float32 softmax without a KV cache, and supports `last_token_only=True` in `Qwen2Model.forward` to project only the final hidden state through `lm_head`.
- **`engine.py`**: Runs a naive autoregressive loop (`NaiveEngine`) that concatenates each sampled token and passes the entire `[1, L + t]` sequence through `Qwen2Model` on every step, forcing $O(S^2)$ attention recomputation across prefill and decode. Each forward + sampling + `.item()` sync is wrapped in `prefill_step` or `decode_step` NVTX ranges.
- **`server.py`**: Exposes `/health`, `/v1/models`, and `/v1/chat/completions` (streaming SSE and non-streaming) in FastAPI. Serializes concurrent requests with a global `asyncio.Lock` and pins all CUDA/NVTX execution to a `ThreadPoolExecutor(max_workers=1)` for strict CUDA thread affinity.
- **`serve.sh` & `test_parity.py`**: Launches single-worker `uvicorn` with `config.yaml` defaults, `CUSTOM_*` env overrides, and `NSYS=ON` support; verifies greedy token ID parity against HuggingFace `AutoModelForCausalLM` (`attn_implementation="eager"`).

---

## Optimization Roadmap

- **Stage 0: Naive PyTorch (`stage-0-baseline/`)** — Full-sequence recomputation, explicit `(Q @ K^T)` attention, serialized `asyncio.Lock` serving.
- **Stage 1: KV Caching** — Cache per-layer K and V tensors across decode steps to eliminate redundant sequence recomputation.
- **Stage 2: Tiled Attention (FlashAttention-style)** — Tile prefill attention in SRAM to avoid materializing the full $S \times S$ attention matrix in HBM.
- **Stage 3: Parallel Decode Attention (Split-K / FlashDecode-style)** — Split long-context KV reductions across thread blocks to improve SM occupancy during decode.
- **Stage 4: Paged KV Cache (PagedAttention-style)** — Store KV cache in fixed-size non-contiguous blocks to eliminate memory fragmentation and support batching.
- **Stage 5: Operator Fusion** — Fuse elementwise, normalization, and rotary kernels to reduce kernel-launch overhead and intermediate DRAM traffic.
