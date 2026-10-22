# Custom Serving - Stage 0 Baseline

This directory represents **Stage 0 of the custom inference track** (the custom counterpart to `vllm-reference/stage-0-baseline`).

### Purpose
* Provide a minimal, unoptimized baseline (e.g., standard PyTorch model forward and generation loop without specialized kernels).
* Establish a lower-bound comparison baseline against which custom optimizations (KV cache, FlashAttention-style tiling, Split-K decode, operator fusion) will be measured.
