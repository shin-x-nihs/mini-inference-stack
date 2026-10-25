# AGENTS.md — custom-serving/

From-scratch engines, one optimization per stage (naive PyTorch → KV cache → tiled prefill attn → Split-K decode
→ paged KV → fusion). No code yet; `stage-0-baseline/README.md` only.

## Stage layout
- `custom-serving/stage-<N>-<desc>/` (exactly two levels deep), with `serve.sh`, engine code, and a README stating
  the single change vs. the previous stage. Result stage name e.g. `stage-0-custom`, `stage-1-naive`.
- Same model, bf16, tokenizer and chat template as root `config.yaml`. Verify output correctness before benchmarking.

## Contract to reuse the shared harness
- `serve.sh`: `cd` to repo root, read `model.*` / `server.*` from `config.yaml`, env overrides optional.
- HTTP: OpenAI-compatible `POST /v1/chat/completions` on `server.port` (8000), SSE `data: …` + `data: [DONE]`,
  supports `stream`, `stream_options.include_usage` (final chunk with `usage`, empty `choices`), `max_tokens`,
  `temperature`, `ignore_eos`; accepts `model` == `model.name`.
- nsys: when `NSYS=ON`, wrap launch in `nsys launch --trace=cuda,nvtx`; emit NVTX ranges named `prefill_step` /
  `decode_step` so `profile_nsys.sh` reports real phase timings.
- ncu: `benchmarks/profile_ncu.sh` only works for vLLM. Write a stage-specific runner producing `ncu_metrics.json`
  with `hardware_ceilings` + `kernels[]` (`label, phase, execution_time_us, arithmetic_intensity_flop_byte,
  attained_tflops, achieved_dram_gbps, …`) as consumed by `plot_dashboard.py`.

If a stage can't meet the shared workload, document the difference — don't edit `benchmarks.*` or `vllm-reference/`.
