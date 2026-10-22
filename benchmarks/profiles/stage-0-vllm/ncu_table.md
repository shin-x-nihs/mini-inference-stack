### Table B: Kernel Details (Nsight Compute - Layer 0)

| Kernel Role | Grid Size | Time (us) | DRAM (%) | Compute (%) | Active Warps (%) | Regs | Spills (B) | AI (FLOP/B) | TFLOP/s | Dominant Bottleneck |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Prefill Fused RMSNorm** | `(2048, 1, 1)` | 46.02 | 0.0% | 0.0% | 91.07% | 38 | 0 | 1.658 | 0.454 | Latency / Issue |
| **Prefill Linear GEMM** | `(128, 9, 1)` | 4142.56 | 0.0% | 0.0% | 16.65% | 232 | 0 | 968.366 | 27.233 | Low Occupancy / Latency |
| **Prefill RoPE Embedding** | `(2048, 1, 1)` | 52.93 | 0.0% | 0.0% | 64.41% | 31 | 0 | 1.729 | 0.258 | Latency / Issue |
| **Prefill void vllm::reshape_and_cache** | `(2048, 1, 1)` | 27.87 | 0.0% | 0.0% | 44.51% | 48 | 0 | 0.982 | 0.075 | Latency / Issue |
| **Prefill Attention (FlashDecode/SplitKV)** | `(32, 1, 12)` | 737.54 | 0.0% | 0.0% | 8.34% | 254 | 0 | 1631.024 | 18.712 | Low Occupancy / Latency |
| **Prefill SwiGLU Activation** | `(2048, 1, 1)` | 243.71 | 0.0% | 0.0% | 64.88% | 37 | 0 | 5.674 | 1.958 | Latency / Issue |
| **Decode Fused RMSNorm** | `(1, 1, 1)` | 7.07 | 0.0% | 0.0% | 49.4% | 38 | 0 | 0.58 | 0.002 | Latency / Issue |
| **Decode Linear GEMV** | `(4480, 1, 1)` | 178.05 | 0.0% | 0.0% | 24.44% | 164 | 0 | 1.001 | 0.321 | Latency / Issue |
| **Decode RoPE Embedding** | `(1, 1, 1)` | 7.23 | 0.0% | 0.0% | 23.41% | 31 | 0 | 0.722 | 0.001 | Latency / Issue |
| **Decode void vllm::reshape_and_cache** | `(1, 1, 1)` | 8.86 | 0.0% | 0.0% | 17.29% | 48 | 0 | 0.044 | 0.0 | Low Occupancy / Latency |
| **Decode Attention (FlashDecode/SplitKV)** | `(1, 17, 2)` | 25.98 | 0.0% | 0.0% | 9.31% | 241 | 0 | 66.677 | 5.532 | Low Occupancy / Latency |
| **Decode Attention Combine (Split-K)** | `(3, 1, 1)` | 21.7 | 0.0% | 0.0% | 8.27% | 60 | 0 | 0.496 | 0.003 | Low Occupancy / Latency |
| **Decode SwiGLU Activation** | `(1, 1, 1)` | 10.02 | 0.0% | 0.0% | 53.05% | 27 | 0 | 5.2 | 0.023 | Latency / Issue |
