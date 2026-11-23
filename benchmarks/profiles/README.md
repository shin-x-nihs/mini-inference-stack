# Profiler Traces

Raw profiler outputs (`.nsys-rep`, `.sqlite`, `nsys_stats.txt`, `.ncu-rep`) are stored per stage under `benchmarks/profiles/<stage-id>/` and gitignored due to file size. Open `.nsys-rep` in NVIDIA Nsight Systems and `.ncu-rep` in NVIDIA Nsight Compute for GUI inspection; parsed summary JSONs live in `benchmarks/results/<stage-id>/`.
