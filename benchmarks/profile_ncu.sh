#!/usr/bin/env bash
set -euo pipefail

# 1. Resolve repository root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# 2. Extract model and profiling parameters from config.yaml
eval "$(uv run python3 - << 'EOF'
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

m = cfg.get("model", {})
b = cfg.get("benchmarks", {})
p = b.get("profiling", {})

print(f"MODEL_NAME='{m.get('name', 'Qwen/Qwen2.5-Coder-1.5B-Instruct')}'")
print(f"DTYPE='{m.get('dtype', 'bfloat16')}'")
print(f"MAX_LEN={m.get('max_model_len', 9216)}")
print(f"GPU_MEM={cfg.get('server', {}).get('gpu_memory_utilization', 0.70)}")
# Default L=2048 so KV cache spills out of L2 into DRAM
print(f"PROF_IN={p.get('ncu_input_len', 2048)}")
EOF
)"

STAGE="${1:-stage-0-vllm}"
PROF_DIR="benchmarks/profiles/${STAGE}"
RES_DIR="benchmarks/results/${STAGE}"
mkdir -p "${PROF_DIR}" "${RES_DIR}"

REP_FILE="${PROF_DIR}/kernels.ncu-rep"
CSV_FILE="${PROF_DIR}/ncu_raw.csv"
TABLE_MD="${PROF_DIR}/ncu_table.md"
METRICS_JSON="${RES_DIR}/ncu_metrics.json"
RUNNER_PY="${PROF_DIR}/_ncu_runner.py"

echo "=========================================================="
echo " Nsight Compute Kernel & Roofline Profiler"
echo " Target Stage:   ${STAGE}"
echo " Model:          ${MODEL_NAME} (${DTYPE})"
echo " Workload:       1 Prefill (L=${PROF_IN}) + 1 Decode Step (C=1)"
echo " Profile Out:    ${PROF_DIR}"
echo " Results JSON:   ${METRICS_JSON}"
echo "=========================================================="

# 3. Create temporary single-process runner that gates profiling to Layer 0 + lm_head
cat << 'EOF' > "${RUNNER_PY}"
import os
import sys
import torch
from vllm import LLM, SamplingParams

model_name, dtype, max_len, gpu_mem, in_len = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
)

llm = LLM(
    model=model_name,
    dtype=dtype,
    max_model_len=max_len,
    gpu_memory_utilization=gpu_mem,
    enforce_eager=True,
    enable_prefix_caching=False,
)

# Locate Layer 0 and lm_head so NCU only profiles 1 representative layer instead of all 28
driver_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
target_modules = []
for name, mod in driver_model.named_modules():
    if name.endswith(".layers.0") or name.endswith("lm_head"):
        target_modules.append((name, mod))

step_Tracker = {"phase": "Prefill"}

def pre_hook(module, args):
    torch.cuda.nvtx.range_push(f"{step_Tracker['phase']}")
    torch.cuda.cudart().cudaProfilerStart()

def post_hook(module, args, output):
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.nvtx.range_pop()

# Warmup pass (unprofiled)
warmup_params = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
llm.generate(["warmup " * 64], warmup_params, use_tqdm=False)

# Attach hooks for surgical Layer 0 + lm_head capture
hooks = []
for name, mod in target_modules:
    hooks.append(mod.register_forward_pre_hook(pre_hook))
    hooks.append(mod.register_forward_hook(post_hook))
    if name.endswith("lm_head"):
        # After lm_head finishes in pass 1 (Prefill), switch NVTX tag to Decode for pass 2
        hooks.append(mod.register_forward_hook(lambda m, a, o: step_Tracker.update({"phase": "Decode"})))

# Profiled pass: max_tokens=2 runs exactly 1 Prefill step + 1 Decode step
prompt_token_ids = [100] * in_len
prof_params = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)

if not target_modules:
    torch.cuda.cudart().cudaProfilerStart()
llm.generate([{"prompt_token_ids": prompt_token_ids}], prof_params, use_tqdm=False)
if not target_modules:
    torch.cuda.cudart().cudaProfilerStop()

for h in hooks:
    h.remove()
EOF

# 4. Run Nsight Compute collecting Table B + Roofline (FLOPs & DRAM Bytes) metrics
METRICS_LIST=(
    "gpu__time_duration.sum"
    "dram__bytes.sum"
    "dram__throughput.avg.pct_of_peak_sustained_active"
    "sm__throughput.avg.pct_of_peak_sustained_active"
    "sm__warps_active.avg.pct_of_peak_sustained_active"
    "launch__registers_per_thread"
    "l1tex__m_xbar2l1tex_read_bytes_mem_local.sum"
    "l1tex__m_l1tex2xbar_write_bytes_mem_local.sum"
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum"
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum"
    "smsp__inst_executed_pipe_tensor.sum"
)
IFS=,; METRICS_ARG="${METRICS_LIST[*]}"; unset IFS

echo ">>> [1/3] Running Nsight Compute capture (Layer 0 + lm_head across Prefill & Decode)..."
VLLM_ENABLE_V1_MULTIPROCESSING=0 ncu \
    --target-processes all \
    --profile-from-start off \
    --nvtx \
    -k regex:"gemvx|cutlass.*gemm|flash_fwd|fused_add_rms_norm|mul_silu" \
    --metrics "${METRICS_ARG}" \
    --export "${REP_FILE}" \
    --force-overwrite \
    uv run python3 "${RUNNER_PY}" "${MODEL_NAME}" "${DTYPE}" "${MAX_LEN}" "${GPU_MEM}" "${PROF_IN}"

echo ">>> [2/3] Exporting raw counters to CSV..."
ncu --import "${REP_FILE}" --csv --page raw > "${CSV_FILE}"

echo ">>> [3/3] Building Table B and Roofline JSON..."
uv run python3 - "${CSV_FILE}" "${TABLE_MD}" "${METRICS_JSON}" "${STAGE}" << 'EOF'
import csv, json, sys
from collections import defaultdict

csv_path, md_path, json_path, stage = sys.argv[1:5]

# Strip any non-CSV preamble lines emitted by ncu
with open(csv_path, "r", encoding="utf-8") as f:
    lines = [line for line in f if line.startswith('"ID"') or (line and line[0].isdigit()) or line.startswith('"0"')]

reader = csv.DictReader(lines)
invocations = defaultdict(dict)

for row in reader:
    kid = row.get("ID")
    if not kid:
        continue
    inv = invocations[kid]
    inv["kernel_name"] = row.get("Kernel Name", "unknown")
    inv["nvtx"] = row.get("NVTX Push/Pop Range", "")
    m_name = row.get("Metric Name", "")
    m_val_str = row.get("Metric Value", "0").replace(",", "")
    try:
        inv[m_name] = float(m_val_str)
    except ValueError:
        pass

def classify_kernel(kname, phase):
    kl = kname.lower()
    if "flash_fwd_splitkv" in kl:
        return "Decode Attention (FlashDecode)"
    if "flash_fwd" in kl:
        return f"{phase} Attention (FlashAttention)"
    if "gemvx" in kl:
        return f"{phase} GEMV (cuBLAS)"
    if "cutlass" in kl and "gemm" in kl:
        return f"{phase} GEMM (CUTLASS TensorOp)"
    if "rms_norm" in kl:
        return f"{phase} Fused Add+RMSNorm"
    if "silu" in kl:
        return f"{phase} Fused SwiGLU"
    return f"{phase} {kname[:30]}"

records = []
peak_bw_estimates = []
peak_tflops_estimates = []

for kid, d in invocations.items():
    dur_ns = d.get("gpu__time_duration.sum", 0.0)
    if dur_ns <= 0:
        continue

    nvtx_txt = d.get("nvtx", "")
    phase = "Decode" if "Decode" in nvtx_txt or "gemvx" in d["kernel_name"].lower() or "splitkv" in d["kernel_name"].lower() else "Prefill"
    label = classify_kernel(d["kernel_name"], phase)

    dram_bytes = d.get("dram__bytes.sum", 0.0)
    dram_pct = d.get("dram__throughput.avg.pct_of_peak_sustained_active", 0.0)
    sm_pct = d.get("sm__throughput.avg.pct_of_peak_sustained_active", 0.0)
    warps_pct = d.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0)
    regs = int(d.get("launch__registers_per_thread", 0))
    spills = int(
        d.get("l1tex__m_xbar2l1tex_read_bytes_mem_local.sum", 0.0) +
        d.get("l1tex__m_l1tex2xbar_write_bytes_mem_local.sum", 0.0)
    )

    # Compute Total FLOPs (CUDA Core FP32/FP16 + Tensor Core HMMA s16816 = 4096 FLOPs/inst)
    fp32_flops = (
        d.get("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", 0.0) +
        d.get("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", 0.0) +
        2.0 * d.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", 0.0)
    )
    fp16_flops = (
        d.get("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum", 0.0) +
        d.get("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum", 0.0) +
        2.0 * d.get("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum", 0.0)
    )
    tensor_flops = d.get("smsp__inst_executed_pipe_tensor.sum", 0.0) * 4096.0
    total_flops = fp32_flops + fp16_flops + tensor_flops

    dur_s = dur_ns * 1e-9
    dur_us = dur_ns * 1e-3
    achieved_bw_gbps = (dram_bytes / dur_ns) if dur_ns > 0 else 0.0
    ai = (total_flops / dram_bytes) if dram_bytes > 0 else 0.0
    tflops = (total_flops / (dur_s * 1e12)) if dur_s > 0 else 0.0

    if dram_pct > 20.0:
        peak_bw_estimates.append(achieved_bw_gbps / (dram_pct / 100.0))
    if sm_pct > 20.0 and tensor_flops > 0:
        peak_tflops_estimates.append(tflops / (sm_pct / 100.0))

    if dram_pct >= sm_pct and dram_pct >= 40.0:
        bound = "Memory Bandwidth"
    elif sm_pct > dram_pct and sm_pct >= 40.0:
        bound = "Compute Pipeline"
    elif warps_pct < 25.0:
        bound = "Low Occupancy / Latency"
    else:
        bound = "Latency / Issue"

    records.append({
        "label": label,
        "phase": phase,
        "raw_kernel_name": d["kernel_name"],
        "execution_time_us": round(dur_us, 2),
        "dram_utilization_pct": round(dram_pct, 2),
        "compute_utilization_pct": round(sm_pct, 2),
        "active_warps_pct": round(warps_pct, 2),
        "registers_per_thread": regs,
        "local_spills_bytes": spills,
        "dominant_bottleneck": bound,
        "arithmetic_intensity_flop_byte": round(ai, 3),
        "attained_tflops": round(tflops, 3),
        "achieved_dram_gbps": round(achieved_bw_gbps, 2),
    })

# Deduplicate by label (keep representative invocation with highest duration per category)
dedup = {}
for r in records:
    lbl = r["label"]
    if lbl not in dedup or r["execution_time_us"] > dedup[lbl]["execution_time_us"]:
        dedup[lbl] = r

final_kernels = list(dedup.values())

payload = {
    "stage": stage,
    "hardware_ceilings": {
        "estimated_peak_dram_gbps": round(sum(peak_bw_estimates) / len(peak_bw_estimates), 1) if peak_bw_estimates else 512.0,
        "estimated_peak_tensor_tflops": round(sum(peak_tflops_estimates) / len(peak_tflops_estimates), 1) if peak_tflops_estimates else 100.0,
    },
    "kernels": final_kernels
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

md_lines = [
    "### Table B: Kernel Details (Nsight Compute)",
    "",
    "| Kernel Category | Time (us) | DRAM (%) | Compute (%) | Active Warps (%) | Regs | Spills (B) | AI (FLOP/B) | TFLOP/s | Dominant Bottleneck |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
]
for k in final_kernels:
    md_lines.append(
        f"| **{k['label']}** | {k['execution_time_us']} | {k['dram_utilization_pct']}% | "
        f"{k['compute_utilization_pct']}% | {k['active_warps_pct']}% | {k['registers_per_thread']} | "
        f"{k['local_spills_bytes']} | {k['arithmetic_intensity_flop_byte']} | {k['attained_tflops']} | {k['dominant_bottleneck']} |"
    )

md_text = "\n".join(md_lines) + "\n"
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_text)

print("\n" + md_text)
EOF

rm -f "${RUNNER_PY}" "${CSV_FILE}"
echo "Saved NCU GUI report: ${REP_FILE}"
echo "Saved Table B MD:     ${TABLE_MD}"
echo "Saved Roofline JSON:  ${METRICS_JSON}"