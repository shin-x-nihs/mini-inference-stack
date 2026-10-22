#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

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
print(f"PROF_IN={p.get('ncu_input_len', 2048)}")
EOF
)"

STAGE="${1:-stage-0-vllm}"
PROF_DIR="benchmarks/profiles/${STAGE}"
RES_DIR="benchmarks/results/${STAGE}"
mkdir -p "${PROF_DIR}" "${RES_DIR}"

REP_FILE="${PROF_DIR}/kernels.ncu-rep"
CSV_FILE="${PROF_DIR}/ncu_raw.csv"
METRICS_JSON="${RES_DIR}/ncu_metrics.json"
RUNNER_PY="${PROF_DIR}/_ncu_runner.py"

echo "=========================================================="
echo " Nsight Compute Kernel & Roofline Profiler"
echo " Target Stage:   ${STAGE}"
echo " Model:          ${MODEL_NAME} (${DTYPE})"
echo " Workload:       Layer 0 | 1 Prefill (L=${PROF_IN}) + 1 Decode Step (C=1)"
echo " Profile Out:    ${PROF_DIR}"
echo " Results JSON:   ${METRICS_JSON}"
echo "=========================================================="

# 3. Create single-process runner that profiles strictly Layer 0 (Pass 1 = Prefill, Pass 2 = Decode)
cat << 'EOF' > "${RUNNER_PY}"
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

# Locate Layer 0 inside vLLM model runner
driver_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
layer0 = None
for name, mod in driver_model.named_modules():
    if name.endswith(".layers.0") or name == "layers.0":
        layer0 = mod
        break

pass_counter = {"count": 0}

def pre_hook(module, args):
    phase = "Prefill" if pass_counter["count"] == 0 else "Decode"
    torch.cuda.nvtx.range_push(phase)
    torch.cuda.cudart().cudaProfilerStart()

def post_hook(module, args, output):
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.nvtx.range_pop()
    pass_counter["count"] += 1

# Unprofiled warmup
warmup_params = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
llm.generate(["warmup " * 64], warmup_params, use_tqdm=False)

# Attach surgical hook to Layer 0
hooks = []
if layer0 is not None:
    hooks.append(layer0.register_forward_pre_hook(pre_hook))
    hooks.append(layer0.register_forward_hook(post_hook))

# Profiled run: max_tokens=2 triggers exactly 1 Prefill pass (count=0) and 1 Decode pass (count=1)
prompt_token_ids = [100] * in_len
prof_params = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
llm.generate([{"prompt_token_ids": prompt_token_ids}], prof_params, use_tqdm=False)

for h in hooks:
    h.remove()
EOF

# 4. Run Nsight Compute with --set detailed (enables full GUI Roofline, SOL, Memory & Warp charts)
# plus explicit FLOP and local memory spill counters for Table B & JSON export
EXTRA_METRICS=(
    "dram__bytes.sum"
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
IFS=,; METRICS_ARG="${EXTRA_METRICS[*]}"; unset IFS

echo ">>> [1/3] Running Nsight Compute capture on Layer 0 (Prefill & Decode)..."
VLLM_ENABLE_V1_MULTIPROCESSING=0 ncu \
    --target-processes all \
    --profile-from-start off \
    --nvtx \
    --set detailed \
    --metrics "${METRICS_ARG}" \
    --export "${REP_FILE}" \
    --force-overwrite \
    uv run python3 "${RUNNER_PY}" "${MODEL_NAME}" "${DTYPE}" "${MAX_LEN}" "${GPU_MEM}" "${PROF_IN}"

echo ">>> [2/3] Exporting raw counters to CSV..."
ncu --import "${REP_FILE}" --csv --page raw > "${CSV_FILE}"

echo ">>> [3/3] Building Roofline and Kernel JSON..."
uv run python3 - "${CSV_FILE}" "${METRICS_JSON}" "${STAGE}" << 'EOF'
import csv, json, sys

csv_path, json_path, stage = sys.argv[1:4]

# Keep only valid CSV lines starting from the "ID" header
with open(csv_path, "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

start_idx = next((i for i, l in enumerate(raw_lines) if l.startswith('"ID"')), None)
if start_idx is None:
    raise RuntimeError("Could not find CSV header in NCU export.")

reader = list(csv.DictReader(raw_lines[start_idx:]))
if len(reader) < 2:
    raise RuntimeError("NCU CSV contains no kernel rows.")

units_row = reader[0]  # Row 0 after header holds metric units (e.g., ns, us, ms, byte)
data_rows = reader[1:]

def get_float(row, key):
    val = row.get(key, "")
    if val is None or val == "" or val == "N/A":
        return 0.0
    try:
        return float(str(val).replace(",", ""))
    except ValueError:
        return 0.0

def get_duration_ns(row, units):
    val = get_float(row, "gpu__time_duration.sum")
    unit = units.get("gpu__time_duration.sum", "ns").strip()
    if unit == "ms":
        return val * 1e6
    if unit in ("us", "µs", "μs"):
        return val * 1e3
    if unit == "s":
        return val * 1e9
    return val

def classify_kernel(kname, phase):
    kl = kname.lower()
    if "splitkv_combine" in kl:
        return f"{phase} Attention Combine (Split-K)"
    if "flash_fwd_splitkv" in kl:
        return f"{phase} Attention (FlashDecode/SplitKV)"
    if "flash_fwd" in kl:
        return f"{phase} Attention (FlashAttention)"
    if "gemv" in kl:
        return f"{phase} Linear GEMV"
    if "gemm" in kl or "nvjet" in kl or "cutlass" in kl:
        return f"{phase} Linear GEMM"
    if "rms_norm" in kl:
        return f"{phase} Fused RMSNorm"
    if "rope" in kl or "rotary" in kl:
        return f"{phase} RoPE Embedding"
    if "silu" in kl or "act_and_mul" in kl:
        return f"{phase} SwiGLU Activation"
    return f"{phase} {kname[:28]}"

records = []
peak_bw_estimates = []
peak_tflops_estimates = []

total_rows = len(data_rows)
for idx, row in enumerate(data_rows):
    dur_ns = get_duration_ns(row, units_row)
    if dur_ns <= 0:
        continue

    kname = row.get("Kernel Name", "unknown")
    nvtx_txt = row.get("NVTX Push/Pop Range", "")
    grid_str = row.get("Grid Size", "")

    # Determine phase via NVTX tag, or fallback to launch order / grid size
    if "Decode" in nvtx_txt:
        phase = "Decode"
    elif "Prefill" in nvtx_txt:
        phase = "Prefill"
    else:
        phase = "Prefill" if idx < (total_rows // 2) else "Decode"

    label = classify_kernel(kname, phase)

    dram_bytes = get_float(row, "dram__bytes.sum")
    # Convert DRAM unit if reported in KB/MB/GB
    dram_unit = units_row.get("dram__bytes.sum", "byte").lower()
    if "gbyte" in dram_unit or dram_unit == "gb":
        dram_bytes *= 1e9
    elif "mbyte" in dram_unit or dram_unit == "mb":
        dram_bytes *= 1e6
    elif "kbyte" in dram_unit or dram_unit == "kb":
        dram_bytes *= 1e3

    dram_pct = get_float(row, "dram__throughput.avg.pct_of_peak_sustained_active")
    sm_pct = get_float(row, "sm__throughput.avg.pct_of_peak_sustained_active")
    warps_pct = get_float(row, "sm__warps_active.avg.pct_of_peak_sustained_active")
    regs = int(get_float(row, "launch__registers_per_thread"))
    spills = int(
        get_float(row, "l1tex__m_xbar2l1tex_read_bytes_mem_local.sum") +
        get_float(row, "l1tex__m_l1tex2xbar_write_bytes_mem_local.sum")
    )

    fp32_flops = (
        get_float(row, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum") +
        get_float(row, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum") +
        2.0 * get_float(row, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
    )
    fp16_flops = (
        get_float(row, "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum") +
        get_float(row, "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum") +
        2.0 * get_float(row, "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum")
    )
    tensor_flops = get_float(row, "smsp__inst_executed_pipe_tensor.sum") * 4096.0
    total_flops = fp32_flops + fp16_flops + tensor_flops

    dur_s = dur_ns * 1e-9
    dur_us = dur_ns * 1e-3
    achieved_bw_gbps = (dram_bytes / dur_ns) if dur_ns > 0 else 0.0
    ai = (total_flops / dram_bytes) if dram_bytes > 0 else 0.0
    tflops = (total_flops / (dur_s * 1e12)) if dur_s > 0 else 0.0

    if dram_pct > 20.0 and achieved_bw_gbps > 0:
        peak_bw_estimates.append(achieved_bw_gbps / (dram_pct / 100.0))
    if sm_pct > 20.0 and tensor_flops > 0:
        peak_tflops_estimates.append(tflops / (sm_pct / 100.0))

    if dram_pct >= sm_pct and dram_pct >= 35.0:
        bound = "Memory Bandwidth"
    elif sm_pct > dram_pct and sm_pct >= 35.0:
        bound = "Compute Pipeline"
    elif warps_pct < 20.0:
        bound = "Low Occupancy / Latency"
    else:
        bound = "Latency / Issue"

    records.append({
        "id": int(row.get("ID", idx)),
        "label": label,
        "phase": phase,
        "raw_kernel_name": kname,
        "grid_size": grid_str,
        " block_size": row.get("Block Size", ""),
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

# Deduplicate by label (keep highest duration invocation per kernel role)
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

print(f"Extracted {len(final_kernels)} kernel metrics into {json_path}")
EOF

rm -f "${RUNNER_PY}" "${CSV_FILE}"
echo "Saved NCU GUI report: ${REP_FILE}"
echo "Saved Roofline JSON:  ${METRICS_JSON}"