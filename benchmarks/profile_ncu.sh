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
d = cfg.get("draft_model", {})
b = cfg.get("benchmarks", {})
p = b.get("profiling", {})
hc = cfg.get("hardware_ceilings", {})

print(f"MODEL_NAME='{m.get('name', 'Qwen/Qwen2.5-Coder-1.5B-Instruct')}'")
print(f"DTYPE='{m.get('dtype', 'bfloat16')}'")
print(f"MAX_LEN={m.get('max_model_len', 9216)}")
print(f"DRAFT_MODEL='{d.get('name', 'Qwen/Qwen2.5-Coder-0.5B-Instruct')}'")
print(f"CFG_SPEC_TOKENS={d.get('num_speculative_tokens', 5)}")
print(f"TEMPERATURE={b.get('temperature', 0.0)}")
print(f"GPU_MEM={cfg.get('server', {}).get('gpu_memory_utilization', 0.70)}")
print(f"PROF_IN={p.get('ncu_input_len', 2048)}")
print(f"PEAK_DRAM_GBPS={float(hc.get('estimated_peak_dram_gbps', 896.0))}")
print(f"PEAK_TENSOR_TFLOPS={float(hc.get('estimated_peak_tensor_tflops', 40.0))}")
EOF
)"

STAGE="stage-0-vllm"
IMPORT_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --import-only)
            IMPORT_ONLY=1
            ;;
        *)
            STAGE="${arg}"
            ;;
    esac
done

PROF_DIR="benchmarks/profiles/${STAGE}"
RES_DIR="benchmarks/results/${STAGE}"
mkdir -p "${PROF_DIR}" "${RES_DIR}"

IS_CUSTOM=0
CUSTOM_STAGE_DIR=""
if [[ "${STAGE}" == *-custom ]]; then
    IS_CUSTOM=1
    STAGE_PREFIX="${STAGE%-custom}"
    MATCHES=(custom-serving/"${STAGE_PREFIX}"-*)
    if [[ -d "custom-serving/${STAGE}" ]]; then
        CUSTOM_STAGE_DIR="custom-serving/${STAGE}"
    elif [[ ${#MATCHES[@]} -eq 1 && -d "${MATCHES[0]}" ]]; then
        CUSTOM_STAGE_DIR="${MATCHES[0]}"
    else
        echo "Error: Could not resolve custom-serving directory for stage '${STAGE}' (expected custom-serving/${STAGE_PREFIX}-*)" >&2
        exit 1
    fi
fi

# Enable speculative decoding automatically for vLLM "stage-1*" or "*spec*" stages (or via NCU_SPEC_TOKENS override)
DEFAULT_SPEC_TOKENS=0
if [[ "${IS_CUSTOM}" -eq 0 && ( "${STAGE}" == *"spec"* || "${STAGE}" == "stage-1"* ) ]]; then
    DEFAULT_SPEC_TOKENS="${CFG_SPEC_TOKENS}"
fi
SPEC_TOKENS="${NCU_SPEC_TOKENS:-$DEFAULT_SPEC_TOKENS}"

REP_FILE="${PROF_DIR}/kernels.ncu-rep"
CSV_FILE="${PROF_DIR}/ncu_raw.csv"
METRICS_JSON="${RES_DIR}/ncu_metrics.json"
RUNNER_PY="${PROF_DIR}/_ncu_runner.py"
trap 'rm -f "${RUNNER_PY}" "${CSV_FILE}"' EXIT

echo "=========================================================="
echo " Nsight Compute Kernel & Roofline Profiler"
echo " Target Stage:   ${STAGE}"
if [[ "${IS_CUSTOM}" -eq 1 ]]; then
    echo " Custom Engine:  ${CUSTOM_STAGE_DIR}"
fi
echo " Target Model:   ${MODEL_NAME} (${DTYPE})"
if [[ "${SPEC_TOKENS}" -gt 0 ]]; then
    echo " Draft Model:    ${DRAFT_MODEL} (K=${SPEC_TOKENS})"
    echo " Workload:       Target Layer 0 | 1 Prefill (L=${PROF_IN}) + 1 Verify Step (M=$((SPEC_TOKENS + 1)), C=1)"
else
    echo " Workload:       Layer 0 | 1 Prefill (L=${PROF_IN}) + 1 Decode Step (M=1, C=1)"
fi
echo " Temperature:    ${TEMPERATURE}"
echo " Profile Out:    ${PROF_DIR}"
echo " Results JSON:   ${METRICS_JSON}"
echo "=========================================================="

if [[ "${IMPORT_ONLY}" -eq 1 ]]; then
    if [[ ! -f "${REP_FILE}" ]]; then
        echo "Error: Saved NCU report not found at ${REP_FILE}" >&2
        exit 1
    fi
    echo ">>> [1/2] Skipping GPU capture (--import-only); reusing ${REP_FILE}"
else
    # 3. Create single-process runner that profiles strictly Target Layer 0 (Pass 1 = Prefill, Pass 2 = Decode/Verify)
    if [[ "${IS_CUSTOM}" -eq 1 ]]; then
        cat << 'EOF' > "${RUNNER_PY}"
from pathlib import Path
import sys
import torch
import yaml
from transformers import AutoTokenizer

stage_dir, model_name, dtype_str, in_len = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
)

sys.path.insert(0, str(Path(stage_dir).resolve()))
import model as model_mod  # noqa: E402

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}
torch_dtype = _DTYPE_MAP.get(dtype_str.lower(), torch.bfloat16)

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
base_prompt = cfg["benchmarks"].get("prompt", "Write a Python module to benchmark matrix multiplication.").strip()

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
reps = (in_len // max(1, len(base_ids))) + 2
prompt_token_ids = (base_ids * reps)[:in_len]

device = torch.device("cuda")
model = model_mod.Qwen2Model.from_pretrained(model_name, dtype=torch_dtype, device=device)
layer0 = model.layers[0]

# Unprofiled warmup (1 prefill + 1 decode step)
warmup_ids = torch.tensor([prompt_token_ids[:64]], dtype=torch.long, device=device)
with torch.inference_mode():
    warm_out = model(warmup_ids, last_token_only=True)
    if isinstance(warm_out, tuple):
        warm_logits, warm_kv = warm_out
        warm_next = torch.argmax(warm_logits[:, -1, :], dim=-1, keepdim=True)
        _ = model(warm_next, past_key_values=warm_kv, last_token_only=True)
        del warm_kv
    else:
        warm_next = torch.argmax(warm_out[:, -1, :], dim=-1, keepdim=True)
        _ = model(torch.cat([warmup_ids, warm_next], dim=1), last_token_only=True)
    del warm_out
torch.cuda.synchronize(device)

pass_counter = {"count": 0, "active": False}

def pre_hook(module, args):
    if pass_counter["count"] < 2:
        phase = "Prefill" if pass_counter["count"] == 0 else "Decode"
        pass_counter["active"] = True
        torch.cuda.nvtx.range_push(phase)
        torch.cuda.cudart().cudaProfilerStart()

def post_hook(module, args, output):
    if pass_counter["active"]:
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()
        pass_counter["active"] = False
    pass_counter["count"] += 1

def norm_pre_hook(module, args):
    if pass_counter["active"]:
        torch.cuda.nvtx.range_push("rmsnorm_op")

def norm_post_hook(module, args, output):
    if pass_counter["active"]:
        torch.cuda.nvtx.range_pop()

orig_apply_rope = getattr(model_mod, "apply_rotary_pos_emb", None)
if orig_apply_rope is not None:
    def wrapped_apply_rope(*args, **kwargs):
        if pass_counter["active"]:
            torch.cuda.nvtx.range_push("rope_op")
            try:
                return orig_apply_rope(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()
        return orig_apply_rope(*args, **kwargs)
    model_mod.apply_rotary_pos_emb = wrapped_apply_rope

hooks = [
    layer0.register_forward_pre_hook(pre_hook),
    layer0.register_forward_hook(post_hook),
    layer0.input_layernorm.register_forward_pre_hook(norm_pre_hook),
    layer0.input_layernorm.register_forward_hook(norm_post_hook),
    layer0.post_attention_layernorm.register_forward_pre_hook(norm_pre_hook),
    layer0.post_attention_layernorm.register_forward_hook(norm_post_hook),
]

input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
with torch.inference_mode():
    # Pass 0: Prefill (L=in_len)
    out = model(input_ids, last_token_only=True)
    # Pass 1: Decode (M=1 using returned past_key_values when supported)
    if isinstance(out, tuple):
        logits, past_kv = out
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        _ = model(next_tok, past_key_values=past_kv, last_token_only=True)
        del past_kv
    else:
        next_tok = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
        _ = model(torch.cat([input_ids, next_tok], dim=1), last_token_only=True)
torch.cuda.synchronize(device)

for h in hooks:
    h.remove()
EOF
    else
        cat << 'EOF' > "${RUNNER_PY}"
import sys
import torch
import yaml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_name, dtype, max_len, gpu_mem, in_len, draft_model, spec_tokens, temp = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    float(sys.argv[4]),
    int(sys.argv[5]),
    sys.argv[6],
    int(sys.argv[7]),
    float(sys.argv[8]),
)

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
base_prompt = cfg["benchmarks"].get("prompt", "Write a Python module to benchmark matrix multiplication.").strip()

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
reps = (in_len // max(1, len(base_ids))) + 2
prompt_token_ids = (base_ids * reps)[:in_len]

llm_kwargs = dict(
    model=model_name,
    dtype=dtype,
    max_model_len=max_len,
    gpu_memory_utilization=gpu_mem,
    enforce_eager=True,
    enable_prefix_caching=False,
)

if spec_tokens > 0:
    llm_kwargs["speculative_config"] = {
        "method": "draft_model",
        "model": draft_model,
        "num_speculative_tokens": spec_tokens,
    }

llm = LLM(**llm_kwargs)

# Locate Target Model Layer 0 across standard and speculative worker hierarchies
worker = llm.llm_engine.model_executor.driver_worker
if hasattr(worker, "model_runner") and hasattr(worker.model_runner, "model"):
    driver_model = worker.model_runner.model
elif hasattr(worker, "scorer_worker"):
    driver_model = worker.scorer_worker.model_runner.model
elif hasattr(worker, "target_worker"):
    driver_model = worker.target_worker.model_runner.model
else:
    raise RuntimeError(f"Could not locate target model on worker: {type(worker)}")

layer0 = None
for name, mod in driver_model.named_modules():
    if name.endswith(".layers.0") or name == "layers.0":
        layer0 = mod
        break

pass_counter = {"count": 0, "active": False}

def pre_hook(module, args):
    if pass_counter["count"] < 2:
        phase = "Prefill" if pass_counter["count"] == 0 else "Decode"
        pass_counter["active"] = True
        torch.cuda.nvtx.range_push(phase)
        torch.cuda.cudart().cudaProfilerStart()

def post_hook(module, args, output):
    if pass_counter["active"]:
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_pop()
        pass_counter["active"] = False
    pass_counter["count"] += 1

# Ensure enough token budget so speculative decoding drafts the full K tokens on Pass 2
gen_tokens = 2 + spec_tokens if spec_tokens > 0 else 2

# Unprofiled warmup
warmup_params = SamplingParams(temperature=temp, max_tokens=gen_tokens, ignore_eos=True)
llm.generate([{"prompt_token_ids": prompt_token_ids[:64]}], warmup_params, use_tqdm=False)

# Attach surgical hook to Target Layer 0
hooks = []
if layer0 is not None:
    hooks.append(layer0.register_forward_pre_hook(pre_hook))
    hooks.append(layer0.register_forward_hook(post_hook))

# Profiled run: captures strictly Pass 0 (Prefill, L=in_len) and Pass 1 (Decode/Verify, M=1 or K+1)
prof_params = SamplingParams(temperature=temp, max_tokens=gen_tokens, ignore_eos=True)
llm.generate([{"prompt_token_ids": prompt_token_ids}], prof_params, use_tqdm=False)

for h in hooks:
    h.remove()
EOF
    fi

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
if [[ "${IS_CUSTOM}" -eq 1 ]]; then
    ncu \
        --target-processes all \
        --profile-from-start off \
        --nvtx \
        --set detailed \
        --metrics "${METRICS_ARG}" \
        --export "${REP_FILE}" \
        --force-overwrite \
        uv run --frozen python3 "${RUNNER_PY}" "${CUSTOM_STAGE_DIR}" "${MODEL_NAME}" "${DTYPE}" "${PROF_IN}"
else
    VLLM_ENABLE_V1_MULTIPROCESSING=0 ncu \
        --target-processes all \
        --profile-from-start off \
        --nvtx \
        --set detailed \
        --metrics "${METRICS_ARG}" \
        --export "${REP_FILE}" \
        --force-overwrite \
        uv run python3 "${RUNNER_PY}" "${MODEL_NAME}" "${DTYPE}" "${MAX_LEN}" "${GPU_MEM}" "${PROF_IN}" "${DRAFT_MODEL}" "${SPEC_TOKENS}" "${TEMPERATURE}"
fi
fi

echo ">>> [2/3] Exporting raw counters to CSV..."
ncu --import "${REP_FILE}" --csv --page raw > "${CSV_FILE}"

echo ">>> [3/3] Building Roofline and Kernel JSON..."
uv run python3 - "${CSV_FILE}" "${METRICS_JSON}" "${STAGE}" "${PEAK_DRAM_GBPS}" "${PEAK_TENSOR_TFLOPS}" << 'EOF'
import csv, json, sys

csv_path, json_path, stage = sys.argv[1:4]
peak_dram_gbps = float(sys.argv[4])
peak_tensor_tflops = float(sys.argv[5])

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

def get_first_float(row, keys):
    for key in keys:
        val = row.get(key, "")
        if val not in (None, "", "N/A"):
            return get_float(row, key)
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

def classify_kernel(kname, phase, nvtx_txt=""):
    kl = kname.lower()
    nl = (nvtx_txt or "").lower()
    if "flash_attention" in kl or "_flash_attention_fwd" in kl:
        return f"{phase} Attention (Triton FlashAttention)"
    if "attn_compute" in nl:
        return f"{phase} Attention"
    if "rmsnorm_op" in nl:
        return f"{phase} Fused RMSNorm"
    if "rope_op" in nl:
        return f"{phase} RoPE Embedding"
    if "splitkv_combine" in kl:
        return f"{phase} Attention Combine (Split-K)"
    if "flash_fwd_splitkv" in kl:
        return f"{phase} Attention (FlashDecode/SplitKV)"
    if "flash_fwd" in kl:
        return f"{phase} Attention (FlashAttention)"
    if "softmax" in kl:
        return f"{phase} Attention"
    if "gemv" in kl:
        return f"{phase} Linear GEMV"
    if "gemm" in kl or "nvjet" in kl or "cutlass" in kl:
        return f"{phase} Linear GEMM"
    if (
        "rms_norm" in kl
        or "rsqrt_kernel" in kl
        or "meanops" in kl
        or "pow_tensor_scalar" in kl
        or "cudafunctor_add" in kl
        or "cudafunctoronself_add" in kl
    ):
        return f"{phase} Fused RMSNorm"
    if "rope" in kl or "rotary" in kl or "neg_kernel" in kl:
        return f"{phase} RoPE Embedding"
    if "reshape_and_cache" in kl or "catarraybatchedcopy" in kl:
        return f"{phase} KV Cache Write"
    if "silu" in kl or "act_and_mul" in kl or "binaryfunctor" in kl:
        return f"{phase} SwiGLU Activation"
    return f"{phase} {kname[:28]}"

records = []

total_rows = len(data_rows)
for idx, row in enumerate(data_rows):
    dur_ns = get_duration_ns(row, units_row)
    if dur_ns <= 0:
        continue

    kname = row.get("Kernel Name", "unknown")
    nvtx_txt = (
        row.get("NVTX Push/Pop Range")
        or row.get("thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg")
        or ""
    )
    grid_str = row.get("Grid Size", "")

    # Determine phase via NVTX tag, or fallback to launch order / grid size
    if "Decode" in nvtx_txt:
        phase = "Decode"
    elif "Prefill" in nvtx_txt:
        phase = "Prefill"
    else:
        phase = "Prefill" if idx < (total_rows // 2) else "Decode"

    label = classify_kernel(kname, phase, nvtx_txt)

    dram_bytes = get_float(row, "dram__bytes.sum")
    # Convert DRAM unit if reported in KB/MB/GB
    dram_unit = units_row.get("dram__bytes.sum", "byte").lower()
    if "gbyte" in dram_unit or dram_unit == "gb":
        dram_bytes *= 1e9
    elif "mbyte" in dram_unit or dram_unit == "mb":
        dram_bytes *= 1e6
    elif "kbyte" in dram_unit or dram_unit == "kb":
        dram_bytes *= 1e3

    dram_pct = get_first_float(row, (
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_active",
    ))
    sm_pct = get_first_float(row, (
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_active",
    ))
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
        "block_size": row.get("Block Size", ""),
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

# Deduplicate by label (keep highest duration non-copy compute invocation per kernel role)
def is_compute_kernel(rec):
    rn = rec["raw_kernel_name"].lower()
    return (
        "direct_copy_kernel" not in rn
        and "bfloat16_copy_kernel" not in rn
        and "catarraybatchedcopy" not in rn
        and rec["arithmetic_intensity_flop_byte"] > 0.0
    )

dedup = {}
for r in records:
    lbl = r["label"]
    r_key = (is_compute_kernel(r), r["arithmetic_intensity_flop_byte"] > 0.0, r["execution_time_us"])
    if lbl not in dedup:
        dedup[lbl] = r
    else:
        cur = dedup[lbl]
        cur_key = (is_compute_kernel(cur), cur["arithmetic_intensity_flop_byte"] > 0.0, cur["execution_time_us"])
        if r_key > cur_key:
            dedup[lbl] = r

final_kernels = list(dedup.values())

payload = {
    "stage": stage,
    "hardware_ceilings": {
        "estimated_peak_dram_gbps": peak_dram_gbps,
        "estimated_peak_tensor_tflops": peak_tensor_tflops,
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