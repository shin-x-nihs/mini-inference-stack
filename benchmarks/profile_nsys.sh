#!/usr/bin/env bash
set -euo pipefail

# 1. Resolve repository root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# 2. Extract configuration from config.yaml
eval "$(uv run python3 - << 'EOF'
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

host = cfg["server"].get("host", "127.0.0.1")
if host == "0.0.0.0":
    host = "127.0.0.1"
port = cfg["server"].get("port", 8000)

d = cfg.get("draft_model", {})
b = cfg.get("benchmarks", {})
p = b.get("profiling", {})

in_len = p.get("input_len", 1024)
out_len = p.get("output_len", 64)

print(f"MODEL_NAME='{cfg['model']['name']}'")
print(f"DRAFT_MODEL='{d.get('name', 'Qwen/Qwen2.5-Coder-0.5B-Instruct')}'")
print(f"NUM_SPEC_TOKENS={d.get('num_speculative_tokens', 5)}")
print(f"TEMPERATURE={b.get('temperature', 0.0)}")
print(f"BASE_URL='http://{host}:{port}'")
print(f"PROF_IN={in_len}")
print(f"PROF_OUT={out_len}")
EOF
)"

STAGE="${1:-stage-0-vllm}"
PROF_DIR="benchmarks/profiles/${STAGE}"
RES_DIR="benchmarks/results/${STAGE}"
mkdir -p "${PROF_DIR}" "${RES_DIR}"

REP_FILE="${PROF_DIR}/timeline.nsys-rep"
SQLITE_FILE="${PROF_DIR}/timeline.sqlite"
STATS_TXT="${PROF_DIR}/nsys_stats.txt"
METRICS_JSON="${RES_DIR}/nsys_metrics.json"

echo "=========================================================="
echo " Nsight Systems Timeline Profiler"
echo " Target Stage:   ${STAGE}"
echo " Target Model:   ${MODEL_NAME}"
if [[ "${STAGE}" == *"spec"* || "${STAGE}" == "stage-1"* ]]; then
    echo " Draft Model:    ${DRAFT_MODEL} (K=${NUM_SPEC_TOKENS})"
fi
echo " Temperature:    ${TEMPERATURE}"
echo " Trace Output:   ${REP_FILE}"
echo " Stats Output:   ${STATS_TXT}"
echo " Metrics JSON:   ${METRICS_JSON}"
echo " Workload:       C=1 | In=${PROF_IN} tokens | Out=${PROF_OUT} tokens"
echo "=========================================================="

run_probe() {
    uv run python3 - "${BASE_URL}" "${MODEL_NAME}" "${PROF_IN}" "${PROF_OUT}" "${TEMPERATURE}" << 'EOF'
import json, sys, urllib.request, yaml
from transformers import AutoTokenizer

base_url, model, in_len, out_len, temp = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
)

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

base_prompt = cfg["benchmarks"].get("prompt", "Write a Python module to benchmark matrix multiplication.").strip()
tokenizer = AutoTokenizer.from_pretrained(model)
base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
reps = (in_len // max(1, len(base_ids))) + 2
prompt = tokenizer.decode((base_ids * reps)[:in_len], skip_special_tokens=True)

payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": out_len,
    "temperature": temp,
    "ignore_eos": True,
    "stream": True
}).encode("utf-8")

req = urllib.request.Request(
    f"{base_url}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req, timeout=120) as resp:
    for _ in resp:
        pass
EOF
}

# Reset any stuck collection from a previous interrupted run
nsys stop >/dev/null 2>&1 || true

# 3. Warmup pass (unprofiled)
echo ">>> [1/5] Running unprofiled warmup request..."
run_probe

# 4. Profiled pass (bare start/stop to avoid reconfiguring the launched session)
echo ">>> [2/5] Capturing Nsight Systems trace..."
nsys start
run_probe
nsys stop

LATEST_REP="$(ls -t "${REPO_ROOT}"/report*.nsys-rep 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_REP}" ]]; then
    echo "Error: Could not find generated report*.nsys-rep in ${REPO_ROOT}" >&2
    exit 1
fi
mv -f "${LATEST_REP}" "${REP_FILE}"

# 5. Export to SQLite and populate the 4 core summary tables
echo ">>> [3/5] Exporting trace to SQLite..."
nsys export --type=sqlite --output="${SQLITE_FILE}" --force-overwrite=true "${REP_FILE}"

echo ">>> [4/5] Building 4 core summary tables (Kernel, API, MemOps, OSRT)..."
nsys stats \
    --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,osrt_sum \
    "${SQLITE_FILE}" > "${STATS_TXT}"

# 6. Parse SQLite database and compute timeline & kernel breakdown metrics into JSON
echo ">>> [5/5] Computing timeline metrics & kernel category breakdown..."
uv run python3 - "${SQLITE_FILE}" "${METRICS_JSON}" "${STAGE}" "${PROF_IN}" "${PROF_OUT}" << 'EOF'
import json, sqlite3, sys

sqlite_path, out_json, stage_name, in_len, expected_out = sys.argv[1:6]
in_len = int(in_len)
expected_out = int(expected_out)
decode_steps = max(1, expected_out - 1)

conn = sqlite3.connect(sqlite_path)
cur = conn.cursor()

# 1. Fetch all GPU kernel executions with demangled names and gridX
cur.execute("""
    SELECT k.start, k.end, COALESCE(s.value, ''), COALESCE(k.gridX, 0)
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    LEFT JOIN StringIds s ON k.demangledName = s.id
    ORDER BY k.start ASC
""")
kernels = cur.fetchall()

if not kernels:
    raise RuntimeError("No GPU kernels found in capture. Ensure server was started with 'NSYS=1 ./serve.sh'.")

w_start = kernels[0][0]
w_end = max(k[1] for k in kernels)
wall_ns = max(1, w_end - w_start)
num_kernels = len(kernels)

# 2. Interval union for exact non-overlapping GPU Active vs. Idle time
merged = []
for s, e, _, _ in kernels:
    if not merged or s > merged[-1][1]:
        merged.append([s, e])
    else:
        merged[-1][1] = max(merged[-1][1], e)

gpu_active_ns = sum(e - s for s, e in merged)
gpu_idle_ns = max(0, wall_ns - gpu_active_ns)
gpu_active_pct = (gpu_active_ns / wall_ns) * 100.0
gpu_idle_pct = 100.0 - gpu_active_pct

# 3. Classify GPU kernel time by operator category (Linear vs. Attention vs. Norm/Other)
cat_ns = {"linear": 0, "attention": 0, "norm_act_other": 0}
for s, e, name, _ in kernels:
    dur = max(0, e - s)
    low = name.lower()
    if any(k in low for k in ("gemm", "gemv", "cutlass", "cublas", "matmul", "wgrad", "dgrad")):
        cat_ns["linear"] += dur
    elif any(k in low for k in ("flash", "attention", "fmha", "splitkv", "sdpa", "paged_attention")):
        cat_ns["attention"] += dur
    else:
        cat_ns["norm_act_other"] += dur

total_kern_ns = max(1, sum(cat_ns.values()))

# 4. CPU CUDA Runtime overhead (Launch overhead vs. Host Sync wait) within active window
launch_ns = 0
launch_calls = 0
graph_launches = 0
sync_ns = 0

cur.execute("""
    SELECT r.start, r.end, s.value
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN StringIds s ON r.nameId = s.id
    WHERE r.end >= ? AND r.start <= ?
""", (w_start, w_end))

for s, e, name in cur.fetchall():
    clipped_s = max(s, w_start)
    clipped_e = min(e, w_end)
    dur = max(0, clipped_e - clipped_s)
    if "Launch" in name:
        launch_ns += dur
        launch_calls += 1
        if "GraphLaunch" in name:
            graph_launches += 1
    elif "Synchronize" in name:
        sync_ns += dur

# 5. Prefill vs. Decode phase split (NVTX -> Grid-Shape Boundary -> Gap Heuristic)
prefill_ms = None
decode_total_ms = None
decode_step_ms = None
phase_source = "NVTX"

cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='NVTX_EVENTS'")
if cur.fetchone():
    cur.execute("""
        SELECT e.start, e.end, COALESCE(e.text, s.value, '')
        FROM NVTX_EVENTS e
        LEFT JOIN StringIds s ON e.textId = s.id
        WHERE e.end > e.start
    """)
    nvtx_rows = cur.fetchall()
    prefill_ns = [e - s for s, e, txt in nvtx_rows if "prefill_step" in txt.lower()]
    decode_ns = [e - s for s, e, txt in nvtx_rows if "decode_step" in txt.lower()]
    if prefill_ns and decode_ns:
        prefill_ms = sum(prefill_ns) / 1e6
        decode_total_ms = sum(decode_ns) / 1e6
        decode_step_ms = decode_total_ms / len(decode_ns) if len(decode_ns) > 1 else decode_total_ms / decode_steps

if prefill_ms is None or decode_step_ms is None:
    phase_source = "GPU Kernel Timeline Split"
    # Token-wise kernels launch 1 block per token (gridX >= in_len // 2 during prefill,
    # vs. gridX == 1 or K+1 during baseline/speculative decode).
    token_ops = ("rms_norm", "rotary", "reshape_and_cache", "act_and_mul")
    prefill_limit = max(1, int(num_kernels * 0.35))
    prefill_token_idxs = [
        i for i in range(prefill_limit)
        if kernels[i][3] >= (in_len // 2)
        and any(op in kernels[i][2].lower() for op in token_ops)
    ]

    if prefill_token_idxs:
        last_pref_idx = prefill_token_idxs[-1]
        lo = last_pref_idx
        hi = min(num_kernels - 1, last_pref_idx + 40)
    else:
        approx_step_kernels = max(1, num_kernels // expected_out)
        lo = max(1, int(approx_step_kernels * 0.5))
        hi = min(num_kernels - 1, int(approx_step_kernels * 1.6))

    best_split_ns = kernels[lo][1]
    max_gap = -1
    running_end = kernels[0][1]
    for i in range(1, hi + 1):
        s, e, _, _ = kernels[i]
        if i >= lo:
            gap = s - running_end
            if gap > max_gap:
                max_gap = gap
                best_split_ns = running_end
        if e > running_end:
            running_end = e

    prefill_ms = (best_split_ns - w_start) / 1e6
    decode_total_ms = (w_end - best_split_ns) / 1e6
    decode_step_ms = decode_total_ms / decode_steps

wall_ms = wall_ns / 1e6

metrics = {
    "stage": stage_name,
    "workload": {
        "concurrency": 1,
        "input_len": in_len,
        "output_len": expected_out
    },
    "wall_time_ms": round(wall_ms, 2),
    "num_kernels": num_kernels,
    "gpu_active_ms": round(gpu_active_ns / 1e6, 2),
    "gpu_active_pct": round(gpu_active_pct, 2),
    "gpu_idle_ms": round(gpu_idle_ns / 1e6, 2),
    "gpu_idle_pct": round(gpu_idle_pct, 2),
    "cpu_launch_overhead_ms": round(launch_ns / 1e6, 2),
    "cpu_launch_calls": launch_calls,
    "cuda_graph_launches": graph_launches,
    "cuda_sync_wait_ms": round(sync_ns / 1e6, 2),
    "prefill_duration_ms": round(prefill_ms, 2),
    "prefill_wall_pct": round((prefill_ms / wall_ms) * 100.0, 1),
    "decode_total_ms": round(decode_total_ms, 2),
    "decode_step_avg_ms": round(decode_step_ms, 2),
    "kernel_time_breakdown": {
        "linear_ms": round(cat_ns["linear"] / 1e6, 2),
        "linear_pct": round((cat_ns["linear"] / total_kern_ns) * 100.0, 1),
        "attention_ms": round(cat_ns["attention"] / 1e6, 2),
        "attention_pct": round((cat_ns["attention"] / total_kern_ns) * 100.0, 1),
        "norm_act_other_ms": round(cat_ns["norm_act_other"] / 1e6, 2),
        "norm_act_other_pct": round((cat_ns["norm_act_other"] / total_kern_ns) * 100.0, 1)
    },
    "phase_measurement_source": phase_source
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
EOF

echo "----------------------------------------------------------"
echo "Saved GUI trace: ${REP_FILE}"
echo "Saved SQLite:    ${SQLITE_FILE}"
echo "Saved Stats:     ${STATS_TXT}"
echo "Saved JSON:      ${METRICS_JSON}"