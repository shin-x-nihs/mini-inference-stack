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

b = cfg.get("benchmarks", {})
p = b.get("profiling", {})

in_len = p.get("input_len", 1024)
out_len = p.get("output_len", 64)

print(f"MODEL_NAME='{cfg['model']['name']}'")
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
PROBE_JSON="${PROF_DIR}/probe_client.json"
METRICS_JSON="${RES_DIR}/nsys_metrics.json"

echo "=========================================================="
echo " Nsight Systems Timeline Profiler"
echo " Target Stage:   ${STAGE}"
echo " Trace Output:   ${REP_FILE}"
echo " Metrics JSON:   ${METRICS_JSON}"
echo " Workload:       C=1 | In=${PROF_IN} tokens | Out=${PROF_OUT} tokens"
echo "=========================================================="

run_probe() {
    local save_path=$1
    uv run python3 - "${BASE_URL}" "${MODEL_NAME}" "${PROF_IN}" "${PROF_OUT}" "${save_path}" << 'EOF'
import json, sys, time, urllib.request

base_url, model, in_len, out_len, save_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
prompt = " repeat" * max(1, in_len - 5)

payload = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": out_len,
    "temperature": 0.0,
    "ignore_eos": True,
    "stream": True
}).encode("utf-8")

req = urllib.request.Request(
    f"{base_url}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"}
)

t0 = time.perf_counter()
t_first = None
chunks = 0

with urllib.request.urlopen(req, timeout=120) as resp:
    for line in resp:
        line_str = line.decode("utf-8").strip()
        if not line_str.startswith("data: ") or line_str == "data: [DONE]":
            continue
        data = json.loads(line_str[6:])
        delta = data["choices"][0].get("delta", {})
        if delta.get("content"):
            if t_first is None:
                t_first = time.perf_counter()
            chunks += 1

t_end = time.perf_counter()
ttft_ms = (t_first - t0) * 1000.0 if t_first else 0.0
decode_total_ms = (t_end - t_first) * 1000.0 if t_first else 0.0
decode_avg_ms = decode_total_ms / max(1, chunks - 1)

if save_path != "/dev/null":
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "client_wall_ms": (t_end - t0) * 1000.0,
            "client_ttft_ms": ttft_ms,
            "client_decode_avg_ms": decode_avg_ms,
            "output_tokens": chunks
        }, f, indent=2)
EOF
}

# Reset any stuck collection from a previous interrupted run
nsys stop >/dev/null 2>&1 || true

# 3. Warmup pass (unprofiled)
echo ">>> [1/4] Running unprofiled warmup request..."
run_probe "/dev/null"

# 4. Profiled pass (bare start/stop to avoid reconfiguring the launched session)
echo ">>> [2/4] Capturing Nsight Systems trace..."
nsys start
run_probe "${PROBE_JSON}"
nsys stop

LATEST_REP="$(ls -t "${REPO_ROOT}"/report*.nsys-rep 2>/dev/null | head -n 1 || true)"
if [[ -z "${LATEST_REP}" ]]; then
    echo "Error: Could not find generated report*.nsys-rep in ${REPO_ROOT}" >&2
    exit 1
fi
mv -f "${LATEST_REP}" "${REP_FILE}"

# 5. Export to SQLite for exact interval union calculation
echo ">>> [3/4] Exporting trace to SQLite..."
nsys export --type=sqlite --output="${SQLITE_FILE}" --force-overwrite=true "${REP_FILE}"

# 6. Parse SQLite database and compute Table A metrics into JSON
echo ">>> [4/4] Computing timeline metrics (interval union & sync overhead)..."
uv run python3 - "${SQLITE_FILE}" "${PROBE_JSON}" "${METRICS_JSON}" "${PROF_OUT}" << 'EOF'
import json, sqlite3, sys

sqlite_path, probe_path, out_json, expected_out = sys.argv[1:5]
expected_out = int(expected_out)

with open(probe_path, "r", encoding="utf-8") as f:
    probe = json.load(f)

conn = sqlite3.connect(sqlite_path)
cur = conn.cursor()

cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start ASC")
kernels = cur.fetchall()

if not kernels:
    raise RuntimeError("No GPU kernels found in capture. Ensure server was started with 'NSYS=1 ./serve.sh'.")

w_start = kernels[0][0]
w_end = max(k[1] for k in kernels)
wall_ns = max(1, w_end - w_start)

merged = []
for s, e in kernels:
    if not merged or s > merged[-1][1]:
        merged.append([s, e])
    else:
        merged[-1][1] = max(merged[-1][1], e)

gpu_active_ns = sum(e - s for s, e in merged)
gpu_active_pct = (gpu_active_ns / wall_ns) * 100.0
gpu_idle_pct = 100.0 - gpu_active_pct

sync_ns = 0
cur.execute("""
    SELECT r.start, r.end, s.value
    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
    JOIN StringIds s ON r.nameId = s.id
    WHERE r.end >= ? AND r.start <= ?
""", (w_start, w_end))

for s, e, name in cur.fetchall():
    if "Synchronize" in name:
        clipped_s = max(s, w_start)
        clipped_e = min(e, w_end)
        if clipped_e > clipped_s:
            sync_ns += (clipped_e - clipped_s)

prefill_ms = None
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
    if prefill_ns:
        prefill_ms = sum(prefill_ns) / 1e6
    if decode_ns:
        total_dec_ms = sum(decode_ns) / 1e6
        decode_step_ms = total_dec_ms / len(decode_ns) if len(decode_ns) > 1 else total_dec_ms / max(1, expected_out - 1)

if prefill_ms is None or decode_step_ms is None:
    phase_source = "Client Stream Fallback"
    prefill_ms = probe["client_ttft_ms"]
    decode_step_ms = probe["client_decode_avg_ms"]

metrics = {
    "wall_time_ms": round(wall_ns / 1e6, 2),
    "gpu_active_pct": round(gpu_active_pct, 2),
    "gpu_idle_pct": round(gpu_idle_pct, 2),
    "cuda_sync_wait_ms": round(sync_ns / 1e6, 2),
    "prefill_duration_ms": round(prefill_ms, 2),
    "decode_step_avg_ms": round(decode_step_ms, 2),
    "phase_measurement_source": phase_source
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
EOF

rm -f "${SQLITE_FILE}" "${PROBE_JSON}"
echo "----------------------------------------------------------"
echo "Saved GUI trace: ${REP_FILE}"
echo "Saved JSON:      ${METRICS_JSON}"