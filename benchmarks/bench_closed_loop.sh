#!/usr/bin/env bash
set -euo pipefail

# 1. Resolve repository root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# 2. Extract configuration values directly from config.yaml
eval "$(uv run python3 - << 'EOF'
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

host = cfg["server"].get("host", "127.0.0.1")
if host == "0.0.0.0":
    host = "127.0.0.1"
port = cfg["server"].get("port", 8000)

b = cfg["benchmarks"]
p = b["pareto"]
c = b["context_sweep"]

concurrencies = " ".join(map(str, p["concurrencies"]))
lengths = " ".join(map(str, c["lengths"]))

print(f"MODEL_NAME='{cfg['model']['name']}'")
print(f"BASE_URL='http://{host}:{port}'")
print(f"SEED={b['random_seed']}")
print(f"PARETO_IN={p['input_len']}")
print(f"PARETO_OUT={p['output_len']}")
print(f"CONCURRENCIES=({concurrencies})")
print(f"CTX_LENGTHS=({lengths})")
print(f"CTX_OUT={c['output_len']}")
print(f"CTX_PROMPTS={c['num_prompts']}")
EOF
)"

# 3. Arguments: ./bench_closed_loop.sh [stage_name] [all|pareto|context]
STAGE="${1:-stage-0-vllm}"
MODE="${2:-all}"
OUT_DIR="benchmarks/results/${STAGE}"
mkdir -p "${OUT_DIR}"

run_vllm_bench() {
    local in_len=$1
    local out_len=$2
    local conc=$3
    local num_prompts=$4
    local filename=$5

    uv run vllm bench serve \
        --backend openai-chat \
        --base-url "${BASE_URL}" \
        --endpoint /v1/chat/completions \
        --model "${MODEL_NAME}" \
        --dataset-name random \
        --random-input-len "${in_len}" \
        --random-output-len "${out_len}" \
        --ignore-eos \
        --seed "${SEED}" \
        --max-concurrency "${conc}" \
        --num-prompts "${num_prompts}" \
        --request-rate inf \
        --percentile-metrics ttft,tpot,itl \
        --metric-percentiles 50,90,99 \
        --save-result \
        --result-dir "${OUT_DIR}" \
        --result-filename "${filename}"
}

echo "=========================================================="
echo " Inference Benchmark Suite"
echo " Target Stage:   ${STAGE}"
echo " Mode:           ${MODE}"
echo " Output Dir:     ${OUT_DIR}"
echo " Model:          ${MODEL_NAME}"
echo " Target URL:     ${BASE_URL}"
echo "=========================================================="

# Sweep 1: Plot 1 (System Capacity / Pareto Frontier)
if [[ "${MODE}" == "all" || "${MODE}" == "pareto" ]]; then
    echo ""
    echo ">>> [1/2] Running Pareto Concurrency Sweep (In=${PARETO_IN}, Out=${PARETO_OUT})"
    for C in "${CONCURRENCIES[@]}"; do
        NUM_PROMPTS=$(( C * 10 ))
        if [ "${NUM_PROMPTS}" -lt 20 ]; then NUM_PROMPTS=20; fi

        echo "--- Concurrency C = ${C} (${NUM_PROMPTS} requests) ---"
        run_vllm_bench "${PARETO_IN}" "${PARETO_OUT}" "${C}" "${NUM_PROMPTS}" "pareto_c${C}.json"
    done
fi

# Sweep 2: Plots 2 & 3 (Prefill TTFT & Decode TPOT across Context Lengths at C=1)
if [[ "${MODE}" == "all" || "${MODE}" == "context" ]]; then
    echo ""
    echo ">>> [2/2] Running Context Length Sweep at C=1 (Out=${CTX_OUT}, Prompts=${CTX_PROMPTS})"
    for L in "${CTX_LENGTHS[@]}"; do
        echo "--- Context Length L = ${L} tokens ---"
        run_vllm_bench "${L}" "${CTX_OUT}" 1 "${CTX_PROMPTS}" "context_l${L}.json"
    done
fi

echo ""
echo "Benchmark complete. JSON files saved in: ${OUT_DIR}"