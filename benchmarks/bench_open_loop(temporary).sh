#!/usr/bin/env bash
set -euo pipefail

# 1. Resolve repository root so paths remain consistent regardless of working directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# 2. Extract configuration values directly from config.yaml
eval "$(uv run python3 -c "
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

print(f\"MODEL_NAME='{cfg['model']['name']}'\")
host = cfg['server'].get('host', '127.0.0.1')
if host == '0.0.0.0': host = '127.0.0.1'
port = cfg['server'].get('port', 8000)
print(f\"BASE_URL='http://{host}:{port}'\")
print(f\"INPUT_LEN={cfg['benchmarks']['input_len']}\")
print(f\"OUTPUT_LEN={cfg['benchmarks']['output_len']}\")
print(f\"SEED={cfg['benchmarks']['random_seed']}\")
")"

# 3. Determine target results directory (defaults to stage-0-vllm)
STAGE="${1:-stage-0-vllm}"
OUT_DIR="benchmarks/results/${STAGE}"
mkdir -p "${OUT_DIR}"

# Offered arrival rates (requests per second) to test queue buildup
QPS_RATES=(1 2 4 6 8)
NUM_PROMPTS=50  # Fixed volume of requests across each arrival rate test

echo "=========================================================="
echo " Open-Loop Arrival Rate (Queue Buildup) Benchmark"
echo " Target Stage:   ${STAGE}"
echo " Output Dir:     ${OUT_DIR}"
echo " Model:          ${MODEL_NAME}"
echo " Target URL:     ${BASE_URL}"
echo " Prompt Length:  ${INPUT_LEN} in / ${OUTPUT_LEN} out"
echo " Determinism:    Seed ${SEED} (--ignore-eos)"
echo " Fixed Prompts:  ${NUM_PROMPTS} per rate"
echo " Arrival Rates:  ${QPS_RATES[*]} req/s"
echo "=========================================================="

for QPS in "${QPS_RATES[@]}"; do
  echo ""
  echo "--- Testing Arrival Rate QPS = ${QPS} (${NUM_PROMPTS} requests) ---"

  uv run vllm bench serve \
    --backend openai-chat \
    --base-url "${BASE_URL}" \
    --endpoint /v1/chat/completions \
    --model "${MODEL_NAME}" \
    --dataset-name random \
    --random-input-len "${INPUT_LEN}" \
    --random-output-len "${OUTPUT_LEN}" \
    --ignore-eos \
    --seed "${SEED}" \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${QPS}" \
    --percentile-metrics ttft,tpot,itl \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir "${OUT_DIR}" \
    --result-filename "bench_qps${QPS}.json"
done

echo ""
echo "Benchmark complete. Results saved in: ${OUT_DIR}"