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

d = cfg.get("draft_model", {})
b = cfg["benchmarks"]
p = b["pareto"]
c = b["context_sweep"]

concurrencies = " ".join(map(str, p["concurrencies"]))
lengths = " ".join(map(str, c["lengths"]))

print(f"MODEL_NAME='{cfg['model']['name']}'")
print(f"DRAFT_MODEL='{d.get('name', 'Qwen/Qwen2.5-Coder-0.5B-Instruct')}'")
print(f"NUM_SPEC_TOKENS={d.get('num_speculative_tokens', 5)}")
print(f"BASE_URL='http://{host}:{port}'")
print(f"SEED={b['random_seed']}")
print(f"TEMPERATURE={b.get('temperature', 0.0)}")
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

TMP_JSONL="${OUT_DIR}/_tmp_prompt.jsonl"
trap 'rm -f "${TMP_JSONL}"' EXIT

# Build a custom JSONL file repeating/slicing benchmarks.prompt to exact token length
build_prompt_jsonl() {
    local target_len=$1
    local num_prompts=$2
    uv run python3 - "${target_len}" "${num_prompts}" "${TMP_JSONL}" << 'EOF'
import json, sys, yaml
from transformers import AutoTokenizer

target_len, num_prompts, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model_name = cfg["model"]["name"]
base_prompt = cfg["benchmarks"].get("prompt", "Write a Python module to benchmark matrix multiplication.").strip()

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
reps = (target_len // max(1, len(base_ids))) + 2
exact_ids = (base_ids * reps)[:target_len]
prompt_text = tokenizer.decode(exact_ids, skip_special_tokens=True)

with open(out_path, "w", encoding="utf-8") as out:
    for _ in range(num_prompts):
        out.write(json.dumps({"prompt": prompt_text}) + "\n")
EOF
}

run_vllm_bench() {
    local in_len=$1
    local out_len=$2
    local conc=$3
    local num_prompts=$4
    local filename=$5

    build_prompt_jsonl "${in_len}" "${num_prompts}"

    uv run vllm bench serve \
        --backend openai-chat \
        --base-url "${BASE_URL}" \
        --endpoint /v1/chat/completions \
        --model "${MODEL_NAME}" \
        --dataset-name custom \
        --dataset-path "${TMP_JSONL}" \
        --custom-output-len "${out_len}" \
        --skip-chat-template \
        --temperature "${TEMPERATURE}" \
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
echo " Target Model:   ${MODEL_NAME}"
if [[ "${STAGE}" == *"spec"* || "${STAGE}" == "stage-1"* ]]; then
    echo " Draft Model:    ${DRAFT_MODEL} (K=${NUM_SPEC_TOKENS})"
fi
echo " Temperature:    ${TEMPERATURE}"
echo " Target URL:     ${BASE_URL}"
echo "=========================================================="

# Unmeasured warmup request to ensure dual-model/speculative buffers are initialized
echo ">>> Running warmup probe..."
curl -s -o /dev/null "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup\"}],\"max_tokens\":16,\"temperature\":${TEMPERATURE}}" || true

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