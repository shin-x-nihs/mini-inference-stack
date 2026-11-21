#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_FILE="${REPO_ROOT}/config.yaml"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed or is not on PATH." >&2
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Error: Configuration file not found at ${CONFIG_FILE}" >&2
    exit 1
fi

# Ingest shared defaults from root config.yaml safely via heredoc
eval "$(uv run --frozen python - "${CONFIG_FILE}" << 'EOF'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

m = cfg.get("model", {})
s = cfg.get("server", {})
b = cfg.get("benchmarks", {})

print(f'CFG_MODEL="{m.get("name", "Qwen/Qwen2.5-Coder-1.5B-Instruct")}"')
print(f'CFG_DTYPE="{m.get("dtype", "bfloat16")}"')
print(f'CFG_MAX_MODEL_LEN="{m.get("max_model_len", 9216)}"')
print(f'CFG_HOST="{s.get("host", "0.0.0.0")}"')
print(f'CFG_PORT="{s.get("port", 8000)}"')
print(f'CFG_TEMPERATURE="{b.get("temperature", 0.0)}"')
EOF
)"

# Apply hierarchy: CLI Environment Variable > root config.yaml
MODEL="${CUSTOM_MODEL:-$CFG_MODEL}"
HOST="${CUSTOM_HOST:-$CFG_HOST}"
PORT="${CUSTOM_PORT:-$CFG_PORT}"
DTYPE="${CUSTOM_DTYPE:-$CFG_DTYPE}"
MAX_MODEL_LEN="${CUSTOM_MAX_MODEL_LEN:-$CFG_MAX_MODEL_LEN}"
TEMPERATURE="${CUSTOM_TEMPERATURE:-$CFG_TEMPERATURE}"
NUM_KV_BLOCKS="${CUSTOM_NUM_KV_BLOCKS:-4096}"
MAX_BATCH_SIZE="${CUSTOM_MAX_BATCH_SIZE:-64}"

export CUSTOM_MODEL="${MODEL}"
export CUSTOM_HOST="${HOST}"
export CUSTOM_PORT="${PORT}"
export CUSTOM_DTYPE="${DTYPE}"
export CUSTOM_MAX_MODEL_LEN="${MAX_MODEL_LEN}"
export CUSTOM_TEMPERATURE="${TEMPERATURE}"
export CUSTOM_NUM_KV_BLOCKS="${NUM_KV_BLOCKS}"
export CUSTOM_MAX_BATCH_SIZE="${MAX_BATCH_SIZE}"

echo "Starting custom inference server (Stage 5: Operator Fusion)"
echo "  Configuration file:  ${CONFIG_FILE}"
echo "  Model:               ${MODEL}"
echo "  Address:             ${HOST}:${PORT}"
echo "  Dtype:               ${DTYPE}"
echo "  Max model length:    ${MAX_MODEL_LEN}"
echo "  Default temperature: ${TEMPERATURE}"
echo "  Num KV blocks:       ${NUM_KV_BLOCKS}"
echo "  Max batch size:      ${MAX_BATCH_SIZE}"

NSYS_CMD=()
if [[ "${NSYS:-0}" == "ON" || "${NSYS:-0}" == "1" ]]; then
    echo "  Profiler mode:       ENABLED (nsys launch dormant session)"
    NSYS_CMD=(nsys launch --trace=cuda,nvtx --cuda-graph-trace=node)
fi

exec "${NSYS_CMD[@]}" uv run --frozen uvicorn server:app \
    --app-dir "${SCRIPT_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers 1 \
    "$@"
