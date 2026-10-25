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

# Ingest defaults from config.yaml safely via heredoc
eval "$(uv run python - "${CONFIG_FILE}" << 'EOF'
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
print(f'CFG_GPU_MEM="{s.get("gpu_memory_utilization", 0.70)}"')
print(f'CFG_PREFIX_CACHE="{s.get("enable_prefix_caching", False)}"')
print(f'CFG_TEMPERATURE="{b.get("temperature", 0.0)}"')
EOF
)"

# Apply hierarchy: CLI Environment Variable > config.yaml value
MODEL="${VLLM_MODEL:-$CFG_MODEL}"
HOST="${VLLM_HOST:-$CFG_HOST}"
PORT="${VLLM_PORT:-$CFG_PORT}"
DTYPE="${VLLM_DTYPE:-$CFG_DTYPE}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$CFG_MAX_MODEL_LEN}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-$CFG_GPU_MEM}"
PREFIX_CACHE="${VLLM_PREFIX_CACHE:-$CFG_PREFIX_CACHE}"
TEMPERATURE="${VLLM_TEMPERATURE:-$CFG_TEMPERATURE}"

# Build prefix caching flag based on resolved config
PREFIX_CACHE_FLAG="--no-enable-prefix-caching"
if [[ "${PREFIX_CACHE}" == "True" || "${PREFIX_CACHE}" == "true" || "${PREFIX_CACHE}" == "1" ]]; then
    PREFIX_CACHE_FLAG="--enable-prefix-caching"
fi

GEN_CONFIG=$(printf '{"temperature": %s}' "${TEMPERATURE}")

echo "Starting vLLM reference server (Stage 0: Baseline)"
echo "  Configuration file:  ${CONFIG_FILE}"
echo "  Model:               ${MODEL}"
echo "  Address:             ${HOST}:${PORT}"
echo "  Dtype:               ${DTYPE}"
echo "  Max model length:    ${MAX_MODEL_LEN}"
echo "  GPU memory target:   ${GPU_MEMORY_UTILIZATION}"
echo "  Default temperature: ${TEMPERATURE}"
echo "  Prefix caching flag: ${PREFIX_CACHE_FLAG}"

NSYS_CMD=()
if [[ "${NSYS:-0}" == "ON" || "${NSYS:-0}" == "1" ]]; then
    echo "  Profiler mode:       ENABLED (nsys launch dormant session)"
    NSYS_CMD=(nsys launch --trace=cuda,nvtx --cuda-graph-trace=node)
fi

exec "${NSYS_CMD[@]}" uv run vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --override-generation-config "${GEN_CONFIG}" \
    "${PREFIX_CACHE_FLAG}"