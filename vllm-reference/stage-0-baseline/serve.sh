
#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root, regardless of the caller's working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Defaults match the model already tested successfully.
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed or is not on PATH." >&2
    exit 1
fi

echo "Starting vLLM reference server"
echo "  Model:             ${MODEL}"
echo "  Address:           ${HOST}:${PORT}"
echo "  Dtype:             ${DTYPE}"
echo "  Max model length:  ${MAX_MODEL_LEN}"
echo "  GPU memory target: ${GPU_MEMORY_UTILIZATION}"
echo
echo "Prefix caching is disabled for the baseline."
echo "GPU clock locking is intentionally not enabled."

exec uv run vllm serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --no-enable-prefix-caching 
