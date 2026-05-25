#!/usr/bin/env bash
# Start vLLM with prefix caching (run in WSL terminal 1)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

echo "Starting vLLM: $MODEL on :$PORT (prefix caching on)"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --enable-prefix-caching \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN"
