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

# PromptFuse vars — not vLLM settings (avoids "Unknown vLLM environment variable VLLM_URL")
unset VLLM_URL PF_URL

MODEL="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${VLLM_PORT:-8000}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

# WSL2: avoid FlashInfer JIT sampler issues
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

_pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
_pyinc="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')"

need_dev_headers() {
  [ ! -f "${_pyinc}/Python.h" ] && [ ! -f "/usr/include/python${_pyver}/Python.h" ]
}

install_dev_headers() {
  echo "Python development headers missing (required for vLLM / Triton JIT)."
  echo "Installing python${_pyver}-dev and build-essential (sudo)..."
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    "python${_pyver}-dev" \
    python3-dev \
    build-essential
}

if need_dev_headers; then
  if command -v apt-get >/dev/null 2>&1; then
    install_dev_headers
  else
    echo "ERROR: Python.h not found. On Ubuntu/WSL run:"
    echo "  sudo apt-get install -y python${_pyver}-dev build-essential"
    exit 1
  fi
fi

# Help gcc find headers when venv Python is used
if [ -f "/usr/include/python${_pyver}/Python.h" ]; then
  export CFLAGS="${CFLAGS:-} -I/usr/include/python${_pyver}"
  export CPPFLAGS="${CPPFLAGS:-} -I/usr/include/python${_pyver}"
fi

if need_dev_headers; then
  echo "ERROR: Still no Python.h after apt install. Check: ls /usr/include/python${_pyver}/"
  exit 1
fi

echo "Starting vLLM: $MODEL on :$PORT (prefix caching, enforce-eager for WSL)"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --enable-prefix-caching \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --enforce-eager
