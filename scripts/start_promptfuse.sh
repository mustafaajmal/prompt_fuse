#!/usr/bin/env bash
# Start PromptFuse middleware (run in WSL terminal 2 — after vLLM is healthy on :8000)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export PROMPTFUSE_CONFIG="${PROMPTFUSE_CONFIG:-configs/demo.yaml}"
# Do not load Llama-3.2-1B at startup (vLLM already uses GPU); loads on first request
export PROMPTFUSE_PRELOAD_COMPRESSOR="${PROMPTFUSE_PRELOAD_COMPRESSOR:-0}"

echo "PromptFuse config: $PROMPTFUSE_CONFIG"
grep -E '^\s*vllm_base_url:|^\s*port:' "$PROMPTFUSE_CONFIG" 2>/dev/null || true
echo ""
echo "Starting PromptFuse (Python -u, unbuffered logs)..."
echo "  Expect: bi-encoder load → 'Uvicorn running on http://0.0.0.0:8080' (~30s–2min)"
echo "  Then:   curl http://localhost:8080/health"
echo ""

exec python -u -m promptfuse.middleware.server
