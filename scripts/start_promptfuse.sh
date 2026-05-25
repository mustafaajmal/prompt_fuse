#!/usr/bin/env bash
# Start PromptFuse middleware (run in WSL terminal 2)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

export PROMPTFUSE_CONFIG="${PROMPTFUSE_CONFIG:-configs/demo.yaml}"

echo "PromptFuse config: $PROMPTFUSE_CONFIG"
echo "Proxy → vLLM at $(grep vllm_base_url "$PROMPTFUSE_CONFIG" 2>/dev/null || echo configs/demo.yaml)"

exec python -m promptfuse.middleware.server
