#!/usr/bin/env bash
# Live in-class demo helper for PromptFuse
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
PF_URL="${PF_URL:-http://localhost:8080}"

echo "=== PromptFuse Live Demo ==="
echo ""

# Health checks
echo "[1/5] Health checks..."
if curl -sf "$VLLM_URL/health" >/dev/null 2>&1; then
  echo "  ✓ vLLM running at $VLLM_URL"
else
  echo "  ✗ vLLM not reachable at $VLLM_URL"
  echo "    Start with: python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --enable-prefix-caching --port 8000"
  exit 1
fi

if curl -sf "$PF_URL/health" >/dev/null 2>&1; then
  echo "  ✓ PromptFuse running at $PF_URL"
else
  echo "  ✗ PromptFuse not reachable at $PF_URL"
  echo "    Start with: promptfuse-serve  (or: python -m promptfuse.middleware.server)"
  exit 1
fi

echo ""
echo "[2/5] Warm demo inventory..."
python scripts/warm_demo_inventory.py --config configs/demo.yaml

echo ""
echo "[3/5] Paraphrase A (first request — expect cache miss)..."
curl -s "$PF_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Summarize the following paragraph in three sentences."}],
    "max_tokens": 64,
    "temperature": 0
  }' | python -c "import sys,json; d=json.load(sys.stdin); print('  Output:', d['choices'][0]['message']['content'][:120]+'...')"

echo ""
echo "[4/5] Paraphrase B (same meaning — expect unifier hit)..."
curl -s "$PF_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":"Please give a three-sentence summary of the text below."}],
    "max_tokens": 64,
    "temperature": 0
  }' | python -c "import sys,json; d=json.load(sys.stdin); print('  Output:', d['choices'][0]['message']['content'][:120]+'...')"

echo ""
echo "[5/5] PromptFuse stats..."
curl -s "$PF_URL/stats" | python -m json.tool

echo ""
echo "=== Demo complete ==="
echo "Run full experiment: python scripts/run_demo_experiment.py --config configs/demo.yaml"
