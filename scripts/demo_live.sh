#!/usr/bin/env bash
# Live in-class demo helper for PromptFuse
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "ERROR: python3 not found. Run: source .venv/bin/activate"
  exit 1
fi

export PROMPTFUSE_CONFIG="${PROMPTFUSE_CONFIG:-configs/demo.yaml}"
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
  echo "    Start with: ./scripts/start_vllm.sh"
  exit 1
fi

if curl -sf "$PF_URL/health" >/dev/null 2>&1; then
  echo "  ✓ PromptFuse running at $PF_URL"
else
  echo "  ✗ PromptFuse not reachable at $PF_URL"
  echo "    Start with: ./scripts/start_promptfuse.sh"
  exit 1
fi

echo ""
echo "[2/5] Demo inventory..."
if [ "${PROMPTFUSE_WARM:-0}" = "1" ]; then
  echo "  Rebuilding inventory on disk (PROMPTFUSE_WARM=1)..."
  "$PYTHON" scripts/warm_demo_inventory.py --config configs/demo.yaml
else
  echo "  Using inventory already loaded by PromptFuse (set PROMPTFUSE_WARM=1 to rebuild)"
fi

echo ""
echo "[3/5] Paraphrase A (first request — expect cache miss)..."
"$PYTHON" - "$PF_URL" <<'PY'
import json, sys, urllib.request
url = sys.argv[1]
prompt = (
    "You are a document assistant. Background: Users paste long articles; "
    "redundant boilerplate is common. Policy: preserve facts. Format: complete sentences.\n\n"
    "Task: Summarize the following paragraph in three sentences."
)
body = json.dumps({"messages":[{"role":"user","content":prompt}],"max_tokens":64,"temperature":0}).encode()
req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    d = json.load(r)
print("  Output:", d["choices"][0]["message"]["content"][:120] + "...")
PY

echo ""
echo "[4/5] Paraphrase B (same meaning — expect unifier hit)..."
"$PYTHON" - "$PF_URL" <<'PY'
import json, sys, urllib.request
url = sys.argv[1]
prompt = (
    "You are a document assistant. Background: Users paste long articles; "
    "redundant boilerplate is common. Policy: preserve facts. Format: complete sentences.\n\n"
    "Task: Please give a three-sentence summary of the text below."
)
body = json.dumps({"messages":[{"role":"user","content":prompt}],"max_tokens":64,"temperature":0}).encode()
req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    d = json.load(r)
print("  Output:", d["choices"][0]["message"]["content"][:120] + "...")
PY

echo ""
echo "[5/5] PromptFuse stats..."
curl -s "$PF_URL/stats" | "$PYTHON" -m json.tool

echo ""
echo "=== Demo complete ==="
echo "Full experiment: $PYTHON scripts/run_demo_experiment.py --config configs/demo.yaml"
