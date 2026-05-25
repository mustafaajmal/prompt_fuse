#!/usr/bin/env bash
# PromptFuse one-time setup (Linux / WSL2)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== PromptFuse Setup ==="

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found"
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

if [ ! -f data/synthetic_paraphrases.json ]; then
  python scripts/generate_synthetic_paraphrases.py
fi

export PROMPTFUSE_CONFIG=configs/demo.yaml
python scripts/warm_demo_inventory.py --config configs/demo.yaml

echo ""
echo "=== Setup complete ==="
echo "Next: huggingface-cli login, start vLLM, then promptfuse-serve"
echo "Eval:  python scripts/run_full_eval.py --no-vllm"
