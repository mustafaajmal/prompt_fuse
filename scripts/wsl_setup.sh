#!/usr/bin/env bash
# PromptFuse — full WSL2 environment setup (run inside WSL, not PowerShell)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== PromptFuse WSL Setup ==="
echo "Project: $ROOT"

# --- Prerequisites ---
for cmd in python3 nvidia-smi; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    exit 1
  fi
done

if ! nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi failed — ensure NVIDIA drivers + WSL2 CUDA are installed."
fi

# --- Virtualenv ---
if [ ! -d .venv ]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools

echo "Installing PyTorch (CUDA 12.4)..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installing PromptFuse + vLLM + dependencies..."
python -m pip install -r requirements.txt

echo "NLTK tokenizers..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# --- Data ---
if [ ! -f data/synthetic_paraphrases.json ]; then
  python scripts/generate_synthetic_paraphrases.py
fi

# --- Hugging Face (optional: only if HF_TOKEN set) ---
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -n "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN found — logging into Hugging Face and prefetching models..."
  python -m pip install -q huggingface_hub
  HF_TOKEN="$HF_TOKEN" python - <<'PY'
import os
from huggingface_hub import login, snapshot_download

login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)
for repo in (
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "sentence-transformers/all-MiniLM-L6-v2",
):
    print(f"Downloading {repo}...")
    snapshot_download(repo_id=repo)
print("Model prefetch complete.")
PY
else
  echo ""
  echo ">>> HF_TOKEN not set — skipped Llama downloads."
  echo ">>> Add your token to .env then run:  ./scripts/wsl_prefetch_models.sh"
  echo ""
fi

# --- Warm canonical inventory ---
export PROMPTFUSE_CONFIG=configs/demo.yaml
echo "Warming demo canonical inventory..."
if [ -n "${HF_TOKEN:-}" ]; then
  python scripts/warm_demo_inventory.py --config configs/demo.yaml --full-pipeline
else
  python scripts/warm_demo_inventory.py --config configs/demo.yaml --no-compress
fi

# --- Shell helpers ---
chmod +x scripts/*.sh 2>/dev/null || true

echo ""
echo "=== WSL setup complete ==="
echo ""
echo "If you have not yet added HF_TOKEN:"
echo "  cp .env.example .env   # then edit .env"
echo "  ./scripts/wsl_prefetch_models.sh"
echo ""
echo "Start services (two WSL terminals):"
echo "  ./scripts/start_vllm.sh"
echo "  ./scripts/start_promptfuse.sh"
echo ""
echo "Live demo:"
echo "  ./scripts/demo_live.sh"
