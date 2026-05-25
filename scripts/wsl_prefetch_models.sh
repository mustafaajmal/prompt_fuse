#!/usr/bin/env bash
# Download gated Llama weights after you set HF_TOKEN in .env
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

if [ ! -f .env ]; then
  echo "Create .env from .env.example and set HF_TOKEN=hf_..."
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is empty in .env"
  exit 1
fi

python -m pip install -q huggingface_hub

echo "Logging into Hugging Face (via huggingface_hub)..."
python - <<'PY'
import os
from huggingface_hub import login, snapshot_download

token = os.environ["HF_TOKEN"]
login(token=token, add_to_git_credential=True)
print("HF login OK.")

for repo in (
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.1-8B-Instruct",
):
    print(f"Downloading {repo}...")
    snapshot_download(repo_id=repo)
print("Model download complete.")
PY

echo "Re-warming inventory with compress → unify (production path)..."
export PROMPTFUSE_CONFIG=configs/demo.yaml
python scripts/warm_demo_inventory.py --config configs/demo.yaml --full-pipeline

echo ""
echo "Done. Start vLLM: ./scripts/start_vllm.sh"
