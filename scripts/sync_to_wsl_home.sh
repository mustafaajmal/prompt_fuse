#!/usr/bin/env bash
# Copy WSL setup scripts from this repo into ~/prompt_fuse (if you use a separate WSL clone)
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="${1:-$HOME/prompt_fuse}"

if [ ! -d "$DST" ]; then
  echo "Destination not found: $DST"
  exit 1
fi

echo "Syncing $SRC -> $DST"
mkdir -p "$DST/scripts"

cp -v "$SRC/scripts/wsl_setup.sh" \
      "$SRC/scripts/wsl_prefetch_models.sh" \
      "$SRC/scripts/start_vllm.sh" \
      "$SRC/scripts/start_promptfuse.sh" \
      "$SRC/scripts/demo_live.sh" \
      "$SRC/scripts/warm_demo_inventory.py" \
      "$DST/scripts/"

cp -v "$SRC/WSL.md" "$SRC/.env.example" "$DST/" 2>/dev/null || true
chmod +x "$DST/scripts/"*.sh
echo "Done. Run: cd $DST && ./scripts/wsl_prefetch_models.sh"
