#!/usr/bin/env bash
# One-time WSL packages for vLLM (Python.h + compiler for Triton JIT)
set -euo pipefail

PYVER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 3.12)"

echo "Installing python${PYVER}-dev, python3-dev, build-essential..."
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  "python${PYVER}-dev" \
  python3-dev \
  build-essential

echo "Verify Python.h:"
ls -la "/usr/include/python${PYVER}/Python.h"
echo "Done. Re-run: ./scripts/start_vllm.sh"
