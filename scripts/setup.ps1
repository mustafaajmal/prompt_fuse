# PromptFuse one-time setup (Windows / PowerShell)
# Usage: .\scripts\setup.ps1
# Requires: Python 3.10+ on PATH, NVIDIA driver + CUDA for GPU workloads

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "=== PromptFuse Setup ===" -ForegroundColor Cyan
Write-Host "Project root: $Root"

function Find-Python {
    foreach ($cmd in @("python", "python3")) {
        $exe = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($exe) { return $exe.Source }
    }
    $venvPy = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venvPy) { return $venvPy }
    throw "Python not found. Install Python 3.10+ and add to PATH, or create .venv manually."
}

$Python = Find-Python
Write-Host "Using Python: $Python"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    & $Python -m venv .venv
}
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip wheel setuptools

Write-Host "Installing PyTorch (CUDA 12.4 wheel)..."
& $VenvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installing PromptFuse + dependencies..."
& $VenvPython -m pip install -r requirements.txt

Write-Host "Downloading NLTK punkt tokenizer..."
& $VenvPython -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

Write-Host "Ensuring synthetic paraphrase dataset exists..."
if (-not (Test-Path "data\synthetic_paraphrases.json")) {
    & $VenvPython scripts\generate_synthetic_paraphrases.py
}

Write-Host "Warming demo canonical inventory..."
$env:PROMPTFUSE_CONFIG = "configs\demo.yaml"
& $VenvPython scripts\warm_demo_inventory.py --config configs\demo.yaml

Write-Host ""
Write-Host "=== Setup complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. huggingface-cli login   (accept Llama 3.1 / 3.2 licenses on HF)"
Write-Host "  2. Terminal 1 - vLLM:"
Write-Host '     .\.venv\Scripts\python.exe -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --enable-prefix-caching --port 8000 --gpu-memory-utilization 0.85'
Write-Host "  3. Terminal 2 - PromptFuse:"
Write-Host '     $env:PROMPTFUSE_CONFIG="configs\demo.yaml"; .\.venv\Scripts\promptfuse-serve'
Write-Host "  4. Eval (no vLLM):"
Write-Host '     .\.venv\Scripts\python.exe scripts\run_full_eval.py --no-vllm'
Write-Host "  5. Live demo:"
Write-Host '     .\scripts\demo_live.ps1'
