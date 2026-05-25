# Start vLLM with prefix caching (Windows)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Run scripts\setup.ps1 first"
}

$Model = if ($env:VLLM_MODEL) { $env:VLLM_MODEL } else { "meta-llama/Llama-3.1-8B-Instruct" }
$Port = if ($env:VLLM_PORT) { $env:VLLM_PORT } else { "8000" }
$GpuUtil = if ($env:VLLM_GPU_UTIL) { $env:VLLM_GPU_UTIL } else { "0.85" }

Write-Host "Starting vLLM: $Model on port $Port"
& $Python -m vllm.entrypoints.openai.api_server `
  --model $Model `
  --enable-prefix-caching `
  --port $Port `
  --gpu-memory-utilization $GpuUtil `
  --max-model-len 4096
