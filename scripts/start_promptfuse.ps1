# Start PromptFuse middleware (Windows)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:PROMPTFUSE_CONFIG = "configs\demo.yaml"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Run scripts\setup.ps1 first"
}

Write-Host "Starting PromptFuse (config=$env:PROMPTFUSE_CONFIG)"
& $Python -m promptfuse.middleware.server
