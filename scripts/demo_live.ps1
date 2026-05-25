# Live in-class demo helper (Windows PowerShell)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error "Run scripts\setup.ps1 first (.venv missing)"
}

$VLLM_URL = if ($env:VLLM_URL) { $env:VLLM_URL } else { "http://localhost:8000" }
$PF_URL = if ($env:PF_URL) { $env:PF_URL } else { "http://localhost:8080" }

Write-Host "=== PromptFuse Live Demo ===" -ForegroundColor Cyan

function Test-Health($Url, $Name) {
    try {
        Invoke-RestMethod -Uri "$Url/health" -Method Get -TimeoutSec 5 | Out-Null
        Write-Host "  OK $Name at $Url" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "  FAIL $Name not reachable at $Url" -ForegroundColor Red
        return $false
    }
}

if (-not (Test-Health $VLLM_URL "vLLM")) {
    Write-Host 'Start vLLM: python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --enable-prefix-caching --port 8000'
    exit 1
}
if (-not (Test-Health $PF_URL "PromptFuse")) {
    Write-Host 'Start PromptFuse: $env:PROMPTFUSE_CONFIG="configs\demo.yaml"; promptfuse-serve'
    exit 1
}

$env:PROMPTFUSE_CONFIG = "configs\demo.yaml"
& $VenvPython scripts\warm_demo_inventory.py --config configs\demo.yaml

$promptA = "You are a document assistant. Background: Users paste long articles; redundant boilerplate is common. Policy: preserve facts. Format: complete sentences.`n`nTask: Summarize the following paragraph in three sentences."

$bodyA = @{
    messages = @(@{ role = "user"; content = $promptA })
    max_tokens = 64
    temperature = 0
} | ConvertTo-Json -Depth 5

Write-Host "`nParaphrase A (first request)..."
$rA = Invoke-RestMethod -Uri "$PF_URL/v1/chat/completions" -Method Post -ContentType "application/json" -Body $bodyA
Write-Host ("  Output: " + $rA.choices[0].message.content.Substring(0, [Math]::Min(120, $rA.choices[0].message.content.Length)) + "...")

$promptB = "You are a document assistant. Background: Users paste long articles; redundant boilerplate is common. Policy: preserve facts. Format: complete sentences.`n`nTask: Please give a three-sentence summary of the text below."

$bodyB = @{
    messages = @(@{ role = "user"; content = $promptB })
    max_tokens = 64
    temperature = 0
} | ConvertTo-Json -Depth 5

Write-Host "`nParaphrase B (expect unifier hit)..."
$rB = Invoke-RestMethod -Uri "$PF_URL/v1/chat/completions" -Method Post -ContentType "application/json" -Body $bodyB
Write-Host ("  Output: " + $rB.choices[0].message.content.Substring(0, [Math]::Min(120, $rB.choices[0].message.content.Length)) + "...")

Write-Host "`nPromptFuse stats:"
Invoke-RestMethod -Uri "$PF_URL/stats" | ConvertTo-Json -Depth 5

Write-Host "`n=== Demo complete ===" -ForegroundColor Green
Write-Host "Full experiment: .\venv\Scripts\python.exe scripts\run_demo_experiment.py --config configs\demo.yaml"
