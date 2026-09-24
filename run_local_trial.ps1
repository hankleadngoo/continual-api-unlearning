param(
    [string]$Output = "results/numpy_4bit_trial"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
$logDir = Join-Path $PSScriptRoot ".cache/local_trial"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$statusFile = Join-Path $logDir "status.json"

function Write-Status([string]$Phase, [string]$Message) {
    @{ phase = $Phase; message = $Message; pid = $PID;
       updated = (Get-Date).ToString("o"); output = $Output } |
        ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8
}

# Keep a second invocation from downloading/assembling the same files concurrently.
$lock = $null
try {
    $lock = [System.IO.File]::Open((Join-Path $logDir "run.lock"),
        [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None)
} catch {
    throw "A local trial is already running. See .cache/local_trial/status.json."
}

try {
    if (-not (Test-Path -LiteralPath $python)) { throw "Missing project .venv/Scripts/python.exe" }
    if (Test-Path -LiteralPath (Join-Path $Output "baseline.json")) {
        throw "Output already contains a baseline. Choose a new -Output directory."
    }
    Write-Status "downloading" "Resuming and verifying pinned CodeLlama weights"
    & $python -u download_codellama.py
    if ($LASTEXITCODE -ne 0) { throw "Model download failed (exit $LASTEXITCODE). Rerun to resume." }

    Write-Status "evaluating" "NumPy: 100 training rows; paired evaluation on 50 validation rows"
    & $python -u algo.py trial --model .cache/models/CodeLlama-7b-hf --output $Output
    if ($LASTEXITCODE -ne 0) { throw "Trial failed (exit $LASTEXITCODE). Inspect stderr.log." }
    if (-not (Test-Path -LiteralPath (Join-Path $Output "comparison.json"))) {
        throw "Trial exited without comparison.json."
    }
    Write-Status "complete" "Evaluation saved to $Output/comparison.json"
} catch {
    Write-Status "failed" $_.Exception.Message
    throw
} finally {
    $lock.Dispose()
}
