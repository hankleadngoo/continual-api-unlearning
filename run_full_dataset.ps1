$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$runDir = 'results/codellama_full_4bit'
$checkpoints = 'checkpoints/codellama_full_4bit'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$lock = [System.IO.File]::Open((Join-Path $PSScriptRoot "$runDir/run.lock"),
    [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None)
function Write-Status([string]$Phase, [string]$Message) {
    @{phase=$Phase; message=$Message; pid=$PID; updated=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath "$runDir/status.json" -Encoding UTF8
}
try {
    if (Test-Path "$checkpoints/step_000.pt") {
        throw 'Full run already exists. Refusing to overwrite; inspect checkpoints before resuming.'
    }
    $modelArgs = @('--model', '.cache/models/CodeLlama-7b-hf', '--quantization', '4bit',
        '--device', 'cuda', '--dtype', 'float16', '--max-length', '512')
    Write-Status 'training' 'All 8 tasks; all prepared training and validation samples'
    & $python -u algo.py train @modelArgs --max-samples 0 --validation-samples 0 --output $checkpoints
    if ($LASTEXITCODE -ne 0) { throw "Training failed (exit $LASTEXITCODE). See stderr.log." }
    if (-not (Test-Path "$checkpoints/step_008.pt")) { throw 'Final checkpoint is missing.' }
    Write-Status 'evaluating' 'Final cumulative checkpoint on all valid D_forget and D_test rows'
    & $python -u algo.py evaluate-api @modelArgs --checkpoint "$checkpoints/step_008.pt" --max-samples 0 --max-new-tokens 64 --output "$runDir/api_counts.json"
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed (exit $LASTEXITCODE). See stderr.log." }
    $report = Get-Content "$runDir/api_counts.json" -Raw | ConvertFrom-Json
    if ($null -eq $report.stages[0].datasets.D_forget -or $null -eq $report.stages[0].datasets.D_test) {
        throw 'Evaluation report is missing a dataset.'
    }
    Write-Status 'complete' 'Final checkpoint evaluated on both D_forget and D_test'
} catch {
    Write-Status 'failed' $_.Exception.Message
    throw
} finally {
    $lock.Dispose()
}
