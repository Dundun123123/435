param(
    [ValidateSet('main', 'verify', 'benchmark', 'prepare_data', 'prepare_large_data', 'study', 'summarize_study', 'validate_delivery', 'validate_large')]
    [string]$Task = 'main',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TaskArguments
)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$existingPython = Join-Path $PSScriptRoot '..\..\.venv\Scripts\python.exe'
if (Test-Path $projectPython) { $taskPython = $projectPython }
elseif (Test-Path $existingPython) { $taskPython = $existingPython }
else { throw 'Run setup.ps1 first to install the project environment.' }
Push-Location $PSScriptRoot
try {
    & $taskPython "$Task.py" @TaskArguments
    if ($LASTEXITCODE -ne 0) { throw "Python exited with code $LASTEXITCODE" }
} finally { Pop-Location }
