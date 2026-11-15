$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create virtual environment.' }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Could not install dependencies.' }
} finally { Pop-Location }
