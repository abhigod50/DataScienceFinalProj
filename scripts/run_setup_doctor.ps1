$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonCandidates = @(
    (Join-Path $projectRoot '.venv\Scripts\python.exe'),
    (Join-Path (Split-Path $projectRoot -Parent) '.venv\Scripts\python.exe')
)
$python = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) {
    $python = 'python'
}

Push-Location $projectRoot
try {
    Write-Host "Using Python: $python"
    & $python 'setup_doctor.py' '--quick'
} finally {
    Pop-Location
}