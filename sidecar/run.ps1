# Windows launcher for the SAM3 sidecar (uv + Python 3.12)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
  uv python install 3.12
  uv sync
}

$env:SAM3_BACKEND = if ($env:SAM3_BACKEND) { $env:SAM3_BACKEND } else { "mock" }
$env:PORT = if ($env:PORT) { $env:PORT } else { "9001" }

uv run uvicorn app.main:app --host 0.0.0.0 --port $env:PORT --reload
