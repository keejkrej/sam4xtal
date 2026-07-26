# Windows launcher for the SAM3 sidecar (uv + Python 3.12)
#   .\run.ps1            → real Hugging Face SAM (transformers + CUDA)
#   .\run.ps1 --mock     → flood-fill stub (no model weights)
#   .\run.ps1 --reload   → enable uvicorn auto-reload (dev only)
param(
  [switch]$Mock,
  [switch]$Reload
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
  uv python install 3.12
  uv sync
}

if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $k, $v = $_ -split '=', 2
    if (-not $k -or -not $v) { return }
    $name = $k.Trim()
    if ($name -eq "SAM3_BACKEND") { return }
    if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
      Set-Item -Path "Env:$name" -Value $v.Trim()
    }
  }
}

if ($Mock) {
  $env:SAM3_BACKEND = "mock"
} else {
  $env:SAM3_BACKEND = "transformers"
}

if (-not $env:PORT) { $env:PORT = "9001" }
$port = [int]$env:PORT

$listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {
  $pids = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
  Write-Error @"
Port $port is already in use by PID(s): $($pids -join ', ').
A previous sidecar is likely still answering requests (often the mock backend).
Stop it first, e.g.:
  taskkill /F /T /PID $($pids[0])
Then re-run .\run.ps1
"@
  exit 1
}

Write-Host "SAM3_BACKEND=$($env:SAM3_BACKEND) PORT=$port"
$uviArgs = @(
  "run", "uvicorn", "app.main:app",
  "--host", "0.0.0.0",
  "--port", "$port"
)
if ($Reload) { $uviArgs += "--reload" }
& uv @uviArgs
