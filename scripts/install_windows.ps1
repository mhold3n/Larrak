Param(
  [string]$EnvFile = "$(Split-Path $PSScriptRoot -Parent)\environment.yml"
)

Write-Host "[Windows] Larrak installer"

function Has-Cmd($cmd) {
  $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue)
}

if (Has-Cmd "mamba") {
  $pm = "mamba"
} elseif (Has-Cmd "conda") {
  $pm = "conda"
} else {
  Write-Host "Conda/Mamba not found. Install Miniforge or Miniconda first." -ForegroundColor Red
  Write-Host "- Miniforge: https://github.com/conda-forge/miniforge"
  Write-Host "- Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
}

if (-not (Test-Path $EnvFile)) {
  Write-Host "environment.yml not found at $EnvFile" -ForegroundColor Red
  exit 1
}

if ($pm -eq "conda") {
  conda config --set solver libmamba | Out-Null
  conda config --set channel_priority strict | Out-Null
}

# Create local conda environment in project directory
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$LocalEnvPath = Join-Path $ProjectRoot "conda_env_windows"

if (Test-Path $LocalEnvPath) {
  Write-Host "Updating existing local environment at '$LocalEnvPath'..."
  & $pm env update -f $EnvFile --prefix $LocalEnvPath
} else {
  Write-Host "Creating local environment at '$LocalEnvPath'..."
  & $pm env create -f $EnvFile --prefix $LocalEnvPath
}

Write-Host "Verifying IPOPT availability..."
& conda activate $LocalEnvPath
python - << 'PY'
import casadi as ca
try:
    x = ca.SX.sym('x'); f = x**2
    ca.nlpsol('s','ipopt',{'x':x,'f':f})
    print('✓ ipopt is available')
except Exception as e:
    print('✗ ipopt not available:', e)
    raise SystemExit(1)
PY

Write-Host "Done. Activate with: conda activate $LocalEnvPath"
