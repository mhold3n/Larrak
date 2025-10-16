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

$envName = "larrak"
$envs = & $pm env list
if ($envs -match "^$envName\s") {
  Write-Host "Updating existing environment '$envName'..."
  & $pm env update -f $EnvFile --name $envName
} else {
  Write-Host "Creating environment '$envName'..."
  & $pm env create -f $EnvFile
}

Write-Host "Verifying IPOPT availability..."
& conda activate $envName
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

Write-Host "Done. Activate with: conda activate $envName"
