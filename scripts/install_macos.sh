#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$PROJECT_DIR/environment.yml"

echo "[macOS] Larrak installer"

if command -v mamba >/dev/null 2>&1; then
  PM=mamba
elif command -v conda >/dev/null 2>&1; then
  PM=conda
else
  echo "Conda/Mamba not found. Install Miniforge or Miniconda first."
  echo "- Miniforge: https://github.com/conda-forge/miniforge"
  echo "- Miniconda: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "environment.yml not found at $ENV_FILE"
  exit 1
fi

if [ "$PM" = "conda" ]; then
  conda config --set solver libmamba || true
  conda config --set channel_priority strict || true
fi

# Create or update env
if $PM env list | grep -q "^larrak\s"; then
  echo "Updating existing environment 'larrak'..."
  $PM env update -f "$ENV_FILE" --name larrak
else
  echo "Creating environment 'larrak'..."
  $PM env create -f "$ENV_FILE"
fi

echo "Verifying IPOPT availability..."
source "$HOME"/.bashrc 2>/dev/null || true
source "$HOME"/.zshrc 2>/dev/null || true
conda activate larrak 2>/dev/null || source activate larrak
python - <<'PY'
import casadi as ca
try:
    x = ca.SX.sym('x'); f = x**2
    ca.nlpsol('s','ipopt',{'x':x,'f':f})
    print('✓ ipopt is available')
except Exception as e:
    print('✗ ipopt not available:', e)
    raise SystemExit(1)
PY

echo "Done. Activate with: conda activate larrak"
