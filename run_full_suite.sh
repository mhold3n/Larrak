#!/bin/bash
set -e

# Use active python environment (assumes user has activated 'larrak' or similar)
# Fallback to anaconda path if 'python' is not found
if command -v python &> /dev/null; then
    PYTHON_EXEC="python"
else
    PYTHON_EXEC="/Users/maxholden/anaconda3/bin/python3"
fi

# Ensure the current directory is in PYTHONPATH so module imports (tests.infra) work
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=========================================================="
echo "Starting Full DOE & Analysis Suite"
echo "Target Python: $(which $PYTHON_EXEC)"
echo "PYTHONPATH: $PYTHONPATH"
echo "=========================================================="

# 1. Phase 1 DOE Generation (Adaptive)
echo ""
echo "[1/3] Running Phase 1 DOE Generation..."
echo "      - Adaptive sampling with slope-dependent density optimization"
echo "      - Min slope limit enabled to prevent noise overfitting"
$PYTHON_EXEC tests/goldens/phase1/generate_doe.py

# 2. Phase 1 Interpretation
echo ""
echo "[2/3] Running Phase 1 Interpretation..."
$PYTHON_EXEC tests/goldens/phase1/interpret.py

# 3. Sensitivity Dashboard Update
echo ""
echo "[3/3] Updating Sensitivity Dashboard HTML..."
$PYTHON_EXEC tests/infra/sensitivity_dashboard.py

echo ""
echo "=========================================================="
echo "Suite Completed Successfully."
echo "Dashboard available at: goldens/SENSITIVITY_REPORT_FULL.html"
echo "=========================================================="
