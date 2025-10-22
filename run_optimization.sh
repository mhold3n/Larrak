#!/bin/bash
"""
Bash script to run optimization in the correct conda environment.

This script automatically activates the 'larrak' conda environment and runs the optimization.
"""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not available. Please install Anaconda or Miniconda."
    exit 1
fi

# Activate the larrak environment and run optimization
echo "🔄 Activating 'larrak' conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate larrak

# Check if activation was successful
if [[ "$CONDA_DEFAULT_ENV" != "larrak" ]]; then
    echo "❌ Error: Failed to activate 'larrak' environment."
    echo "   Please ensure the 'larrak' environment exists:"
    echo "   conda create -n larrak python=3.11"
    echo "   conda activate larrak"
    echo "   pip install -r requirements.txt"
    exit 1
fi

echo "✅ Successfully activated 'larrak' environment"
echo "🚀 Running optimization..."

# Run the optimization with all passed arguments
python scripts/run_optimization_cli.py "$@"

