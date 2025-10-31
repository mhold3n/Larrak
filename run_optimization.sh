#!/bin/bash
"""
Bash script to run optimization in the correct conda environment.

This script automatically activates the local or global 'larrak' conda environment
and runs the optimization, preferring local OS-specific environments.
"""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not available. Please install Anaconda or Miniconda."
    exit 1
fi

# Initialize conda
source $(conda info --base)/etc/profile.d/conda.sh

# Detect OS and set local environment path
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_ENV_PATH="$PROJECT_DIR/conda_env_macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LOCAL_ENV_PATH="$PROJECT_DIR/conda_env_linux"
else
    LOCAL_ENV_PATH=""
fi

# Try to activate local environment first, fallback to global
if [[ -n "$LOCAL_ENV_PATH" && -d "$LOCAL_ENV_PATH" ]]; then
    echo "🔄 Activating local conda environment at '$LOCAL_ENV_PATH'..."
    conda activate "$LOCAL_ENV_PATH"
    
    if [[ "$CONDA_PREFIX" == "$LOCAL_ENV_PATH" ]]; then
        echo "✅ Successfully activated local conda environment"
        ENV_ACTIVATED=true
    else
        echo "⚠️  Failed to activate local environment, trying global 'larrak'..."
        ENV_ACTIVATED=false
    fi
fi

# Fallback to global environment if local activation failed or doesn't exist
if [[ "${ENV_ACTIVATED:-false}" != "true" ]]; then
    echo "🔄 Activating global 'larrak' conda environment..."
    conda activate larrak
    
    # Check if activation was successful
    if [[ "$CONDA_DEFAULT_ENV" != "larrak" ]]; then
        echo "❌ Error: Failed to activate 'larrak' environment."
        echo "   Please ensure the environment exists:"
        if [[ -n "$LOCAL_ENV_PATH" && -d "$LOCAL_ENV_PATH" ]]; then
            echo "   conda activate $LOCAL_ENV_PATH"
        else
            echo "   Or create a local environment using:"
            echo "   scripts/install_macos.sh  (on macOS)"
            echo "   scripts/install_linux.sh  (on Linux)"
            echo "   scripts/install_windows.ps1  (on Windows)"
        fi
        exit 1
    fi
    
    echo "✅ Successfully activated 'larrak' environment"
fi

echo "🚀 Running optimization..."

# Run the optimization with all passed arguments
python scripts/run_optimization_cli.py "$@"


