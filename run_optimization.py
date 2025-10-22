#!/usr/bin/env python3
"""
Python script to run optimization in the correct conda environment.

This script automatically activates the 'larrak' conda environment and runs the optimization.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_optimization_in_larrak_env():
    """Run optimization in the larrak conda environment."""
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Check if conda is available
    try:
        subprocess.run(['conda', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: conda is not available. Please install Anaconda or Miniconda.")
        sys.exit(1)
    
    # Prepare the command to run in larrak environment
    cmd = [
        'conda', 'run', '-n', 'larrak', 
        'python', str(script_dir / 'scripts' / 'run_optimization_cli.py')
    ] + sys.argv[1:]
    
    print("🔄 Running optimization in 'larrak' conda environment...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the optimization
        result = subprocess.run(cmd, cwd=script_dir)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⏹️  Optimization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running optimization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_optimization_in_larrak_env()

