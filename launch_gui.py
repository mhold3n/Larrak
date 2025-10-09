#!/usr/bin/env python3
"""
Launcher script for the Cam Motion Law GUI.

This script provides a simple way to launch the GUI application.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from cam_motion_gui import main

    if __name__ == "__main__":
        print("Starting Cam Motion Law GUI...")
        main()

except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting GUI: {e}")
    sys.exit(1)





