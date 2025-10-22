#!/usr/bin/env python3
"""
Launcher script for the Cam Motion Law GUI.

This script provides a simple way to launch the GUI application with
environment validation.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def validate_environment_before_launch():
    """Validate environment before launching GUI."""
    try:
        # TEMPORARILY DISABLE VALIDATION TO PREVENT UNEXPECTED LINEAR SOLVER CHANGES
        # The validation process may create solvers that conflict with the MA27/MA57 policy
        print("⚠️  Environment validation temporarily disabled to avoid linear-solver conflicts")
        print("Continuing with GUI launch...")
        return
        
        # Original validation code (commented out to avoid unintended solver initialization)
        # from campro.environment.validator import validate_environment
        # 
        # print("Validating environment...")
        # results = validate_environment()
        # overall_status = results["summary"]["overall_status"]
        # 
        # if overall_status.value == "error":
        #     print("❌ Environment validation failed!")
        #     print("Required dependencies are missing or incompatible.")
        #     print("\nTo fix this issue:")
        #     print("1. Run: python scripts/setup_environment.py")
        #     print("2. Or run: python scripts/check_environment.py")
        #     print("\nExiting...")
        #     sys.exit(1)
        # elif overall_status.value == "warning":
        #     print("⚠️  Environment validation passed with warnings.")
        #     print("Run 'python scripts/check_environment.py' for details.")
        #     print("Continuing with GUI launch...")
        # else:
        #     print("✅ Environment validation passed!")

    except ImportError as e:
        print(f"⚠️  Warning: Could not import environment validator: {e}")
        print("Environment validation skipped.")
    except Exception as e:
        print(f"⚠️  Warning: Error during environment validation: {e}")
        print("Environment validation failed.")

try:
    # Validate environment first
    validate_environment_before_launch()

    from cam_motion_gui import main

    if __name__ == "__main__":
        print("Starting Cam Motion Law GUI...")
        main()

except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("Run: python scripts/setup_environment.py")
    print("Or: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting GUI: {e}")
    sys.exit(1)




