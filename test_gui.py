#!/usr/bin/env python3
"""
Test script to verify GUI functionality without actually launching the GUI.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import tkinter as tk  # noqa: F401

        print("+ tkinter imported successfully")
    except ImportError as e:
        print(f"- tkinter import failed: {e}")
        return False

    try:
        import matplotlib.pyplot as plt  # noqa: F401

        print("+ matplotlib imported successfully")
    except ImportError as e:
        print(f"- matplotlib import failed: {e}")
        return False

    try:
        import numpy as np  # noqa: F401

        print("+ numpy imported successfully")
    except ImportError as e:
        print(f"- numpy import failed: {e}")
        return False

    try:
        from CamPro_OptimalMotion import (  # noqa: F401
            CollocationSettings,
            solve_cam_motion_law,
        )

        print("+ CamPro_OptimalMotion imported successfully")
    except ImportError as e:
        print(f"- CamPro_OptimalMotion import failed: {e}")
        return False

    try:
        from campro.logging import get_logger  # noqa: F401

        print("+ campro.logging imported successfully")
    except ImportError as e:
        print(f"- campro.logging import failed: {e}")
        return False

    return True


def test_cam_motion_law():
    """Test basic cam motion law functionality."""
    print("\nTesting cam motion law solving...")

    try:
        from CamPro_OptimalMotion import solve_cam_motion_law

        # Test basic cam motion law
        solution = solve_cam_motion_law(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            motion_type="minimum_jerk",
            cycle_time=1.0,
        )

        print("+ Basic cam motion law solved successfully")
        print(f"  - Generated {len(solution['cam_angle'])} points")
        print(
            f"  - Cam angle range: {solution['cam_angle'][0]:.1f}° to {solution['cam_angle'][-1]:.1f}°",
        )
        print(f"  - Max velocity: {max(abs(v) for v in solution['velocity']):.3f} mm/s")
        print(
            f"  - Max acceleration: {max(abs(a) for a in solution['acceleration']):.3f} mm/s²",
        )

        return True

    except Exception as e:
        print(f"- Cam motion law solving failed: {e}")
        return False


def test_gui_creation():
    """Test GUI creation without showing it."""
    print("\nTesting GUI creation...")

    try:
        import tkinter as tk

        from cam_motion_gui import CamMotionGUI

        # Create root window but don't show it
        root = tk.Tk()
        root.withdraw()  # Hide the window

        # Create GUI instance
        app = CamMotionGUI(root)

        print("+ GUI created successfully")
        print(f"  - Variables created: {len(app.variables)}")
        print(f"  - Default stroke: {app.variables['stroke'].get()}")
        print(
            f"  - Default upstroke duration: {app.variables['upstroke_duration'].get()}",
        )

        # Test parameter setting
        app.variables["stroke"].set(25.0)
        app.variables["upstroke_duration"].set(50.0)

        print(f"  - Updated stroke: {app.variables['stroke'].get()}")
        print(
            f"  - Updated upstroke duration: {app.variables['upstroke_duration'].get()}",
        )

        # Clean up
        root.destroy()

        return True

    except Exception as e:
        print(f"- GUI creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_matplotlib_integration():
    """Test matplotlib integration."""
    print("\nTesting matplotlib integration...")

    try:
        import numpy as np
        from matplotlib.figure import Figure

        # Create a simple figure
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Create some test data
        x = np.linspace(0, 360, 100)
        y = np.sin(np.radians(x))

        ax.plot(x, y)
        ax.set_xlabel("Cam Angle (degrees)")
        ax.set_ylabel("Value")
        ax.set_title("Test Plot")
        ax.grid(True)

        print("+ Matplotlib figure created successfully")
        print(f"  - Figure size: {fig.get_size_inches()}")
        print(f"  - DPI: {fig.dpi}")

        return True

    except Exception as e:
        print(f"- Matplotlib integration failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Cam Motion Law GUI Test Suite")
    print("=" * 40)

    tests = [
        test_imports,
        test_cam_motion_law,
        test_gui_creation,
        test_matplotlib_integration,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("+ All tests passed! GUI should work correctly.")
        return True
    print("- Some tests failed. Check the errors above.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
