#!/usr/bin/env python3
"""
Demo script for the Cam Motion Law GUI.

This script demonstrates how to use the GUI programmatically
and provides examples of different cam motion law configurations.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import tkinter as tk

from cam_motion_gui import CamMotionGUI
from CamPro_OptimalMotion import solve_cam_motion_law


def demo_basic_cam():
    """Demonstrate basic cam motion law solving."""
    print("=== Basic Cam Motion Law Demo ===")

    # Basic cam parameters
    solution = solve_cam_motion_law(
        stroke=20.0,
        upstroke_duration_percent=60.0,
        motion_type="minimum_jerk",
        cycle_time=1.0,
    )

    print(f"Solution generated with {len(solution['cam_angle'])} points")
    print(f"Cam angle range: {solution['cam_angle'][0]:.1f}° to {solution['cam_angle'][-1]:.1f}°")
    print(f"Position range: {solution['position'][0]:.3f} to {solution['position'][-1]:.3f} mm")
    print(f"Max velocity: {max(abs(v) for v in solution['velocity']):.3f} mm/s")
    print(f"Max acceleration: {max(abs(a) for a in solution['acceleration']):.3f} mm/s²")
    print(f"Max jerk: {max(abs(j) for j in solution['control']):.3f} mm/s³")


def demo_constrained_cam():
    """Demonstrate cam motion law with constraints."""
    print("\n=== Constrained Cam Motion Law Demo ===")

    # Cam with constraints
    solution = solve_cam_motion_law(
        stroke=15.0,
        upstroke_duration_percent=50.0,
        motion_type="minimum_jerk",
        cycle_time=0.5,
        max_velocity=100.0,
        max_acceleration=500.0,
        dwell_at_tdc=True,
        dwell_at_bdc=False,
    )

    print(f"Solution generated with {len(solution['cam_angle'])} points")
    print(f"Max velocity: {max(abs(v) for v in solution['velocity']):.3f} mm/s (limit: 100.0)")
    print(f"Max acceleration: {max(abs(a) for a in solution['acceleration']):.3f} mm/s² (limit: 500.0)")

    # Check if constraints are satisfied
    max_vel = max(abs(v) for v in solution["velocity"])
    max_acc = max(abs(a) for a in solution["acceleration"])

    print(f"Velocity constraint satisfied: {max_vel <= 100.0 + 1e-6}")
    print(f"Acceleration constraint satisfied: {max_acc <= 500.0 + 1e-6}")


def demo_zero_acceleration_cam():
    """Demonstrate cam motion law with zero acceleration phase."""
    print("\n=== Zero Acceleration Phase Cam Demo ===")

    # Cam with zero acceleration phase
    solution = solve_cam_motion_law(
        stroke=25.0,
        upstroke_duration_percent=70.0,
        motion_type="minimum_energy",
        cycle_time=1.5,
        zero_accel_duration_percent=20.0,
        max_velocity=50.0,
    )

    print(f"Solution generated with {len(solution['cam_angle'])} points")
    print("Zero acceleration duration: 20% of cycle")
    print(f"Max velocity: {max(abs(v) for v in solution['velocity']):.3f} mm/s")

    # Find regions with near-zero acceleration
    near_zero_accel = [abs(a) < 1e-3 for a in solution["acceleration"]]
    zero_accel_percentage = sum(near_zero_accel) / len(near_zero_accel) * 100
    print(f"Actual zero acceleration percentage: {zero_accel_percentage:.1f}%")


def demo_motion_types():
    """Demonstrate different motion law types."""
    print("\n=== Motion Law Types Comparison ===")

    motion_types = ["minimum_jerk", "minimum_energy", "minimum_time"]

    for motion_type in motion_types:
        print(f"\n{motion_type.upper()}:")

        solution = solve_cam_motion_law(
            stroke=20.0,
            upstroke_duration_percent=60.0,
            motion_type=motion_type,
            cycle_time=1.0,
        )

        max_vel = max(abs(v) for v in solution["velocity"])
        max_acc = max(abs(a) for a in solution["acceleration"])
        max_jerk = max(abs(j) for j in solution["control"])

        print(f"  Max velocity: {max_vel:.3f} mm/s")
        print(f"  Max acceleration: {max_acc:.3f} mm/s²")
        print(f"  Max jerk: {max_jerk:.3f} mm/s³")


def create_gui_with_preset():
    """Create GUI with preset parameters."""
    print("\n=== Creating GUI with Preset Parameters ===")

    root = tk.Tk()
    app = CamMotionGUI(root)

    # Set some preset values
    app.variables["stroke"].set(30.0)
    app.variables["upstroke_duration"].set(45.0)
    app.variables["zero_accel_duration"].set(15.0)
    app.variables["cycle_time"].set(0.8)
    app.variables["max_velocity"].set(150.0)
    app.variables["max_acceleration"].set(800.0)
    app.variables["motion_type"].set("minimum_jerk")
    app.variables["dwell_at_tdc"].set(True)
    app.variables["dwell_at_bdc"].set(False)

    print("GUI created with preset parameters:")
    print("- Stroke: 30.0 mm")
    print("- Upstroke duration: 45%")
    print("- Zero acceleration duration: 15%")
    print("- Cycle time: 0.8 s")
    print("- Max velocity: 150.0 mm/s")
    print("- Max acceleration: 800.0 mm/s²")
    print("- Motion type: minimum_jerk")
    print("- Dwell at TDC: True")
    print("- Dwell at BDC: False")

    return root, app


def main():
    """Run all demos."""
    print("Cam Motion Law GUI Demo")
    print("=" * 40)

    try:
        # Run demos
        demo_basic_cam()
        demo_constrained_cam()
        demo_zero_acceleration_cam()
        demo_motion_types()

        # Create GUI with preset
        print("\n" + "=" * 40)
        print("Creating GUI with preset parameters...")
        root, app = create_gui_with_preset()

        # Center the window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")

        print("\nGUI is ready! You can:")
        print("1. Modify the preset parameters")
        print("2. Click 'Solve Motion Law' to generate curves")
        print("3. Save the plot using 'Save Plot' button")
        print("4. Try different motion types and constraints")

        # Start GUI
        root.mainloop()

    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()





