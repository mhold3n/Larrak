#!/usr/bin/env python3
"""
Demo script for the enhanced GUI with cam-ring mapping functionality.

This script demonstrates how to use the enhanced GUI that now supports both:
1. Primary optimization: Linear follower motion law creation
2. Secondary optimization: Cam-ring mapping for circular follower design
"""

import sys
from pathlib import Path

# Add the current directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Launch the enhanced GUI with preset parameters."""
    try:
        import tkinter as tk

        from cam_motion_gui import CamMotionGUI

        print("Launching Enhanced Cam Motion Law & Ring Design GUI...")
        print("=" * 60)
        print("Features:")
        print("  Tab 1: Linear Follower Motion Law (Optimization 1)")
        print("    - Create optimal motion laws for linear followers")
        print("    - Support for minimum jerk, energy, and time motion laws")
        print("    - Constraint handling and boundary conditions")
        print()
        print("  Tab 2: Cam-Ring Design (Optimization 2)")
        print("    - Design circular ring followers based on linear motion")
        print("    - Cam connected to linear follower via connecting rod")
        print("    - Cam directly contacts ring follower surface")
        print("    - Ring profile mathematically determined by meshing law")
        print("    - Polar plots showing cam and ring follower profiles")
        print("=" * 60)

        # Create and configure the GUI
        root = tk.Tk()

        # Set style
        style = ttk.Style()
        style.theme_use("clam")  # Use a modern theme

        # Create the enhanced GUI
        app = CamMotionGUI(root)

        # Set some default parameters for demonstration
        app.variables["stroke"].set(25.0)
        app.variables["upstroke_duration"].set(50.0)
        app.variables["cycle_time"].set(1.2)
        app.variables["motion_type"].set("minimum_jerk")

        # Set cam-ring system parameters
        app.variables["base_radius"].set(18.0)
        app.variables["connecting_rod_length"].set(25.0)

        # Center the window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")

        print("GUI launched successfully!")
        print("Instructions:")
        print(
            "1. Go to Tab 1 and click 'Solve Motion Law' to create the linear follower motion",
        )
        print(
            "2. Go to Tab 2 and click 'Design Ring Follower' to create the circular follower",
        )
        print("3. The plot will show polar plots of the cam and ring follower profiles")
        print("4. Use the 'Save Plot' button to export your results")
        print()
        print("Close the GUI window to exit this demo.")

        # Start the GUI
        root.mainloop()

    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
