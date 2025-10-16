"""
Clean Cam-Ring System Designer GUI

A comprehensive GUI with 5 tabs for three-stage optimization:
1. Motion Law - Linear follower motion visualization (Run 1)
2. Cam/Ring Motion - Motion relationships (Run 2)
3. 2D Profiles - Cam and ring profile plots (Run 2)
4. Animation - 60-frame discrete animation
5. Crank Center - Torque and side-loading optimization (Run 3)
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from campro.logging import get_logger

# Validate environment before starting GUI
def _validate_gui_environment():
    """Validate environment and show user-friendly error dialog if needed."""
    try:
        from campro.environment.validator import validate_environment
        
        results = validate_environment()
        overall_status = results["summary"]["overall_status"]
        
        if overall_status.value == "error":
            # Show error dialog
            root = tk.Tk()
            root.withdraw()  # Hide main window
            
            error_msg = (
                "Environment validation failed!\n\n"
                "Required dependencies are missing or incompatible.\n\n"
                "To fix this issue:\n"
                "1. Run: python scripts/setup_environment.py\n"
                "2. Or run: python scripts/check_environment.py\n\n"
                "Would you like to continue anyway?\n"
                "(This will likely cause errors)"
            )
            
            result = messagebox.askyesno(
                "Environment Error",
                error_msg,
                icon="warning"
            )
            
            root.destroy()
            
            if not result:
                print("GUI startup cancelled due to environment issues.")
                sys.exit(1)
            else:
                print("Warning: Continuing with environment issues. Errors may occur.")
                
        elif overall_status.value == "warning":
            # Show warning dialog
            root = tk.Tk()
            root.withdraw()  # Hide main window
            
            warning_msg = (
                "Environment validation passed with warnings.\n\n"
                "Some dependencies may not be optimal.\n\n"
                "Run 'python scripts/check_environment.py' for details.\n\n"
                "Continue with GUI?"
            )
            
            result = messagebox.askyesno(
                "Environment Warning",
                warning_msg,
                icon="warning"
            )
            
            root.destroy()
            
            if not result:
                print("GUI startup cancelled due to environment warnings.")
                sys.exit(1)
                
    except ImportError as e:
        print(f"Warning: Could not import environment validator: {e}")
        print("Environment validation skipped.")
    except Exception as e:
        print(f"Warning: Error during environment validation: {e}")
        print("Environment validation failed.")

# Perform validation
_validate_gui_environment()
from campro.optimization.unified_framework import (
    OptimizationMethod,
    UnifiedOptimizationConstraints,
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
    UnifiedOptimizationTargets,
)
from campro.physics.kinematics.litvin_assembly import (
    AssemblyInputs,
    compute_assembly_state,
    compute_global_rmax,
    transform_to_world_polar,
)
from campro.physics.kinematics.phase2_relationships import (
    Phase2AnimationInputs,
    build_phase2_relationships,
)
from campro.storage import OptimizationRegistry

log = get_logger(__name__)


class CamMotionGUI:
    """Clean, focused GUI for cam-ring system design."""

    def __init__(self, root):
        self.root = root
        self.root.title("Cam-Ring System Designer")
        self.root.geometry("1400x1000")  # Increased height to accommodate tertiary controls

        # Configure ttk styles for better contrast
        self._configure_styles()

        # Variables to store input values
        self.variables = self._create_variables()
        # Cache values that may be accessed from worker threads
        self._contact_type_cache = self.variables["contact_type"].get()

        # Storage for optimization results
        self.unified_result = None
        self.registry = OptimizationRegistry()

        # Initialize unified optimization framework
        self.unified_framework = UnifiedOptimizationFramework("UnifiedCamRingOptimizer")

        # Create GUI elements
        self._create_widgets()
        self._layout_widgets()

        # Initialize plots for each tab
        self._setup_plots()

        # Set initial guesses based on default stroke
        self._on_stroke_changed()

        log.info("Cam-Ring System Designer GUI initialized")

        # Animation assembly state (initialized later after optimization)
        self.animation_state = None
        self._anim_rmax = None

    def _configure_styles(self):
        """Configure ttk styles for better contrast and readability."""
        style = ttk.Style()

        # Configure Combobox style for better text contrast
        style.configure("TCombobox",
                       fieldbackground="white",
                       background="white",
                       foreground="black",
                       selectbackground="lightblue",
                       selectforeground="black")

        # Configure Entry style for better text contrast
        style.configure("TEntry",
                       fieldbackground="white",
                       background="white",
                       foreground="black",
                       selectbackground="lightblue",
                       selectforeground="black")

        # Configure Button style for better contrast
        style.configure("TButton",
                       background="lightgray",
                       foreground="black",
                       focuscolor="none")

        # Configure Checkbutton style
        style.configure("TCheckbutton",
                       background="SystemButtonFace",
                       foreground="black",
                       focuscolor="none")

        # Configure Label style
        style.configure("TLabel",
                       background="SystemButtonFace",
                       foreground="black")

        # Configure LabelFrame style
        style.configure("TLabelFrame",
                       background="SystemButtonFace",
                       foreground="black")

        # Configure Notebook style
        style.configure("TNotebook",
                       background="SystemButtonFace",
                       foreground="black")

        style.configure("TNotebook.Tab",
                       background="lightgray",
                       foreground="black",
                       padding=[10, 5])

        # Map active tab style
        style.map("TNotebook.Tab",
                 background=[("selected", "white")],
                 foreground=[("selected", "black")])

        # Configure Frame style for better contrast
        style.configure("TFrame",
                       background="SystemButtonFace")

        # Configure Separator style
        style.configure("TSeparator",
                       background="gray")

        # Additional button styling for better visibility
        style.map("TButton",
                 background=[("active", "lightblue"),
                           ("pressed", "darkblue")],
                 foreground=[("active", "white"),
                           ("pressed", "white")])

    def _create_variables(self):
        """Create Tkinter variables for input fields."""
        return {
            # Core system parameters
            "stroke": tk.DoubleVar(value=20.0),
            "cycle_time": tk.DoubleVar(value=1.0),
            "upstroke_duration": tk.DoubleVar(value=60.0),
            "zero_accel_duration": tk.DoubleVar(value=0.0),
            "motion_type": tk.StringVar(value="minimum_jerk"),

            # Cam-ring system parameters
            "base_radius": tk.DoubleVar(value=15.0),
            "connecting_rod_length": tk.DoubleVar(value=25.0),
            "contact_type": tk.StringVar(value="external"),

            # Sun gear parameters (for complete system)
            "sun_gear_radius": tk.DoubleVar(value=15.0),
            "ring_gear_radius": tk.DoubleVar(value=45.0),
            "gear_ratio": tk.DoubleVar(value=3.0),

            # Optimization method
            "optimization_method": tk.StringVar(value="legendre_collocation"),

            # Animation settings
            "animation_speed": tk.DoubleVar(value=1.0),

            # Tertiary optimization parameters (crank center optimization)
            "crank_center_x_min": tk.DoubleVar(value=-50.0),
            "crank_center_x_max": tk.DoubleVar(value=50.0),
            "crank_center_y_min": tk.DoubleVar(value=-50.0),
            "crank_center_y_max": tk.DoubleVar(value=50.0),
            "crank_radius_min": tk.DoubleVar(value=20.0),
            "crank_radius_max": tk.DoubleVar(value=100.0),

            # Tertiary optimization targets
            "maximize_torque": tk.BooleanVar(value=True),
            "minimize_side_loading": tk.BooleanVar(value=True),
            "minimize_compression_side_load": tk.BooleanVar(value=True),
            "minimize_combustion_side_load": tk.BooleanVar(value=True),
            "minimize_torque_ripple": tk.BooleanVar(value=True),
            "maximize_power_output": tk.BooleanVar(value=True),
        }

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")

        # Control panel (top)
        self._create_control_panel()

        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)

        # Tab 1: Motion Law
        self.motion_law_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.motion_law_frame, text="Motion Law")

        # Tab 2: Cam/Ring Motion
        self.motion_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.motion_frame, text="Cam/Ring Motion")

        # Tab 3: 2D Profiles
        self.profiles_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.profiles_frame, text="2D Profiles")

        # Tab 4: Animation
        self.animation_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.animation_frame, text="Animation")

        # Tab 5: Crank Center Optimization
        self.tertiary_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.tertiary_frame, text="Crank Center (Run 3)")

        # Status bar
        self.status_var = tk.StringVar(value="Ready - Configure parameters and run optimization")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN)

    def _create_control_panel(self):
        """Create the main control panel."""
        self.control_frame = ttk.LabelFrame(self.main_frame, text="System Parameters", padding="10")

        # Row 1: Core motion law parameters
        ttk.Label(self.control_frame, text="Stroke (mm):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.stroke_entry = ttk.Entry(self.control_frame, textvariable=self.variables["stroke"], width=8)
        self.stroke_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Upstroke Duration (%):").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        self.upstroke_entry = ttk.Entry(self.control_frame, textvariable=self.variables["upstroke_duration"], width=8)
        self.upstroke_entry.grid(row=0, column=3, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Zero Accel Duration (%):").grid(row=0, column=4, sticky=tk.W, pady=2, padx=(20, 0))
        self.zero_accel_entry = ttk.Entry(self.control_frame, textvariable=self.variables["zero_accel_duration"], width=8)
        self.zero_accel_entry.grid(row=0, column=5, sticky=tk.W, padx=(5, 0), pady=2)

        # Row 2: Additional motion law parameters
        ttk.Label(self.control_frame, text="Cycle Time (s):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.cycle_time_entry = ttk.Entry(self.control_frame, textvariable=self.variables["cycle_time"], width=8)
        self.cycle_time_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Motion Type:").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        self.motion_type_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.variables["motion_type"],
            values=["minimum_jerk", "minimum_energy", "minimum_time"],
            state="readonly",
            width=12,
        )
        self.motion_type_combo.grid(row=1, column=3, sticky=tk.W, padx=(5, 0), pady=2)
        self.motion_type_combo.set("minimum_jerk")  # Set default value

        # Row 3: Cam-ring parameters
        ttk.Label(self.control_frame, text="Cam Base Radius (mm):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.base_radius_entry = ttk.Entry(self.control_frame, textvariable=self.variables["base_radius"], width=8)
        self.base_radius_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Rod Length (mm):").grid(row=2, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        self.rod_length_entry = ttk.Entry(self.control_frame, textvariable=self.variables["connecting_rod_length"], width=8)
        self.rod_length_entry.grid(row=2, column=3, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Contact Type:").grid(row=2, column=4, sticky=tk.W, pady=2, padx=(20, 0))
        self.contact_type_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.variables["contact_type"],
            values=["external", "internal"],
            state="readonly",
            width=12,
        )
        self.contact_type_combo.grid(row=2, column=5, sticky=tk.W, padx=(5, 0), pady=2)
        self.contact_type_combo.set("external")  # Set default value

        # Row 4: Sun gear parameters
        ttk.Label(self.control_frame, text="Sun Gear Radius (mm):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.sun_gear_entry = ttk.Entry(self.control_frame, textvariable=self.variables["sun_gear_radius"], width=8)
        self.sun_gear_entry.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Ring Gear Radius (mm):").grid(row=3, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        self.ring_gear_entry = ttk.Entry(self.control_frame, textvariable=self.variables["ring_gear_radius"], width=8)
        self.ring_gear_entry.grid(row=3, column=3, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Gear Ratio:").grid(row=3, column=4, sticky=tk.W, pady=2, padx=(20, 0))
        self.gear_ratio_entry = ttk.Entry(self.control_frame, textvariable=self.variables["gear_ratio"], width=8)
        self.gear_ratio_entry.grid(row=3, column=5, sticky=tk.W, padx=(5, 0), pady=2)

        # Row 5: Control buttons
        self.optimize_button = ttk.Button(
            self.control_frame,
            text="🚀 Optimize System",
            command=self._run_optimization,
            style="Accent.TButton",
        )
        self.optimize_button.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=10)

        self.reset_button = ttk.Button(
            self.control_frame,
            text="Reset Parameters",
            command=self._reset_parameters,
        )
        self.reset_button.grid(row=4, column=2, columnspan=2, sticky=tk.W, pady=10, padx=(20, 0))

        self.save_button = ttk.Button(
            self.control_frame,
            text="Save Results",
            command=self._save_results,
        )
        self.save_button.grid(row=4, column=4, columnspan=2, sticky=tk.W, pady=10, padx=(20, 0))

        # Row 5: Tertiary optimization header
        ttk.Separator(self.control_frame, orient="horizontal").grid(row=5, column=0, columnspan=6, sticky="ew", pady=5)
        ttk.Label(self.control_frame, text="Crank Center Optimization (Run 3)", font=("TkDefaultFont", 10, "bold")).grid(row=6, column=0, columnspan=6, sticky=tk.W, pady=2)

        # Row 6: Crank center position bounds
        ttk.Label(self.control_frame, text="Crank X Range (mm):").grid(row=7, column=0, sticky=tk.W, pady=2)
        crank_x_frame = ttk.Frame(self.control_frame)
        crank_x_frame.grid(row=7, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Entry(crank_x_frame, textvariable=self.variables["crank_center_x_min"], width=5).pack(side=tk.LEFT)
        ttk.Label(crank_x_frame, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(crank_x_frame, textvariable=self.variables["crank_center_x_max"], width=5).pack(side=tk.LEFT)

        ttk.Label(self.control_frame, text="Crank Y Range (mm):").grid(row=7, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        crank_y_frame = ttk.Frame(self.control_frame)
        crank_y_frame.grid(row=7, column=3, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Entry(crank_y_frame, textvariable=self.variables["crank_center_y_min"], width=5).pack(side=tk.LEFT)
        ttk.Label(crank_y_frame, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(crank_y_frame, textvariable=self.variables["crank_center_y_max"], width=5).pack(side=tk.LEFT)

        # Row 7: Crank radius bounds
        ttk.Label(self.control_frame, text="Crank Radius Range (mm):").grid(row=8, column=0, sticky=tk.W, pady=2)
        crank_r_frame = ttk.Frame(self.control_frame)
        crank_r_frame.grid(row=8, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Entry(crank_r_frame, textvariable=self.variables["crank_radius_min"], width=5).pack(side=tk.LEFT)
        ttk.Label(crank_r_frame, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(crank_r_frame, textvariable=self.variables["crank_radius_max"], width=5).pack(side=tk.LEFT)

        # Row 8: Optimization objectives (checkboxes)
        ttk.Label(self.control_frame, text="Objectives:").grid(row=9, column=0, sticky=tk.W, pady=2)
        objectives_frame = ttk.Frame(self.control_frame)
        objectives_frame.grid(row=9, column=1, columnspan=5, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Checkbutton(objectives_frame, text="Max Torque", variable=self.variables["maximize_torque"]).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(objectives_frame, text="Min Side Load", variable=self.variables["minimize_side_loading"]).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(objectives_frame, text="Min Compression Side Load", variable=self.variables["minimize_compression_side_load"]).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(objectives_frame, text="Min Combustion Side Load", variable=self.variables["minimize_combustion_side_load"]).pack(side=tk.LEFT, padx=5)

        # Add callback to update initial guesses when stroke changes
        self.variables["stroke"].trace("w", self._on_stroke_changed)
        # Cache contact type changes (thread-safe reads use this cache)
        self.variables["contact_type"].trace("w", self._on_contact_type_changed)

    def _layout_widgets(self):
        """Layout all widgets in the main window."""
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.control_frame.pack(fill=tk.X, pady=(0, 10))
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _setup_plots(self):
        """Setup matplotlib plots for each tab."""
        # Tab 1: Motion Law
        self.motion_law_fig = Figure(figsize=(12, 6), dpi=100)
        self.motion_law_canvas = FigureCanvasTkAgg(self.motion_law_fig, self.motion_law_frame)
        self.motion_law_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 2: Cam/Ring Motion
        self.motion_fig = Figure(figsize=(12, 6), dpi=100)
        self.motion_canvas = FigureCanvasTkAgg(self.motion_fig, self.motion_frame)
        self.motion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 3: 2D Profiles
        self.profiles_fig = Figure(figsize=(12, 6), dpi=100)
        self.profiles_canvas = FigureCanvasTkAgg(self.profiles_fig, self.profiles_frame)
        self.profiles_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 4: Animation - Two side-by-side subplots
        self.animation_fig = Figure(figsize=(16, 8), dpi=100)
        self.animation_canvas = FigureCanvasTkAgg(self.animation_fig, self.animation_frame)
        self.animation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 5: Crank Center Optimization
        self.tertiary_fig = Figure(figsize=(12, 6), dpi=100)
        self.tertiary_canvas = FigureCanvasTkAgg(self.tertiary_fig, self.tertiary_frame)
        self.tertiary_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Animation controls
        self._create_animation_controls()

        # Initialize empty plots
        self._clear_all_plots()

    def _create_animation_controls(self):
        """Create animation control buttons."""
        self.animation_control_frame = ttk.Frame(self.animation_frame)
        self.animation_control_frame.pack(fill=tk.X, pady=(0, 10))

        self.play_button = ttk.Button(
            self.animation_control_frame,
            text="▶ Play Animation",
            command=self._play_animation,
        )
        self.play_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = ttk.Button(
            self.animation_control_frame,
            text="⏸ Pause",
            command=self._pause_animation,
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            self.animation_control_frame,
            text="⏹ Stop",
            command=self._stop_animation,
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Label(self.animation_control_frame, text="Speed:").pack(side=tk.LEFT, padx=(20, 5))
        self.speed_scale = ttk.Scale(
            self.animation_control_frame,
            from_=0.1, to=3.0,
            variable=self.variables["animation_speed"],
            orient=tk.HORIZONTAL,
            length=100,
        )
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        # Frame count will be displayed dynamically based on collocation points
        self.frames_label = ttk.Label(self.animation_control_frame, text="Frames: Auto (from collocation)")
        self.frames_label.pack(side=tk.LEFT, padx=(20, 5))

        # Animation state
        self.animation_playing = False
        self.animation_frame_index = 0
        self.animation_timer = None

    def _clear_all_plots(self):
        """Clear all plots and show placeholder text."""
        # Motion Law tab
        self.motion_law_fig.clear()
        ax = self.motion_law_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Motion Law\n\nRun optimization to see results",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title("Linear Follower Motion Law")
        self.motion_law_canvas.draw()

        # Cam/Ring Motion tab
        self.motion_fig.clear()
        ax = self.motion_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Cam/Ring Motion\n\nRun optimization to see results",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title("Cam and Ring Motion Relationships")
        self.motion_canvas.draw()

        # 2D Profiles tab
        self.profiles_fig.clear()
        ax = self.profiles_fig.add_subplot(111)
        ax.text(0.5, 0.5, "2D Profiles\n\nRun optimization to see results",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title("Cam and Ring 2D Profiles")
        self.profiles_canvas.draw()

        # Animation tab - Two subplots
        self.animation_fig.clear()

        # Left subplot: Full assembly view
        ax1 = self.animation_fig.add_subplot(121, projection="polar")
        ax1.text(0.5, 0.5, "Full Assembly View\n\nRun optimization to see collocation-based animation",
                ha="center", va="center", fontsize=10, transform=ax1.transAxes)
        ax1.set_title("Full Assembly View")

        # Right subplot: Detailed tooth meshing view
        ax2 = self.animation_fig.add_subplot(122, projection="polar")
        ax2.text(0.5, 0.5, "Tooth Contact Detail\n\nShows 3-4 teeth at contact point",
                ha="center", va="center", fontsize=10, transform=ax2.transAxes)
        ax2.set_title("Tooth Contact Detail")

        self.animation_fig.suptitle("Collocation-Based Animation", fontsize=14, fontweight="bold")
        self.animation_canvas.draw()

        # Crank Center Optimization tab
        self.tertiary_fig.clear()
        ax = self.tertiary_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Crank Center Optimization\n\nRun optimization to see results",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title("Torque and Side-Loading Analysis")
        self.tertiary_canvas.draw()

    def _validate_inputs(self):
        """Validate input parameters."""
        try:
            stroke = self.variables["stroke"].get()
            upstroke_duration = self.variables["upstroke_duration"].get()
            zero_accel_duration = self.variables["zero_accel_duration"].get()

            if stroke <= 0:
                self.status_var.set("Error: Stroke must be positive")
                return False

            if upstroke_duration < 0 or upstroke_duration > 100:
                self.status_var.set("Error: Upstroke duration must be between 0 and 100%")
                return False

            if zero_accel_duration < 0 or zero_accel_duration > 100:
                self.status_var.set("Error: Zero acceleration duration must be between 0 and 100%")
                return False

            if zero_accel_duration > upstroke_duration:
                self.status_var.set("Warning: Zero acceleration duration exceeds upstroke duration")

            return True

        except Exception as e:
            self.status_var.set(f"Error: {e}")
            return False

    def _on_stroke_changed(self, *args):
        """Update initial guesses when stroke changes."""
        try:
            stroke = self.variables["stroke"].get()
            if stroke > 0:
                # Set intelligent initial guesses
                self.variables["base_radius"].set(stroke)
                self.variables["connecting_rod_length"].set(stroke * 1.25)

                print(f"DEBUG: Updated initial guesses based on stroke {stroke} mm:")
                print(f"  - Cam base radius: {stroke} mm")
                print(f"  - Connecting rod length: {stroke * 1.25} mm")
        except Exception as e:
            print(f"DEBUG: Error updating initial guesses: {e}")

    def _on_contact_type_changed(self, *args):
        """Cache contact type for use in worker threads."""
        try:
            self._contact_type_cache = self.variables["contact_type"].get()
        except Exception as e:
            # Keep previous cache; log for debug only
            print(f"DEBUG: Unable to cache contact_type: {e}")

    def _run_optimization(self):
        """Run the complete system optimization."""
        if not self._validate_inputs():
            return

        # Disable optimize button during computation
        self.optimize_button.config(state="disabled")
        self.status_var.set("Running system optimization...")

        # Run in separate thread to prevent GUI freezing
        thread = threading.Thread(target=self._optimization_thread)
        thread.daemon = True
        thread.start()

    def _optimization_thread(self):
        """Thread function for system optimization."""
        try:
            print("DEBUG: Starting system optimization thread...")

            # Configure unified framework
            self._configure_unified_framework()

            # Prepare input data
            input_data = {
                "stroke": self.variables["stroke"].get(),
                "cycle_time": self.variables["cycle_time"].get(),
                "upstroke_duration_percent": self.variables["upstroke_duration"].get(),
                "zero_accel_duration_percent": self.variables["zero_accel_duration"].get(),
                "motion_type": self.variables["motion_type"].get(),
            }

            print(f"DEBUG: System optimization input data: {input_data}")

            # Perform cascaded optimization
            result_data = self.unified_framework.optimize_cascaded(input_data)

            # Store the result
            self.unified_result = result_data

            # Update plots on main thread
            self.root.after(0, self._update_all_plots, result_data)

            # Update frames label if we have frame count
            if hasattr(self, "_total_frames"):
                self.root.after(0, lambda: self.frames_label.config(text=f"Frames: {self._total_frames} (collocation points)"))

        except Exception as e:
            print(f"DEBUG: Error in optimization thread: {e}")
            import traceback
            traceback.print_exc()
            error_message = str(e)
            # Promote common TE-requirement failures to a clear dialog
            if "Thermal-efficiency path" in error_message or "CasADi" in error_message or "IPOPT" in error_message:
                self.root.after(0, lambda: messagebox.showerror(
                    "Thermal Efficiency Required",
                    "Thermal-efficiency optimization is required but unavailable or failed.\n\n"
                    "Please install CasADi with IPOPT and retry.\n\n"
                    "Conda (recommended):\n  conda install -c conda-forge casadi ipopt\n\n"
                    "Pip (may lack IPOPT):\n  pip install 'casadi>=3.6,<3.7'",
                ))
            self.root.after(0, self._enable_optimize_button)
            self.root.after(0, lambda: self.status_var.set(f"Optimization failed: {error_message}"))

    def _enable_optimize_button(self):
        """Re-enable the optimize button."""
        self.optimize_button.config(state="normal")

    def _configure_unified_framework(self):
        """Configure the unified optimization framework with current settings."""
        # Get optimization method
        method_name = self.variables["optimization_method"].get()
        method = OptimizationMethod(method_name)

        # Create settings
        settings = UnifiedOptimizationSettings(
            method=method,
            collocation_degree=3,
            max_iterations=100,
            tolerance=1e-6,
            lagrangian_tolerance=1e-8,
            penalty_weight=1.0,
            # Enable CasADi/IPOPT path via thermal efficiency adapter
            use_thermal_efficiency=True,
            require_thermal_efficiency=True,
        )

        # Create constraints
        constraints = UnifiedOptimizationConstraints(
            stroke_min=1.0,
            stroke_max=100.0,
            max_velocity=100.0,
            max_acceleration=1000.0,
            max_jerk=10000.0,
            base_radius_min=5.0,
            base_radius_max=100.0,
            # Add tertiary constraints
            crank_center_x_min=self.variables["crank_center_x_min"].get(),
            crank_center_x_max=self.variables["crank_center_x_max"].get(),
            crank_center_y_min=self.variables["crank_center_y_min"].get(),
            crank_center_y_max=self.variables["crank_center_y_max"].get(),
            crank_radius_min=self.variables["crank_radius_min"].get(),
            crank_radius_max=self.variables["crank_radius_max"].get(),
        )

        # Create targets
        targets = UnifiedOptimizationTargets(
            minimize_jerk=True,
            minimize_time=False,
            minimize_energy=False,
            minimize_ring_size=True,
            minimize_cam_size=True,
            minimize_curvature_variation=True,
            # Add tertiary targets
            maximize_torque=self.variables["maximize_torque"].get(),
            minimize_side_loading=self.variables["minimize_side_loading"].get(),
            minimize_side_loading_during_compression=self.variables["minimize_compression_side_load"].get(),
            minimize_side_loading_during_combustion=self.variables["minimize_combustion_side_load"].get(),
            minimize_torque_ripple=self.variables["minimize_torque_ripple"].get(),
            maximize_power_output=self.variables["maximize_power_output"].get(),
        )

        # Configure framework
        self.unified_framework.configure(settings=settings, constraints=constraints, targets=targets)
        print(f"DEBUG: Configured unified framework with method: {method_name} (use_thermal_efficiency=True)")

    def _update_all_plots(self, result_data):
        """Update all plots with optimization results."""
        try:
            print("DEBUG: Updating all plots with result data")

            # Update Motion Law tab
            self._update_motion_law_plot(result_data)

            # Update Cam/Ring Motion tab
            self._update_motion_plot(result_data)

            # Update 2D Profiles tab
            self._update_profiles_plot(result_data)

            # Update Animation tab
            self._update_animation_plot(result_data)

            # Update Crank Center Optimization tab
            self._update_tertiary_plot(result_data)

            # Update status
            self.status_var.set(f"Optimization completed - Total time: {result_data.total_solve_time:.3f}s")

        except Exception as e:
            print(f"DEBUG: Error updating plots: {e}")
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Plot update failed: {e}")
        finally:
            self._enable_optimize_button()

    def _update_motion_law_plot(self, result_data):
        """Update the motion law plot (Tab 1)."""
        self.motion_law_fig.clear()

        if result_data.primary_theta is not None and result_data.primary_position is not None:
            # Create subplots for position, velocity, acceleration, and jerk
            ax1 = self.motion_law_fig.add_subplot(411)
            ax2 = self.motion_law_fig.add_subplot(412)
            ax3 = self.motion_law_fig.add_subplot(413)
            ax4 = self.motion_law_fig.add_subplot(414)

            # Plot position
            ax1.plot(result_data.primary_theta, result_data.primary_position, "b-", linewidth=2, label="Position")
            ax1.set_ylabel("Position (mm)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Plot velocity
            if result_data.primary_velocity is not None:
                ax2.plot(result_data.primary_theta, result_data.primary_velocity, "g-", linewidth=2, label="Velocity")
                ax2.set_ylabel("Velocity (mm/s)")
                ax2.grid(True, alpha=0.3)
                ax2.legend()

            # Plot acceleration
            if result_data.primary_acceleration is not None:
                ax3.plot(result_data.primary_theta, result_data.primary_acceleration, "r-", linewidth=2, label="Acceleration")
                ax3.set_ylabel("Acceleration (mm/s²)")
                ax3.grid(True, alpha=0.3)
                ax3.legend()

            # Plot jerk
            if result_data.primary_jerk is not None:
                ax4.plot(result_data.primary_theta, result_data.primary_jerk, "m-", linewidth=2, label="Jerk")
                ax4.set_ylabel("Jerk (mm/s³)")
                ax4.set_xlabel("Cam Angle (degrees)")
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            # If jerk data is not available, compute it from acceleration
            elif result_data.primary_acceleration is not None and len(result_data.primary_acceleration) > 1:
                # Compute jerk as derivative of acceleration
                jerk = np.gradient(result_data.primary_acceleration, result_data.primary_theta)
                ax4.plot(result_data.primary_theta, jerk, "m-", linewidth=2, label="Jerk (computed)")
                ax4.set_ylabel("Jerk (mm/s³)")
                ax4.set_xlabel("Cam Angle (degrees)")
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, "Jerk data not available", ha="center", va="center", fontsize=10)
                ax4.set_ylabel("Jerk (mm/s³)")
                ax4.set_xlabel("Cam Angle (degrees)")

            self.motion_law_fig.suptitle("Linear Follower Motion Law", fontsize=14, fontweight="bold")
        else:
            ax = self.motion_law_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No motion law data available", ha="center", va="center", fontsize=12)
            ax.set_title("Linear Follower Motion Law")

        self.motion_law_fig.tight_layout()
        self.motion_law_canvas.draw()

    def _update_motion_plot(self, result_data):
        """Update the cam/ring motion plot (Tab 2)."""
        self.motion_fig.clear()

        if result_data.secondary_psi is not None and result_data.secondary_R_psi is not None:
            # Create subplots for ring motion and cam motion
            ax1 = self.motion_fig.add_subplot(211)
            ax2 = self.motion_fig.add_subplot(212)

            # Plot ring motion (psi vs R_psi)
            ax1.plot(result_data.secondary_psi, result_data.secondary_R_psi, "purple", linewidth=2, label="Ring Radius")
            ax1.set_ylabel("Ring Radius (mm)")
            ax1.set_xlabel("Ring Angle ψ (rad)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_title("Ring Follower Motion")

            # Plot cam motion (if available)
            if result_data.secondary_cam_curves is not None:
                cam_theta = result_data.secondary_cam_curves.get("theta", result_data.primary_theta)
                cam_radius = result_data.secondary_cam_curves.get("profile_radius")
                if cam_radius is not None:
                    ax2.plot(cam_theta, cam_radius, "orange", linewidth=2, label="Cam Radius")
                    ax2.set_ylabel("Cam Radius (mm)")
                    ax2.set_xlabel("Cam Angle θ (degrees)")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    ax2.set_title("Cam Profile Motion")

            self.motion_fig.suptitle("Cam and Ring Motion Relationships", fontsize=14, fontweight="bold")
        else:
            ax = self.motion_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No cam/ring motion data available", ha="center", va="center", fontsize=12)
            ax.set_title("Cam and Ring Motion Relationships")

        self.motion_fig.tight_layout()
        self.motion_canvas.draw()

    def _update_profiles_plot(self, result_data):
        """Update the 2D profiles plot (Tab 3)."""
        self.profiles_fig.clear()

        if result_data.secondary_cam_curves is not None and result_data.secondary_psi is not None:
            # Create subplots for cam profile and ring profile
            ax1 = self.profiles_fig.add_subplot(121, projection="polar")
            ax2 = self.profiles_fig.add_subplot(122, projection="polar")

            # Plot cam profile (polar) - use the theta from cam_curves
            if "theta" in result_data.secondary_cam_curves and "profile_radius" in result_data.secondary_cam_curves:
                cam_theta_rad = np.radians(result_data.secondary_cam_curves["theta"])
                cam_radius = result_data.secondary_cam_curves["profile_radius"]
                ax1.plot(cam_theta_rad, cam_radius, "orange", linewidth=2, label="Cam Profile")
                ax1.set_title("Cam Profile (Polar)", pad=20)
                ax1.grid(True)
                ax1.set_ylim(0, max(cam_radius) * 1.1)  # Set appropriate radial limits

            # Plot ring profile (polar)
            ax2.plot(result_data.secondary_psi, result_data.secondary_R_psi, "purple", linewidth=2, label="Ring Profile")
            ax2.set_title("Ring Profile (Polar)", pad=20)
            ax2.grid(True)
            ax2.set_ylim(0, max(result_data.secondary_R_psi) * 1.1)  # Set appropriate radial limits

            self.profiles_fig.suptitle("Cam and Ring 2D Profiles", fontsize=14, fontweight="bold")
        else:
            ax = self.profiles_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No profile data available", ha="center", va="center", fontsize=12)
            ax.set_title("Cam and Ring 2D Profiles")

        self.profiles_fig.tight_layout()
        self.profiles_canvas.draw()

    def _update_animation_plot(self, result_data):
        """Update the animation plot (Tab 4)."""
        self.animation_fig.clear()

        if result_data.secondary_psi is not None and result_data.secondary_R_psi is not None:
            # Store animation data
            self.animation_data = result_data

            # Deterministic Phase-2 relationships: no solving at runtime
            contact_type = getattr(self, "_contact_type_cache", "external")
            try:
                bundle = self.unified_framework.get_phase2_animation_inputs()
                rel_inputs = Phase2AnimationInputs(
                    theta_deg=np.asarray(bundle["theta_deg"]),
                    x_theta_mm=np.asarray(bundle["x_theta_mm"]),
                    base_radius_mm=float(bundle["base_radius_mm"]),
                    psi_rad=np.asarray(bundle["psi_rad"]),
                    R_psi_mm=np.asarray(bundle["R_psi_mm"]),
                    gear_geometry=bundle.get("gear_geometry", {}) or {},
                    contact_type=contact_type,
                    constrain_center_to_x_axis=True,
                    align_tdc_at_theta0=True,
                )
                self.animation_state = build_phase2_relationships(rel_inputs)
            except Exception:
                # Fallback to legacy assembly if framework bundle is unavailable
                gear = getattr(result_data, "secondary_gear_geometry", None) or {}
                base_circle_cam = float(gear.get("base_circle_cam", result_data.secondary_base_radius or 20.0))
                base_circle_ring = float(gear.get("base_circle_ring", (result_data.secondary_base_radius or 20.0) * 1.7))
                z_cam = int(gear.get("z_cam", 20))
                theta_cam_deg = None
                if isinstance(result_data.secondary_cam_curves, dict):
                    theta_cam_deg = result_data.secondary_cam_curves.get("theta")
                theta_cam_rad = np.radians(theta_cam_deg) if theta_cam_deg is not None else np.linspace(0, 2*np.pi, len(result_data.secondary_psi))
                inputs = AssemblyInputs(
                    base_circle_cam=base_circle_cam,
                    base_circle_ring=base_circle_ring,
                    z_cam=z_cam,
                    contact_type=contact_type,
                    psi=result_data.secondary_psi,
                    R_psi=result_data.secondary_R_psi,
                    theta_cam_rad=theta_cam_rad,
                )
                self.animation_state = compute_assembly_state(inputs)
            self._anim_rmax = compute_global_rmax(self.animation_state)

            # Frames from psi
            self._total_frames = len(result_data.secondary_psi)

            # Create initial frame directly (we're already on main thread when this is called)
            self._draw_animation_frame(0)

            self.animation_fig.suptitle("Collocation-Based Animation", fontsize=14, fontweight="bold")

            # Auto-start playback for convenience
            self._play_animation()
        else:
            # Show placeholder with two subplots
            ax1 = self.animation_fig.add_subplot(121, projection="polar")
            ax1.text(0.5, 0.5, "Full Assembly View\n\nRun optimization to see collocation-based animation",
                    ha="center", va="center", fontsize=10, transform=ax1.transAxes)
            ax1.set_title("Full Assembly View")

            ax2 = self.animation_fig.add_subplot(122, projection="polar")
            ax2.text(0.5, 0.5, "Tooth Contact Detail\n\nShows 3-4 teeth at contact point",
                    ha="center", va="center", fontsize=10, transform=ax2.transAxes)
            ax2.set_title("Tooth Contact Detail")

            self.animation_fig.suptitle("Collocation-Based Animation", fontsize=14, fontweight="bold")

        self.animation_fig.tight_layout()
        self.animation_canvas.draw()

    def _draw_animation_frame(self, frame_index):
        """Draw a single animation frame using collocation point data."""
        if not hasattr(self, "animation_data") or self.animation_data is None:
            return

        # Prefer ring contact angles (psi, radians) for animation; fallback to primary theta (degrees)
        psi_data = getattr(self.animation_data, "secondary_psi", None)
        theta_deg_data = getattr(self.animation_data, "primary_theta", None)

        if psi_data is not None and len(psi_data) > 0:
            angle_series = psi_data  # radians
        elif theta_deg_data is not None and len(theta_deg_data) > 0:
            angle_series = np.radians(theta_deg_data)  # convert degrees -> radians
        else:
            return

        # Calculate total frames from available angle series
        total_frames = len(angle_series)

        # Ensure frame_index is within bounds
        if frame_index >= total_frames:
            frame_index = total_frames - 1

        # Get the current collocation/contact angle (radians)
        current_theta = angle_series[frame_index]

        # Clear the plot and create two subplots
        self.animation_fig.clear()

        # Left subplot: Full assembly view
        ax1 = self.animation_fig.add_subplot(121, projection="polar")

        # Right subplot: Detailed tooth meshing view
        ax2 = self.animation_fig.add_subplot(122, projection="polar")

        # Draw full assembly view
        self._draw_full_assembly_view(ax1, frame_index, current_theta, total_frames)

        # Draw detailed tooth meshing view
        self._draw_tooth_contact_detail(ax2, frame_index, current_theta, total_frames)

        # Update the canvas
        self.animation_canvas.draw()

    def _draw_full_assembly_view(self, ax, frame_index, current_theta, total_frames):
        """Draw the full assembly view showing cam-ring contact at current collocation point."""
        # Set up the polar plot with fixed scaling
        if self._anim_rmax is None:
            self._anim_rmax = 100.0
        ax.set_ylim(0, self._anim_rmax)
        ax.set_title(f"Full Assembly - Frame {frame_index + 1}/{total_frames}\nθ = {np.degrees(current_theta):.1f}°")

        # Draw ring gear (fixed, centered): phase-2 profile directly
        ring_psi = self.animation_data.secondary_psi
        ring_R_psi = self.animation_data.secondary_R_psi
        ax.plot(ring_psi, ring_R_psi, "r-", linewidth=3, label="Ring Gear")

        # Draw 3 planet teeth using assembly transform
        st = self.animation_state
        if st is not None and isinstance(self.animation_data.secondary_gear_geometry, dict):
            flanks = self.animation_data.secondary_gear_geometry.get("flanks")
            drew_planet = False
            if isinstance(flanks, dict) and ("tooth" in flanks):
                m0 = int(round(st.z_cam * st.planet_spin_angle[self.animation_frame_index] / (2*np.pi)))
                for dm, color in [(-1, "b-"), (0, "b-"), (1, "b-")]:
                    idx = m0 + dm
                    tooth = flanks.get("tooth")
                    if isinstance(tooth, dict):
                        tooth_theta = np.asarray(tooth.get("theta"))
                        tooth_r = np.asarray(tooth.get("r"))
                        if tooth_theta is not None and tooth_r is not None and len(tooth_theta) > 0:
                            thw, rw = transform_to_world_polar(
                                tooth_theta,
                                tooth_r,
                                st.planet_center_radius[self.animation_frame_index],
                                st.planet_center_angle[self.animation_frame_index],
                                st.planet_spin_angle[self.animation_frame_index],
                                idx,
                                st.z_cam,
                            )
                            ax.plot(thw, rw, color, linewidth=2, label="Cam Teeth" if dm == 0 else None)
                            drew_planet = True
            # Fallback: draw cam pitch profile aligned as a single body if no explicit tooth data
            if not drew_planet and isinstance(self.animation_data.secondary_cam_curves, dict):
                cam_theta_deg = self.animation_data.secondary_cam_curves.get("theta")
                cam_r = self.animation_data.secondary_cam_curves.get("profile_radius")
                if cam_theta_deg is not None and cam_r is not None:
                    cam_theta_rad = np.radians(np.asarray(cam_theta_deg))
                    thw, rw = transform_to_world_polar(
                        cam_theta_rad,
                        np.asarray(cam_r),
                        st.planet_center_radius[self.animation_frame_index],
                        st.planet_center_angle[self.animation_frame_index],
                        st.planet_spin_angle[self.animation_frame_index],
                        0,
                        st.z_cam,
                    )
                    ax.plot(thw, rw, "b-", linewidth=2, label="Cam Profile")

        # Draw contact point
        # Contact from phase-2 data directly
        contact_radius = float(self.animation_state.contact_radius[self.animation_frame_index]) if self.animation_state is not None else float(np.mean(self.animation_data.secondary_R_psi))
        ax.plot(current_theta, contact_radius, "go", markersize=8, label="Contact Point")

        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    def _draw_tooth_contact_detail(self, ax, frame_index, current_theta, total_frames):
        """Draw detailed tooth meshing view showing 3-4 teeth at contact region."""
        # Set up zoomed polar plot (fixed scaling)
        detail_angle_range = np.radians(10.0)
        if self._anim_rmax is None:
            self._anim_rmax = 100.0
        ax.set_ylim(0, self._anim_rmax)
        ax.set_xlim(current_theta - detail_angle_range, current_theta + detail_angle_range)
        ax.set_title(f"Tooth Contact Detail - Frame {frame_index + 1}/{total_frames}")

        # Draw ring gear teeth in detail region
        ring_psi = self.animation_data.secondary_psi
        ring_R_psi = self.animation_data.secondary_R_psi
        mask = (ring_psi >= current_theta - detail_angle_range) & (ring_psi <= current_theta + detail_angle_range)
        ax.plot(ring_psi[mask], ring_R_psi[mask], "r-", linewidth=3, label="Ring Teeth")

        # Draw cam teeth in detail region using assembly transform
        st = self.animation_state
        if st is not None and isinstance(self.animation_data.secondary_gear_geometry, dict):
            flanks = self.animation_data.secondary_gear_geometry.get("flanks")
            drew_planet = False
            if isinstance(flanks, dict) and ("tooth" in flanks):
                m0 = int(round(st.z_cam * st.planet_spin_angle[self.animation_frame_index] / (2*np.pi)))
                for dm, color in [(-1, "b-"), (0, "b-"), (1, "b-")]:
                    idx = m0 + dm
                    tooth = flanks.get("tooth")
                    if isinstance(tooth, dict):
                        tooth_theta = np.asarray(tooth.get("theta"))
                        tooth_r = np.asarray(tooth.get("r"))
                        if tooth_theta is not None and tooth_r is not None and len(tooth_theta) > 0:
                            thw, rw = transform_to_world_polar(
                                tooth_theta,
                                tooth_r,
                                st.planet_center_radius[self.animation_frame_index],
                                st.planet_center_angle[self.animation_frame_index],
                                st.planet_spin_angle[self.animation_frame_index],
                                idx,
                                st.z_cam,
                            )
                            maskw = (thw >= current_theta - detail_angle_range) & (thw <= current_theta + detail_angle_range)
                            ax.plot(thw[maskw], rw[maskw], color, linewidth=2, label="Cam Teeth" if dm == 0 else None)
                            drew_planet = True
            if not drew_planet and isinstance(self.animation_data.secondary_cam_curves, dict):
                cam_theta_deg = self.animation_data.secondary_cam_curves.get("theta")
                cam_r = self.animation_data.secondary_cam_curves.get("profile_radius")
                if cam_theta_deg is not None and cam_r is not None:
                    cam_theta_rad = np.radians(np.asarray(cam_theta_deg))
                    thw, rw = transform_to_world_polar(
                        cam_theta_rad,
                        np.asarray(cam_r),
                        st.planet_center_radius[self.animation_frame_index],
                        st.planet_center_angle[self.animation_frame_index],
                        st.planet_spin_angle[self.animation_frame_index],
                        0,
                        st.z_cam,
                    )
                    maskw = (thw >= current_theta - detail_angle_range) & (thw <= current_theta + detail_angle_range)
                    ax.plot(thw[maskw], rw[maskw], "b-", linewidth=2, label="Cam Profile")

        # Highlight the exact contact point
        contact_radius = float(self.animation_state.contact_radius[self.animation_frame_index]) if self.animation_state is not None else float(np.mean(self.animation_data.secondary_R_psi))
        ax.plot(current_theta, contact_radius, "go", markersize=10, label="Contact Point")

        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    def _draw_system_components(self, ax, theta, position):
        """Draw the cam-ring system components."""
        # Draw linear follower
        ax.plot([0, 0], [0, position], "b-", linewidth=3, label="Linear Follower")
        ax.plot(0, position, "bo", markersize=8, label="Follower Position")

        # Draw connecting rod (if available) - using fixed length for phase 2
        if hasattr(self, "animation_data"):
            rod_length = 25.0  # Fixed connecting rod length for phase 2 simplification
            # Simple connecting rod representation
            ax.plot([0, 10], [position, position + 5], "g-", linewidth=2, label="Connecting Rod")

        # Draw cam (simplified representation)
        if hasattr(self, "animation_data") and self.animation_data.secondary_base_radius is not None:
            base_radius = self.animation_data.secondary_base_radius
            cam_x = base_radius * np.cos(np.radians(theta))
            cam_y = base_radius * np.sin(np.radians(theta))
            ax.plot(cam_x, cam_y, "ro", markersize=10, label="Cam")

        ax.legend()

    def _update_tertiary_plot(self, result_data):
        """Update the crank center optimization plot (Tab 5)."""
        self.tertiary_fig.clear()

        if (result_data.tertiary_crank_center_x is not None and
            result_data.tertiary_torque_output is not None):

            # Create subplots for torque and side-loading
            ax1 = self.tertiary_fig.add_subplot(221)
            ax2 = self.tertiary_fig.add_subplot(222)
            ax3 = self.tertiary_fig.add_subplot(223)
            ax4 = self.tertiary_fig.add_subplot(224)

            # Plot 1: Crank center position
            ax1.plot(result_data.tertiary_crank_center_x, result_data.tertiary_crank_center_y,
                    "ro", markersize=12, label="Optimized Position")
            ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            ax1.axvline(x=0, color="k", linestyle="--", alpha=0.3)
            ax1.set_xlabel("X Position (mm)")
            ax1.set_ylabel("Y Position (mm)")
            ax1.set_title("Optimized Crank Center Position")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_aspect("equal")

            # Plot 2: Torque metrics
            metrics = ["Avg Torque", "Max Torque", "Torque Ripple"]
            values = [
                result_data.tertiary_torque_output or 0,
                result_data.tertiary_max_torque or 0,
                result_data.tertiary_torque_ripple or 0,
            ]
            ax2.bar(metrics, values, color=["green", "blue", "orange"])
            ax2.set_ylabel("Torque (N⋅m)")
            ax2.set_title("Torque Performance Metrics")
            ax2.grid(True, alpha=0.3, axis="y")

            # Plot 3: Side-loading metrics
            side_load_metrics = ["Total Penalty", "Max Side Load"]
            side_load_values = [
                result_data.tertiary_side_load_penalty or 0,
                result_data.tertiary_max_side_load or 0,
            ]
            ax3.bar(side_load_metrics, side_load_values, color=["red", "darkred"])
            ax3.set_ylabel("Side Load (N)")
            ax3.set_title("Side-Loading Metrics")
            ax3.grid(True, alpha=0.3, axis="y")

            # Plot 4: Summary text
            ax4.axis("off")
            summary_text = f"""
Crank Center Optimization Results

Optimized Position:
  X: {result_data.tertiary_crank_center_x:.2f} mm
  Y: {result_data.tertiary_crank_center_y:.2f} mm

Crank Geometry:
  Radius: {result_data.tertiary_crank_radius:.2f} mm
  Rod Length: {result_data.tertiary_rod_length:.2f} mm

Performance:
  Avg Torque: {result_data.tertiary_torque_output:.2f} N⋅m
  Max Torque: {result_data.tertiary_max_torque:.2f} N⋅m
  Power Output: {result_data.tertiary_power_output:.2f} W
  
Side-Loading:
  Total Penalty: {result_data.tertiary_side_load_penalty:.2f} N
  Max Side Load: {result_data.tertiary_max_side_load:.2f} N
            """
            ax4.text(0.1, 0.5, summary_text, fontsize=10, family="monospace",
                    verticalalignment="center")

            self.tertiary_fig.suptitle("Crank Center Optimization (Run 3)",
                                       fontsize=14, fontweight="bold")
        else:
            ax = self.tertiary_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No crank center optimization data available",
                   ha="center", va="center", fontsize=12)
            ax.set_title("Crank Center Optimization")

        self.tertiary_fig.tight_layout()
        self.tertiary_canvas.draw()

    def _play_animation(self):
        """Start animation playback."""
        if not hasattr(self, "animation_data") or self.animation_data is None:
            self.status_var.set("No animation data available. Run optimization first.")
            return

        # Calculate total frames from collocation points
        theta_data = getattr(self.animation_data, "primary_theta", None)
        if theta_data is None or len(theta_data) == 0:
            self.status_var.set("No collocation data available for animation.")
            return

        total_frames = len(theta_data)

        self.animation_playing = True
        self.animation_frame_index = 0

        # Store frame count for later update (thread-safe)
        self._total_frames = total_frames

        self._animate_next_frame()

    def _animate_next_frame(self):
        """Animate to the next frame."""
        if not self.animation_playing:
            return

        # Calculate total frames from collocation points
        theta_data = getattr(self.animation_data, "primary_theta", None)
        if theta_data is None or len(theta_data) == 0:
            return

        total_frames = len(theta_data)

        self._draw_animation_frame(self.animation_frame_index)

        self.animation_frame_index = (self.animation_frame_index + 1) % total_frames

        # Schedule next frame
        delay = int(1000 / (30 * self.variables["animation_speed"].get()))  # 30 FPS base
        self.animation_timer = self.root.after(delay, self._animate_next_frame)

    def _pause_animation(self):
        """Pause animation."""
        self.animation_playing = False
        if self.animation_timer:
            self.root.after_cancel(self.animation_timer)
            self.animation_timer = None

    def _stop_animation(self):
        """Stop animation and reset to frame 0."""
        self._pause_animation()
        self.animation_frame_index = 0
        if hasattr(self, "animation_data") and self.animation_data is not None:
            self._draw_animation_frame(0)

    def _save_results(self):
        """Save optimization results to file."""
        if self.unified_result is None:
            self.status_var.set("No results to save. Run optimization first.")
            return

        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Optimization Results",
            )

            if filename:
                import json
                # Convert numpy arrays to lists for JSON serialization
                result_dict = {
                    "primary_theta": self.unified_result.primary_theta.tolist() if self.unified_result.primary_theta is not None else None,
                    "primary_position": self.unified_result.primary_position.tolist() if self.unified_result.primary_position is not None else None,
                    "primary_velocity": self.unified_result.primary_velocity.tolist() if self.unified_result.primary_velocity is not None else None,
                    "primary_acceleration": self.unified_result.primary_acceleration.tolist() if self.unified_result.primary_acceleration is not None else None,
                    "secondary_psi": self.unified_result.secondary_psi.tolist() if self.unified_result.secondary_psi is not None else None,
                    "secondary_R_psi": self.unified_result.secondary_R_psi.tolist() if self.unified_result.secondary_R_psi is not None else None,
                    "secondary_base_radius": self.unified_result.secondary_base_radius,
                    "secondary_rod_length": 25.0,  # Fixed connecting rod length for phase 2 simplification
                    "total_solve_time": self.unified_result.total_solve_time,
                    "tertiary_crank_center_x": self.unified_result.tertiary_crank_center_x,
                    "tertiary_crank_center_y": self.unified_result.tertiary_crank_center_y,
                    "tertiary_crank_radius": self.unified_result.tertiary_crank_radius,
                    "tertiary_rod_length": self.unified_result.tertiary_rod_length,
                    "tertiary_torque_output": self.unified_result.tertiary_torque_output,
                    "tertiary_side_load_penalty": self.unified_result.tertiary_side_load_penalty,
                    "tertiary_max_torque": self.unified_result.tertiary_max_torque,
                    "tertiary_power_output": self.unified_result.tertiary_power_output,
                }

                with open(filename, "w") as f:
                    json.dump(result_dict, f, indent=2)

                self.status_var.set(f"Results saved to {filename}")

        except Exception as e:
            self.status_var.set(f"Error saving results: {e}")

    def _reset_parameters(self):
        """Reset all parameters to default values."""
        self.variables["stroke"].set(20.0)
        self.variables["cycle_time"].set(1.0)
        self.variables["upstroke_duration"].set(60.0)
        self.variables["zero_accel_duration"].set(0.0)
        self.variables["motion_type"].set("minimum_jerk")
        self.variables["base_radius"].set(15.0)
        self.variables["connecting_rod_length"].set(25.0)
        self.variables["contact_type"].set("external")
        self.variables["sun_gear_radius"].set(15.0)
        self.variables["ring_gear_radius"].set(45.0)
        self.variables["gear_ratio"].set(3.0)
        self.variables["optimization_method"].set("legendre_collocation")
        self.variables["animation_speed"].set(1.0)

        # Reset tertiary parameters
        self.variables["crank_center_x_min"].set(-50.0)
        self.variables["crank_center_x_max"].set(50.0)
        self.variables["crank_center_y_min"].set(-50.0)
        self.variables["crank_center_y_max"].set(50.0)
        self.variables["crank_radius_min"].set(20.0)
        self.variables["crank_radius_max"].set(100.0)
        self.variables["maximize_torque"].set(True)
        self.variables["minimize_side_loading"].set(True)
        self.variables["minimize_compression_side_load"].set(True)
        self.variables["minimize_combustion_side_load"].set(True)
        self.variables["minimize_torque_ripple"].set(True)
        self.variables["maximize_power_output"].set(True)

        # Clear results and plots
        self.unified_result = None
        self._clear_all_plots()

        # Update initial guesses
        self._on_stroke_changed()

        self.status_var.set("Parameters reset to defaults")


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = CamMotionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
