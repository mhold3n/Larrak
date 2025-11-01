"""
Clean Cam-Ring System Designer GUI

A comprehensive GUI with 5 tabs for three-stage optimization:
1. Motion Law - Linear follower motion visualization (Run 1)
2. Cam/Ring Motion - Motion relationships (Run 2)
3. 2D Profiles - Cam and ring profile plots (Run 2)
4. Animation - 60-frame discrete animation
5. Crank Center - Torque and side-loading optimization (Run 3)
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from campro.logging import get_logger
from campro.optimization.solver_analysis import analyze_ipopt_run

log = get_logger(__name__)


# Validate environment before starting GUI
def _validate_gui_environment():
    """Validate environment and show user-friendly error dialog if needed."""
    # Skip validation if explicitly disabled (avoids HSL solver clobbering warnings)
    if os.getenv("CAMPRO_SKIP_VALIDATION") == "1":
        print("[DEBUG] Environment validation skipped (CAMPRO_SKIP_VALIDATION=1)")
        return
    
    try:
        from campro.environment.validator import (
            validate_casadi_ipopt,
            validate_python_version,
            validate_required_packages,
        )
        
        # Perform lightweight validation without HSL solver tests (which cause clobbering)
        # We skip validate_environment() because it calls validate_hsl_solvers() which
        # creates multiple IPOPT solvers with different linear solvers, causing warnings
        print("[DEBUG] Performing lightweight environment validation (skipping HSL tests)...")
        
        # Check critical dependencies only
        python_ok = validate_python_version()
        casadi_ok = validate_casadi_ipopt()
        packages_ok = validate_required_packages()
        
        # Determine overall status from critical checks only
        if python_ok.status.value == "error" or casadi_ok.status.value == "error":
            overall_status = "error"
        elif any(p.status.value == "error" for p in packages_ok):
            overall_status = "error"
        elif python_ok.status.value == "warning" or casadi_ok.status.value == "warning":
            overall_status = "warning"
        elif any(p.status.value == "warning" for p in packages_ok):
            overall_status = "warning"
        else:
            overall_status = "pass"
        
        # Create a mock results dict with overall_status
        class MockStatus:
            def __init__(self, value):
                self.value = value
        
        overall_status_obj = MockStatus(overall_status)
        results = {"summary": {"overall_status": overall_status_obj}}

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
                icon="warning",
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
                icon="warning",
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


# Skip validation at module load - it will run when user presses optimize button
# _validate_gui_environment()  # Commented out - validation moved to before optimization
from campro.optimization.unified_framework import (  # noqa: E402
    OptimizationMethod,
    UnifiedOptimizationConstraints,
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
    UnifiedOptimizationTargets,
)
from campro.physics.kinematics.litvin_assembly import (  # noqa: E402
    AssemblyInputs,
    compute_assembly_state,
    compute_global_rmax,
    transform_to_world_polar,
)
from campro.physics.kinematics.phase2_relationships import (  # noqa: E402
    Phase2AnimationInputs,
    build_phase2_relationships,
)
from campro.storage import OptimizationRegistry  # noqa: E402

log = get_logger(__name__)


class CamMotionGUI:
    """Clean, focused GUI for cam-ring system design."""

    def __init__(self, root):
        self.root = root
        self.root.title("Cam-Ring System Designer")
        self.root.geometry(
            "1400x1000",
        )  # Increased height to accommodate tertiary controls

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

        # Store latest Ipopt log path for analysis
        self.latest_ipopt_log = None

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
        style.configure(
            "TCombobox",
            fieldbackground="white",
            background="white",
            foreground="black",
            selectbackground="lightblue",
            selectforeground="black",
        )

        # Configure Entry style for better text contrast
        style.configure(
            "TEntry",
            fieldbackground="white",
            background="white",
            foreground="black",
            selectbackground="lightblue",
            selectforeground="black",
        )

        # Configure Button style for better contrast
        style.configure(
            "TButton", background="lightgray", foreground="black", focuscolor="none",
        )

        # Configure Checkbutton style
        style.configure(
            "TCheckbutton",
            background="SystemButtonFace",
            foreground="black",
            focuscolor="none",
        )

        # Configure Label style
        style.configure("TLabel", background="SystemButtonFace", foreground="black")

        # Configure LabelFrame style
        style.configure(
            "TLabelFrame", background="SystemButtonFace", foreground="black",
        )

        # Configure Notebook style
        style.configure("TNotebook", background="SystemButtonFace", foreground="black")

        style.configure(
            "TNotebook.Tab", background="lightgray", foreground="black", padding=[10, 5],
        )

        # Map active tab style
        style.map(
            "TNotebook.Tab",
            background=[("selected", "white")],
            foreground=[("selected", "black")],
        )

        # Configure Frame style for better contrast
        style.configure("TFrame", background="SystemButtonFace")

        # Configure Separator style
        style.configure("TSeparator", background="gray")

        # Additional button styling for better visibility
        style.map(
            "TButton",
            background=[("active", "lightblue"), ("pressed", "darkblue")],
            foreground=[("active", "white"), ("pressed", "white")],
        )

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
            # CasADi physics validation mode
            "enable_casadi_validation_mode": tk.BooleanVar(value=False),
            "casadi_validation_tolerance": tk.DoubleVar(value=1e-4),
            # ORDER2 (CasADi) micro optimization toggle
            "enable_order2_micro": tk.BooleanVar(value=False),
            # CasADi optimization options
            "use_casadi_optimizer": tk.BooleanVar(value=False),
            "enable_warmstart": tk.BooleanVar(value=True),
            "casadi_n_segments": tk.IntVar(value=50),
            "casadi_poly_order": tk.IntVar(value=3),
            "casadi_collocation_method": tk.StringVar(value="legendre"),
            "thermal_efficiency_target": tk.DoubleVar(value=0.55),
            "enable_thermal_efficiency": tk.BooleanVar(value=True),
            # Universal grid and mapper selections
            "universal_n_points": tk.IntVar(value=360),
            "mapper_method": tk.StringVar(value="linear"),  # linear, pchip, barycentric, projection
            # Diagnostics toggles
            "enable_grid_diagnostics": tk.BooleanVar(value=False),
            "enable_grid_plots": tk.BooleanVar(value=False),
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

        # Tab 6: Diagnostics (read-only)
        self.diagnostics_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.diagnostics_frame, text="Diagnostics")
        self._build_diagnostics_tab()

        # Status bar
        self.status_var = tk.StringVar(
            value="Ready - Configure parameters and run optimization",
        )
        self.status_bar = ttk.Label(
            self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN,
        )

    def _create_control_panel(self):
        """Create the main control panel."""
        self.control_frame = ttk.LabelFrame(
            self.main_frame, text="System Parameters", padding="10",
        )

        # Row 1: Core motion law parameters
        ttk.Label(self.control_frame, text="Stroke (mm):").grid(
            row=0, column=0, sticky=tk.W, pady=2,
        )
        self.stroke_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["stroke"], width=8,
        )
        self.stroke_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Upstroke Duration (%):").grid(
            row=0, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.upstroke_entry = ttk.Entry(
            self.control_frame,
            textvariable=self.variables["upstroke_duration"],
            width=8,
        )
        self.upstroke_entry.grid(row=0, column=3, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Zero Accel Duration (%):").grid(
            row=0, column=4, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.zero_accel_entry = ttk.Entry(
            self.control_frame,
            textvariable=self.variables["zero_accel_duration"],
            width=8,
        )
        self.zero_accel_entry.grid(row=0, column=5, sticky=tk.W, padx=(5, 0), pady=2)

        # Row 2: Additional motion law parameters
        ttk.Label(self.control_frame, text="Cycle Time (s):").grid(
            row=1, column=0, sticky=tk.W, pady=2,
        )
        self.cycle_time_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["cycle_time"], width=8,
        )
        self.cycle_time_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Motion Type:").grid(
            row=1, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.motion_type_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.variables["motion_type"],
            values=["minimum_jerk", "minimum_energy", "minimum_time", "pcurve_te"],
            state="readonly",
            width=12,
        )
        self.motion_type_combo.grid(row=1, column=3, sticky=tk.W, padx=(5, 0), pady=2)
        self.motion_type_combo.set("minimum_jerk")  # Set default value

        # Row 3: Cam-ring parameters
        ttk.Label(self.control_frame, text="Cam Base Radius (mm):").grid(
            row=2, column=0, sticky=tk.W, pady=2,
        )
        self.base_radius_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["base_radius"], width=8,
        )
        self.base_radius_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Rod Length (mm):").grid(
            row=2, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.rod_length_entry = ttk.Entry(
            self.control_frame,
            textvariable=self.variables["connecting_rod_length"],
            width=8,
        )
        self.rod_length_entry.grid(row=2, column=3, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Contact Type:").grid(
            row=2, column=4, sticky=tk.W, pady=2, padx=(20, 0),
        )
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
        ttk.Label(self.control_frame, text="Sun Gear Radius (mm):").grid(
            row=3, column=0, sticky=tk.W, pady=2,
        )
        self.sun_gear_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["sun_gear_radius"], width=8,
        )
        self.sun_gear_entry.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Ring Gear Radius (mm):").grid(
            row=3, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.ring_gear_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["ring_gear_radius"], width=8,
        )
        self.ring_gear_entry.grid(row=3, column=3, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Gear Ratio:").grid(
            row=3, column=4, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.gear_ratio_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["gear_ratio"], width=8,
        )
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
        self.reset_button.grid(
            row=4, column=2, columnspan=2, sticky=tk.W, pady=10, padx=(20, 0),
        )

        self.save_button = ttk.Button(
            self.control_frame,
            text="Save Results",
            command=self._save_results,
        )
        self.save_button.grid(
            row=4, column=4, columnspan=2, sticky=tk.W, pady=10, padx=(20, 0),
        )

        # Diagnose button: quick access to Ipopt/solver diagnostics
        self.diagnose_button = ttk.Button(
            self.control_frame,
            text="Diagnose NLP",
            command=self._diagnose_nlp,
        )
        self.diagnose_button.grid(row=4, column=6, sticky=tk.W, pady=10, padx=(20, 0))

        # Row 5: Tertiary optimization header
        ttk.Separator(self.control_frame, orient="horizontal").grid(
            row=5, column=0, columnspan=6, sticky="ew", pady=5,
        )
        ttk.Label(
            self.control_frame,
            text="Crank Center Optimization (Run 3)",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=6, column=0, columnspan=6, sticky=tk.W, pady=2)

        # Row 6: Crank center position bounds
        ttk.Label(self.control_frame, text="Crank X Range (mm):").grid(
            row=7, column=0, sticky=tk.W, pady=2,
        )
        crank_x_frame = ttk.Frame(self.control_frame)
        crank_x_frame.grid(row=7, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Entry(
            crank_x_frame, textvariable=self.variables["crank_center_x_min"], width=5,
        ).pack(side=tk.LEFT)
        ttk.Label(crank_x_frame, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(
            crank_x_frame, textvariable=self.variables["crank_center_x_max"], width=5,
        ).pack(side=tk.LEFT)

        ttk.Label(self.control_frame, text="Crank Y Range (mm):").grid(
            row=7, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        crank_y_frame = ttk.Frame(self.control_frame)
        crank_y_frame.grid(row=7, column=3, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Entry(
            crank_y_frame, textvariable=self.variables["crank_center_y_min"], width=5,
        ).pack(side=tk.LEFT)
        ttk.Label(crank_y_frame, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(
            crank_y_frame, textvariable=self.variables["crank_center_y_max"], width=5,
        ).pack(side=tk.LEFT)

        # Row 7: Crank radius bounds
        ttk.Label(self.control_frame, text="Crank Radius Range (mm):").grid(
            row=8, column=0, sticky=tk.W, pady=2,
        )
        crank_r_frame = ttk.Frame(self.control_frame)
        crank_r_frame.grid(row=8, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Entry(
            crank_r_frame, textvariable=self.variables["crank_radius_min"], width=5,
        ).pack(side=tk.LEFT)
        ttk.Label(crank_r_frame, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(
            crank_r_frame, textvariable=self.variables["crank_radius_max"], width=5,
        ).pack(side=tk.LEFT)

        # Row 8: Optimization objectives (checkboxes)
        ttk.Label(self.control_frame, text="Objectives:").grid(
            row=9, column=0, sticky=tk.W, pady=2,
        )
        objectives_frame = ttk.Frame(self.control_frame)
        objectives_frame.grid(
            row=9, column=1, columnspan=5, sticky=tk.W, padx=(5, 0), pady=2,
        )

        ttk.Checkbutton(
            objectives_frame,
            text="Max Torque",
            variable=self.variables["maximize_torque"],
        ).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(
            objectives_frame,
            text="Min Side Load",
            variable=self.variables["minimize_side_loading"],
        ).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(
            objectives_frame,
            text="Min Compression Side Load",
            variable=self.variables["minimize_compression_side_load"],
        ).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(
            objectives_frame,
            text="Min Combustion Side Load",
            variable=self.variables["minimize_combustion_side_load"],
        ).pack(side=tk.LEFT, padx=5)

        # Row 9: CasADi validation mode controls
        ttk.Label(self.control_frame, text="CasADi Validation:").grid(
            row=10, column=0, sticky=tk.W, pady=2,
        )
        validation_frame = ttk.Frame(self.control_frame)
        validation_frame.grid(
            row=10, column=1, columnspan=3, sticky=tk.W, padx=(5, 0), pady=2,
        )
        ttk.Checkbutton(
            validation_frame,
            text="Enable Validation Mode",
            variable=self.variables["enable_casadi_validation_mode"],
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(validation_frame, text="Tolerance:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(
            validation_frame,
            textvariable=self.variables["casadi_validation_tolerance"],
            width=8,
        ).pack(side=tk.LEFT, padx=5)

        # Row 10: ORDER2 (CasADi) toggle
        ttk.Label(self.control_frame, text="ORDER2 (CasADi):").grid(
            row=11, column=0, sticky=tk.W, pady=2,
        )
        order2_frame = ttk.Frame(self.control_frame)
        order2_frame.grid(
            row=11, column=1, columnspan=3, sticky=tk.W, padx=(5, 0), pady=2,
        )
        ttk.Checkbutton(
            order2_frame,
            text="Enable ORDER2 (micro)",
            variable=self.variables["enable_order2_micro"],
        ).pack(side=tk.LEFT, padx=5)

        # Row 11: CasADi Optimization Options
        ttk.Separator(self.control_frame, orient="horizontal").grid(
            row=12, column=0, columnspan=6, sticky="ew", pady=5,
        )
        ttk.Label(
            self.control_frame,
            text="CasADi Optimization Options",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=13, column=0, columnspan=6, sticky=tk.W, pady=2)

        # CasADi optimizer toggle
        ttk.Label(self.control_frame, text="CasADi Optimizer:").grid(
            row=14, column=0, sticky=tk.W, pady=2,
        )
        casadi_frame = ttk.Frame(self.control_frame)
        casadi_frame.grid(
            row=14, column=1, columnspan=5, sticky=tk.W, padx=(5, 0), pady=2,
        )
        ttk.Checkbutton(
            casadi_frame,
            text="Use CasADi Optimizer",
            variable=self.variables["use_casadi_optimizer"],
        ).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(
            casadi_frame,
            text="Enable Warm-start",
            variable=self.variables["enable_warmstart"],
        ).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(
            casadi_frame,
            text="Thermal Efficiency",
            variable=self.variables["enable_thermal_efficiency"],
        ).pack(side=tk.LEFT, padx=5)

        # CasADi parameters
        ttk.Label(self.control_frame, text="Segments:").grid(
            row=15, column=0, sticky=tk.W, pady=2,
        )
        ttk.Entry(
            self.control_frame,
            textvariable=self.variables["casadi_n_segments"],
            width=8,
        ).grid(row=15, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Poly Order:").grid(
            row=15, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        ttk.Entry(
            self.control_frame,
            textvariable=self.variables["casadi_poly_order"],
            width=8,
        ).grid(row=15, column=3, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Method:").grid(
            row=15, column=4, sticky=tk.W, pady=2, padx=(20, 0),
        )
        casadi_method_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.variables["casadi_collocation_method"],
            values=["legendre", "radau"],
            state="readonly",
            width=10,
        )
        casadi_method_combo.grid(row=15, column=5, sticky=tk.W, padx=(5, 0), pady=2)
        casadi_method_combo.set("legendre")

        # Universal grid controls
        ttk.Label(self.control_frame, text="Universal Points:").grid(
            row=15, column=6, sticky=tk.W, pady=2, padx=(20, 0),
        )
        universal_points_entry = ttk.Spinbox(
            self.control_frame,
            from_=90,
            to=4096,
            increment=10,
            textvariable=self.variables["universal_n_points"],
            width=6,
        )
        universal_points_entry.grid(row=15, column=7, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Mapper Method:").grid(
            row=15, column=8, sticky=tk.W, pady=2, padx=(20, 0),
        )
        mapper_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.variables["mapper_method"],
            values=["linear", "pchip", "barycentric", "projection"],
            state="readonly",
            width=12,
        )
        mapper_combo.grid(row=15, column=9, sticky=tk.W, padx=(5, 0), pady=2)
        mapper_combo.set("linear")

        # Diagnostics toggles
        diag_frame = ttk.Frame(self.control_frame)
        diag_frame.grid(row=16, column=0, columnspan=10, sticky=tk.W, pady=(4, 2))
        diag_chk = ttk.Checkbutton(
            diag_frame,
            text="Grid Diagnostics",
            variable=self.variables["enable_grid_diagnostics"],
        )
        diag_chk.pack(side=tk.LEFT, padx=(0, 10))
        plots_chk = ttk.Checkbutton(
            diag_frame,
            text="Grid Plots",
            variable=self.variables["enable_grid_plots"],
        )
        plots_chk.pack(side=tk.LEFT)

        # Thermal efficiency target
        ttk.Label(self.control_frame, text="Efficiency Target:").grid(
            row=16, column=0, sticky=tk.W, pady=2,
        )
        ttk.Entry(
            self.control_frame,
            textvariable=self.variables["thermal_efficiency_target"],
            width=8,
        ).grid(row=16, column=1, sticky=tk.W, padx=(5, 0), pady=2)

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

    def _build_diagnostics_tab(self):
        """Create read-only diagnostics UI elements."""
        # Summary group
        summary = ttk.LabelFrame(self.diagnostics_frame, text="Summary", padding="10")
        summary.pack(fill=tk.X, expand=False)

        self.diag_run_id_var = tk.StringVar(value="-")
        self.diag_status_var = tk.StringVar(value="-")
        self.diag_iter_var = tk.StringVar(value="0")

        ttk.Label(summary, text="Run ID:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(summary, textvariable=self.diag_run_id_var).grid(
            row=0, column=1, sticky=tk.W, pady=2,
        )

        ttk.Label(summary, text="Status:").grid(
            row=0, column=2, sticky=tk.W, padx=(20, 0), pady=2,
        )
        ttk.Label(summary, textvariable=self.diag_status_var).grid(
            row=0, column=3, sticky=tk.W, pady=2,
        )

        ttk.Label(summary, text="Iterations:").grid(
            row=0, column=4, sticky=tk.W, padx=(20, 0), pady=2,
        )
        ttk.Label(summary, textvariable=self.diag_iter_var).grid(
            row=0, column=5, sticky=tk.W, pady=2,
        )

        # Residuals group
        residuals = ttk.LabelFrame(
            self.diagnostics_frame, text="Residuals", padding="10",
        )
        residuals.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.diag_residuals_text = tk.Text(residuals, height=10, wrap="none")
        self.diag_residuals_text.pack(fill=tk.BOTH, expand=True)
        self.diag_residuals_text.config(state="disabled")

        # KKT/Constraint stats group
        kkt_frame = ttk.LabelFrame(
            self.diagnostics_frame, text="KKT / Constraint Stats", padding="10",
        )
        kkt_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.diag_kkt_text = tk.Text(kkt_frame, height=6, wrap="none")
        self.diag_kkt_text.pack(fill=tk.BOTH, expand=True)
        self.diag_kkt_text.config(state="disabled")

        # Artifacts group
        artifacts = ttk.LabelFrame(
            self.diagnostics_frame, text="Artifacts", padding="10",
        )
        artifacts.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.diag_artifacts_text = tk.Text(artifacts, height=6, wrap="none")
        self.diag_artifacts_text.pack(fill=tk.BOTH, expand=True)
        self.diag_artifacts_text.config(state="disabled")
        # Buttons for quick actions
        art_btns = ttk.Frame(artifacts)
        art_btns.pack(fill=tk.X, expand=False, pady=(6, 0))
        ttk.Button(
            art_btns,
            text="Open Ipopt Log",
            command=lambda: self._open_artifact("ipopt_log"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            art_btns,
            text="Copy Log Path",
            command=lambda: self._copy_artifact("ipopt_log"),
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(
            art_btns,
            text="Open Run Meta",
            command=lambda: self._open_artifact("run_meta"),
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            art_btns,
            text="Copy Meta Path",
            command=lambda: self._copy_artifact("run_meta"),
        ).pack(side=tk.LEFT)

        # Feasibility group
        feas = ttk.LabelFrame(
            self.diagnostics_frame, text="Feasibility (Phase 0)", padding="10",
        )
        feas.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.diag_feas_text = tk.Text(feas, height=8, wrap="none")
        self.diag_feas_text.pack(fill=tk.BOTH, expand=True)
        self.diag_feas_text.config(state="disabled")

    def _update_diagnostics_tab(self, solve_report):
        """Populate diagnostics tab from a SolveReport-like object."""
        try:
            if not solve_report:
                return
            # Summary
            self.diag_run_id_var.set(getattr(solve_report, "run_id", "-") or "-")
            self.diag_status_var.set(getattr(solve_report, "status", "-") or "-")
            try:
                self.diag_iter_var.set(
                    str(int(getattr(solve_report, "n_iter", 0) or 0)),
                )
            except Exception:
                self.diag_iter_var.set("0")

            # Residuals
            try:
                self.diag_residuals_text.config(state="normal")
                self.diag_residuals_text.delete("1.0", tk.END)
                res = getattr(solve_report, "residuals", {}) or {}
                if res:
                    # Stable order for common keys first
                    keys = [
                        "primal_inf",
                        "dual_inf",
                        "kkt_error",
                        "constraint_violation",
                    ]
                    printed = set()
                    for k in keys:
                        if k in res:
                            try:
                                self.diag_residuals_text.insert(
                                    tk.END, f"{k}: {float(res[k]):.6e}\n",
                                )
                                printed.add(k)
                            except Exception:
                                pass
                    # Print remaining keys
                    for k, v in res.items():
                        if k in printed:
                            continue
                        try:
                            if isinstance(v, (int, float)):
                                self.diag_residuals_text.insert(
                                    tk.END, f"{k}: {float(v):.6e}\n",
                                )
                            else:
                                self.diag_residuals_text.insert(tk.END, f"{k}: {v}\n")
                        except Exception:
                            pass
                else:
                    self.diag_residuals_text.insert(tk.END, "No residuals available.\n")
            finally:
                self.diag_residuals_text.config(state="disabled")

            # Artifacts
            try:
                self.diag_artifacts_text.config(state="normal")
                self.diag_artifacts_text.delete("1.0", tk.END)
                arts = getattr(solve_report, "artifacts", {}) or {}
                if arts:
                    for k, v in arts.items():
                        self.diag_artifacts_text.insert(tk.END, f"{k}: {v}\n")
                else:
                    self.diag_artifacts_text.insert(tk.END, "No artifacts recorded.\n")
            finally:
                self.diag_artifacts_text.config(state="disabled")

            # KKT stats
            try:
                self.diag_kkt_text.config(state="normal")
                self.diag_kkt_text.delete("1.0", tk.END)
                kkt = getattr(solve_report, "kkt", {}) or {}
                if kkt:
                    order = ["primal_inf", "dual_inf", "compl_inf"]
                    printed = set()
                    for k in order:
                        if k in kkt:
                            try:
                                self.diag_kkt_text.insert(
                                    tk.END, f"{k}: {float(kkt[k]):.6e}\n",
                                )
                                printed.add(k)
                            except Exception:
                                pass
                    for k, v in kkt.items():
                        if k in printed:
                            continue
                        try:
                            self.diag_kkt_text.insert(tk.END, f"{k}: {float(v):.6e}\n")
                        except Exception:
                            self.diag_kkt_text.insert(tk.END, f"{k}: {v}\n")
                else:
                    self.diag_kkt_text.insert(tk.END, "No KKT stats available.\n")
            finally:
                self.diag_kkt_text.config(state="disabled")

            # Feasibility (from unified_result if available)
            try:
                self.diag_feas_text.config(state="normal")
                self.diag_feas_text.delete("1.0", tk.END)
                feas_info = None
                if hasattr(self, "unified_result") and self.unified_result is not None:
                    ci = getattr(self.unified_result, "convergence_info", {}) or {}
                    feas_info = ci.get("feasibility_primary")
                if isinstance(feas_info, dict):
                    feasible = feas_info.get("feasible", False)
                    max_violation = feas_info.get("max_violation", 0.0)
                    self.diag_feas_text.insert(tk.END, f"Feasible: {feasible}\n")
                    self.diag_feas_text.insert(
                        tk.END, f"Max Violation: {max_violation:.3e}\n",
                    )
                    viol = feas_info.get("violations", {}) or {}
                    if viol:
                        self.diag_feas_text.insert(tk.END, "Violations:\n")
                        for k, v in viol.items():
                            try:
                                self.diag_feas_text.insert(
                                    tk.END, f"  - {k}: {float(v):.3e}\n",
                                )
                            except Exception:
                                self.diag_feas_text.insert(tk.END, f"  - {k}: {v}\n")
                    recs = feas_info.get("recommendations", []) or []
                    if recs:
                        self.diag_feas_text.insert(tk.END, "Recommendations:\n")
                        for r in recs:
                            self.diag_feas_text.insert(tk.END, f"  - {r}\n")
                else:
                    self.diag_feas_text.insert(
                        tk.END, "Feasibility info not available.\n",
                    )
            finally:
                self.diag_feas_text.config(state="disabled")
        except Exception as e:
            print(f"DEBUG: Failed to update diagnostics tab: {e}")

    # --- Diagnostics helpers: artifact open/copy ---------------------------------
    def _get_artifact_path(self, kind: str) -> str | None:
        try:
            if (
                hasattr(self, "solve_report")
                and self.solve_report
                and getattr(self.solve_report, "artifacts", None)
            ):
                path = self.solve_report.artifacts.get(kind)
                if path:
                    return str(path)
        except Exception:
            pass
        if kind == "ipopt_log" and getattr(self, "latest_ipopt_log", None):
            return self.latest_ipopt_log
        return None

    def _open_artifact(self, kind: str) -> None:
        path = self._get_artifact_path(kind)
        if not path:
            self.status_var.set(f"No {kind.replace('_', ' ')} path available")
            return
        p = Path(path)
        if not p.exists():
            self.status_var.set(f"Path not found: {p}")
            return
        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", str(p)], check=False)
            elif os.name == "nt":  # Windows
                os.startfile(str(p))  # type: ignore[attr-defined]  # noqa: S606
            else:
                subprocess.run(["xdg-open", str(p)], check=False)
            self.status_var.set(f"Opened: {p}")
        except Exception as e:
            self.status_var.set(f"Open failed: {e}")

    def _copy_artifact(self, kind: str) -> None:
        path = self._get_artifact_path(kind)
        if not path:
            self.status_var.set(f"No {kind.replace('_', ' ')} path available")
            return
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(path)
            self.root.update()  # Keep clipboard after window loses focus
            self.status_var.set(f"Copied path: {path}")
        except Exception as e:
            self.status_var.set(f"Copy failed: {e}")

    def _setup_plots(self):
        """Setup matplotlib plots for each tab."""
        # Tab 1: Motion Law
        self.motion_law_fig = Figure(figsize=(12, 6), dpi=100)
        self.motion_law_canvas = FigureCanvasTkAgg(
            self.motion_law_fig, self.motion_law_frame,
        )
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
        self.animation_canvas = FigureCanvasTkAgg(
            self.animation_fig, self.animation_frame,
        )
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

        ttk.Label(self.animation_control_frame, text="Speed:").pack(
            side=tk.LEFT, padx=(20, 5),
        )
        self.speed_scale = ttk.Scale(
            self.animation_control_frame,
            from_=0.1,
            to=3.0,
            variable=self.variables["animation_speed"],
            orient=tk.HORIZONTAL,
            length=100,
        )
        self.speed_scale.pack(side=tk.LEFT, padx=5)

        # Frame count will be displayed dynamically based on collocation points
        self.frames_label = ttk.Label(
            self.animation_control_frame, text="Frames: Auto (from collocation)",
        )
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
        ax.text(
            0.5,
            0.5,
            "Motion Law\n\nRun optimization to see results",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title("Linear Follower Motion Law")
        self.motion_law_canvas.draw()

        # Cam/Ring Motion tab
        self.motion_fig.clear()
        ax = self.motion_fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Cam/Ring Motion\n\nRun optimization to see results",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title("Cam and Ring Motion Relationships")
        self.motion_canvas.draw()

        # 2D Profiles tab
        self.profiles_fig.clear()
        ax = self.profiles_fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "2D Profiles\n\nRun optimization to see results",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title("Cam and Ring 2D Profiles")
        self.profiles_canvas.draw()

        # Animation tab - Two subplots
        self.animation_fig.clear()

        # Left subplot: Full assembly view
        ax1 = self.animation_fig.add_subplot(121, projection="polar")
        ax1.text(
            0.5,
            0.5,
            "Full Assembly View\n\nRun optimization to see collocation-based animation",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax1.transAxes,
        )
        ax1.set_title("Full Assembly View")

        # Right subplot: Detailed tooth meshing view
        ax2 = self.animation_fig.add_subplot(122, projection="polar")
        ax2.text(
            0.5,
            0.5,
            "Tooth Contact Detail\n\nShows 3-4 teeth at contact point",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax2.transAxes,
        )
        ax2.set_title("Tooth Contact Detail")

        self.animation_fig.suptitle(
            "Collocation-Based Animation", fontsize=14, fontweight="bold",
        )
        self.animation_canvas.draw()

        # Crank Center Optimization tab
        self.tertiary_fig.clear()
        ax = self.tertiary_fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Crank Center Optimization\n\nRun optimization to see results",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
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
                self.status_var.set(
                    "Error: Upstroke duration must be between 0 and 100%",
                )
                return False

            if zero_accel_duration < 0 or zero_accel_duration > 100:
                self.status_var.set(
                    "Error: Zero acceleration duration must be between 0 and 100%",
                )
                return False

            if zero_accel_duration > upstroke_duration:
                self.status_var.set(
                    "Warning: Zero acceleration duration exceeds upstroke duration",
                )

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

    def _validate_environment_before_optimization(self) -> bool:
        """Validate environment (CasADi/HSL) before optimization starts.
        
        Returns:
            True if validation passes, False otherwise.
        """
        try:
            # 1) Check CasADi import
            try:
                import casadi  # noqa: F401
                casadi_ok = True
            except ImportError as e:
                casadi_ok = False
                error_msg = (
                    "CasADi module not found.\n\n"
                    "Please ensure CasADi is installed or select the folder containing the 'casadi' package."
                )
                result = messagebox.askyesno(
                    "CasADi Not Found",
                    error_msg + "\n\nWould you like to select the CasADi folder?",
                    icon="warning",
                )
                if result:
                    pkg_dir = filedialog.askdirectory(title="Select folder containing 'casadi' package")
                    if pkg_dir:
                        if pkg_dir not in sys.path:
                            sys.path.insert(0, pkg_dir)
                        try:
                            import importlib
                            importlib.invalidate_caches()
                            import casadi  # type: ignore  # noqa: F401
                            casadi_ok = True
                        except Exception:
                            messagebox.showerror(
                                "CasADi Import Failed",
                                f"Failed to import CasADi from selected folder.\n\nError: {e}",
                            )
                            return False
                else:
                    return False

            # 2) Check HSL library path
            try:
                from campro import constants as _c
                hsl_path = getattr(_c, "HSLLIB_PATH", "")
            except Exception:
                hsl_path = ""

            def _valid_hsl(p: str) -> bool:
                return bool(p) and Path(p).exists()

            if not _valid_hsl(hsl_path):
                error_msg = (
                    "HSL (libcoinhsl) library path not set or invalid.\n\n"
                    "Please select the HSL library file (DLL/DYLIB/SO)."
                )
                result = messagebox.askyesno(
                    "HSL Library Not Found",
                    error_msg + "\n\nWould you like to select the HSL library file?",
                    icon="warning",
                )
                if result:
                    filetypes = [
                        ("Windows DLL", "*.dll"),
                        ("macOS dylib", "*.dylib"),
                        ("Linux so", "*.so"),
                        ("All files", "*.*"),
                    ]
                    sel = filedialog.askopenfilename(
                        title="Select HSL library (libcoinhsl)", filetypes=filetypes
                    )
                    if sel:
                        os.environ["HSLLIB_PATH"] = sel
                        try:
                            from campro import constants as _c2
                            _c2.HSLLIB_PATH = sel  # override runtime constant
                        except Exception:
                            pass
                        hsl_path = sel
                    else:
                        return False
                else:
                    return False

            # Both checks passed
            return True

        except Exception as e:
            messagebox.showerror(
                "Validation Error",
                f"Error during environment validation:\n\n{e}",
            )
            return False

    def _run_optimization(self):
        """Run the complete system optimization."""
        if not self._validate_inputs():
            return

        # Validate environment before optimization
        if not self._validate_environment_before_optimization():
            self.status_var.set("Optimization cancelled - environment validation failed")
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
                "zero_accel_duration_percent": self.variables[
                    "zero_accel_duration"
                ].get(),
                "motion_type": self.variables["motion_type"].get(),
            }

            print(f"DEBUG: System optimization input data: {input_data}")

            # Perform cascaded optimization
            print("DEBUG: Starting cascaded optimization...")
            result_data = self.unified_framework.optimize_cascaded(input_data)
            print("DEBUG: Cascaded optimization completed successfully!")

            # Store the result and convert to SolveReport for diagnostics
            self.unified_result = result_data
            try:
                from campro.api.adapters import unified_data_to_solve_report

                self.solve_report = unified_data_to_solve_report(result_data)
                # Cache latest Ipopt log for diagnostics panel
                try:
                    if hasattr(self.solve_report, "artifacts") and isinstance(
                        self.solve_report.artifacts, dict,
                    ):
                        self.latest_ipopt_log = self.solve_report.artifacts.get(
                            "ipopt_log", None,
                        )
                except Exception:
                    pass
                # Update diagnostics tab on main thread
                self.root.after(0, self._update_diagnostics_tab, self.solve_report)
            except Exception as _e:
                # Keep GUI responsive even if adapter fails
                print(f"DEBUG: SolveReport adapter failed: {_e}")

            # Update plots on main thread
            self.root.after(0, self._update_all_plots, result_data)

            # Update frames label if we have frame count
            if hasattr(self, "_total_frames"):
                self.root.after(
                    0,
                    lambda: self.frames_label.config(
                        text=f"Frames: {self._total_frames} (collocation points)",
                    ),
                )

            # Re-enable optimize button
            self.root.after(0, self._enable_optimize_button)

        except Exception as e:
            print(f"DEBUG: Error in optimization thread: {e}")
            import traceback

            traceback.print_exc()
            error_message = str(e)
            self.root.after(0, self._enable_optimize_button)
            self.root.after(
                0, lambda: self.status_var.set(f"Optimization failed: {error_message}"),
            )

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
        )

        # Enable Ipopt analysis for MA57 readiness grading
        settings.enable_ipopt_analysis = True

        # Thermal efficiency: only enable if explicitly requested (not hardcoded)
        # Default to False unless a GUI control exists and is checked
        settings.use_thermal_efficiency = bool(
            self.variables.get("use_thermal_efficiency", tk.BooleanVar(value=False)).get()
        ) if "use_thermal_efficiency" in self.variables else False

        # Configure CasADi validation mode from GUI
        settings.enable_casadi_validation_mode = self.variables[
            "enable_casadi_validation_mode"
        ].get()
        settings.casadi_validation_tolerance = self.variables[
            "casadi_validation_tolerance"
        ].get()

        # The Lobatto/Radau/Legendre dropdown controls the SHARED collocation method used
        # across all modules (primary motion-law and MotionConstraints). This is a temporary
        # global toggle for simplicity. In a future session we will support granular, per-stage
        # method selection (e.g., primary=Radau, secondary=Lobatto) and per-stage degrees.
        settings.collocation_method = self.variables["casadi_collocation_method"].get()

        # Universal grid controls from GUI
        settings.universal_n_points = int(self.variables["universal_n_points"].get())

        # Mapper method selection from GUI (linear, pchip, barycentric, projection)
        # Currently used inside invariance mappings and available for future stage wrappers.
        settings.mapper_method = self.variables["mapper_method"].get()

        # Diagnostics toggles from GUI
        settings.enable_grid_diagnostics = bool(self.variables["enable_grid_diagnostics"].get())
        settings.enable_grid_plots = bool(self.variables["enable_grid_plots"].get())

        # Exposing optimization weights in the GUI:
        # - To tune secondary tracking toward the golden profile, surface a numeric input bound to
        #   a Tk variable (e.g., `tracking_weight_var = tk.DoubleVar(value=1.0)`) and assign here:
        #   `settings.tracking_weight = tracking_weight_var.get()`
        # - Similarly, the primary phase-1 flow supports weights like `dpdt_weight`, `jerk_weight`,
        #   and `imep_weight` in `UnifiedOptimizationSettings`. These can be exposed with sliders
        #   and assigned before calling `configure()` below, enabling quick experimentation from the GUI.

        # Configure CasADi optimizer settings
        settings.use_casadi = self.variables["use_casadi_optimizer"].get()
        if settings.use_casadi:
            settings.casadi_n_segments = self.variables["casadi_n_segments"].get()
            settings.casadi_poly_order = self.variables["casadi_poly_order"].get()
            settings.casadi_collocation_method = self.variables[
                "casadi_collocation_method"
            ].get()
            settings.enable_warmstart = self.variables["enable_warmstart"].get()
            settings.thermal_efficiency_target = self.variables[
                "thermal_efficiency_target"
            ].get()
            settings.enable_thermal_efficiency = self.variables[
                "enable_thermal_efficiency"
            ].get()

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
            minimize_side_loading_during_compression=self.variables[
                "minimize_compression_side_load"
            ].get(),
            minimize_side_loading_during_combustion=self.variables[
                "minimize_combustion_side_load"
            ].get(),
            minimize_torque_ripple=self.variables["minimize_torque_ripple"].get(),
            maximize_power_output=self.variables["maximize_power_output"].get(),
        )

        # Configure framework
        self.unified_framework.configure(
            settings=settings, constraints=constraints, targets=targets,
        )

        # Enable ORDER2_MICRO (CasADi) if requested via checkbox
        try:
            enable_order2 = bool(self.variables["enable_order2_micro"].get())
            if (
                hasattr(self.unified_framework, "secondary_optimizer")
                and self.unified_framework.secondary_optimizer is not None
            ):
                self.unified_framework.secondary_optimizer.enable_order2_micro = (
                    enable_order2
                )
                log.info(f"ORDER2_MICRO enabled: {enable_order2}")
        except Exception as _e:
            print(f"DEBUG: Failed to set ORDER2 flag: {_e}")

        # Enable CasADi validation mode if requested
        if settings.enable_casadi_validation_mode:
            self.unified_framework.enable_casadi_validation_mode(
                settings.casadi_validation_tolerance,
            )
            print(
                f"DEBUG: CasADi validation mode enabled with tolerance: {settings.casadi_validation_tolerance}",
            )
        else:
            self.unified_framework.disable_casadi_validation_mode()

        print(f"DEBUG: Configured unified framework with method: {method_name}")

    def _update_all_plots(self, result_data):
        """Update all plots with optimization results."""
        try:
            print("DEBUG: Updating all plots with result data")
            print(f"DEBUG: Result data type: {type(result_data)}")
            print(f"DEBUG: Result data attributes: {dir(result_data)}")

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
            self.status_var.set(
                f"Optimization completed - Total time: {result_data.total_solve_time:.3f}s",
            )

            # Append residuals summary from SolveReport if available
            try:
                sr = getattr(self, "solve_report", None)
                if sr and getattr(sr, "residuals", None):
                    # Prefer common keys; fallback to first two residuals
                    ordered = []
                    for k in (
                        "primal_inf",
                        "dual_inf",
                        "kkt_error",
                        "constraint_violation",
                    ):
                        if k in sr.residuals:
                            try:
                                ordered.append(f"{k}={float(sr.residuals[k]):.1e}")
                            except Exception:
                                pass
                    if not ordered:
                        items = list(sr.residuals.items())[:2]
                        ordered = [
                            f"{k}={float(v):.1e}"
                            for k, v in items
                            if isinstance(v, (int, float))
                        ]
                    if ordered:
                        cur = self.status_var.get()
                        self.status_var.set(cur + " | Residuals: " + ", ".join(ordered))
            except Exception as _e:
                print(f"DEBUG: Failed to append residuals to status: {_e}")

            # Show detailed per-phase analysis
            print("DEBUG: Showing detailed analysis...")
            self._show_detailed_analysis(result_data)

            # Show MA57 readiness analysis
            print("DEBUG: Showing MA57 readiness analysis...")
            self._show_ma57_readiness()

            print("DEBUG: All plots updated successfully!")

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

        if (
            result_data.primary_theta is not None
            and result_data.primary_position is not None
        ):
            # Create subplots for position, velocity, acceleration, and jerk
            ax1 = self.motion_law_fig.add_subplot(411)
            ax2 = self.motion_law_fig.add_subplot(412)
            ax3 = self.motion_law_fig.add_subplot(413)
            ax4 = self.motion_law_fig.add_subplot(414)

            # Plot position
            ax1.plot(
                result_data.primary_theta,
                result_data.primary_position,
                "b-",
                linewidth=2,
                label="Position",
            )
            ax1.set_ylabel("Position (mm)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Plot velocity
            if result_data.primary_velocity is not None:
                ax2.plot(
                    result_data.primary_theta,
                    result_data.primary_velocity,
                    "g-",
                    linewidth=2,
                    label="Velocity",
                )
                ax2.set_ylabel("Velocity (mm/s)")
                ax2.grid(True, alpha=0.3)
                ax2.legend()

            # Plot acceleration
            if result_data.primary_acceleration is not None:
                ax3.plot(
                    result_data.primary_theta,
                    result_data.primary_acceleration,
                    "r-",
                    linewidth=2,
                    label="Acceleration",
                )
                ax3.set_ylabel("Acceleration (mm/s²)")
                ax3.grid(True, alpha=0.3)
                ax3.legend()

            # Plot jerk
            if result_data.primary_jerk is not None:
                ax4.plot(
                    result_data.primary_theta,
                    result_data.primary_jerk,
                    "m-",
                    linewidth=2,
                    label="Jerk",
                )
                ax4.set_ylabel("Jerk (mm/s³)")
                ax4.set_xlabel("Cam Angle (degrees)")
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            # If jerk data is not available, compute it from acceleration
            elif (
                result_data.primary_acceleration is not None
                and len(result_data.primary_acceleration) > 1
            ):
                # Compute jerk as derivative of acceleration
                jerk = np.gradient(
                    result_data.primary_acceleration, result_data.primary_theta,
                )
                ax4.plot(
                    result_data.primary_theta,
                    jerk,
                    "m-",
                    linewidth=2,
                    label="Jerk (computed)",
                )
                ax4.set_ylabel("Jerk (mm/s³)")
                ax4.set_xlabel("Cam Angle (degrees)")
                ax4.grid(True, alpha=0.3)
                ax4.legend()
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "Jerk data not available",
                    ha="center",
                    va="center",
                    fontsize=10,
                )
                ax4.set_ylabel("Jerk (mm/s³)")
                ax4.set_xlabel("Cam Angle (degrees)")

            self.motion_law_fig.suptitle(
                "Linear Follower Motion Law", fontsize=14, fontweight="bold",
            )
        else:
            ax = self.motion_law_fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No motion law data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Linear Follower Motion Law")

        self.motion_law_fig.tight_layout()
        self.motion_law_canvas.draw()

    def _update_motion_plot(self, result_data):
        """Update the cam/ring motion plot (Tab 2)."""
        self.motion_fig.clear()

        if (
            result_data.secondary_psi is not None
            and result_data.secondary_R_psi is not None
        ):
            # Create subplots for ring motion and cam motion
            ax1 = self.motion_fig.add_subplot(211)
            ax2 = self.motion_fig.add_subplot(212)

            # Plot ring motion (psi vs R_psi)
            ax1.plot(
                result_data.secondary_psi,
                result_data.secondary_R_psi,
                "purple",
                linewidth=2,
                label="Ring Radius",
            )
            ax1.set_ylabel("Ring Radius (mm)")
            ax1.set_xlabel("Ring Angle ψ (rad)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_title("Ring Follower Motion")

            # Plot cam motion (if available)
            if result_data.secondary_cam_curves is not None:
                cam_theta = result_data.secondary_cam_curves.get(
                    "theta", result_data.primary_theta,
                )
                cam_radius = result_data.secondary_cam_curves.get("profile_radius")
                if cam_radius is not None:
                    ax2.plot(
                        cam_theta, cam_radius, "orange", linewidth=2, label="Cam Radius",
                    )
                    ax2.set_ylabel("Cam Radius (mm)")
                    ax2.set_xlabel("Cam Angle θ (degrees)")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    ax2.set_title("Cam Profile Motion")

            self.motion_fig.suptitle(
                "Cam and Ring Motion Relationships", fontsize=14, fontweight="bold",
            )
        else:
            ax = self.motion_fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No cam/ring motion data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Cam and Ring Motion Relationships")

        self.motion_fig.tight_layout()
        self.motion_canvas.draw()

    def _update_profiles_plot(self, result_data):
        """Update the 2D profiles plot (Tab 3)."""
        self.profiles_fig.clear()

        if (
            result_data.secondary_cam_curves is not None
            and result_data.secondary_psi is not None
        ):
            # Create subplots for cam profile and ring profile
            ax1 = self.profiles_fig.add_subplot(121, projection="polar")
            ax2 = self.profiles_fig.add_subplot(122, projection="polar")

            # Plot cam profile (polar) - use the theta from cam_curves
            if (
                "theta" in result_data.secondary_cam_curves
                and "profile_radius" in result_data.secondary_cam_curves
            ):
                cam_theta_rad = np.radians(result_data.secondary_cam_curves["theta"])
                cam_radius = result_data.secondary_cam_curves["profile_radius"]
                ax1.plot(
                    cam_theta_rad,
                    cam_radius,
                    "orange",
                    linewidth=2,
                    label="Cam Profile",
                )
                ax1.set_title("Cam Profile (Polar)", pad=20)
                ax1.grid(True)
                ax1.set_ylim(0, max(cam_radius) * 1.1)  # Set appropriate radial limits

            # Plot ring profile (polar)
            ax2.plot(
                result_data.secondary_psi,
                result_data.secondary_R_psi,
                "purple",
                linewidth=2,
                label="Ring Profile",
            )
            ax2.set_title("Ring Profile (Polar)", pad=20)
            ax2.grid(True)
            ax2.set_ylim(
                0, max(result_data.secondary_R_psi) * 1.1,
            )  # Set appropriate radial limits

            self.profiles_fig.suptitle(
                "Cam and Ring 2D Profiles", fontsize=14, fontweight="bold",
            )
        else:
            ax = self.profiles_fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No profile data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Cam and Ring 2D Profiles")

        self.profiles_fig.tight_layout()
        self.profiles_canvas.draw()

    def _update_animation_plot(self, result_data):
        """Update the animation plot (Tab 4)."""
        self.animation_fig.clear()

        if (
            result_data.secondary_psi is not None
            and result_data.secondary_R_psi is not None
        ):
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
                base_circle_cam = float(
                    gear.get(
                        "base_circle_cam", result_data.secondary_base_radius or 20.0,
                    ),
                )
                base_circle_ring = float(
                    gear.get(
                        "base_circle_ring",
                        (result_data.secondary_base_radius or 20.0) * 1.7,
                    ),
                )
                z_cam = int(gear.get("z_cam", 20))
                theta_cam_deg = None
                if isinstance(result_data.secondary_cam_curves, dict):
                    theta_cam_deg = result_data.secondary_cam_curves.get("theta")
                theta_cam_rad = (
                    np.radians(theta_cam_deg)
                    if theta_cam_deg is not None
                    else np.linspace(0, 2 * np.pi, len(result_data.secondary_psi))
                )
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

            self.animation_fig.suptitle(
                "Collocation-Based Animation", fontsize=14, fontweight="bold",
            )

            # Auto-start playback for convenience
            self._play_animation()
        else:
            # Show placeholder with two subplots
            ax1 = self.animation_fig.add_subplot(121, projection="polar")
            ax1.text(
                0.5,
                0.5,
                "Full Assembly View\n\nRun optimization to see collocation-based animation",
                ha="center",
                va="center",
                fontsize=10,
                transform=ax1.transAxes,
            )
            ax1.set_title("Full Assembly View")

            ax2 = self.animation_fig.add_subplot(122, projection="polar")
            ax2.text(
                0.5,
                0.5,
                "Tooth Contact Detail\n\nShows 3-4 teeth at contact point",
                ha="center",
                va="center",
                fontsize=10,
                transform=ax2.transAxes,
            )
            ax2.set_title("Tooth Contact Detail")

            self.animation_fig.suptitle(
                "Collocation-Based Animation", fontsize=14, fontweight="bold",
            )

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
        ax.set_title(
            f"Full Assembly - Frame {frame_index + 1}/{total_frames}\nθ = {np.degrees(current_theta):.1f}°",
        )

        # Draw ring gear (fixed, centered): phase-2 profile directly
        ring_psi = self.animation_data.secondary_psi
        ring_R_psi = self.animation_data.secondary_R_psi
        ax.plot(ring_psi, ring_R_psi, "r-", linewidth=3, label="Ring Gear")

        # Draw 3 planet teeth using assembly transform
        st = self.animation_state
        if st is not None and isinstance(
            self.animation_data.secondary_gear_geometry, dict,
        ):
            flanks = self.animation_data.secondary_gear_geometry.get("flanks")
            drew_planet = False
            if isinstance(flanks, dict) and ("tooth" in flanks):
                m0 = int(
                    round(
                        st.z_cam
                        * st.planet_spin_angle[self.animation_frame_index]
                        / (2 * np.pi),
                    ),
                )
                for dm, color in [(-1, "b-"), (0, "b-"), (1, "b-")]:
                    idx = m0 + dm
                    tooth = flanks.get("tooth")
                    if isinstance(tooth, dict):
                        tooth_theta = np.asarray(tooth.get("theta"))
                        tooth_r = np.asarray(tooth.get("r"))
                        if (
                            tooth_theta is not None
                            and tooth_r is not None
                            and len(tooth_theta) > 0
                        ):
                            thw, rw = transform_to_world_polar(
                                tooth_theta,
                                tooth_r,
                                st.planet_center_radius[self.animation_frame_index],
                                st.planet_center_angle[self.animation_frame_index],
                                st.planet_spin_angle[self.animation_frame_index],
                                idx,
                                st.z_cam,
                            )
                            ax.plot(
                                thw,
                                rw,
                                color,
                                linewidth=2,
                                label="Cam Teeth" if dm == 0 else None,
                            )
                            drew_planet = True
            # Fallback: draw cam pitch profile aligned as a single body if no explicit tooth data
            if not drew_planet and isinstance(
                self.animation_data.secondary_cam_curves, dict,
            ):
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
        contact_radius = (
            float(self.animation_state.contact_radius[self.animation_frame_index])
            if self.animation_state is not None
            else float(np.mean(self.animation_data.secondary_R_psi))
        )
        ax.plot(
            current_theta, contact_radius, "go", markersize=8, label="Contact Point",
        )

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
        ax.set_xlim(
            current_theta - detail_angle_range, current_theta + detail_angle_range,
        )
        ax.set_title(f"Tooth Contact Detail - Frame {frame_index + 1}/{total_frames}")

        # Draw ring gear teeth in detail region
        ring_psi = self.animation_data.secondary_psi
        ring_R_psi = self.animation_data.secondary_R_psi
        mask = (ring_psi >= current_theta - detail_angle_range) & (
            ring_psi <= current_theta + detail_angle_range
        )
        ax.plot(ring_psi[mask], ring_R_psi[mask], "r-", linewidth=3, label="Ring Teeth")

        # Draw cam teeth in detail region using assembly transform
        st = self.animation_state
        if st is not None and isinstance(
            self.animation_data.secondary_gear_geometry, dict,
        ):
            flanks = self.animation_data.secondary_gear_geometry.get("flanks")
            drew_planet = False
            if isinstance(flanks, dict) and ("tooth" in flanks):
                m0 = int(
                    round(
                        st.z_cam
                        * st.planet_spin_angle[self.animation_frame_index]
                        / (2 * np.pi),
                    ),
                )
                for dm, color in [(-1, "b-"), (0, "b-"), (1, "b-")]:
                    idx = m0 + dm
                    tooth = flanks.get("tooth")
                    if isinstance(tooth, dict):
                        tooth_theta = np.asarray(tooth.get("theta"))
                        tooth_r = np.asarray(tooth.get("r"))
                        if (
                            tooth_theta is not None
                            and tooth_r is not None
                            and len(tooth_theta) > 0
                        ):
                            thw, rw = transform_to_world_polar(
                                tooth_theta,
                                tooth_r,
                                st.planet_center_radius[self.animation_frame_index],
                                st.planet_center_angle[self.animation_frame_index],
                                st.planet_spin_angle[self.animation_frame_index],
                                idx,
                                st.z_cam,
                            )
                            maskw = (thw >= current_theta - detail_angle_range) & (
                                thw <= current_theta + detail_angle_range
                            )
                            ax.plot(
                                thw[maskw],
                                rw[maskw],
                                color,
                                linewidth=2,
                                label="Cam Teeth" if dm == 0 else None,
                            )
                            drew_planet = True
            if not drew_planet and isinstance(
                self.animation_data.secondary_cam_curves, dict,
            ):
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
                    maskw = (thw >= current_theta - detail_angle_range) & (
                        thw <= current_theta + detail_angle_range
                    )
                    ax.plot(
                        thw[maskw], rw[maskw], "b-", linewidth=2, label="Cam Profile",
                    )

        # Highlight the exact contact point
        contact_radius = (
            float(self.animation_state.contact_radius[self.animation_frame_index])
            if self.animation_state is not None
            else float(np.mean(self.animation_data.secondary_R_psi))
        )
        ax.plot(
            current_theta, contact_radius, "go", markersize=10, label="Contact Point",
        )

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
            ax.plot(
                [0, 10],
                [position, position + 5],
                "g-",
                linewidth=2,
                label="Connecting Rod",
            )

        # Draw cam (simplified representation)
        if (
            hasattr(self, "animation_data")
            and self.animation_data.secondary_base_radius is not None
        ):
            base_radius = self.animation_data.secondary_base_radius
            cam_x = base_radius * np.cos(np.radians(theta))
            cam_y = base_radius * np.sin(np.radians(theta))
            ax.plot(cam_x, cam_y, "ro", markersize=10, label="Cam")

        ax.legend()

    def _update_tertiary_plot(self, result_data):
        """Update the crank center optimization plot (Tab 5)."""
        self.tertiary_fig.clear()

        if (
            result_data.tertiary_crank_center_x is not None
            and result_data.tertiary_torque_output is not None
        ):
            # Create subplots for torque and side-loading
            ax1 = self.tertiary_fig.add_subplot(221)
            ax2 = self.tertiary_fig.add_subplot(222)
            ax3 = self.tertiary_fig.add_subplot(223)
            ax4 = self.tertiary_fig.add_subplot(224)

            # Plot 1: Crank center position
            ax1.plot(
                result_data.tertiary_crank_center_x,
                result_data.tertiary_crank_center_y,
                "ro",
                markersize=12,
                label="Optimized Position",
            )
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
            ax4.text(
                0.1,
                0.5,
                summary_text,
                fontsize=10,
                family="monospace",
                verticalalignment="center",
            )

            self.tertiary_fig.suptitle(
                "Crank Center Optimization (Run 3)", fontsize=14, fontweight="bold",
            )
        else:
            ax = self.tertiary_fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No crank center optimization data available",
                ha="center",
                va="center",
                fontsize=12,
            )
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
        delay = int(
            1000 / (30 * self.variables["animation_speed"].get()),
        )  # 30 FPS base
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
                    "primary_theta": self.unified_result.primary_theta.tolist()
                    if self.unified_result.primary_theta is not None
                    else None,
                    "primary_position": self.unified_result.primary_position.tolist()
                    if self.unified_result.primary_position is not None
                    else None,
                    "primary_velocity": self.unified_result.primary_velocity.tolist()
                    if self.unified_result.primary_velocity is not None
                    else None,
                    "primary_acceleration": self.unified_result.primary_acceleration.tolist()
                    if self.unified_result.primary_acceleration is not None
                    else None,
                    "secondary_psi": self.unified_result.secondary_psi.tolist()
                    if self.unified_result.secondary_psi is not None
                    else None,
                    "secondary_R_psi": self.unified_result.secondary_R_psi.tolist()
                    if self.unified_result.secondary_R_psi is not None
                    else None,
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

        # Reset CasADi optimization parameters
        self.variables["use_casadi_optimizer"].set(False)
        self.variables["enable_warmstart"].set(True)
        self.variables["casadi_n_segments"].set(50)
        self.variables["casadi_poly_order"].set(3)
        self.variables["casadi_collocation_method"].set("legendre")
        self.variables["thermal_efficiency_target"].set(0.55)
        self.variables["enable_thermal_efficiency"].set(True)

        # Clear results and plots
        self.unified_result = None
        self._clear_all_plots()

        # Update initial guesses
        self._on_stroke_changed()

        self.status_var.set("Parameters reset to defaults")

    def _show_detailed_analysis(self, result_data):
        """Show detailed per-phase analysis in console output."""
        try:
            print("\n" + "=" * 60)
            print("DETAILED OPTIMIZATION ANALYSIS")
            print("=" * 60)

            # Phase 1 Analysis
            p1_analysis = getattr(result_data, "primary_ipopt_analysis", None)
            print("\nPhase 1 (Thermal Efficiency):")
            if p1_analysis:
                print(f"  MA57 Readiness: {p1_analysis.grade.upper()}")
                print(f"  Reasons: {', '.join(p1_analysis.reasons)}")
                print(f"  Suggested Action: {p1_analysis.suggested_action}")
                if "iterations" in p1_analysis.stats:
                    print(f"  Iterations: {p1_analysis.stats['iterations']}")
                if "solve_time" in p1_analysis.stats:
                    print(f"  Solve Time: {p1_analysis.stats['solve_time']:.3f}s")
            else:
                print("  Analysis: Not available (thermal efficiency adapter)")

            # Phase 2 Analysis
            p2_analysis = getattr(result_data, "secondary_ipopt_analysis", None)
            print("\nPhase 2 (Litvin/Cam-Ring):")
            if p2_analysis:
                print(f"  MA57 Readiness: {p2_analysis.grade.upper()}")
                print(f"  Reasons: {', '.join(p2_analysis.reasons)}")
                print(f"  Suggested Action: {p2_analysis.suggested_action}")
                if "iterations" in p2_analysis.stats:
                    print(f"  Iterations: {p2_analysis.stats['iterations']}")
                if "solve_time" in p2_analysis.stats:
                    print(f"  Solve Time: {p2_analysis.stats['solve_time']:.3f}s")
            else:
                print("  Analysis: Not available")

            # Phase 3 Analysis
            p3_analysis = getattr(result_data, "tertiary_ipopt_analysis", None)
            print("\nPhase 3 (Crank Center):")
            if p3_analysis:
                print(f"  MA57 Readiness: {p3_analysis.grade.upper()}")
                print(f"  Reasons: {', '.join(p3_analysis.reasons)}")
                print(f"  Suggested Action: {p3_analysis.suggested_action}")
                if "iterations" in p3_analysis.stats:
                    print(f"  Iterations: {p3_analysis.stats['iterations']}")
                if "solve_time" in p3_analysis.stats:
                    print(f"  Solve Time: {p3_analysis.stats['solve_time']:.3f}s")
            else:
                print("  Analysis: Not available")

            # Overall Summary
            print("\nOverall Summary:")
            print(f"  Total Solve Time: {result_data.total_solve_time:.3f}s")
            print(f"  Optimization Method: {result_data.optimization_method}")

            # Count phases with analysis
            analysis_count = sum(
                1
                for analysis in [p1_analysis, p2_analysis, p3_analysis]
                if analysis is not None
            )
            print(f"  Phases with Analysis: {analysis_count}/3")

            print("=" * 60)

        except Exception as e:
            print(f"DEBUG: Detailed analysis display failed: {e}")

    def _show_ma57_readiness(self):
        """Analyze all optimization phases and show comprehensive MA57 readiness."""
        try:
            if not hasattr(self, "unified_result") or not self.unified_result:
                return

            # Extract analysis from all phases
            p1_analysis = getattr(self.unified_result, "primary_ipopt_analysis", None)
            p2_analysis = getattr(self.unified_result, "secondary_ipopt_analysis", None)
            p3_analysis = getattr(self.unified_result, "tertiary_ipopt_analysis", None)

            # Display each phase separately
            status_lines = []
            if p1_analysis:
                status_lines.append(f"P1 MA57: {p1_analysis.grade.upper()}")
            if p2_analysis:
                status_lines.append(f"P2 MA57: {p2_analysis.grade.upper()}")
            if p3_analysis:
                status_lines.append(f"P3 MA57: {p3_analysis.grade.upper()}")

            # Update status bar with all three phases
            if status_lines:
                current_status = self.status_var.get()
                # Remove any existing MA57 info
                if " | MA57" in current_status:
                    current_status = current_status.split(" | MA57")[0]
                grade_info = " | " + " | ".join(status_lines)
                self.status_var.set(current_status + grade_info)

            # Log detailed analysis for debugging
            print("DEBUG: Multi-phase MA57 readiness analysis:")
            if p1_analysis:
                print(f"  Phase 1: {p1_analysis.grade} - {p1_analysis.reasons}")
            if p2_analysis:
                print(f"  Phase 2: {p2_analysis.grade} - {p2_analysis.reasons}")
            if p3_analysis:
                print(f"  Phase 3: {p3_analysis.grade} - {p3_analysis.reasons}")

        except Exception as e:
            # If analysis fails, keep normal status
            print(f"DEBUG: Multi-phase MA57 readiness analysis failed: {e}")

    def _diagnose_nlp(self):
        """Run quick diagnostics: residuals and MA57 readiness from last run."""
        try:
            # Ensure we have results
            if not hasattr(self, "unified_result") or self.unified_result is None:
                self.status_var.set("No results to diagnose. Run optimization first.")
                return

            # Build or reuse SolveReport
            try:
                from campro.api.adapters import unified_data_to_solve_report

                sr = getattr(self, "solve_report", None)
                if sr is None:
                    sr = unified_data_to_solve_report(self.unified_result)
                    self.solve_report = sr
            except Exception as _e:
                sr = None
                print(f"DEBUG: Unable to build SolveReport for diagnostics: {_e}")

            # Determine Ipopt log path for analysis
            ipopt_log = None
            if sr and getattr(sr, "artifacts", None):
                try:
                    ipopt_log = sr.artifacts.get("ipopt_log")
                except Exception:
                    ipopt_log = None
            if not ipopt_log and hasattr(self, "latest_ipopt_log"):
                ipopt_log = self.latest_ipopt_log

            # Analyze Ipopt output heuristically (stats may be empty)
            readiness = None
            try:
                readiness = analyze_ipopt_run({}, ipopt_log)
            except Exception as _e:
                print(f"DEBUG: analyze_ipopt_run failed: {_e}")

            # Compose report text
            lines = []
            if sr:
                lines.append(f"Run ID: {getattr(sr, 'run_id', 'n/a')}")
                lines.append(f"Status: {getattr(sr, 'status', 'n/a')}")
                lines.append(f"Iterations: {getattr(sr, 'n_iter', 0)}")
                if getattr(sr, "residuals", None):
                    # Show a few key residuals
                    keys = [
                        "primal_inf",
                        "dual_inf",
                        "kkt_error",
                        "constraint_violation",
                    ]
                    shown = []
                    for k in keys:
                        if k in sr.residuals:
                            try:
                                shown.append(f"{k}={float(sr.residuals[k]):.2e}")
                            except Exception:
                                pass
                    if not shown:
                        # show first two entries
                        items = list(sr.residuals.items())[:2]
                        for k, v in items:
                            if isinstance(v, (int, float)):
                                shown.append(f"{k}={v:.2e}")
                    if shown:
                        lines.append("Residuals: " + ", ".join(shown))
                if ipopt_log:
                    lines.append(f"Ipopt log: {ipopt_log}")
            if readiness:
                lines.append("")
                lines.append(f"MA57 Readiness: {readiness.grade.upper()}")
                if readiness.reasons:
                    lines.append("Reasons: " + ", ".join(readiness.reasons))
                if readiness.suggested_action:
                    lines.append("Suggested: " + readiness.suggested_action)

            text = "\n".join(lines) if lines else "No diagnostics available."

            # Show modal info dialog and refresh Diagnostics tab
            try:
                messagebox.showinfo("NLP Diagnostics", text)
            except Exception:
                # Fallback: print to console
                print("NLP Diagnostics:\n" + text)

            try:
                if sr is not None:
                    self._update_diagnostics_tab(sr)
            except Exception as _e:
                print(f"DEBUG: Diagnostics tab refresh failed: {_e}")

            # Update status bar with short summary
            if readiness:
                current = self.status_var.get()
                if " | Diagnose:" in current:
                    current = current.split(" | Diagnose:")[0]
                self.status_var.set(
                    current + f" | Diagnose: MA57 {readiness.grade.upper()}",
                )
        except Exception as e:
            try:
                messagebox.showwarning("NLP Diagnostics", f"Diagnostics failed: {e}")
            except Exception:
                print(f"NLP Diagnostics failed: {e}")


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = CamMotionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
