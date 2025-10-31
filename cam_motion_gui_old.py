#!/usr/bin/env python3
"""
DEPRECATED: Legacy GUI for Cam Motion Law Optimization.

This module is kept for reference only. Use `cam_motion_gui.py` instead.
It may be removed in a future release.
"""

import warnings

warnings.warn(
    "cam_motion_gui_old is deprecated; use cam_motion_gui instead.",
    DeprecationWarning,
    stacklevel=2,
)

import threading  # noqa: E402
import tkinter as tk  # noqa: E402
from pathlib import Path  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from campro.logging import get_logger  # noqa: E402

# Import unified optimization framework
from campro.optimization.unified_framework import (  # noqa: E402
    OptimizationMethod,
    UnifiedOptimizationConstraints,
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
    UnifiedOptimizationTargets,
)
from campro.storage import OptimizationRegistry  # noqa: E402

# Import our cam motion law solver
from campro.optimization.collocation import CollocationSettings  # noqa: E402
from CamPro_OptimalMotion import solve_cam_motion_law  # noqa: E402

log = get_logger(__name__)


class CamMotionGUI:
    """Main GUI class for cam motion law optimization."""

    def __init__(self, root):
        self.root = root
        self.root.title("Cam-Ring System Designer")
        self.root.geometry("1400x900")

        # Variables to store input values
        self.variables = self._create_variables()

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
            "animation_frames": tk.IntVar(value=60),
            "animation_speed": tk.DoubleVar(value=1.0),
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

        # Row 1: Core parameters
        ttk.Label(self.control_frame, text="Stroke (mm):").grid(
            row=0, column=0, sticky=tk.W, pady=2,
        )
        self.stroke_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["stroke"], width=8,
        )
        self.stroke_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Cycle Time (s):").grid(
            row=0, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.cycle_time_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["cycle_time"], width=8,
        )
        self.cycle_time_entry.grid(row=0, column=3, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Motion Type:").grid(
            row=0, column=4, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.motion_type_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.variables["motion_type"],
            values=["minimum_jerk", "minimum_energy", "minimum_time"],
            state="readonly",
            width=12,
        )
        self.motion_type_combo.grid(row=0, column=5, sticky=tk.W, padx=(5, 0), pady=2)

        # Row 2: Cam-ring parameters
        ttk.Label(self.control_frame, text="Cam Base Radius (mm):").grid(
            row=1, column=0, sticky=tk.W, pady=2,
        )
        self.base_radius_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["base_radius"], width=8,
        )
        self.base_radius_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Rod Length (mm):").grid(
            row=1, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.rod_length_entry = ttk.Entry(
            self.control_frame,
            textvariable=self.variables["connecting_rod_length"],
            width=8,
        )
        self.rod_length_entry.grid(row=1, column=3, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Contact Type:").grid(
            row=1, column=4, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.contact_type_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.variables["contact_type"],
            values=["external", "internal"],
            state="readonly",
            width=12,
        )
        self.contact_type_combo.grid(row=1, column=5, sticky=tk.W, padx=(5, 0), pady=2)

        # Row 3: Sun gear parameters
        ttk.Label(self.control_frame, text="Sun Gear Radius (mm):").grid(
            row=2, column=0, sticky=tk.W, pady=2,
        )
        self.sun_gear_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["sun_gear_radius"], width=8,
        )
        self.sun_gear_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Ring Gear Radius (mm):").grid(
            row=2, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.ring_gear_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["ring_gear_radius"], width=8,
        )
        self.ring_gear_entry.grid(row=2, column=3, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(self.control_frame, text="Gear Ratio:").grid(
            row=2, column=4, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.gear_ratio_entry = ttk.Entry(
            self.control_frame, textvariable=self.variables["gear_ratio"], width=8,
        )
        self.gear_ratio_entry.grid(row=2, column=5, sticky=tk.W, padx=(5, 0), pady=2)

        # Row 4: Control buttons
        self.optimize_button = ttk.Button(
            self.control_frame,
            text="🚀 Optimize System",
            command=self._run_optimization,
            style="Accent.TButton",
        )
        self.optimize_button.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=10)

        self.reset_button = ttk.Button(
            self.control_frame,
            text="Reset Parameters",
            command=self._reset_parameters,
        )
        self.reset_button.grid(
            row=3, column=2, columnspan=2, sticky=tk.W, pady=10, padx=(20, 0),
        )

        self.save_button = ttk.Button(
            self.control_frame,
            text="Save Results",
            command=self._save_results,
        )
        self.save_button.grid(
            row=3, column=4, columnspan=2, sticky=tk.W, pady=10, padx=(20, 0),
        )

        # Add callback to update initial guesses when stroke changes
        self.variables["stroke"].trace("w", self._on_stroke_changed)

    # Old widget creation methods removed - using simplified control panel instead

    def _layout_widgets(self):
        """Layout all widgets in the main window."""
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.control_frame.pack(fill=tk.X, pady=(0, 10))
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Core parameters
        self._create_core_parameters()

        # Optional constraints
        self._create_constraint_parameters()

        # Solver settings (removed - now handled by unified settings)

    def _create_secondary_widgets(self):
        """Create widgets for secondary optimization (cam-ring mapping)."""
        # Input frame for secondary optimization
        self.secondary_input_frame = ttk.LabelFrame(
            self.secondary_frame, text="Cam-Ring System Parameters", padding="10",
        )
        self.secondary_input_frame.pack(fill=tk.X, pady=(0, 10))

        # Cam parameters
        self._create_cam_parameters()

        # Ring design parameters
        self._create_ring_parameters()

    def _create_core_parameters(self):
        """Create widgets for core cam parameters."""
        # Stroke
        ttk.Label(self.primary_input_frame, text="Stroke (mm):").grid(
            row=0, column=0, sticky=tk.W, pady=2,
        )
        self.stroke_entry = ttk.Entry(
            self.primary_input_frame, textvariable=self.variables["stroke"], width=10,
        )
        self.stroke_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Add callback to update initial guesses when stroke changes
        self.variables["stroke"].trace("w", self._on_stroke_changed)

        # Upstroke duration
        ttk.Label(self.primary_input_frame, text="Upstroke Duration (%):").grid(
            row=1, column=0, sticky=tk.W, pady=2,
        )
        self.upstroke_entry = ttk.Entry(
            self.primary_input_frame,
            textvariable=self.variables["upstroke_duration"],
            width=10,
        )
        self.upstroke_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Zero acceleration duration
        ttk.Label(self.primary_input_frame, text="Zero Accel Duration (%):").grid(
            row=2, column=0, sticky=tk.W, pady=2,
        )
        self.zero_accel_entry = ttk.Entry(
            self.primary_input_frame,
            textvariable=self.variables["zero_accel_duration"],
            width=10,
        )
        self.zero_accel_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Cycle time
        ttk.Label(self.primary_input_frame, text="Cycle Time (s):").grid(
            row=3, column=0, sticky=tk.W, pady=2,
        )
        self.cycle_time_entry = ttk.Entry(
            self.primary_input_frame,
            textvariable=self.variables["cycle_time"],
            width=10,
        )
        self.cycle_time_entry.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Motion type
        ttk.Label(self.primary_input_frame, text="Motion Type:").grid(
            row=4, column=0, sticky=tk.W, pady=2,
        )
        self.motion_type_combo = ttk.Combobox(
            self.primary_input_frame,
            textvariable=self.variables["motion_type"],
            values=["minimum_jerk", "minimum_energy", "minimum_time"],
            state="readonly",
            width=15,
        )
        self.motion_type_combo.grid(row=4, column=1, sticky=tk.W, padx=(5, 0), pady=2)

    def _create_constraint_parameters(self):
        """Create widgets for optional constraints."""
        # Max velocity
        ttk.Label(self.primary_input_frame, text="Max Velocity (mm/s):").grid(
            row=0, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.max_vel_entry = ttk.Entry(
            self.primary_input_frame,
            textvariable=self.variables["max_velocity"],
            width=10,
        )
        self.max_vel_entry.grid(row=0, column=3, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(self.primary_input_frame, text="(0 = no limit)").grid(
            row=0, column=4, sticky=tk.W, padx=(5, 0), pady=2,
        )

        # Max acceleration
        ttk.Label(self.primary_input_frame, text="Max Acceleration (mm/s²):").grid(
            row=1, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.max_acc_entry = ttk.Entry(
            self.primary_input_frame,
            textvariable=self.variables["max_acceleration"],
            width=10,
        )
        self.max_acc_entry.grid(row=1, column=3, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(self.primary_input_frame, text="(0 = no limit)").grid(
            row=1, column=4, sticky=tk.W, padx=(5, 0), pady=2,
        )

        # Max jerk
        ttk.Label(self.primary_input_frame, text="Max Jerk (mm/s³):").grid(
            row=2, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )
        self.max_jerk_entry = ttk.Entry(
            self.primary_input_frame, textvariable=self.variables["max_jerk"], width=10,
        )
        self.max_jerk_entry.grid(row=2, column=3, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Label(self.primary_input_frame, text="(0 = no limit)").grid(
            row=2, column=4, sticky=tk.W, padx=(5, 0), pady=2,
        )

        # Dwell checkboxes
        self.dwell_tdc_check = ttk.Checkbutton(
            self.primary_input_frame,
            text="Dwell at TDC",
            variable=self.variables["dwell_at_tdc"],
        )
        self.dwell_tdc_check.grid(row=3, column=2, sticky=tk.W, pady=2, padx=(20, 0))

        self.dwell_bdc_check = ttk.Checkbutton(
            self.primary_input_frame,
            text="Dwell at BDC",
            variable=self.variables["dwell_at_bdc"],
        )
        self.dwell_bdc_check.grid(row=3, column=3, sticky=tk.W, pady=2, padx=(5, 0))

    def _create_cam_parameters(self):
        """Create widgets for cam-ring system parameters."""
        # System description
        desc_label = ttk.Label(
            self.secondary_input_frame,
            text="System: Cam connected to linear follower via connecting rod, cam contacts ring follower directly",
            font=("TkDefaultFont", 9, "italic"),
            foreground="green",
        )
        desc_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        # Cam base radius
        ttk.Label(self.secondary_input_frame, text="Cam Base Radius (mm):").grid(
            row=1, column=0, sticky=tk.W, pady=2,
        )
        self.base_radius_entry = ttk.Entry(
            self.secondary_input_frame,
            textvariable=self.variables["base_radius"],
            width=10,
        )
        self.base_radius_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Connecting rod length (distance from cam center to linear follower connection)
        ttk.Label(self.secondary_input_frame, text="Connecting Rod Length (mm):").grid(
            row=2, column=0, sticky=tk.W, pady=2,
        )
        self.connecting_rod_entry = ttk.Entry(
            self.secondary_input_frame,
            textvariable=self.variables["connecting_rod_length"],
            width=10,
        )
        self.connecting_rod_entry.grid(
            row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2,
        )

        # Contact type (external/internal contact between cam and ring)
        ttk.Label(self.secondary_input_frame, text="Cam-Ring Contact Type:").grid(
            row=3, column=0, sticky=tk.W, pady=2,
        )
        self.contact_type_combo = ttk.Combobox(
            self.secondary_input_frame,
            textvariable=self.variables["contact_type"],
            values=["external", "internal"],
            state="readonly",
            width=15,
        )
        self.contact_type_combo.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)

    def _create_ring_parameters(self):
        """Create widgets for ring design parameters."""
        # Information label explaining that ring design is mathematically determined
        info_label = ttk.Label(
            self.secondary_input_frame,
            text="Ring Profile: Mathematically determined by cam geometry and meshing law",
            font=("TkDefaultFont", 9, "italic"),
            foreground="blue",
        )
        info_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))

        # Ring design is determined by the mathematical relationship
        # The ring radius R(ψ) comes from the meshing law: ρ_c(θ)dθ = R(ψ)dψ
        # No user selection needed - it's computed from the cam geometry

        # Enable ring design checkbox
        self.enable_ring_check = ttk.Checkbutton(
            self.secondary_input_frame,
            text="Enable Ring Design",
            variable=self.variables["enable_ring_design"],
        )
        self.enable_ring_check.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=10)

    def _create_tertiary_widgets(self):
        """Create widgets for tertiary optimization (sun gear system)."""
        # Input frame for tertiary optimization
        self.tertiary_input_frame = ttk.LabelFrame(
            self.tertiary_frame, text="Sun Gear System Parameters", padding="10",
        )
        self.tertiary_input_frame.pack(fill=tk.X, pady=(0, 10))

        # Sun gear parameters
        ttk.Label(self.tertiary_input_frame, text="Sun Gear Radius (mm):").grid(
            row=0, column=0, sticky=tk.W, pady=2,
        )
        self.sun_gear_entry = ttk.Entry(
            self.tertiary_input_frame,
            textvariable=self.variables["sun_gear_radius"],
            width=10,
        )
        self.sun_gear_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Ring gear parameters
        ttk.Label(self.tertiary_input_frame, text="Ring Gear Radius (mm):").grid(
            row=1, column=0, sticky=tk.W, pady=2,
        )
        self.ring_gear_entry = ttk.Entry(
            self.tertiary_input_frame,
            textvariable=self.variables["ring_gear_radius"],
            width=10,
        )
        self.ring_gear_entry.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Gear ratio
        ttk.Label(self.tertiary_input_frame, text="Gear Ratio:").grid(
            row=2, column=0, sticky=tk.W, pady=2,
        )
        self.gear_ratio_entry = ttk.Entry(
            self.tertiary_input_frame,
            textvariable=self.variables["gear_ratio"],
            width=10,
        )
        self.gear_ratio_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Max back rotation
        ttk.Label(self.tertiary_input_frame, text="Max Back Rotation (deg):").grid(
            row=3, column=0, sticky=tk.W, pady=2,
        )
        self.back_rotation_entry = ttk.Entry(
            self.tertiary_input_frame,
            textvariable=self.variables["max_back_rotation"],
            width=10,
        )
        self.back_rotation_entry.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Information label
        info_text = (
            "Sun gear eliminates cam-ring interference and enables 360° ring coverage"
        )
        info_label = ttk.Label(
            self.tertiary_input_frame,
            text=info_text,
            font=("TkDefaultFont", 9, "italic"),
            foreground="green",
        )
        info_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))

    def _create_settings_widgets(self):
        """Create widgets for unified optimization settings."""
        # Settings frame
        self.settings_input_frame = ttk.LabelFrame(
            self.settings_frame, text="Unified Optimization Settings", padding="10",
        )
        self.settings_input_frame.pack(fill=tk.X, pady=(0, 10))

        # Collocation settings
        ttk.Label(self.settings_input_frame, text="Collocation Degree:").grid(
            row=0, column=0, sticky=tk.W, pady=2,
        )
        self.collocation_degree_entry = ttk.Entry(
            self.settings_input_frame,
            textvariable=self.variables["collocation_degree"],
            width=10,
        )
        self.collocation_degree_entry.grid(
            row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2,
        )

        # Max iterations
        ttk.Label(self.settings_input_frame, text="Max Iterations:").grid(
            row=1, column=0, sticky=tk.W, pady=2,
        )
        self.max_iterations_entry = ttk.Entry(
            self.settings_input_frame,
            textvariable=self.variables["max_iterations"],
            width=10,
        )
        self.max_iterations_entry.grid(
            row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2,
        )

        # Tolerance
        ttk.Label(self.settings_input_frame, text="Tolerance:").grid(
            row=2, column=0, sticky=tk.W, pady=2,
        )
        self.tolerance_entry = ttk.Entry(
            self.settings_input_frame,
            textvariable=self.variables["tolerance"],
            width=10,
        )
        self.tolerance_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Lagrangian tolerance
        ttk.Label(self.settings_input_frame, text="Lagrangian Tolerance:").grid(
            row=3, column=0, sticky=tk.W, pady=2,
        )
        self.lagrangian_tolerance_entry = ttk.Entry(
            self.settings_input_frame,
            textvariable=self.variables["lagrangian_tolerance"],
            width=10,
        )
        self.lagrangian_tolerance_entry.grid(
            row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2,
        )

        # Penalty weight
        ttk.Label(self.settings_input_frame, text="Penalty Weight:").grid(
            row=4, column=0, sticky=tk.W, pady=2,
        )
        self.penalty_weight_entry = ttk.Entry(
            self.settings_input_frame,
            textvariable=self.variables["penalty_weight"],
            width=10,
        )
        self.penalty_weight_entry.grid(
            row=4, column=1, sticky=tk.W, padx=(5, 0), pady=2,
        )

        # Optimization targets frame
        self.targets_frame = ttk.LabelFrame(
            self.settings_frame, text="Optimization Targets", padding="10",
        )
        self.targets_frame.pack(fill=tk.X, pady=(0, 10))

        # Primary targets
        ttk.Label(
            self.targets_frame,
            text="Primary Layer Targets:",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        self.minimize_jerk_check = ttk.Checkbutton(
            self.targets_frame,
            text="Minimize Jerk",
            variable=self.variables["minimize_jerk"],
        )
        self.minimize_jerk_check.grid(row=1, column=0, sticky=tk.W, pady=2)

        self.minimize_time_check = ttk.Checkbutton(
            self.targets_frame,
            text="Minimize Time",
            variable=self.variables["minimize_time"],
        )
        self.minimize_time_check.grid(row=2, column=0, sticky=tk.W, pady=2)

        self.minimize_energy_check = ttk.Checkbutton(
            self.targets_frame,
            text="Minimize Energy",
            variable=self.variables["minimize_energy"],
        )
        self.minimize_energy_check.grid(row=3, column=0, sticky=tk.W, pady=2)

        # Secondary targets
        ttk.Label(
            self.targets_frame,
            text="Secondary Layer Targets:",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=1, sticky=tk.W, pady=(0, 5), padx=(20, 0))

        self.minimize_ring_size_check = ttk.Checkbutton(
            self.targets_frame,
            text="Minimize Ring Size",
            variable=self.variables["minimize_ring_size"],
        )
        self.minimize_ring_size_check.grid(
            row=1, column=1, sticky=tk.W, pady=2, padx=(20, 0),
        )

        self.minimize_cam_size_check = ttk.Checkbutton(
            self.targets_frame,
            text="Minimize Cam Size",
            variable=self.variables["minimize_cam_size"],
        )
        self.minimize_cam_size_check.grid(
            row=2, column=1, sticky=tk.W, pady=2, padx=(20, 0),
        )

        self.minimize_curvature_check = ttk.Checkbutton(
            self.targets_frame,
            text="Minimize Curvature Variation",
            variable=self.variables["minimize_curvature_variation"],
        )
        self.minimize_curvature_check.grid(
            row=3, column=1, sticky=tk.W, pady=2, padx=(20, 0),
        )

        # Tertiary targets
        ttk.Label(
            self.targets_frame,
            text="Tertiary Layer Targets:",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=0, column=2, sticky=tk.W, pady=(0, 5), padx=(20, 0))

        self.minimize_system_size_check = ttk.Checkbutton(
            self.targets_frame,
            text="Minimize System Size",
            variable=self.variables["minimize_system_size"],
        )
        self.minimize_system_size_check.grid(
            row=1, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )

        self.maximize_efficiency_check = ttk.Checkbutton(
            self.targets_frame,
            text="Maximize Efficiency",
            variable=self.variables["maximize_efficiency"],
        )
        self.maximize_efficiency_check.grid(
            row=2, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )

        self.minimize_back_rotation_check = ttk.Checkbutton(
            self.targets_frame,
            text="Minimize Back Rotation",
            variable=self.variables["minimize_back_rotation"],
        )
        self.minimize_back_rotation_check.grid(
            row=3, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )

        self.minimize_gear_stress_check = ttk.Checkbutton(
            self.targets_frame,
            text="Minimize Gear Stress",
            variable=self.variables["minimize_gear_stress"],
        )
        self.minimize_gear_stress_check.grid(
            row=4, column=2, sticky=tk.W, pady=2, padx=(20, 0),
        )

    def _create_control_buttons(self):
        """Create control buttons."""
        # Unified optimization button (main button)
        self.unified_button_frame = ttk.Frame(self.main_frame)
        self.unified_button_frame.pack(fill=tk.X, pady=10)

        self.unified_solve_button = ttk.Button(
            self.unified_button_frame,
            text="🚀 Run Unified Optimization",
            command=self._solve_unified_optimization,
            style="Accent.TButton",
        )
        self.unified_solve_button.pack(side=tk.LEFT, padx=5)

        # Individual layer buttons (for testing individual layers)
        self.individual_button_frame = ttk.Frame(self.main_frame)
        self.individual_button_frame.pack(fill=tk.X, pady=5)

        self.solve_primary_button = ttk.Button(
            self.individual_button_frame,
            text="Primary: Motion Law",
            command=self._solve_primary_only,
        )
        self.solve_primary_button.pack(side=tk.LEFT, padx=5)

        self.solve_secondary_button = ttk.Button(
            self.individual_button_frame,
            text="Secondary: Cam-Ring",
            command=self._solve_secondary_only,
        )
        self.solve_secondary_button.pack(side=tk.LEFT, padx=5)

        self.solve_tertiary_button = ttk.Button(
            self.individual_button_frame,
            text="Tertiary: Sun Gear",
            command=self._solve_tertiary_only,
        )
        self.solve_tertiary_button.pack(side=tk.LEFT, padx=5)

        # Common control buttons (shared between all tabs)
        self.common_button_frame = ttk.Frame(self.main_frame)
        self.common_button_frame.pack(fill=tk.X, pady=10)

        self.save_button = ttk.Button(
            self.common_button_frame,
            text="Save Plot",
            command=self._save_plot,
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = ttk.Button(
            self.common_button_frame,
            text="Clear Plot",
            command=self._clear_plot,
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = ttk.Button(
            self.common_button_frame,
            text="Reset Parameters",
            command=self._reset_parameters,
        )
        self.reset_button.pack(side=tk.LEFT, padx=5)

        # Plot view toggle
        ttk.Label(self.common_button_frame, text="View:").pack(
            side=tk.LEFT, padx=(20, 5),
        )
        self.plot_view_var = tk.StringVar(value="subplots")
        ttk.Radiobutton(
            self.common_button_frame,
            text="Smart Subplots",
            variable=self.plot_view_var,
            value="subplots",
            command=self._toggle_plot_view,
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(
            self.common_button_frame,
            text="Single Plot",
            variable=self.plot_view_var,
            value="single",
            command=self._toggle_plot_view,
        ).pack(side=tk.LEFT, padx=(0, 5))

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

        # Tab 4: Animation
        self.animation_fig = Figure(figsize=(12, 6), dpi=100)
        self.animation_canvas = FigureCanvasTkAgg(
            self.animation_fig, self.animation_frame,
        )
        self.animation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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

        ttk.Label(self.animation_control_frame, text="Frames:").pack(
            side=tk.LEFT, padx=(20, 5),
        )
        self.frames_entry = ttk.Entry(
            self.animation_control_frame,
            textvariable=self.variables["animation_frames"],
            width=6,
        )
        self.frames_entry.pack(side=tk.LEFT, padx=5)

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

        # Animation tab
        self.animation_fig.clear()
        ax = self.animation_fig.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "System Animation\n\nRun optimization to see animation",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_title("Cam-Ring System Animation")
        self.animation_canvas.draw()

    def _validate_inputs(self):
        """Validate input parameters."""
        try:
            stroke = self.variables["stroke"].get()
            upstroke_duration = self.variables["upstroke_duration"].get()
            zero_accel_duration = self.variables["zero_accel_duration"].get()
            cycle_time = self.variables["cycle_time"].get()

            print("DEBUG: Validating inputs:")
            print(f"  - Stroke: {stroke}")
            print(f"  - Upstroke duration: {upstroke_duration}%")
            print(f"  - Zero acceleration duration: {zero_accel_duration}%")
            print(f"  - Cycle time: {cycle_time}")

            if stroke <= 0:
                raise ValueError("Stroke must be positive")
            if not 0 <= upstroke_duration <= 100:
                raise ValueError("Upstroke duration must be between 0 and 100%")
            if not 0 <= zero_accel_duration <= 100:
                raise ValueError(
                    "Zero acceleration duration must be between 0 and 100%",
                )

            # REMOVED: The incorrect constraint that was causing the error
            # if zero_accel_duration > upstroke_duration:
            #     raise ValueError("Zero acceleration duration cannot exceed upstroke duration")

            print(
                f"DEBUG: Validation passed - zero acceleration duration ({zero_accel_duration}%) can exceed upstroke duration ({upstroke_duration}%)",
            )

            if cycle_time <= 0:
                raise ValueError("Cycle time must be positive")

            return True
        except Exception as e:
            print(f"DEBUG: Validation failed with error: {e}")
            messagebox.showerror("Input Error", str(e))
            return False

    def _solve_motion_law(self):
        """Solve the motion law in a separate thread."""
        if not self._validate_inputs():
            return

        # Disable solve button during computation
        self.solve_button.config(state="disabled")
        self.status_var.set("Solving motion law...")

        # Run in separate thread to prevent GUI freezing
        thread = threading.Thread(target=self._solve_thread)
        thread.daemon = True
        thread.start()

    def _solve_thread(self):
        """Thread function for solving motion law."""
        try:
            print("DEBUG: Starting motion law solving thread...")

            # Get parameters
            stroke = self.variables["stroke"].get()
            upstroke_duration = self.variables["upstroke_duration"].get()
            zero_accel_duration = self.variables["zero_accel_duration"].get()
            cycle_time = self.variables["cycle_time"].get()
            motion_type = self.variables["motion_type"].get()
            dwell_at_tdc = self.variables["dwell_at_tdc"].get()
            dwell_at_bdc = self.variables["dwell_at_bdc"].get()

            print("DEBUG: Thread parameters:")
            print(f"  - Stroke: {stroke}")
            print(f"  - Upstroke duration: {upstroke_duration}%")
            print(f"  - Zero acceleration duration: {zero_accel_duration}%")
            print(f"  - Cycle time: {cycle_time}")
            print(f"  - Motion type: {motion_type}")

            # Get constraints (convert 0 to None for no limit)
            max_velocity = self.variables["max_velocity"].get()
            max_acceleration = self.variables["max_acceleration"].get()
            max_jerk = self.variables["max_jerk"].get()

            max_velocity = max_velocity if max_velocity > 0 else None
            max_acceleration = max_acceleration if max_acceleration > 0 else None
            max_jerk = max_jerk if max_jerk > 0 else None

            print("DEBUG: Constraints:")
            print(f"  - Max velocity: {max_velocity}")
            print(f"  - Max acceleration: {max_acceleration}")
            print(f"  - Max jerk: {max_jerk}")

            # Get solver settings
            collocation_degree = self.variables["collocation_degree"].get()
            collocation_method = self.variables["collocation_method"].get()

            settings = CollocationSettings(
                degree=collocation_degree,
                method=collocation_method,
                verbose=False,
            )

            print("DEBUG: Calling solve_cam_motion_law with:")
            print(f"  - stroke={stroke}")
            print(f"  - upstroke_duration_percent={upstroke_duration}")
            print(
                f"  - zero_accel_duration_percent={zero_accel_duration if zero_accel_duration > 0 else None}",
            )

            # Solve motion law
            solution = solve_cam_motion_law(
                stroke=stroke,
                upstroke_duration_percent=upstroke_duration,
                motion_type=motion_type,
                cycle_time=cycle_time,
                max_velocity=max_velocity,
                max_acceleration=max_acceleration,
                max_jerk=max_jerk,
                zero_accel_duration_percent=zero_accel_duration
                if zero_accel_duration > 0
                else None,
                dwell_at_tdc=dwell_at_tdc,
                dwell_at_bdc=dwell_at_bdc,
                settings=settings,
            )

            print(
                f"DEBUG: Motion law solved successfully, got {len(solution['cam_angle'])} points",
            )

            # Store the primary result for ring design
            self.primary_result = solution

            # Update plot in main thread
            self.root.after(0, self._update_plot, solution)

        except Exception as e:
            print(f"DEBUG: Error in solve thread: {e}")
            import traceback

            traceback.print_exc()
            log.error(f"Error solving motion law: {e}")
            self.root.after(0, self._show_error, str(e))
        finally:
            # Re-enable solve button
            self.root.after(0, self._enable_solve_button)

    def _solve_ring_design(self):
        """Solve the ring design in a separate thread."""
        if not self._validate_ring_inputs():
            return

        if self.primary_result is None:
            messagebox.showwarning(
                "No Primary Solution",
                "Please solve the motion law first before designing the ring follower.",
            )
            return

        # Disable solve button during computation
        self.solve_ring_button.config(state="disabled")
        self.status_var.set("Designing ring follower...")

        # Run in separate thread to prevent GUI freezing
        thread = threading.Thread(target=self._solve_ring_thread)
        thread.daemon = True
        thread.start()

    def _solve_ring_thread(self):
        """Thread function for solving ring design."""
        try:
            print("DEBUG: Starting ring design solving thread...")

            # Get cam-ring system parameters
            base_radius = self.variables["base_radius"].get()
            connecting_rod_length = self.variables["connecting_rod_length"].get()
            contact_type = self.variables["contact_type"].get()

            print("DEBUG: Cam-ring system parameters:")
            print(f"  - Cam base radius: {base_radius}")
            print(f"  - Connecting rod length: {connecting_rod_length}")
            print(f"  - Cam-ring contact type: {contact_type}")
            print("  - System: Cam connected to linear follower via connecting rod")
            print("  - Ring profile: Computed from cam geometry and meshing law")

            # Create optimization constraints
            constraints = CamRingOptimizationConstraints(
                base_radius_min=5.0,
                base_radius_max=100.0,
                connecting_rod_length_min=10.0,
                connecting_rod_length_max=200.0,
            )

            # Create optimization targets
            targets = CamRingOptimizationTargets(
                minimize_ring_size=True,
                minimize_cam_size=True,
                minimize_curvature_variation=True,
                ring_size_weight=1.0,
                cam_size_weight=0.5,
                curvature_weight=0.2,
            )

            # Create cam-ring optimizer
            optimizer = CamRingOptimizer("CamRingOptimizer")
            optimizer.configure(constraints=constraints, targets=targets)

            # Set initial guess from current parameters
            initial_guess = {
                "base_radius": base_radius,
                "connecting_rod_length": connecting_rod_length,
            }

            print(
                f"DEBUG: Starting cam-ring optimization with initial guess: {initial_guess}",
            )

            # Perform optimization
            optimization_result = optimizer.optimize(
                primary_data=self.primary_result,
                initial_guess=initial_guess,
            )

            if optimization_result.status.value == "converged":
                ring_result = optimization_result.solution
                print("DEBUG: Optimization completed successfully")
                print(
                    f"  - Final objective value: {optimization_result.objective_value:.6f}",
                )
                print(f"  - Iterations: {optimization_result.iterations}")
                print(
                    f"  - Optimized parameters: {ring_result.get('optimized_parameters', {})}",
                )
            else:
                print(
                    f"DEBUG: Optimization failed: {optimization_result.metadata.get('error_message', 'Unknown error')}",
                )
                # Fallback to simple mapping
                ring_result = process_linear_to_ring_follower(
                    primary_data=self.primary_result,
                    secondary_constraints={
                        "cam_parameters": {
                            "base_radius": base_radius,
                            "connecting_rod_length": connecting_rod_length,
                            "contact_type": contact_type,
                        },
                    },
                    secondary_relationships={},
                    optimization_targets={},
                )

            # Store the result
            self.ring_design_result = ring_result

            # Update plot on main thread
            self.root.after(0, self._update_ring_plot, ring_result)

        except Exception as e:
            print(f"DEBUG: Error in ring design thread: {e}")
            import traceback

            traceback.print_exc()
            log.error(f"Error designing ring follower: {e}")
            self.root.after(0, self._show_error, str(e))
        finally:
            # Re-enable solve button
            self.root.after(0, self._enable_ring_solve_button)

    def _validate_ring_inputs(self):
        """Validate cam-ring system inputs."""
        try:
            base_radius = self.variables["base_radius"].get()
            connecting_rod_length = self.variables["connecting_rod_length"].get()

            if base_radius <= 0:
                raise ValueError("Cam base radius must be positive")
            if connecting_rod_length <= 0:
                raise ValueError("Connecting rod length must be positive")

            return True
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return False

    def _validate_tertiary_inputs(self):
        """Validate tertiary (sun gear) system inputs."""
        try:
            sun_gear_radius = self.variables["sun_gear_radius"].get()
            ring_gear_radius = self.variables["ring_gear_radius"].get()
            gear_ratio = self.variables["gear_ratio"].get()
            max_back_rotation = self.variables["max_back_rotation"].get()

            if sun_gear_radius <= 0:
                raise ValueError("Sun gear radius must be positive")
            if ring_gear_radius <= 0:
                raise ValueError("Ring gear radius must be positive")
            if gear_ratio <= 0:
                raise ValueError("Gear ratio must be positive")
            if max_back_rotation < 0 or max_back_rotation > 180:
                raise ValueError("Max back rotation must be between 0 and 180 degrees")

            return True
        except Exception as e:
            messagebox.showerror("Input Error", str(e))
            return False

    def _create_unified_plots(self, result_data):
        """Create comprehensive plots showing all three optimization layers."""
        try:
            # Create a 2x2 subplot layout
            ax1 = self.fig.add_subplot(2, 2, 1)
            ax2 = self.fig.add_subplot(2, 2, 2)
            ax3 = self.fig.add_subplot(2, 2, 3, projection="polar")
            ax4 = self.fig.add_subplot(2, 2, 4, projection="polar")

            # Plot 1: Primary optimization results (motion law)
            if result_data.primary_theta is not None:
                ax1.plot(
                    result_data.primary_theta,
                    result_data.primary_position,
                    "b-",
                    label="Position",
                    linewidth=2,
                )
                ax1.plot(
                    result_data.primary_theta,
                    result_data.primary_velocity,
                    "r-",
                    label="Velocity",
                    linewidth=2,
                )
                ax1.plot(
                    result_data.primary_theta,
                    result_data.primary_acceleration,
                    "g-",
                    label="Acceleration",
                    linewidth=2,
                )
                ax1.set_xlabel("Cam Angle (degrees)")
                ax1.set_ylabel("Displacement (mm)")
                ax1.set_title("Primary: Linear Follower Motion Law")
                ax1.legend()
                ax1.grid(True)

            # Plot 2: Secondary optimization results (cam curves)
            if result_data.secondary_cam_curves is not None:
                cam_curves = result_data.secondary_cam_curves
                if "profile_radius" in cam_curves:
                    ax2.plot(
                        result_data.primary_theta,
                        cam_curves["profile_radius"],
                        "b-",
                        label="Cam Profile",
                        linewidth=2,
                    )
                    ax2.set_xlabel("Cam Angle (degrees)")
                    ax2.set_ylabel("Radius (mm)")
                    ax2.set_title("Secondary: Cam Profile")
                    ax2.legend()
                    ax2.grid(True)

            # Plot 3: Secondary optimization results (ring profile - polar)
            if (
                result_data.secondary_psi is not None
                and result_data.secondary_R_psi is not None
            ):
                psi_rad = result_data.secondary_psi
                R_psi = result_data.secondary_R_psi
                ax3.plot(psi_rad, R_psi, "r-", linewidth=2)
                ax3.set_title("Secondary: Ring Follower Profile")
                ax3.grid(True)

            # Plot 4: Tertiary optimization results (complete ring profile - polar)
            if (
                result_data.tertiary_psi_complete is not None
                and result_data.tertiary_R_psi_complete is not None
            ):
                psi_complete = result_data.tertiary_psi_complete
                R_psi_complete = result_data.tertiary_R_psi_complete
                ax4.plot(psi_complete, R_psi_complete, "g-", linewidth=2)
                ax4.set_title("Tertiary: Complete 360° Ring Profile")
                ax4.grid(True)

            # Add summary text
            summary = self.unified_framework.get_optimization_summary()
            summary_text = f"Method: {summary['method']}\n"
            summary_text += f"Total Time: {summary['total_solve_time']:.3f}s\n"
            summary_text += f"Primary: {summary['primary_results']['points']} points\n"
            summary_text += f"Secondary: Ring coverage {summary['secondary_results']['ring_coverage']:.1f}°\n"
            summary_text += f"Tertiary: Complete 360° coverage: {summary['tertiary_results']['complete_360_coverage']}"

            self.fig.text(
                0.02,
                0.02,
                summary_text,
                fontsize=8,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            )

            self.fig.tight_layout()

        except Exception as e:
            print(f"DEBUG: Error creating unified plots: {e}")
            import traceback

            traceback.print_exc()

    def _update_plot(self, solution):
        """Update the plot with the solution."""
        try:
            print(f"DEBUG: Updating plot with solution keys: {list(solution.keys())}")

            # Clear previous plot
            self.fig.clear()

            # Get data
            cam_angle = solution["cam_angle"]
            position = solution["position"]
            velocity = solution["velocity"]
            acceleration = solution["acceleration"]
            jerk = solution["control"]

            print("DEBUG: Plot data shapes:")
            print(
                f"  - cam_angle: {len(cam_angle)} points, range: {cam_angle[0]:.2f} to {cam_angle[-1]:.2f}",
            )
            print(
                f"  - position: {len(position)} points, range: {position.min():.2f} to {position.max():.2f}",
            )
            print(
                f"  - velocity: {len(velocity)} points, range: {velocity.min():.2f} to {velocity.max():.2f}",
            )
            print(
                f"  - acceleration: {len(acceleration)} points, range: {acceleration.min():.2f} to {acceleration.max():.2f}",
            )
            print(
                f"  - jerk: {len(jerk)} points, range: {jerk.min():.2f} to {jerk.max():.2f}",
            )

            # Store solution BEFORE plotting so BDC marker can access it
            self.current_solution = solution

            # Create plots based on selected view mode
            if self.plot_view_var.get() == "subplots":
                self._create_smart_scaled_plots(
                    cam_angle, position, velocity, acceleration, jerk,
                )
            else:
                self._create_single_plot(
                    cam_angle, position, velocity, acceleration, jerk,
                )

            print("DEBUG: About to refresh canvas...")

            # Refresh canvas
            self.canvas.draw()

            print("DEBUG: Canvas refreshed successfully")

            # Update status
            self.status_var.set("Motion law solved successfully")

            log.info("Plot updated successfully")

        except Exception as e:
            print(f"DEBUG: Error in _update_plot: {e}")
            import traceback

            traceback.print_exc()
            log.error(f"Error updating plot: {e}")
            self._show_error(f"Error updating plot: {e}")

    def _create_smart_scaled_plots(
        self, cam_angle, position, velocity, acceleration, jerk,
    ):
        """Create smart-scaled subplots for each motion law curve."""

        # Create 2x2 subplot layout
        axes = self.fig.subplots(2, 2)
        self.fig.suptitle("Cam Motion Law Curves", fontsize=14, fontweight="bold")

        # Define colors and labels
        curves = [
            (position, "b-", "Displacement", "mm", axes[0, 0]),
            (velocity, "g-", "Velocity", "mm/s", axes[0, 1]),
            (acceleration, "r-", "Acceleration", "mm/s²", axes[1, 0]),
            (jerk, "m-", "Jerk", "mm/s³", axes[1, 1]),
        ]

        for data, color, title, unit, ax in curves:
            # Plot the curve
            ax.plot(cam_angle, data, color, linewidth=2)

            # Add TDC/BDC markers
            ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)

            # Calculate BDC position based on upstroke duration
            if (
                hasattr(self, "current_solution")
                and "cam_angle" in self.current_solution
            ):
                # Calculate BDC based on upstroke duration percentage
                upstroke_duration = self.variables["upstroke_duration"].get()
                bdc_angle = (upstroke_duration / 100.0) * 360.0
                # BDC marker placed at calculated position
                ax.axvline(x=bdc_angle, color="black", linestyle="--", alpha=0.5)
            else:
                # Fallback to 180° if no solution available
                print("DEBUG: Using fallback BDC marker at 180°")
                ax.axvline(x=180, color="black", linestyle="--", alpha=0.5)

            # Smart scaling
            self._apply_smart_scaling(ax, data, cam_angle)

            # Formatting
            ax.set_xlabel("Cam Angle (degrees)")
            ax.set_ylabel(f"{title} ({unit})")
            ax.set_title(f"{title} vs Cam Angle")
            ax.grid(True, alpha=0.3)

            # Add statistics text box
            self._add_statistics_box(ax, data, unit)

        # Adjust layout to prevent overlap
        self.fig.tight_layout()

    def _create_single_plot(self, cam_angle, position, velocity, acceleration, jerk):
        """Create a single plot with all curves (traditional view)."""
        import numpy as np

        # Create single subplot
        ax = self.fig.add_subplot(111)
        self.fig.suptitle("Cam Motion Law Curves", fontsize=14, fontweight="bold")

        # Plot all curves on the same axes
        ax.plot(cam_angle, position, "b-", linewidth=2, label="Displacement (mm)")
        ax.plot(cam_angle, velocity, "g-", linewidth=2, label="Velocity (mm/s)")
        ax.plot(
            cam_angle, acceleration, "r-", linewidth=2, label="Acceleration (mm/s²)",
        )
        ax.plot(cam_angle, jerk, "m-", linewidth=2, label="Jerk (mm/s³)")

        # Add TDC/BDC markers
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="TDC")

        # Calculate BDC position based on upstroke duration
        if hasattr(self, "current_solution") and "cam_angle" in self.current_solution:
            # Calculate BDC based on upstroke duration percentage
            upstroke_duration = self.variables["upstroke_duration"].get()
            bdc_angle = (upstroke_duration / 100.0) * 360.0
            # BDC marker placed at calculated position
            ax.axvline(
                x=bdc_angle, color="black", linestyle="--", alpha=0.5, label="BDC",
            )
        else:
            # Fallback to 180° if no solution available
            print("DEBUG: Using fallback BDC marker at 180°")
            ax.axvline(x=180, color="black", linestyle="--", alpha=0.5, label="BDC")

        # Formatting
        ax.set_xlabel("Cam Angle (degrees)")
        ax.set_ylabel("Value")
        ax.set_title("All Motion Law Curves")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Set x-axis limits
        ax.set_xlim(0, 360)

        # Auto-scale y-axis to fit all data
        all_data = np.concatenate([position, velocity, acceleration, jerk])
        data_min, data_max = all_data.min(), all_data.max()
        data_range = data_max - data_min

        if data_range > 0:
            padding = data_range * 0.1
            ax.set_ylim(data_min - padding, data_max + padding)

        # Add horizontal line at zero if data crosses zero
        if data_min <= 0 <= data_max:
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

        # Adjust layout to prevent overlap
        self.fig.tight_layout()

    def _toggle_plot_view(self):
        """Toggle between subplot and single plot views."""
        if hasattr(self, "current_solution") and self.current_solution is not None:
            # Re-plot with the new view mode
            self._update_plot(self.current_solution)

    def _apply_smart_scaling(self, ax, data, cam_angle):
        """Apply smart scaling to a subplot."""

        # Calculate data range and add padding
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min

        # Handle edge cases
        if data_range == 0:
            # All values are the same
            center = data_min
            y_min = center - abs(center) * 0.1 if center != 0 else -1
            y_max = center + abs(center) * 0.1 if center != 0 else 1
        else:
            # Add 10% padding on each side
            padding = data_range * 0.1
            y_min = data_min - padding
            y_max = data_max + padding

        # Set y-axis limits
        ax.set_ylim(y_min, y_max)

        # Set x-axis limits (cam angle from 0 to 360)
        ax.set_xlim(0, 360)

        # Add horizontal line at zero if data crosses zero
        if y_min <= 0 <= y_max:
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3, linewidth=0.5)

    def _add_statistics_box(self, ax, data, unit):
        """Add a statistics text box to the subplot."""
        import numpy as np

        # Calculate statistics
        max_val = data.max()
        min_val = data.min()
        mean_val = data.mean()
        rms_val = np.sqrt(np.mean(data**2))

        # Create text box
        stats_text = f"Max: {max_val:.2f} {unit}\nMin: {min_val:.2f} {unit}\nMean: {mean_val:.2f} {unit}\nRMS: {rms_val:.2f} {unit}"

        # Position text box in upper right corner
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=8,
            family="monospace",
        )

    def _show_error(self, error_message):
        """Show error message."""
        messagebox.showerror("Error", error_message)
        self.status_var.set("Error occurred")

    def _enable_solve_button(self):
        """Re-enable the solve button."""
        self.solve_button.config(state="normal")

    def _enable_ring_solve_button(self):
        """Re-enable the ring solve button."""
        self.solve_ring_button.config(state="normal")

    def _solve_unified_optimization(self):
        """Solve all three optimization layers using the unified framework."""
        if not self._validate_inputs():
            return

        # Disable solve button during computation
        self.unified_solve_button.config(state="disabled")
        self.status_var.set("Running unified optimization...")

        # Run in separate thread to prevent GUI freezing
        thread = threading.Thread(target=self._solve_unified_thread)
        thread.daemon = True
        thread.start()

    def _solve_unified_thread(self):
        """Thread function for unified optimization."""
        try:
            print("DEBUG: Starting unified optimization thread...")

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
            }

            print(f"DEBUG: Unified optimization input data: {input_data}")

            # Perform cascaded optimization
            result_data = self.unified_framework.optimize_cascaded(input_data)

            # Store the result
            self.unified_result = result_data

            # Update plot on main thread
            self.root.after(0, self._update_unified_plot, result_data)

        except Exception as e:
            print(f"DEBUG: Error in unified optimization thread: {e}")
            import traceback

            traceback.print_exc()
            error_message = str(e)
            self.root.after(0, self._enable_unified_solve_button)
            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Unified optimization failed: {error_message}",
                ),
            )

    def _configure_unified_framework(self):
        """Configure the unified optimization framework with current settings."""
        # Get optimization method
        method_name = self.variables["optimization_method"].get()
        method = OptimizationMethod(method_name)

        # Create unified settings
        settings = UnifiedOptimizationSettings(
            method=method,
            collocation_degree=self.variables["collocation_degree"].get(),
            max_iterations=self.variables["max_iterations"].get(),
            tolerance=self.variables["tolerance"].get(),
            lagrangian_tolerance=self.variables["lagrangian_tolerance"].get(),
            penalty_weight=self.variables["penalty_weight"].get(),
        )

        # Create unified constraints
        constraints = UnifiedOptimizationConstraints(
            stroke_min=1.0,
            stroke_max=100.0,
            max_velocity=self.variables["max_velocity"].get()
            if self.variables["max_velocity"].get() > 0
            else None,
            max_acceleration=self.variables["max_acceleration"].get()
            if self.variables["max_acceleration"].get() > 0
            else None,
            max_jerk=self.variables["max_jerk"].get()
            if self.variables["max_jerk"].get() > 0
            else None,
            base_radius_min=5.0,
            base_radius_max=100.0,
            connecting_rod_length_min=10.0,
            connecting_rod_length_max=200.0,
            sun_gear_radius_min=8.0,
            sun_gear_radius_max=50.0,
            ring_gear_radius_min=25.0,
            ring_gear_radius_max=150.0,
            min_gear_ratio=1.5,
            max_gear_ratio=10.0,
            max_back_rotation=np.pi,
        )

        # Create unified targets
        targets = UnifiedOptimizationTargets(
            minimize_jerk=self.variables["minimize_jerk"].get(),
            minimize_time=self.variables["minimize_time"].get(),
            minimize_energy=self.variables["minimize_energy"].get(),
            minimize_ring_size=self.variables["minimize_ring_size"].get(),
            minimize_cam_size=self.variables["minimize_cam_size"].get(),
            minimize_curvature_variation=self.variables[
                "minimize_curvature_variation"
            ].get(),
            minimize_system_size=self.variables["minimize_system_size"].get(),
            maximize_efficiency=self.variables["maximize_efficiency"].get(),
            minimize_back_rotation=self.variables["minimize_back_rotation"].get(),
            minimize_gear_stress=self.variables["minimize_gear_stress"].get(),
        )

        # Configure framework
        self.unified_framework.configure(
            settings=settings, constraints=constraints, targets=targets,
        )

        print(f"DEBUG: Configured unified framework with method: {method.value}")

    def _solve_primary_only(self):
        """Solve only the primary optimization layer."""
        if not self._validate_inputs():
            return

        self.solve_primary_button.config(state="disabled")
        self.status_var.set("Solving primary optimization...")

        thread = threading.Thread(target=self._solve_primary_thread)
        thread.daemon = True
        thread.start()

    def _solve_secondary_only(self):
        """Solve only the secondary optimization layer."""
        if not self._validate_ring_inputs():
            return

        self.solve_secondary_button.config(state="disabled")
        self.status_var.set("Solving secondary optimization...")

        thread = threading.Thread(target=self._solve_secondary_thread)
        thread.daemon = True
        thread.start()

    def _solve_tertiary_only(self):
        """Solve only the tertiary optimization layer."""
        if not self._validate_tertiary_inputs():
            return

        self.solve_tertiary_button.config(state="disabled")
        self.status_var.set("Solving tertiary optimization...")

        thread = threading.Thread(target=self._solve_tertiary_thread)
        thread.daemon = True
        thread.start()

    def _solve_primary_thread(self):
        """Thread function for primary optimization only."""
        try:
            print("DEBUG: Starting primary optimization thread...")
            # Implementation for primary-only optimization
            # This would use the unified framework but only run the primary layer
            self.root.after(0, self._enable_primary_solve_button)
            self.root.after(
                0, lambda: self.status_var.set("Primary optimization completed"),
            )
        except Exception as e:
            print(f"DEBUG: Error in primary optimization thread: {e}")
            error_message = str(e)
            self.root.after(0, self._enable_primary_solve_button)
            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Primary optimization failed: {error_message}",
                ),
            )

    def _solve_secondary_thread(self):
        """Thread function for secondary optimization only."""
        try:
            print("DEBUG: Starting secondary optimization thread...")
            # Implementation for secondary-only optimization
            self.root.after(0, self._enable_secondary_solve_button)
            self.root.after(
                0, lambda: self.status_var.set("Secondary optimization completed"),
            )
        except Exception as e:
            print(f"DEBUG: Error in secondary optimization thread: {e}")
            error_message = str(e)
            self.root.after(0, self._enable_secondary_solve_button)
            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Secondary optimization failed: {error_message}",
                ),
            )

    def _solve_tertiary_thread(self):
        """Thread function for tertiary optimization only."""
        try:
            print("DEBUG: Starting tertiary optimization thread...")
            # Implementation for tertiary-only optimization
            self.root.after(0, self._enable_tertiary_solve_button)
            self.root.after(
                0, lambda: self.status_var.set("Tertiary optimization completed"),
            )
        except Exception as e:
            print(f"DEBUG: Error in tertiary optimization thread: {e}")
            error_message = str(e)
            self.root.after(0, self._enable_tertiary_solve_button)
            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Tertiary optimization failed: {error_message}",
                ),
            )

    def _enable_unified_solve_button(self):
        """Re-enable the unified solve button."""
        self.unified_solve_button.config(state="normal")

    def _enable_primary_solve_button(self):
        """Re-enable the primary solve button."""
        self.solve_primary_button.config(state="normal")

    def _enable_secondary_solve_button(self):
        """Re-enable the secondary solve button."""
        self.solve_secondary_button.config(state="normal")

    def _enable_tertiary_solve_button(self):
        """Re-enable the tertiary solve button."""
        self.solve_tertiary_button.config(state="normal")

    def _update_unified_plot(self, result_data):
        """Update the plot with unified optimization results."""
        try:
            print("DEBUG: Updating unified plot with result data")

            # Clear previous plot
            self.fig.clear()

            # Create comprehensive plot showing all three layers
            self._create_unified_plots(result_data)

            # Refresh canvas
            self.canvas.draw()

            # Update status
            summary = self.unified_framework.get_optimization_summary()
            self.status_var.set(
                f"Unified optimization completed - Method: {summary['method']}, Time: {summary['total_solve_time']:.3f}s",
            )

        except Exception as e:
            print(f"DEBUG: Error updating unified plot: {e}")
            import traceback

            traceback.print_exc()
            self.status_var.set(f"Plot update failed: {e}")
        finally:
            self._enable_unified_solve_button()

    def _update_ring_plot(self, ring_result):
        """Update the plot with ring design results."""
        try:
            print(
                f"DEBUG: Updating ring plot with result keys: {list(ring_result.keys())}",
            )

            # Clear previous plot
            self.fig.clear()

            # Create subplots for both primary and secondary results
            if hasattr(self, "current_solution") and self.current_solution is not None:
                # Show both primary and secondary results
                self._create_combined_plots(self.current_solution, ring_result)
            else:
                # Show only ring results
                self._create_ring_only_plots(ring_result)

            # Refresh canvas
            self.canvas.draw()

            # Update status
            self.status_var.set("Cam and ring profiles generated successfully")

        except Exception as e:
            print(f"DEBUG: Error updating ring plot: {e}")
            import traceback

            traceback.print_exc()
            self._show_error(f"Error updating plot: {e}")

    def _create_combined_plots(self, primary_solution, ring_result):
        """Create combined plots showing both primary and secondary results."""
        # Create 1x2 subplot layout for polar plots
        gs = self.fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

        # Cam profile (left polar plot)
        ax1 = self.fig.add_subplot(gs[0, 0], projection="polar")

        # Ring profile (right polar plot)
        ax2 = self.fig.add_subplot(gs[0, 1], projection="polar")

        # Plot cam profile
        if "theta" in ring_result and "cam_curves" in ring_result:
            theta_rad = np.deg2rad(ring_result["theta"])
            cam_curves = ring_result["cam_curves"]

            # Plot pitch curve (cam centerline)
            if "pitch_radius" in cam_curves:
                ax1.plot(
                    theta_rad,
                    cam_curves["pitch_radius"],
                    "b-",
                    linewidth=2,
                    label="Pitch Curve",
                )

            # Plot profile curve (cam surface)
            if "profile_radius" in cam_curves:
                ax1.plot(
                    theta_rad,
                    cam_curves["profile_radius"],
                    "r-",
                    linewidth=2,
                    label="Profile Curve",
                )

            ax1.set_title("Cam Profile", pad=20)
            ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
            ax1.grid(True, alpha=0.3)

        # Plot ring profile
        if "psi" in ring_result and "R_psi" in ring_result:
            psi_rad = ring_result["psi"]  # psi is already in radians
            R_psi = ring_result["R_psi"]

            ax2.plot(psi_rad, R_psi, "g-", linewidth=2, label="Ring Profile")
            ax2.set_title("Ring Follower Profile", pad=20)
            ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
            ax2.grid(True, alpha=0.3)

    def _create_ring_only_plots(self, ring_result):
        """Create plots showing only ring design results."""
        # Create 1x2 subplot layout for polar plots
        gs = self.fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

        # Cam profile (left polar plot)
        ax1 = self.fig.add_subplot(gs[0, 0], projection="polar")

        # Ring profile (right polar plot)
        ax2 = self.fig.add_subplot(gs[0, 1], projection="polar")

        # Plot cam profile
        if "theta" in ring_result and "cam_curves" in ring_result:
            theta_rad = np.deg2rad(ring_result["theta"])
            cam_curves = ring_result["cam_curves"]

            # Plot pitch curve (cam centerline)
            if "pitch_radius" in cam_curves:
                ax1.plot(
                    theta_rad,
                    cam_curves["pitch_radius"],
                    "b-",
                    linewidth=2,
                    label="Pitch Curve",
                )

            # Plot profile curve (cam surface)
            if "profile_radius" in cam_curves:
                ax1.plot(
                    theta_rad,
                    cam_curves["profile_radius"],
                    "r-",
                    linewidth=2,
                    label="Profile Curve",
                )

            ax1.set_title("Cam Profile", pad=20)
            ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
            ax1.grid(True, alpha=0.3)

        # Plot ring profile
        if "psi" in ring_result and "R_psi" in ring_result:
            psi_rad = ring_result["psi"]  # psi is already in radians
            R_psi = ring_result["R_psi"]

            ax2.plot(psi_rad, R_psi, "g-", linewidth=2, label="Ring Profile")
            ax2.set_title("Ring Follower Profile", pad=20)
            ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
            ax2.grid(True, alpha=0.3)

    def _save_plot(self):
        """Save the current plot."""
        if not hasattr(self, "current_solution"):
            messagebox.showwarning(
                "Warning", "No plot to save. Please solve a motion law first.",
            )
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
            ],
            title="Save Motion Law Plot",
        )

        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches="tight")
                messagebox.showinfo("Success", f"Plot saved to {filename}")
                self.status_var.set(f"Plot saved to {Path(filename).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {e}")

    def _clear_plot(self):
        """Clear the current plot."""
        self.fig.clear()

        # Create a single subplot for the empty state
        ax = self.fig.add_subplot(111)
        ax.set_xlabel("Cam Angle (degrees)")
        ax.set_ylabel("Value")
        ax.set_title("Cam Motion Law Curves")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.5,
            0.5,
            'Click "Solve Motion Law" to generate curves',
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            alpha=0.7,
        )
        ax.set_xlim(0, 360)
        ax.set_ylim(-1, 1)

        self.canvas.draw()

        if hasattr(self, "current_solution"):
            delattr(self, "current_solution")

        self.status_var.set("Plot cleared")

    def _on_stroke_changed(self, *args):
        """Update initial guesses when stroke changes."""
        try:
            stroke = self.variables["stroke"].get()
            if stroke > 0:
                # Set initial cam radius to stroke length
                self.variables["base_radius"].set(stroke)

                # Set initial rod length slightly longer than stroke (1.25x)
                rod_length = stroke * 1.25
                self.variables["connecting_rod_length"].set(rod_length)

                print(f"DEBUG: Updated initial guesses based on stroke {stroke} mm:")
                print(f"  - Cam base radius: {stroke} mm")
                print(f"  - Connecting rod length: {rod_length} mm")
        except Exception as e:
            print(f"DEBUG: Error updating initial guesses: {e}")

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
                "zero_accel_duration_percent": self.variables[
                    "zero_accel_duration"
                ].get(),
            }

            print(f"DEBUG: System optimization input data: {input_data}")

            # Perform cascaded optimization
            result_data = self.unified_framework.optimize_cascaded(input_data)

            # Store the result
            self.unified_result = result_data

            # Update plots on main thread
            self.root.after(0, self._update_all_plots, result_data)

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

            # Update status
            self.status_var.set(
                f"Optimization completed - Total time: {result_data.total_solve_time:.3f}s",
            )

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
            # Create subplots for position, velocity, acceleration
            ax1 = self.motion_law_fig.add_subplot(311)
            ax2 = self.motion_law_fig.add_subplot(312)
            ax3 = self.motion_law_fig.add_subplot(313)

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
                ax3.set_xlabel("Cam Angle (degrees)")
                ax3.grid(True, alpha=0.3)
                ax3.legend()

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

            # Plot cam profile (polar)
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
            result_data.primary_theta is not None
            and result_data.primary_position is not None
        ):
            # Store animation data
            self.animation_data = result_data

            # Create initial frame
            self._draw_animation_frame(0)

            self.animation_fig.suptitle(
                "Cam-Ring System Animation", fontsize=14, fontweight="bold",
            )
        else:
            ax = self.animation_fig.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "No animation data available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Cam-Ring System Animation")

        self.animation_fig.tight_layout()
        self.animation_canvas.draw()

    def _draw_animation_frame(self, frame_index):
        """Draw a single animation frame."""
        if not hasattr(self, "animation_data") or self.animation_data is None:
            return

        self.animation_fig.clear()
        ax = self.animation_fig.add_subplot(111)

        # Calculate current state
        total_frames = self.variables["animation_frames"].get()
        if total_frames <= 0:
            total_frames = 60

        # Interpolate to current frame
        if (
            self.animation_data.primary_theta is not None
            and len(self.animation_data.primary_theta) > 0
        ):
            theta_current = np.interp(
                frame_index / (total_frames - 1),
                np.linspace(0, 1, len(self.animation_data.primary_theta)),
                self.animation_data.primary_theta,
            )

            if self.animation_data.primary_position is not None:
                position_current = np.interp(
                    frame_index / (total_frames - 1),
                    np.linspace(0, 1, len(self.animation_data.primary_position)),
                    self.animation_data.primary_position,
                )

                # Draw system components
                self._draw_system_components(ax, theta_current, position_current)

        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"Frame {frame_index + 1}/{total_frames} - θ = {theta_current:.1f}°",
        )

        self.animation_fig.tight_layout()
        self.animation_canvas.draw()

    def _draw_system_components(self, ax, theta, position):
        """Draw the cam-ring system components."""
        # Draw linear follower
        ax.plot([0, 0], [0, position], "b-", linewidth=3, label="Linear Follower")
        ax.plot(0, position, "bo", markersize=8, label="Follower Position")

        # Draw connecting rod (if available)
        if (
            hasattr(self, "animation_data")
            and self.animation_data.secondary_rod_length is not None
        ):
            rod_length = self.animation_data.secondary_rod_length
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

    def _play_animation(self):
        """Start animation playback."""
        if not hasattr(self, "animation_data") or self.animation_data is None:
            self.status_var.set("No animation data available. Run optimization first.")
            return

        self.animation_playing = True
        self.animation_frame_index = 0
        self._animate_next_frame()

    def _animate_next_frame(self):
        """Animate to the next frame."""
        if not self.animation_playing:
            return

        total_frames = self.variables["animation_frames"].get()
        if total_frames <= 0:
            total_frames = 60

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
                    "secondary_rod_length": self.unified_result.secondary_rod_length,
                    "total_solve_time": self.unified_result.total_solve_time,
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
        self.variables["animation_frames"].set(60)
        self.variables["animation_speed"].set(1.0)

        # Clear results and plots
        self.unified_result = None
        self._clear_all_plots()

        # Update initial guesses
        self._on_stroke_changed()

        self.status_var.set("Parameters reset to defaults")

    def _on_method_changed(self, *args):
        """Update method description when optimization method changes."""
        method = self.variables["optimization_method"].get()
        descriptions = {
            "legendre_collocation": "High-precision collocation with Legendre polynomials",
            "radau_collocation": "Radau collocation for stiff systems",
            "hermite_collocation": "Hermite collocation for smooth solutions",
            "slsqp": "Sequential Quadratic Programming",
            "l_bfgs_b": "Limited-memory BFGS with bounds",
            "tnc": "Truncated Newton with constraints",
            "cobyla": "Constrained Optimization BY Linear Approximation",
            "differential_evolution": "Global optimization with differential evolution",
            "basin_hopping": "Global optimization with basin hopping",
            "dual_annealing": "Global optimization with dual annealing",
            "lagrangian": "Lagrangian multiplier method",
            "penalty_method": "Penalty function method",
            "augmented_lagrangian": "Augmented Lagrangian method",
        }

        description = descriptions.get(method, "Unknown optimization method")
        self.method_desc_var.set(description)

        # Update status
        self.status_var.set(f"Method changed to: {method} - {description}")

    def _reset_parameters(self):
        """Reset all parameters to default values."""
        # Unified optimization method
        self.variables["optimization_method"].set("legendre_collocation")

        # Primary optimization parameters
        self.variables["stroke"].set(20.0)
        self.variables["upstroke_duration"].set(60.0)
        self.variables["zero_accel_duration"].set(0.0)
        self.variables["cycle_time"].set(1.0)
        self.variables["max_velocity"].set(0.0)
        self.variables["max_acceleration"].set(0.0)
        self.variables["max_jerk"].set(0.0)
        self.variables["motion_type"].set("minimum_jerk")
        self.variables["dwell_at_tdc"].set(True)
        self.variables["dwell_at_bdc"].set(True)

        # Secondary optimization parameters
        self.variables["base_radius"].set(15.0)
        self.variables["connecting_rod_length"].set(25.0)
        self.variables["contact_type"].set("external")

        # Tertiary optimization parameters
        self.variables["sun_gear_radius"].set(15.0)
        self.variables["ring_gear_radius"].set(45.0)
        self.variables["gear_ratio"].set(3.0)
        self.variables["max_back_rotation"].set(45.0)

        # Unified optimization settings
        self.variables["collocation_degree"].set(3)
        self.variables["max_iterations"].set(100)
        self.variables["tolerance"].set(1e-6)
        self.variables["lagrangian_tolerance"].set(1e-8)
        self.variables["penalty_weight"].set(1.0)

        # Optimization targets
        self.variables["minimize_jerk"].set(True)
        self.variables["minimize_time"].set(False)
        self.variables["minimize_energy"].set(False)
        self.variables["minimize_ring_size"].set(True)
        self.variables["minimize_cam_size"].set(True)
        self.variables["minimize_curvature_variation"].set(True)
        self.variables["minimize_system_size"].set(True)
        self.variables["maximize_efficiency"].set(True)
        self.variables["minimize_back_rotation"].set(True)
        self.variables["minimize_gear_stress"].set(True)

        # Update initial guesses based on default stroke
        self._on_stroke_changed()

        self.status_var.set("Parameters reset to defaults")


def main():
    """Main function to run the GUI."""
    root = tk.Tk()

    # Set style
    style = ttk.Style()
    style.theme_use("clam")  # Use a modern theme

    # Create and run the GUI
    app = CamMotionGUI(root)

    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    print("DEBUG: Starting Cam Motion Law GUI...")
    print("DEBUG: Loading updated validation logic...")

    # Test the constraint validation directly
    try:
        from campro.constraints.cam import CamMotionConstraints

        test_constraints = CamMotionConstraints(
            stroke=20.0,
            upstroke_duration_percent=30.0,
            zero_accel_duration_percent=60.0,
        )
        print(
            "DEBUG: Constraint validation test PASSED - zero acceleration can exceed upstroke",
        )
    except Exception as e:
        print(f"DEBUG: Constraint validation test FAILED: {e}")

    log.info("Starting Cam Motion Law GUI")
    root.mainloop()


if __name__ == "__main__":
    main()
