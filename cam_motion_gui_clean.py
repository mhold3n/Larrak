"""
DEPRECATED: Legacy clean GUI prototype.

Use `cam_motion_gui.py` (the main GUI) going forward. This prototype is kept
for reference and may be removed in a future release.
"""

import warnings

warnings.warn(
    "cam_motion_gui_clean is deprecated; use cam_motion_gui instead.",
    DeprecationWarning,
    stacklevel=2,
)

import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from campro.logging import get_logger
from campro.optimization.unified_framework import (
    OptimizationMethod,
    UnifiedOptimizationConstraints,
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
    UnifiedOptimizationTargets,
)
from campro.storage import OptimizationRegistry

log = get_logger(__name__)


class CamMotionGUI:
    """Clean, focused GUI for cam-ring system design."""

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

        # Create constraints
        constraints = UnifiedOptimizationConstraints(
            stroke_min=1.0,
            stroke_max=100.0,
            max_velocity=100.0,
            max_acceleration=1000.0,
            max_jerk=10000.0,
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

        # Create targets
        targets = UnifiedOptimizationTargets(
            minimize_jerk=True,
            minimize_time=False,
            minimize_energy=False,
            minimize_ring_size=True,
            minimize_cam_size=True,
            minimize_curvature_variation=True,
            minimize_system_size=True,
            maximize_efficiency=True,
            minimize_back_rotation=True,
            minimize_gear_stress=True,
        )

        # Configure framework
        self.unified_framework.configure(
            settings=settings, constraints=constraints, targets=targets,
        )
        print(f"DEBUG: Configured unified framework with method: {method_name}")

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


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = CamMotionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
