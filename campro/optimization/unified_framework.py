"""
Unified optimization framework for cascaded cam-ring system optimization.

This module provides a unified framework that homogenizes all three optimization
processes (primary motion law, secondary cam-ring, tertiary sun gear) to use
shared solution methods, libraries, and data structures.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from campro.diagnostics.feasibility import check_feasibility
from campro.diagnostics.scaling import compute_scaling_vector
from campro.logging import get_logger
from campro.optimization.casadi_unified_flow import (
    CasADiOptimizationSettings,
    CasADiUnifiedFlow,
)
from campro.optimization.ma57_migration_analyzer import MA57MigrationAnalyzer
from campro.optimization.parameter_tuning import DynamicParameterTuner
from campro.optimization.solver_analysis import MA57ReadinessReport
from campro.optimization.solver_selection import (
    AdaptiveSolverSelector,
    ProblemCharacteristics,
)
from campro.utils.progress_logger import ProgressLogger

from .base import OptimizationResult, OptimizationStatus
from .cam_ring_optimizer import (
    CamRingOptimizationConstraints,
    CamRingOptimizationTargets,
    CamRingOptimizer,
)
from .collocation import CollocationSettings
from .crank_center_optimizer import (
    CrankCenterOptimizationConstraints,
    CrankCenterOptimizationTargets,
    CrankCenterOptimizer,
)
from .motion import MotionOptimizer
from .grid import UniversalGrid, GridMapper

log = get_logger(__name__)


class OptimizationMethod(Enum):
    """Available optimization methods for all optimization layers.

    All optimization phases (1, 2, 3) use CasADi/IPOPT solution methods.
    Only collocation methods are supported.
    """

    # Collocation methods (used by CasADi/IPOPT)
    LEGENDRE_COLLOCATION = "legendre_collocation"
    RADAU_COLLOCATION = "radau_collocation"
    HERMITE_COLLOCATION = "hermite_collocation"


class OptimizationLayer(Enum):
    """Optimization layers in the cascaded system."""

    PRIMARY = "primary"  # Linear follower motion law
    SECONDARY = "secondary"  # Cam-ring system parameters
    TERTIARY = "tertiary"  # Crank center optimization for torque maximization and side-loading minimization


@dataclass
class UnifiedOptimizationSettings:
    """Unified settings for all optimization layers."""

    # Solution method
    method: OptimizationMethod = OptimizationMethod.LEGENDRE_COLLOCATION

    # Collocation settings (when using collocation methods)
    collocation_degree: int = 3
    collocation_tolerance: float = 1e-6
    # Legacy/Universal-grid controls (predate CasADi ladder system):
    # Universal grid size (GUI-controlled). Used for downstream comparisons and invariant checks.
    # NOTE: This does NOT influence the CasADi ladder's segment counts. The ladder uses
    # casadi_coarse_segments and casadi_resolution_ladder for its adaptive refinement.
    universal_n_points: int = 360
    # CasADi primary optimization toggle/settings
    use_casadi: bool = False
    casadi_poly_order: int = 3
    casadi_collocation_method: str = "legendre"
    casadi_coarse_segments: tuple[int, ...] = (40, 80, 160)
    casadi_resolution_ladder: tuple[int, ...] | None = None
    casadi_target_angle_deg: float = 0.1
    casadi_max_segments: int = 4096
    casadi_retry_failed_level: bool = True
    # Primary phase discretization parameters (for experimentation and debugging)
    primary_min_segments: int = 10  # Minimum number of collocation segments K
    primary_refinement_factor: int = 4  # K ≈ n_points / refinement_factor
    # Legacy/Universal-grid controls (predate CasADi ladder system):
    # Mapper method controls how solutions are resampled onto the universal grid for
    # invariant checks. This mapping existed long before the CasADi flow.
    # Options: linear, pchip, barycentric, projection
    mapper_method: str = "linear"
    # Grid diagnostics and plots: optional instrumentation that runs after mapping
    # regardless of which optimizer produced the solution. Helpful when validating the new ladder.
    enable_grid_diagnostics: bool = False
    enable_grid_plots: bool = False
    # GridSpec metadata for stages
    grid_family: str = "uniform"  # e.g., uniform, LGL, Radau, Chebyshev
    grid_segments: int = 1
    # Shared collocation method selection for all modules (GUI dropdown: 'legendre', 'radau', 'lobatto').
    # Note: This is a temporary global toggle controlled by the GUI. In a future iteration we will
    # allow granular per-stage method selection (primary/secondary/tertiary) and per-stage degrees.
    collocation_method: str = "legendre"

    # General optimization settings (used by constraint objects)
    max_iterations: int = 100
    tolerance: float = 1e-6

    # General settings
    parallel_processing: bool = True
    verbose: bool = True
    save_intermediate_results: bool = True

    # Ipopt analysis settings
    enable_ipopt_analysis: bool = True

    # Thermal-efficiency-focused primary optimization (complex gas optimizer)
    use_thermal_efficiency: bool = False
    thermal_efficiency_config: dict[str, Any] | None = None
    # If true, fail fast when TE is enabled but unavailable/non-converged
    require_thermal_efficiency: bool = False
    # Phase-1: constant load model (free piston against generator)
    constant_load_value: float = 1.0
    # Phase-1: constant operating temperature (free-piston idealization)
    constant_temperature_K: float = 900.0

    # CasADi physics validation mode settings
    enable_casadi_validation_mode: bool = False
    casadi_validation_tolerance: float = 1e-4

    # Phase-1: pressure-slope invariance option (robust to fuel/load via base-pressure scheduling)
    use_pressure_invariance: bool = False
    fuel_sweep: list[float] = field(default_factory=lambda: [0.7, 1.0, 1.3])
    load_sweep: list[float] = field(default_factory=lambda: [50.0, 100.0, 150.0])
    dpdt_weight: float = 1.0
    jerk_weight: float = 1.0
    imep_weight: float = 0.2
    ema_alpha: float = 0.25
    outer_iterations: int = 2
    # Adapter configuration
    bounce_alpha: float = 30.0  # kPa per 1.0 fuel-mult step
    bounce_beta: float = 100.0  # kPa base
    piston_area_mm2: float = 1.0
    clearance_volume_mm3: float = 1.0
    # TE guardrail (Stage B) parameters
    pressure_guard_epsilon: float = 1e-3
    pressure_guard_lambda: float = 1e4

    # PR template parameters (geometry-informed pressure ratio template)
    pr_template_expansion_efficiency: float = 0.85  # Target expansion efficiency (0-1)
    pr_template_peak_scale: float = 1.5  # Peak PR scaling factor relative to baseline
    pr_template_use_template: bool = True  # Use explicit template instead of seed-derived PR

    # Secondary collocation tracking weight (golden profile influence). The GUI can expose this as a
    # user-tunable numeric field to adjust how strongly secondary tracks the golden motion profile.
    tracking_weight: float = 1.0

    # Primary phase-1 weighting knobs available for GUI exposure:
    # - dpdt_weight: pressure-slope alignment weight (guardrails)
    # - jerk_weight: smoothness (jerk-squared) weight
    # - imep_weight: efficiency surrogate (higher is better)
    # These can be surfaced as sliders in the GUI and set here prior to optimization.


@dataclass
class UnifiedOptimizationConstraints:
    """Unified constraints for all optimization layers."""

    # Primary layer constraints (motion law)
    stroke_min: float = 1.0
    stroke_max: float = 100.0
    cycle_time_min: float = 0.1
    cycle_time_max: float = 10.0
    max_velocity: float | None = None
    max_acceleration: float | None = None
    max_jerk: float | None = None

    # Secondary layer constraints (cam-ring)
    base_radius_min: float = 5.0
    base_radius_max: float = 100.0
    # connecting_rod_length_min: float = 10.0  # Removed for phase 2 simplification
    # connecting_rod_length_max: float = 200.0  # Removed for phase 2 simplification
    min_curvature_radius: float = 1.0
    max_curvature: float = 10.0

    # Tertiary layer constraints (crank center optimization)
    crank_center_x_min: float = -50.0  # mm
    crank_center_x_max: float = 50.0  # mm
    crank_center_y_min: float = -50.0  # mm
    crank_center_y_max: float = 50.0  # mm
    crank_radius_min: float = 20.0  # mm
    crank_radius_max: float = 100.0  # mm
    rod_length_min: float = 100.0  # mm
    rod_length_max: float = 300.0  # mm
    min_torque_output: float = 100.0  # N⋅m
    max_side_load_penalty: float = 500.0  # N

    # Secondary layer constraints (for backward compatibility)
    ring_gear_radius: float = 45.0  # GUI target for average ring radius

    # Physical constraints
    min_clearance: float = 2.0
    max_interference: float = 0.0
    min_ring_coverage: float = 2 * np.pi  # 360° coverage required


@dataclass
class UnifiedOptimizationTargets:
    """Unified optimization targets for all layers."""

    # Primary layer targets
    minimize_jerk: bool = True
    minimize_time: bool = False
    minimize_energy: bool = False

    # Secondary layer targets
    minimize_ring_size: bool = True
    minimize_cam_size: bool = True
    minimize_curvature_variation: bool = True

    # Tertiary layer targets (crank center optimization)
    maximize_torque: bool = True
    minimize_side_loading: bool = True
    minimize_side_loading_during_compression: bool = True
    minimize_side_loading_during_combustion: bool = True
    minimize_torque_ripple: bool = True
    maximize_power_output: bool = True

    # Weighting factors
    jerk_weight: float = 1.0
    time_weight: float = 0.5
    energy_weight: float = 0.3
    ring_size_weight: float = 1.0
    cam_size_weight: float = 0.5
    curvature_weight: float = 0.2
    # Crank center optimization weights
    torque_weight: float = 1.0
    side_load_weight: float = 0.8
    compression_side_load_weight: float = 1.2
    combustion_side_load_weight: float = 1.5
    torque_ripple_weight: float = 0.3
    power_output_weight: float = 0.5


@dataclass
class UnifiedOptimizationData:
    """Unified data structure for all optimization layers.
    
    Motion-law inputs are per-degree or unitless:
    - stroke: in mm (or m)
    - duration_angle_deg: required, in degrees
    - upstroke_duration_percent: percentage of cycle
    - Optional actuator limits: in mm/degⁿ (velocity, acceleration, jerk)
    - engine_speed_rpm: optional, used to derive cycle_time
    - cycle_time: derived from engine_speed_rpm and duration_angle_deg, not primary input
    """

    # Input data
    stroke: float = 20.0
    cycle_time: float = 1.0  # Cycle time in seconds (derived from engine_speed_rpm and duration_angle_deg)
    engine_speed_rpm: float | None = None  # Engine speed in RPM (optional, used to derive cycle_time)
    duration_angle_deg: float = 360.0  # Required: motion law duration in degrees
    upstroke_duration_percent: float = 60.0
    zero_accel_duration_percent: float = 0.0
    motion_type: str = "minimum_jerk"
    # Combustion parameters (not wired yet - will be used in next phase)
    afr: float = 18.0
    ignition_timing: float = 0.005
    ignition_deg: float = -5.0
    fuel_mass: float = 5e-4
    ca50_target_deg: float = 0.0
    ca50_weight: float = 0.0
    ca_duration_target_deg: float = 0.0
    ca_duration_weight: float = 0.0
    injector_delay_s: float | None = None  # Injector delay [s]
    injector_delay_deg: float | None = None  # Injector delay [deg]

    # Primary results
    primary_theta: np.ndarray | None = None
    primary_position: np.ndarray | None = None
    primary_velocity: np.ndarray | None = None
    primary_acceleration: np.ndarray | None = None
    primary_jerk: np.ndarray | None = None
    primary_load_profile: np.ndarray | None = None
    primary_constant_load_value: float | None = None
    primary_constant_temperature_K: float | None = None
    primary_ca_markers: dict[str, float] | None = None
    primary_position_units: str | None = None
    primary_velocity_units: str | None = None
    primary_acceleration_units: str | None = None
    primary_jerk_units: str | None = None
    primary_pressure_invariance: dict[str, Any] | None = None

    # Secondary results
    secondary_base_radius: float | None = None
    # secondary_rod_length: Optional[float] = None  # Removed for phase 2 simplification
    secondary_cam_curves: dict[str, np.ndarray] | None = None
    secondary_psi: np.ndarray | None = None
    secondary_R_psi: np.ndarray | None = None
    secondary_gear_geometry: dict[str, Any] | None = None
    secondary_ring_profile: dict[str, Any] | None = None  # Synchronized ring profile on universal theta grid

    # Tertiary results (crank center optimization)
    tertiary_crank_center_x: float | None = None
    tertiary_crank_center_y: float | None = None
    tertiary_crank_radius: float | None = None
    tertiary_rod_length: float | None = None
    tertiary_torque_output: float | None = None
    tertiary_side_load_penalty: float | None = None
    tertiary_max_torque: float | None = None
    tertiary_torque_ripple: float | None = None
    tertiary_power_output: float | None = None
    tertiary_max_side_load: float | None = None

    # Golden profile for downstream tracking (radial follower center)
    golden_profile: dict[str, Any] | None = None

    # Universal grid snapshots
    universal_theta_deg: np.ndarray | None = None
    universal_theta_rad: np.ndarray | None = None

    # Metadata
    optimization_method: OptimizationMethod | None = None
    total_solve_time: float = 0.0
    convergence_info: dict[str, Any] = field(default_factory=dict)

    # Per-phase Ipopt analysis results
    primary_ipopt_analysis: MA57ReadinessReport | None = None
    secondary_ipopt_analysis: MA57ReadinessReport | None = None
    tertiary_ipopt_analysis: MA57ReadinessReport | None = None


class UnifiedOptimizationFramework:
    """
    Unified optimization framework for cascaded cam-ring system optimization.

    This framework homogenizes all three optimization processes to use shared
    solution methods, libraries, and data structures for seamless cascading.
    """

    def __init__(
        self,
        name: str = "UnifiedOptimizationFramework",
        settings: UnifiedOptimizationSettings | None = None,
    ):
        self.name = name
        self.settings = settings or UnifiedOptimizationSettings()
        self.constraints = UnifiedOptimizationConstraints()
        self.targets = UnifiedOptimizationTargets()
        self.data = UnifiedOptimizationData()
        self._is_configured = True

        # Add performance tuning components
        self.solver_selector = AdaptiveSolverSelector()
        self.parameter_tuner = DynamicParameterTuner()
        self.migration_analyzer = MA57MigrationAnalyzer()

        # Initialize optimizers
        self._initialize_optimizers()

    def _initialize_optimizers(self) -> None:
        """Initialize all optimization layers with unified settings."""
        # Create and store universal grid
        ug = UniversalGrid(n_points=int(self.settings.universal_n_points))
        self.data.universal_theta_rad = ug.theta_rad
        self.data.universal_theta_deg = ug.theta_deg
        # Create GridSpec for stages
        from .grid import GridSpec
        self.grid_spec = GridSpec(
            method=self.settings.collocation_method,
            degree=self.settings.collocation_degree,
            n_points=int(self.settings.universal_n_points),
            periodic=True,
            family=self.settings.grid_family,
            segments=self.settings.grid_segments,
        )
        # Create collocation settings (method set by GUI dropdown; shared across modules)
        collocation_settings = CollocationSettings(
            degree=self.settings.collocation_degree,
            tolerance=self.settings.collocation_tolerance,
            method=self.settings.collocation_method,
        )

        # Initialize primary optimizer (motion law)
        # Use free-piston IPOPT flow with integrated combustion model
        from campro.optimization.freepiston_phase1_adapter import FreePistonPhase1Adapter

        self.primary_optimizer = FreePistonPhase1Adapter()
        # Configure with discretization parameters from settings
        self.primary_optimizer.configure(
            min_segments=self.settings.primary_min_segments,
            refinement_factor=self.settings.primary_refinement_factor,
            disable_combustion=False,  # Can be enabled via settings.diagnostics if needed
        )

        # Initialize secondary optimizer (cam-ring)
        self.secondary_optimizer = CamRingOptimizer(
            name="SecondaryCamRingOptimizer",
        )

        # Initialize tertiary optimizer (crank center optimization)
        self.tertiary_optimizer = CrankCenterOptimizer(
            name="TertiaryCrankCenterOptimizer",
        )

        log.info(
            f"Initialized unified optimization framework with {self.settings.method.value} method",
        )

    def configure(
        self,
        settings: UnifiedOptimizationSettings | None = None,
        constraints: UnifiedOptimizationConstraints | None = None,
        targets: UnifiedOptimizationTargets | None = None,
        **kwargs,
    ) -> None:
        """
        Configure the unified optimization framework.

        Parameters
        ----------
        settings : UnifiedOptimizationSettings, optional
            Unified optimization settings
        constraints : UnifiedOptimizationConstraints, optional
            Unified optimization constraints
        targets : UnifiedOptimizationTargets, optional
            Unified optimization targets
        **kwargs
            Additional configuration parameters
        """
        if settings is not None:
            self.settings = settings
        if constraints is not None:
            self.constraints = constraints
        if targets is not None:
            self.targets = targets

        # Update settings from kwargs
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)

        # Reconfigure all optimizers with new settings
        self._configure_optimizers()

        self._is_configured = True
        log.info("Configured unified optimization framework")

    def _configure_optimizers(self) -> None:
        """Configure all optimizers with unified settings."""
        # Configure primary optimizer
        if self.settings.method in [
            OptimizationMethod.LEGENDRE_COLLOCATION,
            OptimizationMethod.RADAU_COLLOCATION,
            OptimizationMethod.HERMITE_COLLOCATION,
        ]:
            collocation_settings = CollocationSettings(
                degree=self.settings.collocation_degree,
                tolerance=self.settings.collocation_tolerance,
                method=self.settings.collocation_method,
            )
            self.primary_optimizer.configure(settings=collocation_settings)

        # Configure secondary optimizer
        secondary_constraints = CamRingOptimizationConstraints(
            base_radius_min=self.constraints.base_radius_min,
            base_radius_max=self.constraints.base_radius_max,
            target_average_ring_radius=self.constraints.ring_gear_radius,  # Use GUI target
            min_curvature_radius=self.constraints.min_curvature_radius,
            max_curvature=self.constraints.max_curvature,
            max_iterations=self.settings.max_iterations,
            tolerance=self.settings.tolerance,
        )

        secondary_targets = CamRingOptimizationTargets(
            minimize_ring_size=self.targets.minimize_ring_size,
            minimize_cam_size=self.targets.minimize_cam_size,
            minimize_curvature_variation=self.targets.minimize_curvature_variation,
            target_average_ring_radius=True,
            ring_size_weight=1.0,
            cam_size_weight=1.0,
            curvature_weight=self.targets.curvature_weight,
            target_radius_weight=0.5,
        )

        self.secondary_optimizer.configure(
            constraints=secondary_constraints,
            targets=secondary_targets,
        )

        # Configure tertiary optimizer (crank center optimization)
        tertiary_constraints = CrankCenterOptimizationConstraints(
            crank_center_x_min=self.constraints.crank_center_x_min,
            crank_center_x_max=self.constraints.crank_center_x_max,
            crank_center_y_min=self.constraints.crank_center_y_min,
            crank_center_y_max=self.constraints.crank_center_y_max,
            crank_radius_min=self.constraints.crank_radius_min,
            crank_radius_max=self.constraints.crank_radius_max,
            rod_length_min=self.constraints.rod_length_min,
            rod_length_max=self.constraints.rod_length_max,
            min_torque_output=self.constraints.min_torque_output,
            max_side_load_penalty=self.constraints.max_side_load_penalty,
            max_iterations=self.settings.max_iterations,
            tolerance=self.settings.tolerance,
        )

        tertiary_targets = CrankCenterOptimizationTargets(
            maximize_torque=self.targets.maximize_torque,
            minimize_side_loading=self.targets.minimize_side_loading,
            minimize_side_loading_during_compression=self.targets.minimize_side_loading_during_compression,
            minimize_side_loading_during_combustion=self.targets.minimize_side_loading_during_combustion,
            minimize_torque_ripple=self.targets.minimize_torque_ripple,
            maximize_power_output=self.targets.maximize_power_output,
            torque_weight=self.targets.torque_weight,
            side_load_weight=self.targets.side_load_weight,
            compression_side_load_weight=self.targets.compression_side_load_weight,
            combustion_side_load_weight=self.targets.combustion_side_load_weight,
            torque_ripple_weight=self.targets.torque_ripple_weight,
            power_output_weight=self.targets.power_output_weight,
        )

        self.tertiary_optimizer.configure(
            constraints=tertiary_constraints,
            targets=tertiary_targets,
        )

        # Configure CasADi validation mode if enabled
        if self.settings.enable_casadi_validation_mode:
            log.info("CasADi validation mode enabled for tertiary optimization")
            # The validation mode is controlled by constants, but we can log the setting
            log.info(
                f"Validation tolerance: {self.settings.casadi_validation_tolerance}",
            )

    def optimize_cascaded(self, input_data: dict[str, Any]) -> UnifiedOptimizationData:
        """
        Perform cascaded optimization across all three layers.

        Parameters
        ----------
        input_data : Dict[str, Any]
            Input data for optimization

        Returns
        -------
        UnifiedOptimizationData
            Complete optimization results from all layers
        """
        if not self._is_configured:
            raise RuntimeError("Framework must be configured before optimization")

        # Create main progress logger
        main_logger = ProgressLogger("CASCADED", flush_immediately=True)
        main_logger.start_phase(total_steps=3)
        main_logger.info(f"Optimization method: {self.settings.method.value}")
        main_logger.info(f"Universal grid points: {self.settings.universal_n_points}")

        start_time = time.time()

        try:
            # Step 1: Initialize and validate inputs
            main_logger.step(1, 3, "Initializing and validating inputs")
            init_start = time.time()
            self._update_data_from_input(input_data)
            main_logger.step_complete("Input initialization", time.time() - init_start)

            # A4: Phase-0 feasibility check for primary constraints
            feas_start = time.time()
            try:
                from campro.diagnostics.feasibility import check_feasibility_nlp

                primary_constraints = {
                    "stroke": self.data.stroke,
                    "cycle_time": self.data.cycle_time,
                    "upstroke_percent": self.data.upstroke_duration_percent,
                    "zero_accel_percent": self.data.zero_accel_duration_percent,
                }
                primary_bounds = {
                    "max_velocity": self.constraints.max_velocity,
                    "max_acceleration": self.constraints.max_acceleration,
                    "max_jerk": self.constraints.max_jerk,
                }
                feas = check_feasibility_nlp(primary_constraints, primary_bounds)
                self.data.convergence_info["feasibility_primary"] = {
                    "feasible": feas.feasible,
                    "max_violation": feas.max_violation,
                    "violations": feas.violations,
                    "recommendations": feas.recommendations,
                }
                if not feas.feasible:
                    main_logger.warning(
                        f"Primary feasibility check failed (max_violation={feas.max_violation:.3e}): "
                        f"{', '.join(feas.recommendations) if feas.recommendations else 'No recommendations'}"
                    )
                else:
                    main_logger.info("Primary feasibility check passed")
                main_logger.step_complete("Feasibility check", time.time() - feas_start)
            except Exception as _e:
                log.debug(f"Feasibility NLP pre-check skipped due to error: {_e}")
                main_logger.step_complete("Feasibility check (skipped)", time.time() - feas_start)

            # Phase 1: Primary optimization (motion law)
            main_logger.separator()
            main_logger.step(1, 3, "Phase 1/3: Primary Optimization (Motion Law)")
            primary_start = time.time()

            # CasADi unified flow now handles its own deterministic seed/polish
            initial_guess = None

            primary_result = self._optimize_primary(initial_guess=initial_guess)
            
            # Check success status using the method
            is_successful = primary_result.is_successful() if hasattr(primary_result, 'is_successful') else False
            status_str = str(primary_result.status) if hasattr(primary_result, 'status') else 'unknown'
            
            main_logger.info(
                f"Primary optimization completed: status={status_str}, "
                f"successful={is_successful}"
            )

            update_start = time.time()
            self._update_data_from_primary(primary_result)
            main_logger.step_complete("Data update from primary", time.time() - update_start)

            # Clear completion summary for Phase 1
            phase1_elapsed = time.time() - primary_start
            main_logger.step_complete("Phase 1: Primary Optimization", phase1_elapsed)
            main_logger.info(
                f"✓ Phase 1 COMPLETE: Primary optimization finished successfully "
                f"(status={status_str}, time={phase1_elapsed:.3f}s)"
            )

            # Phase 2: Secondary optimization (cam-ring)
            main_logger.separator()
            main_logger.info("Moving to Phase 2: Secondary Optimization (Cam-Ring)")
            main_logger.step(2, 3, "Phase 2/3: Secondary Optimization (Cam-Ring)")
            secondary_start = time.time()

            # A4: Feasibility check for secondary bound ordering
            try:
                sec_pairs = {
                    "base_radius": (
                        self.constraints.base_radius_min,
                        self.constraints.base_radius_max,
                    ),
                }
                sec_constraints = {"pairs": sec_pairs}
                _ = check_feasibility(sec_constraints, {})
                # We only record ordering issues if any
            except Exception:
                pass

            try:
                secondary_result = self._optimize_secondary()
                
                # Check if secondary optimization succeeded before updating data
                secondary_success = secondary_result.is_successful() if hasattr(secondary_result, 'is_successful') else False
                if not secondary_success:
                    main_logger.warning(
                        f"Secondary optimization failed (status={secondary_result.status if hasattr(secondary_result, 'status') else 'unknown'}). "
                        "Attempting partial data update, but tertiary optimization will be skipped."
                    )
                    # Still update data if it has partial results, but mark as failed
                    update_start = time.time()
                    try:
                        self._update_data_from_secondary(secondary_result)
                        main_logger.step_complete("Data update from secondary (partial)", time.time() - update_start)
                    except Exception as update_error:
                        main_logger.warning(f"Data update from secondary failed: {update_error}")
                else:
                    update_start = time.time()
                    self._update_data_from_secondary(secondary_result)
                    main_logger.step_complete("Data update from secondary", time.time() - update_start)
            except Exception as e:
                import traceback
                main_logger.error(f"Secondary optimization failed with exception: {e}")
                traceback.print_exc()
                raise

            # Clear completion summary for Phase 2
            phase2_elapsed = time.time() - secondary_start
            main_logger.step_complete("Phase 2: Secondary Optimization", phase2_elapsed)
            # Re-check success status for final summary (may have been updated during data update)
            secondary_success = secondary_result.is_successful() if hasattr(secondary_result, 'is_successful') else False
            secondary_status_str = str(secondary_result.status) if hasattr(secondary_result, 'status') else 'unknown'
            
            if secondary_success:
                main_logger.info(
                    f"✓ Phase 2 COMPLETE: Secondary optimization finished successfully "
                    f"(status={secondary_status_str}, time={phase2_elapsed:.3f}s)"
                )
            else:
                main_logger.warning(
                    f"✗ Phase 2 FAILED: Secondary optimization did not converge "
                    f"(status={secondary_status_str}, time={phase2_elapsed:.3f}s)"
                )

            # Phase 3: Tertiary optimization (sun gear) - only if secondary succeeded
            if not secondary_success:
                main_logger.warning(
                    "Skipping Phase 3 (Tertiary Optimization) because Phase 2 (Secondary) failed. "
                    "Tertiary optimization requires successful secondary optimization."
                )
                raise RuntimeError(
                    "Secondary optimization must be completed successfully before tertiary optimization"
                )

            main_logger.separator()
            main_logger.info("Moving to Phase 3: Tertiary Optimization (Sun Gear)")
            main_logger.step(3, 3, "Phase 3/3: Tertiary Optimization (Sun Gear)")
            tertiary_start = time.time()

            # A4: Feasibility check for tertiary bounds
            try:
                tert_pairs = {
                    "crank_center_x": (
                        self.constraints.crank_center_x_min,
                        self.constraints.crank_center_x_max,
                    ),
                    "crank_center_y": (
                        self.constraints.crank_center_y_min,
                        self.constraints.crank_center_y_max,
                    ),
                    "crank_radius": (
                        self.constraints.crank_radius_min,
                        self.constraints.crank_radius_max,
                    ),
                }
                tert_constraints = {"pairs": tert_pairs}
                tert_feas = check_feasibility(tert_constraints, {})
                if tert_feas.violations:
                    self.data.convergence_info.setdefault("feasibility_tertiary", {})
                    self.data.convergence_info["feasibility_tertiary"].update(
                        {
                            "violations": tert_feas.violations,
                            "recommendations": tert_feas.recommendations,
                        },
                    )
            except Exception:
                pass

            tertiary_result = self._optimize_tertiary()
            update_start = time.time()
            self._update_data_from_tertiary(tertiary_result)
            main_logger.step_complete("Data update from tertiary", time.time() - update_start)
            
            # Clear completion summary for Phase 3
            phase3_elapsed = time.time() - tertiary_start
            main_logger.step_complete("Phase 3: Tertiary Optimization", phase3_elapsed)
            tertiary_success = tertiary_result.is_successful() if hasattr(tertiary_result, 'is_successful') else True
            tertiary_status_str = str(tertiary_result.status) if hasattr(tertiary_result, 'status') else 'unknown'
            main_logger.info(
                f"✓ Phase 3 COMPLETE: Tertiary optimization finished "
                f"(status={tertiary_status_str}, time={phase3_elapsed:.3f}s)"
            )

            # Finalize results
            self.data.total_solve_time = time.time() - start_time
            self.data.optimization_method = self.settings.method

            main_logger.separator()
            main_logger.complete_phase(success=True)
            main_logger.info(f"Total optimization time: {self.data.total_solve_time:.3f}s")

        except Exception as e:
            main_logger.error(f"Cascaded optimization failed: {e}")
            main_logger.complete_phase(success=False)
            log.error(f"Cascaded optimization failed: {e}")
            raise

        return self.data

    def enable_casadi_validation_mode(self, tolerance: float = 1e-4) -> None:
        """
        Enable CasADi physics validation mode for tertiary optimization.

        This enables parallel validation where both Python and CasADi physics
        are evaluated and compared during crank center optimization.

        Parameters
        ----------
        tolerance : float, optional
            Tolerance for validation comparisons, by default 1e-4
        """
        self.settings.enable_casadi_validation_mode = True
        self.settings.casadi_validation_tolerance = tolerance

        # Update constants to enable validation mode

        # Note: We can't directly modify the constants, but we can log the setting
        log.info(f"CasADi validation mode enabled with tolerance {tolerance}")
        log.info(
            "Note: Set CASADI_PHYSICS_VALIDATION_MODE=True in constants.py to activate",
        )

    def disable_casadi_validation_mode(self) -> None:
        """Disable CasADi physics validation mode."""
        self.settings.enable_casadi_validation_mode = False
        log.info("CasADi validation mode disabled")

    def _update_data_from_input(self, input_data: dict[str, Any]) -> None:
        """Update data structure from input parameters."""
        self.data.stroke = input_data.get("stroke", 20.0)
        self.data.cycle_time = input_data.get("cycle_time", 1.0)
        self.data.duration_angle_deg = input_data.get(
            "duration_angle_deg",
            getattr(self.data, "duration_angle_deg", 360.0),
        )
        
        # Phase 6: Compute implied RPM from cycle_time and duration_angle_deg if engine_speed_rpm not provided
        if self.data.engine_speed_rpm is None or self.data.engine_speed_rpm <= 0:
            if self.data.cycle_time > 0 and self.data.duration_angle_deg > 0:
                # rpm = 60 * duration_angle_deg / (360 * cycle_time)
                implied_rpm = 60.0 * self.data.duration_angle_deg / (360.0 * self.data.cycle_time)
                self.data.engine_speed_rpm = implied_rpm
                log.info(
                    f"Computed implied engine speed: {implied_rpm:.1f} RPM "
                    f"(from cycle_time={self.data.cycle_time:.3f} s, "
                    f"duration_angle_deg={self.data.duration_angle_deg:.1f} deg)"
                )
        else:
            # Phase 6: If engine_speed_rpm is provided, derive cycle_time from it
            if self.data.duration_angle_deg > 0:
                derived_cycle_time = (self.data.duration_angle_deg / 360.0) * (60.0 / self.data.engine_speed_rpm)
                self.data.cycle_time = derived_cycle_time
                log.info(
                    f"Derived cycle_time from engine_speed_rpm: {derived_cycle_time:.6f} s "
                    f"(from rpm={self.data.engine_speed_rpm:.1f}, "
                    f"duration_angle_deg={self.data.duration_angle_deg:.1f} deg)"
                )
        self.data.upstroke_duration_percent = input_data.get(
            "upstroke_duration_percent", 60.0,
        )
        self.data.zero_accel_duration_percent = input_data.get(
            "zero_accel_duration_percent", 0.0,
        )
        self.data.motion_type = input_data.get("motion_type", "minimum_jerk")
        self.data.afr = max(1e-3, float(input_data.get("afr", self.data.afr)))
        self.data.fuel_mass = max(
            1e-9,
            float(input_data.get("fuel_mass", self.data.fuel_mass)),
        )
        self.data.ca50_target_deg = float(
            input_data.get("ca50_target_deg", self.data.ca50_target_deg),
        )
        self.data.ca50_weight = max(
            0.0,
            float(input_data.get("ca50_weight", self.data.ca50_weight)),
        )
        self.data.ca_duration_target_deg = float(
            input_data.get("ca_duration_target_deg", self.data.ca_duration_target_deg),
        )
        self.data.ca_duration_weight = max(
            0.0,
            float(input_data.get("ca_duration_weight", self.data.ca_duration_weight)),
        )

        ignition_deg = input_data.get("ignition_deg")
        if ignition_deg is not None:
            ignition_deg = float(ignition_deg)
            self.data.ignition_deg = ignition_deg
            self.data.ignition_timing = self._convert_ignition_deg_to_time(
                ignition_deg,
                self.data.cycle_time,
            )
        # Injector delay
        injector_delay_deg = input_data.get("injector_delay_deg")
        if injector_delay_deg is not None:
            self.data.injector_delay_deg = float(injector_delay_deg)
            # Also convert to seconds for consistency
            omega_avg = 360.0 / max(self.data.cycle_time, 1e-9)
            self.data.injector_delay_s = float(injector_delay_deg) / omega_avg
        # Optional overrides for constant load and temperature from UI/input
        if "constant_load_value" in input_data:
            try:
                self.settings.constant_load_value = float(
                    input_data["constant_load_value"],
                )
            except Exception:
                pass
        if "constant_temperature_K" in input_data:
            try:
                self.settings.constant_temperature_K = float(
                    input_data["constant_temperature_K"],
                )
            except Exception:
                pass

    @staticmethod
    def _convert_ignition_deg_to_time(ignition_deg: float, cycle_time: float) -> float:
        """Convert ignition timing in crank degrees relative to TDC to absolute time."""
        if cycle_time <= 0.0:
            return 0.0
        if ignition_deg >= 0.0:
            theta_deg = (360.0 - ignition_deg) % 360.0
        else:
            theta_deg = (-ignition_deg) % 360.0
        return cycle_time * (theta_deg / 360.0)

    def _optimize_primary(self, initial_guess: dict[str, Any] | None = None) -> OptimizationResult:
        """Perform primary optimization (motion law).
        
        Args:
            initial_guess: Optional initial guess for warm-start optimization
        """
        # Create progress logger for primary optimization
        primary_logger = ProgressLogger("PRIMARY", flush_immediately=True)
        primary_logger.start_phase()

        # Create cam motion constraints with user input parameters
        from campro.constraints.cam import CamMotionConstraints

        # Get constraint values from unified constraints
        # Pass None if not provided (unbounded) - don't use defaults
        max_velocity = self.constraints.max_velocity
        max_acceleration = self.constraints.max_acceleration
        max_jerk = self.constraints.max_jerk

        primary_logger.step(1, None, "Creating motion constraints")
        constraint_start = time.time()
        # Create cam motion constraints with user input parameters
        cam_constraints = CamMotionConstraints(
            stroke=self.data.stroke,
            upstroke_duration_percent=self.data.upstroke_duration_percent,
            zero_accel_duration_percent=self.data.zero_accel_duration_percent,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            max_jerk=max_jerk,
        )
        primary_logger.step_complete("Motion constraints creation", time.time() - constraint_start)
        primary_logger.info(f"Motion type: {self.data.motion_type}")
        primary_logger.info(f"Constraints: stroke={self.data.stroke}mm, max_vel={max_velocity}, max_accel={max_acceleration}, max_jerk={max_jerk}")

        # A3: Compute and record primary scaling vector for diagnostics
        scaling_start = time.time()
        try:
            bounds_for_scaling = {
                "position": (0.0, float(self.data.stroke)),
                "velocity": (-(max_velocity or 0.0), (max_velocity or 0.0)),
                "acceleration": (-(max_acceleration or 0.0), (max_acceleration or 0.0)),
                "jerk": (-(max_jerk or 0.0), (max_jerk or 0.0)),
            }
            scales = compute_scaling_vector(bounds_for_scaling)
            self.data.convergence_info["scaling_primary"] = scales
            primary_logger.step_complete("Scaling vector computation", time.time() - scaling_start)
        except Exception:
            primary_logger.step_complete("Scaling vector computation (skipped)", time.time() - scaling_start)

        # Get motion type from data
        motion_type = self.data.motion_type

        # Optional CasADi Phase 1 flow
        if getattr(self.settings, "use_casadi", False):
            casadi_result = self._run_casadi_primary(
                cam_constraints=cam_constraints,
                max_velocity=max_velocity,
                max_acceleration=max_acceleration,
                max_jerk=max_jerk,
                primary_logger=primary_logger,
            )
            if casadi_result is not None:
                return casadi_result

            primary_logger.warning(
                "CasADi primary flow failed or returned no result; falling back to FreePiston adapter",
            )

        # Always-on Stage A: pressure-invariance robust objective, with optional Stage B TE refine
        try:
            from campro.physics.simple_cycle_adapter import (
                SimpleCycleAdapter,
                CycleGeometry,
                CycleThermo,
                WiebeParams,
            )

            # Initialize adapter and default configs
            adapter = SimpleCycleAdapter(
                wiebe=WiebeParams(a=5.0, m=2.0, start_deg=-5.0, duration_deg=25.0),
                alpha_fuel_to_base=float(self.settings.bounce_alpha),
                beta_base=float(self.settings.bounce_beta),
            )
            geom = CycleGeometry(
                area_mm2=float(self.settings.piston_area_mm2),
                Vc_mm3=float(self.settings.clearance_volume_mm3),
            )
            thermo = CycleThermo(
                gamma_bounce=1.25,
                p_atm_kpa=101.325,
            )

            # Seed s_ref from a nominal minimum-jerk solve
            primary_logger.step(2, None, f"Solving base motion law (type={motion_type}, n_points={int(self.settings.universal_n_points)})")
            seed_start = time.time()
            
            # Log optimizer details and problem size estimates
            optimizer_name = self.primary_optimizer.__class__.__name__
            primary_logger.info(f"Using {optimizer_name} optimizer")
            primary_logger.info(
                f"Problem size estimates: n_points={int(self.settings.universal_n_points)}, "
                f"cycle_time={self.data.cycle_time:.3f}s, stroke={self.data.stroke:.2f}mm"
            )
            primary_logger.info("Starting base motion law solve...")
            
            base_result_seed = self.primary_optimizer.solve_cam_motion_law(
                cam_constraints=cam_constraints,
                motion_type=motion_type,
                cycle_time=self.data.cycle_time,
                n_points=int(self.settings.universal_n_points),
                afr=getattr(self.data, "afr", None),
                ignition_timing=getattr(self.data, "ignition_timing", None),
                fuel_mass=getattr(self.data, "fuel_mass", None),
                ca50_target_deg=getattr(self.data, "ca50_target_deg", None),
                ca50_weight=getattr(self.data, "ca50_weight", None),
                duration_target_deg=getattr(self.data, "ca_duration_target_deg", None),
                duration_weight=getattr(self.data, "ca_duration_weight", None),
            )
            
            seed_elapsed = time.time() - seed_start
            primary_logger.step_complete("Base motion law solve", seed_elapsed)
            primary_logger.info(
                f"Base solve completed in {seed_elapsed:.3f}s: "
                f"status={base_result_seed.status if hasattr(base_result_seed, 'status') else 'unknown'}"
            )
            base_sol = base_result_seed.solution or {}
            theta_seed = base_sol.get("cam_angle")
            x_seed = base_sol.get("position")
            # Map seed to universal grid for consistent downstream comparisons
            if theta_seed is None or x_seed is None:
                # Use 1-360° range to match UniversalGrid (avoids wraparound issues)
                theta_seed = np.linspace(np.pi / 180.0, 2 * np.pi, int(self.settings.universal_n_points), endpoint=True)
                x_seed = np.zeros_like(theta_seed)
            ug_theta = self.data.universal_theta_rad if self.data.universal_theta_rad is not None else theta_seed
            # Mapper selection (states: interpolation by default)
            if self.settings.mapper_method == "pchip":
                x_seed_u = GridMapper.periodic_pchip_resample(theta_seed, x_seed, ug_theta)
            elif self.settings.mapper_method == "barycentric":
                x_seed_u = GridMapper.barycentric_resample(theta_seed, x_seed, ug_theta)
            elif self.settings.mapper_method == "projection":
                # States: still prefer interpolation; projection reserved for conserved fields
                x_seed_u = GridMapper.periodic_linear_resample(theta_seed, x_seed, ug_theta)
            else:
                x_seed_u = GridMapper.periodic_linear_resample(theta_seed, x_seed, ug_theta)
            theta_seed = ug_theta
            x_seed = x_seed_u
            # Derive velocity on U: prefer spectral derivative on uniform periodic U
            try:
                v_seed = GridMapper.fft_derivative(theta_seed, x_seed)
            except Exception:
                v_seed = np.gradient(x_seed, theta_seed)
            # Nominal fuel=1.0, mid load
            mid_idx__ = max(0, (len(self.settings.load_sweep) - 1) // 2)
            c_mid__ = float(self.settings.load_sweep[mid_idx__])

            base_afr__ = getattr(self.data, "afr", None)
            base_fuel_mass__ = getattr(self.data, "fuel_mass", None)
            ignition_time_s__ = getattr(self.data, "ignition_timing", None)
            ignition_theta_deg__ = getattr(self.data, "ignition_deg", None)
            base_temp__ = getattr(self.settings, "constant_temperature_K", None)
            if base_temp__ is None:
                base_temp__ = getattr(self.data, "primary_constant_temperature_K", None)
            if base_temp__ is None:
                base_temp__ = 900.0

            def build_combustion_inputs(
                fuel_scale: float | None,
                *,
                fuel_mass_override: float | None = None,
                ignition_deg_override: float | None = None,
                afr_override: float | None = None,
                air_mass_override: float | None = None,
                injector_delay_deg_override: float | None = None,
                injector_delay_s_override: float | None = None,
            ) -> dict[str, float] | None:
                if base_afr__ is None and fuel_mass_override is None:
                    return None
                fuel_mass_val: float | None = None
                reference_mass = fuel_mass_override if fuel_mass_override is not None else base_fuel_mass__
                if reference_mass is None:
                    fallback_mass = getattr(self.data, "fuel_mass", None)
                    if fallback_mass is not None:
                        reference_mass = fallback_mass
                if reference_mass is not None:
                    fuel_mass_val = float(reference_mass)
                    if fuel_scale is not None:
                        fuel_mass_val = max(fuel_mass_val * float(fuel_scale), 1e-9)
                # Use override AFR if provided, otherwise compute from air_mass_override if available
                afr_val = afr_override
                if afr_val is None:
                    if air_mass_override is not None and fuel_mass_val is not None and fuel_mass_val > 0.0:
                        # Compute AFR from air and fuel masses
                        afr_val = float(air_mass_override) / float(fuel_mass_val)
                    else:
                        afr_val = float(base_afr__ if base_afr__ is not None else self.data.afr)
                
                payload: dict[str, float] = {
                    "afr": float(afr_val),
                    "cycle_time_s": float(self.data.cycle_time),
                    "initial_temperature_K": float(base_temp__),
                    "initial_pressure_Pa": float(thermo.p_atm_kpa) * 1e3,
                }
                if fuel_mass_val is not None:
                    payload["fuel_mass"] = fuel_mass_val
                if ignition_time_s__ is not None:
                    payload["ignition_time_s"] = float(ignition_time_s__)
                ignition_deg_val = ignition_deg_override if ignition_deg_override is not None else ignition_theta_deg__
                if ignition_deg_val is not None:
                    payload["ignition_theta_deg"] = float(ignition_deg_val)
                # Add injector delay if provided
                if injector_delay_s_override is not None:
                    payload["injector_delay_s"] = float(injector_delay_s_override)
                elif injector_delay_deg_override is not None:
                    payload["injector_delay_deg"] = float(injector_delay_deg_override)
                else:
                    if hasattr(self.data, "injector_delay_s") and self.data.injector_delay_s is not None:
                        payload["injector_delay_s"] = float(self.data.injector_delay_s)
                    if hasattr(self.data, "injector_delay_deg") and self.data.injector_delay_deg is not None:
                        payload["injector_delay_deg"] = float(self.data.injector_delay_deg)
                return payload

            area_m2__ = float(geom.area_mm2) * 1e-6
            try:
                constant_load__ = float(getattr(self.settings, "constant_load_value", 0.0))
            except Exception:
                constant_load__ = 0.0
            stroke_m__ = max(float(self.data.stroke) / 1000.0, 1e-6)
            workload_target_j__ = getattr(self.settings, "workload_target", None)
            if workload_target_j__ is None:
                workload_target_j__ = constant_load__ * stroke_m__
            else:
                workload_target_j__ = float(workload_target_j__)

            p_load_kpa__ = 0.0
            if workload_target_j__ and area_m2__ > 0.0:
                p_load_pa = workload_target_j__ / max(area_m2__ * stroke_m__, 1e-12)
                p_load_kpa__ = p_load_pa / 1000.0
            p_cc_kpa__ = float(getattr(self.settings, "crankcase_pressure_kpa", 0.0))
            p_env_kpa__ = float(thermo.p_atm_kpa)

            current_fuel_mass__ = base_fuel_mass__
            current_ignition_deg__ = ignition_theta_deg__
            current_afr__ = base_afr__
            # Initialize air mass from AFR and fuel mass
            current_air_mass__ = None
            if current_fuel_mass__ is not None and current_afr__ is not None:
                current_air_mass__ = float(current_fuel_mass__) * float(current_afr__)
            # Get fuel type for stoichiometric AFR (default: diesel)
            fuel_type_default = getattr(self.settings, "fuel_type", "diesel")
            stoich_afr = 14.5 if fuel_type_default.lower() == "diesel" else 14.7
            tune_iters__ = int(max(0, getattr(self.settings, "combustion_tune_iterations", 3)))
            work_tol__ = float(getattr(self.settings, "workload_tolerance_j", 5.0))
            ca50_target__ = getattr(self.data, "ca50_target_deg", None)
            ca50_tol__ = float(getattr(self.settings, "ca50_tolerance_deg", 1.0))
            ca10_target__ = getattr(self.data, "ca10_target_deg", None)
            ca10_tol__ = float(getattr(self.settings, "ca10_tolerance_deg", 2.0))
            ca90_target__ = getattr(self.data, "ca90_target_deg", None)
            ca90_tol__ = float(getattr(self.settings, "ca90_tolerance_deg", 2.0))
            work_gain__ = float(getattr(self.settings, "combustion_tune_fuel_gain", 0.3))
            ca_gain__ = float(getattr(self.settings, "combustion_tune_ignition_gain", 0.2))
            afr_gain__ = float(getattr(self.settings, "combustion_tune_afr_gain", 0.15))
            air_gain__ = float(getattr(self.settings, "combustion_tune_air_gain", 0.3))
            injector_delay_gain__ = float(getattr(self.settings, "combustion_tune_injector_delay_gain", 0.1))
            omega_avg = 360.0 / max(self.data.cycle_time, 1e-9)
            current_injector_delay_deg__ = getattr(self.data, "injector_delay_deg", None)
            if current_injector_delay_deg__ is None and hasattr(self.data, "injector_delay_s") and self.data.injector_delay_s is not None:
                current_injector_delay_deg__ = float(self.data.injector_delay_s) * omega_avg

            for _tune in range(tune_iters__):
                injector_delay_s_override__ = None
                if current_injector_delay_deg__ is not None:
                    injector_delay_s_override__ = float(current_injector_delay_deg__) / omega_avg
                comb_inputs_tune__ = build_combustion_inputs(
                    1.0,
                    fuel_mass_override=current_fuel_mass__,
                    ignition_deg_override=current_ignition_deg__,
                    afr_override=current_afr__,
                    air_mass_override=current_air_mass__,
                    injector_delay_deg_override=current_injector_delay_deg__,
                    injector_delay_s_override=injector_delay_s_override__,
                )
                if comb_inputs_tune__ is None:
                    break
                out_tune__ = adapter.evaluate(
                    theta_seed,
                    x_seed,
                    v_seed,
                    1.0,
                    c_mid__,
                    geom,
                    thermo,
                    combustion=comb_inputs_tune__,
                    cycle_time_s=self.data.cycle_time,
                )
                cycle_work_tune__ = float(out_tune__.get("cycle_work_j", 0.0))
                work_error__ = cycle_work_tune__ - float(workload_target_j__ or 0.0)
                ca50_error__ = 0.0
                ca10_error__ = 0.0
                ca90_error__ = 0.0
                ca_markers_tune__ = out_tune__.get("ca_markers") or {}
                if ca50_target__ is not None and ca_markers_tune__.get("CA50") is not None:
                    ca50_error__ = float(ca_markers_tune__["CA50"]) - float(ca50_target__)
                if ca10_target__ is not None and ca_markers_tune__.get("CA10") is not None:
                    ca10_error__ = float(ca_markers_tune__["CA10"]) - float(ca10_target__)
                if ca90_target__ is not None and ca_markers_tune__.get("CA90") is not None:
                    ca90_error__ = float(ca_markers_tune__["CA90"]) - float(ca90_target__)
                # Check convergence: all targets must be satisfied
                converged = abs(work_error__) <= work_tol__
                if ca50_target__ is not None:
                    converged = converged and abs(ca50_error__) <= ca50_tol__
                if ca10_target__ is not None:
                    converged = converged and abs(ca10_error__) <= ca10_tol__
                if ca90_target__ is not None:
                    converged = converged and abs(ca90_error__) <= ca90_tol__
                if converged:
                    break
                
                # Co-tune fuel, air, AFR, and ignition based on work and CA50 errors
                denom_work__ = max(abs(float(workload_target_j__ or 0.0)), 1.0)
                
                # Strategy: If work is too low and CA50 is acceptable, increase both air and fuel proportionally
                # If work is too low and fuel is at minimum, enrich mixture (decrease AFR)
                # If CA50 is too early, retard ignition; if too late, advance ignition
                
                if abs(ca50_error__) <= ca50_tol__ or ca50_target__ is None:
                    # CA50 is acceptable, focus on work error
                    if work_error__ > 0.0:
                        # Work too high: reduce fuel and air proportionally (maintain AFR)
                        adjust_factor__ = 1.0 - work_gain__ * (work_error__ / denom_work__)
                        adjust_factor__ = float(np.clip(adjust_factor__, 0.5, 1.5))
                        if current_fuel_mass__ is not None:
                            current_fuel_mass__ = max(current_fuel_mass__ * adjust_factor__, 1e-9)
                        if current_air_mass__ is not None:
                            current_air_mass__ = max(current_air_mass__ * adjust_factor__, 1e-9)
                    elif work_error__ < 0.0:
                        # Work too low: increase both air and fuel proportionally (maintain AFR)
                        adjust_factor__ = 1.0 + air_gain__ * (abs(work_error__) / denom_work__)
                        adjust_factor__ = float(np.clip(adjust_factor__, 1.0, 1.5))
                        if current_fuel_mass__ is not None:
                            current_fuel_mass__ = max(current_fuel_mass__ * adjust_factor__, 1e-9)
                        if current_air_mass__ is not None:
                            current_air_mass__ = max(current_air_mass__ * adjust_factor__, 1e-9)
                        # If fuel is already at a safe minimum and work is still low, enrich mixture
                        if current_fuel_mass__ is not None and current_fuel_mass__ <= 1e-6 and current_afr__ is not None:
                            # Decrease AFR (enrich) to extract more work
                            afr_adjust = 1.0 - afr_gain__ * (abs(work_error__) / denom_work__)
                            afr_adjust = float(np.clip(afr_adjust, 0.8, 1.0))
                            current_afr__ = max(current_afr__ * afr_adjust, stoich_afr * 0.7)  # Don't go too rich
                            # Recompute air mass to maintain fuel mass
                            if current_air_mass__ is not None and current_fuel_mass__ is not None:
                                current_air_mass__ = float(current_fuel_mass__) * float(current_afr__)
                else:
                    # CA50/CA10/CA90 errors are significant, adjust ignition timing and injector delay
                    # Primary adjustment: ignition timing affects CA50
                    if ca50_error__ > 0.0:
                        # CA50 too late: advance ignition (more negative degrees)
                        current_ignition_deg__ = float(current_ignition_deg__ or ignition_theta_deg__ or -5.0) - ca_gain__ * ca50_error__
                    elif ca50_error__ < 0.0:
                        # CA50 too early: retard ignition (less negative/more positive degrees)
                        current_ignition_deg__ = float(current_ignition_deg__ or ignition_theta_deg__ or -5.0) - ca_gain__ * ca50_error__
                    
                    # Secondary adjustment: injector delay affects CA10/CA90 spread
                    # If CA10 is too early or CA90 is too late, adjust injector delay
                    if ca10_target__ is not None and abs(ca10_error__) > ca10_tol__:
                        if ca10_error__ > 0.0:
                            # CA10 too late: inject earlier (positive delay)
                            if current_injector_delay_deg__ is None:
                                current_injector_delay_deg__ = 0.0
                            current_injector_delay_deg__ += injector_delay_gain__ * ca10_error__
                        elif ca10_error__ < 0.0:
                            # CA10 too early: inject later (negative delay)
                            if current_injector_delay_deg__ is None:
                                current_injector_delay_deg__ = 0.0
                            current_injector_delay_deg__ += injector_delay_gain__ * ca10_error__
                    
                    if ca90_target__ is not None and abs(ca90_error__) > ca90_tol__:
                        # CA90 too late: inject earlier; CA90 too early: inject later
                        if current_injector_delay_deg__ is None:
                            current_injector_delay_deg__ = 0.0
                        current_injector_delay_deg__ += injector_delay_gain__ * ca90_error__ * 0.5  # Half weight for CA90
                    
                    # Also adjust fuel/air if work error is significant
                    if abs(work_error__) > work_tol__:
                        if work_error__ > 0.0:
                            adjust_factor__ = 1.0 - work_gain__ * (work_error__ / denom_work__)
                            adjust_factor__ = float(np.clip(adjust_factor__, 0.5, 1.5))
                            if current_fuel_mass__ is not None:
                                current_fuel_mass__ = max(current_fuel_mass__ * adjust_factor__, 1e-9)
                            if current_air_mass__ is not None:
                                current_air_mass__ = max(current_air_mass__ * adjust_factor__, 1e-9)
                        elif work_error__ < 0.0:
                            adjust_factor__ = 1.0 + air_gain__ * (abs(work_error__) / denom_work__)
                            adjust_factor__ = float(np.clip(adjust_factor__, 1.0, 1.5))
                            if current_fuel_mass__ is not None:
                                current_fuel_mass__ = max(current_fuel_mass__ * adjust_factor__, 1e-9)
                            if current_air_mass__ is not None:
                                current_air_mass__ = max(current_air_mass__ * adjust_factor__, 1e-9)

            if current_fuel_mass__ is not None:
                base_fuel_mass__ = current_fuel_mass__
                try:
                    self.data.fuel_mass = float(current_fuel_mass__)
                except Exception:
                    pass
            if current_afr__ is not None:
                base_afr__ = current_afr__
                try:
                    self.data.afr = float(current_afr__)
                except Exception:
                    pass
            if current_air_mass__ is not None:
                try:
                    if not hasattr(self.data, 'air_mass'):
                        self.data.air_mass = None
                    self.data.air_mass = float(current_air_mass__)
                except Exception:
                    pass
            if current_ignition_deg__ is not None:
                ignition_theta_deg__ = float(current_ignition_deg__)
                try:
                    self.data.ignition_deg = float(ignition_theta_deg__)
                    self.data.ignition_timing = float(self.data.cycle_time) * (float(ignition_theta_deg__) / 360.0)
                except Exception:
                    pass
            if current_injector_delay_deg__ is not None:
                try:
                    self.data.injector_delay_deg = float(current_injector_delay_deg__)
                    # Also update injector_delay_s for consistency
                    omega_avg = 360.0 / max(self.data.cycle_time, 1e-9)
                    self.data.injector_delay_s = float(current_injector_delay_deg__) / omega_avg
                except Exception:
                    pass

            combustion_mid__ = build_combustion_inputs(
                1.0,
                fuel_mass_override=current_fuel_mass__,
                ignition_deg_override=ignition_theta_deg__,
                afr_override=current_afr__,
                air_mass_override=current_air_mass__,
                injector_delay_deg_override=current_injector_delay_deg__,
                injector_delay_s_override=(
                    float(current_injector_delay_deg__) / omega_avg if current_injector_delay_deg__ is not None else None
                ),
            )
            out0__ = adapter.evaluate(
                theta_seed,
                x_seed,
                v_seed,
                1.0,
                c_mid__,
                geom,
                thermo,
                combustion=combustion_mid__,
                cycle_time_s=self.data.cycle_time,
            )

            # Compute workload-aligned p_load for reference case
            cycle_work_seed__ = float(out0__.get("cycle_work_j", 0.0))
            p_load_kpa_ref__ = p_load_kpa__
            if cycle_work_seed__ > 0.0 and area_m2__ > 0.0:
                # Use actual cycle work to compute reference load pressure
                p_load_pa_ref = cycle_work_seed__ / max(area_m2__ * stroke_m__, 1e-12)
                p_load_kpa_ref__ = p_load_pa_ref / 1000.0
            elif workload_target_j__ and area_m2__ > 0.0:
                # Fallback to workload target if cycle work not available
                p_load_pa_ref = workload_target_j__ / max(area_m2__ * stroke_m__, 1e-12)
                p_load_kpa_ref__ = p_load_pa_ref / 1000.0
            
            # Compute PR reference: use geometry-informed template or seed-derived
            use_template = getattr(self.settings, "pr_template_use_template", True)
            
            if use_template:
                # Compute geometry-informed PR template
                from campro.physics.pr_template import compute_pr_template
                
                # Calculate compression ratio from geometry
                stroke_volume_mm3 = float(geom.area_mm2) * float(self.data.stroke)
                clearance_volume_mm3 = float(geom.Vc_mm3)
                v_max_mm3 = clearance_volume_mm3 + stroke_volume_mm3
                v_min_mm3 = clearance_volume_mm3
                compression_ratio = v_max_mm3 / max(v_min_mm3, 1e-9)
                
                # Compute bore from area
                bore_mm = np.sqrt(float(geom.area_mm2) * 4.0 / np.pi)
                
                # Get template parameters from settings
                expansion_efficiency_target = float(
                    getattr(self.settings, "pr_template_expansion_efficiency", 0.85)
                )
                pr_peak_scale = float(getattr(self.settings, "pr_template_peak_scale", 1.5))
                
                # Generate PR template
                pi_ref__ = compute_pr_template(
                    theta=theta_seed,
                    stroke_mm=float(self.data.stroke),
                    bore_mm=bore_mm,
                    clearance_volume_mm3=clearance_volume_mm3,
                    compression_ratio=compression_ratio,
                    p_load_kpa=p_load_kpa_ref__,
                    p_cc_kpa=p_cc_kpa__,
                    p_env_kpa=p_env_kpa__,
                    expansion_efficiency_target=expansion_efficiency_target,
                    pr_peak_scale=pr_peak_scale,
                )
                
                # Store seed evaluation data for diagnostics/comparison
                p_cyl_seed_raw__ = out0__.get("p_cyl")
                if p_cyl_seed_raw__ is None:
                    p_cyl_seed_raw__ = out0__.get("p_comb")
                p_cyl_seed__ = np.asarray(p_cyl_seed_raw__, dtype=float)
                p_bounce_seed__ = np.asarray(out0__.get("p_bounce"), dtype=float)
            else:
                # Fallback to seed-derived PR (legacy behavior)
                p_cyl_seed_raw__ = out0__.get("p_cyl")
                if p_cyl_seed_raw__ is None:
                    p_cyl_seed_raw__ = out0__.get("p_comb")
                p_cyl_seed__ = np.asarray(p_cyl_seed_raw__, dtype=float)
                p_bounce_seed__ = np.asarray(out0__.get("p_bounce"), dtype=float)
                denom_ref__ = p_load_kpa_ref__ + p_cc_kpa__ + p_env_kpa__ + p_bounce_seed__
                pi_ref__ = p_cyl_seed__ / np.maximum(denom_ref__, 1e-6)
            last_ca_markers__: dict[str, float] | None = out0__.get("ca_markers") or None

            fuel_sweep__ = [float(x) for x in (self.settings.fuel_sweep or [1.0])]
            load_sweep__ = [float(x) for x in (self.settings.load_sweep or [0.0])]
            wj__ = float(self.settings.jerk_weight)
            wp__ = float(self.settings.dpdt_weight)
            ww__ = float(self.settings.imep_weight)
            wwk__ = float(getattr(self.settings, "workload_weight", 0.1))

            def _pressure_ratio(out: dict[str, Any], workload_override: float | None = None) -> np.ndarray:
                """Compute pressure ratio with workload-aligned denominator.
                
                If workload_override is provided, compute p_load_kpa from it.
                Otherwise, use the cycle_work_j from out to compute effective workload.
                """
                p_cyl_val = out.get("p_cyl")
                if p_cyl_val is None:
                    p_cyl_val = out.get("p_comb")
                p_cyl_arr = np.asarray(p_cyl_val, dtype=float)
                p_bounce_arr = np.asarray(out.get("p_bounce"), dtype=float)
                
                # Compute workload-aligned p_load_kpa for this case
                p_load_kpa_case = p_load_kpa__
                if workload_override is not None and workload_override > 0.0 and area_m2__ > 0.0:
                    # Use provided workload override
                    p_load_pa_case = workload_override / max(area_m2__ * stroke_m__, 1e-12)
                    p_load_kpa_case = p_load_pa_case / 1000.0
                else:
                    # Compute from cycle work if available
                    cycle_work_case = float(out.get("cycle_work_j", 0.0))
                    if cycle_work_case > 0.0 and area_m2__ > 0.0:
                        p_load_pa_case = cycle_work_case / max(area_m2__ * stroke_m__, 1e-12)
                        p_load_kpa_case = p_load_pa_case / 1000.0
                    elif workload_target_j__ and area_m2__ > 0.0:
                        # Fallback to scaling workload_target by fuel multiplier if available
                        fuel_mult = float(out.get("fuel_multiplier", 1.0))
                        if fuel_mult > 0.0:
                            effective_work = workload_target_j__ * fuel_mult
                            p_load_pa_case = effective_work / max(area_m2__ * stroke_m__, 1e-12)
                            p_load_kpa_case = p_load_pa_case / 1000.0
                
                denom_arr = p_load_kpa_case + p_cc_kpa__ + p_env_kpa__ + p_bounce_arr
                return p_cyl_arr / np.maximum(denom_arr, 1e-6)

            def _loss_p__(theta: np.ndarray, x: np.ndarray) -> tuple[float, float, dict[str, Any], dict[str, Any]]:
                theta = np.asarray(theta, dtype=float)
                x = np.asarray(x, dtype=float)
                v_theta = np.gradient(x, theta)
                loss = 0.0
                imeps: list[float] = []
                ratio_traces: list[np.ndarray] = []
                ratio_cases: list[dict[str, float]] = []
                work_cases: list[dict[str, float]] = []
                base_work_j: float | None = None
                for fm in fuel_sweep__:
                    for c in load_sweep__:
                        combustion_payload = build_combustion_inputs(fm)
                        out = adapter.evaluate(
                            theta,
                            x,
                            v_theta,
                            fm,
                            c,
                            geom,
                            thermo,
                            combustion=combustion_payload,
                            cycle_time_s=self.data.cycle_time,
                        )
                        # Compute effective workload for this case (from cycle work or scaled target)
                        cycle_work_case = float(out.get("cycle_work_j", 0.0))
                        workload_override = None
                        if cycle_work_case > 0.0:
                            workload_override = cycle_work_case
                        elif workload_target_j__:
                            # Scale by fuel multiplier as proxy
                            workload_override = workload_target_j__ * float(fm)
                        pi = _pressure_ratio(out, workload_override=workload_override)
                        theta_u = theta_seed
                        if self.settings.mapper_method == "projection":
                            w_src = GridMapper.trapz_weights(theta)
                            pi_u = GridMapper.l2_project(theta, pi, theta_u, weights_from=w_src)
                        elif self.settings.mapper_method == "pchip":
                            pi_u = GridMapper.periodic_pchip_resample(theta, pi, theta_u)
                        elif self.settings.mapper_method == "barycentric":
                            pi_u = GridMapper.barycentric_resample(theta, pi, theta_u)
                        else:
                            pi_u = GridMapper.periodic_linear_resample(theta, pi, theta_u)
                        pi_al = SimpleCycleAdapter.phase_align(pi_u, pi_ref__)
                        w_u = GridMapper.trapz_weights(theta_u)
                        loss += float(np.sum((pi_al - pi_ref__) ** 2 * w_u))
                        imeps.append(float(out["imep"]))
                        ratio_traces.append(pi_al.astype(float))
                        ca_markers_case = out.get("ca_markers") or {}
                        # Capture gain table entry if adapter supports it
                        applied_delta_p_base = adapter.alpha * float(fm) + adapter.beta
                        if hasattr(adapter, '_update_gain_table') and combustion_payload is not None:
                            afr = combustion_payload.get("afr")
                            fuel_mass = combustion_payload.get("fuel_mass") or combustion_payload.get("fuel_mass_kg")
                            if afr is not None and fuel_mass is not None:
                                stoich_afr = 14.5  # Default for diesel
                                phi = stoich_afr / float(afr)
                                # Use current alpha/beta from adapter (may be scheduled)
                                alpha_current, beta_current = adapter._get_scheduled_base_pressure(phi, float(fuel_mass))
                                adapter._update_gain_table(phi, float(fuel_mass), alpha_current, beta_current)
                                applied_delta_p_base = alpha_current * float(fm) + beta_current
                        
                        ratio_case = {
                            "fuel_multiplier": float(fm),
                            "load": float(c),
                            "pi_mean": float(np.mean(pi_al)),
                            "pi_peak": float(np.max(pi_al)),
                            "pi_min": float(np.min(pi_al)),
                            "applied_delta_p_base_kpa": float(applied_delta_p_base),
                        }
                        # Add CA markers if available
                        if ca_markers_case:
                            for key, value in ca_markers_case.items():
                                if value is not None:
                                    ratio_case[f"ca_{key.lower()}"] = float(value)
                        ratio_cases.append(ratio_case)
                        
                        cycle_work_j = float(out.get("cycle_work_j", 0.0))
                        work_case = {
                            "fuel_multiplier": float(fm),
                            "load": float(c),
                            "cycle_work_j": cycle_work_j,
                            "work_error_j": cycle_work_j - float(workload_target_j__ or 0.0),
                        }
                        # Add CA markers to work cases if available
                        if ca_markers_case:
                            for key, value in ca_markers_case.items():
                                if value is not None:
                                    work_case[f"ca_{key.lower()}"] = float(value)
                        work_cases.append(work_case)
                        if base_work_j is None and abs(float(fm) - 1.0) < 1e-6 and float(c) == float(c_mid__):
                            base_work_j = cycle_work_j
                K = max(1, len(fuel_sweep__) * len(load_sweep__))
                if ratio_traces:
                    merged_pi = np.concatenate(ratio_traces)
                    ratio_stats = {
                        "pi_mean": float(np.mean(merged_pi)),
                        "pi_peak": float(np.max(merged_pi)),
                        "pi_min": float(np.min(merged_pi)),
                        "pi_std": float(np.std(merged_pi)),
                        "pi_ref_mean": float(np.mean(pi_ref__)),
                        "pi_ref_peak": float(np.max(pi_ref__)),
                        "pi_ref_min": float(np.min(pi_ref__)),
                        "pi_ref_std": float(np.std(pi_ref__)),
                        "cases": ratio_cases,
                    }
                else:
                    ratio_stats = {
                        "pi_mean": 0.0,
                        "pi_peak": 0.0,
                        "pi_min": 0.0,
                        "pi_std": 0.0,
                        "pi_ref_mean": float(np.mean(pi_ref__)),
                        "pi_ref_peak": float(np.max(pi_ref__)),
                        "pi_ref_min": float(np.min(pi_ref__)),
                        "pi_ref_std": float(np.std(pi_ref__)),
                        "cases": ratio_cases,
                    }
                imep_avg = float(np.mean(imeps)) if imeps else 0.0
                if work_cases:
                    mean_work = float(np.mean([wc["cycle_work_j"] for wc in work_cases]))
                else:
                    mean_work = 0.0
                work_stats = {
                    "target_work_j": float(workload_target_j__ or 0.0),
                    "cycle_work_mean_j": mean_work,
                    "cycle_work_error_j": mean_work - float(workload_target_j__ or 0.0),
                    "base_cycle_work_j": base_work_j,
                    "cases": work_cases,
                }
                return loss / K, imep_avg, ratio_stats, work_stats

            def objective__(t: np.ndarray, x: np.ndarray, v: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
                T = float(self.data.cycle_time)
                theta = (2.0 * np.pi) * (np.asarray(t, dtype=float) / max(T, 1e-9))
                theta[0] = 0.0
                theta[-1] = 2.0 * np.pi
                loss_p, imep_avg, _, work_stats = _loss_p__(theta, x)
                jerk_term = float(np.trapz(u ** 2, t))
                work_error = float(work_stats.get("cycle_work_error_j", 0.0))
                work_penalty = wwk__ * (work_error ** 2)
                return wj__ * jerk_term + wp__ * loss_p - ww__ * imep_avg + work_penalty

            # Detect free-piston adapter and optimize EMA loop
            from campro.optimization.freepiston_phase1_adapter import FreePistonPhase1Adapter
            is_free_piston_adapter = isinstance(self.primary_optimizer, FreePistonPhase1Adapter)
            
            # Outer EMA loop
            outer_iterations = int(max(1, self.settings.outer_iterations))
            # Collapse outer_iterations to 1 for free-piston adapter (custom objective is ignored)
            if is_free_piston_adapter:
                outer_iterations = 1
                primary_logger.info(
                    "Free-piston adapter detected: collapsing outer_iterations to 1 "
                    "(custom objective ignored, using base seed solve)"
                )
            primary_logger.step(3, None, f"Starting EMA loop ({outer_iterations} iterations)")
            ema_start = time.time()
            for _iter in range(outer_iterations):
                iter_start = time.time()
                primary_logger.info(f"EMA iteration {_iter + 1}/{outer_iterations}")
                # Pass initial_guess only on first iteration for warm-start
                kwargs = {}
                if _iter == 0 and initial_guess is not None:
                    kwargs["initial_guess"] = initial_guess
                    primary_logger.info("Using warm-start initial guess for this iteration")
                
                # For free-piston adapter, reuse base seed solve instead of calling solve_custom_objective
                # (custom objective is ignored, so the extra solve is redundant)
                if is_free_piston_adapter:
                    primary_logger.info(
                        "Free-piston adapter: reusing base_result_seed (skipping redundant solve_custom_objective)"
                    )
                    result_stage_a__ = base_result_seed
                else:
                    primary_logger.info("Calling solve_custom_objective...")
                    result_stage_a__ = self.primary_optimizer.solve_custom_objective(
                        objective_function=objective__,
                        constraints=cam_constraints,
                        distance=self.data.stroke,
                        time_horizon=self.data.cycle_time,
                        n_points=int(self.settings.universal_n_points),
                        **kwargs,
                    )
                iter_elapsed = time.time() - iter_start
                status_str = result_stage_a__.status if hasattr(result_stage_a__, 'status') else 'unknown'
                primary_logger.info(f"EMA iteration {_iter + 1} completed (status={status_str}, time={iter_elapsed:.3f}s)")
                sol__ = result_stage_a__.solution or {}
                t_arr__ = sol__.get("time")
                x_arr__ = sol__.get("position")
                th__ = sol__.get("cam_angle")
                if th__ is None and t_arr__ is not None:
                    th__ = (2.0 * np.pi) * (t_arr__ / max(float(self.data.cycle_time), 1e-9))
                if th__ is not None and x_arr__ is not None:
                    # Map current solution to universal grid
                    ug_theta = self.data.universal_theta_rad if self.data.universal_theta_rad is not None else th__
                    if self.settings.mapper_method == "pchip":
                        x_u__ = GridMapper.periodic_pchip_resample(th__, x_arr__, ug_theta)
                    elif self.settings.mapper_method == "barycentric":
                        x_u__ = GridMapper.barycentric_resample(th__, x_arr__, ug_theta)
                    elif self.settings.mapper_method == "projection":
                        # For states, keep interpolation
                        x_u__ = GridMapper.periodic_linear_resample(th__, x_arr__, ug_theta)
                    else:
                        x_u__ = GridMapper.periodic_linear_resample(th__, x_arr__, ug_theta)
                    try:
                        v_arr__ = GridMapper.fft_derivative(ug_theta, x_u__)
                    except Exception:
                        v_arr__ = np.gradient(x_u__, ug_theta)
                    out_nom__ = adapter.evaluate(
                        ug_theta,
                        x_u__,
                        v_arr__,
                        1.0,
                        c_mid__,
                        geom,
                        thermo,
                        combustion=build_combustion_inputs(1.0),
                        cycle_time_s=self.data.cycle_time,
                    )
                    # Use cycle work from evaluation for workload-aligned pressure ratio
                    cycle_work_nom__ = float(out_nom__.get("cycle_work_j", 0.0))
                    workload_override_nom__ = cycle_work_nom__ if cycle_work_nom__ > 0.0 else None
                    pi_best__ = _pressure_ratio(out_nom__, workload_override=workload_override_nom__)
                    alpha__ = float(self.settings.ema_alpha)
                    pi_ref__ = (1.0 - alpha__) * pi_ref__ + alpha__ * pi_best__
                    if out_nom__.get("ca_markers"):
                        last_ca_markers__ = out_nom__["ca_markers"]
            primary_logger.step_complete("EMA loop", time.time() - ema_start)

            # Stage A diagnostics (compute entirely on universal grid)
            diag_loss__ = 0.0
            diag_imep__ = 0.0
            diag_ratio_stats__: dict[str, Any] = {
                "pi_mean": 0.0,
                "pi_peak": 0.0,
                "pi_min": 0.0,
                "pi_std": 0.0,
                "pi_ref_mean": float(np.mean(pi_ref__)),
                "pi_ref_peak": float(np.max(pi_ref__)),
                "pi_ref_min": float(np.min(pi_ref__)),
                "pi_ref_std": float(np.std(pi_ref__)),
                "cases": [],
            }
            diag_work_stats__: dict[str, Any] = {
                "target_work_j": float(workload_target_j__ or 0.0),
                "cycle_work_mean_j": 0.0,
                "cycle_work_error_j": -float(workload_target_j__ or 0.0),
                "cases": [],
            }
            if th__ is not None and x_arr__ is not None:
                # Map state to universal grid
                ug_th = self.data.universal_theta_rad if self.data.universal_theta_rad is not None else th__
                if self.settings.mapper_method == "pchip":
                    x_u_diag__ = GridMapper.periodic_pchip_resample(th__, x_arr__, ug_th)
                elif self.settings.mapper_method == "barycentric":
                    x_u_diag__ = GridMapper.barycentric_resample(th__, x_arr__, ug_th)
                else:
                    x_u_diag__ = GridMapper.periodic_linear_resample(th__, x_arr__, ug_th)
                diag_loss__, diag_imep__, diag_ratio_stats__, diag_work_stats__ = _loss_p__(ug_th, x_u_diag__)
            pressure_meta__ = {
                "loss_p_mean": diag_loss__,
                "imep_avg": diag_imep__,
                "fuel_sweep": fuel_sweep__,
                "load_sweep": load_sweep__,
                "pressure_ratio": diag_ratio_stats__,
                "pressure_ratio_target_mean": float(np.mean(pi_ref__)),
                "pi_reference": pi_ref__.tolist(),
                "theta_deg": np.degrees(theta_seed).tolist(),
                "denominator_base": {
                    "p_load_kpa": p_load_kpa_ref__,  # Use workload-aligned reference
                    "p_env_kpa": p_env_kpa__,
                    "p_cc_kpa": p_cc_kpa__,
                },
                "workload": diag_work_stats__,
                "work_target_j": float(workload_target_j__ or 0.0),
            }
            result_stage_a__.metadata["pressure_invariance"] = pressure_meta__
            if last_ca_markers__:
                result_stage_a__.metadata["ca_markers"] = last_ca_markers__
            try:
                pi_summary__ = pressure_meta__["pressure_ratio"]
                primary_logger.info(
                    "Pressure invariance: pi_mean=%.4f, pi_peak=%.4f, imep_avg=%.2f kPa",
                    float(pi_summary__.get("pi_mean", 0.0)),
                    float(pi_summary__.get("pi_peak", 0.0)),
                    float(diag_imep__),
                )
            except Exception:
                pass

            target_pi_mean__ = pressure_meta__["pressure_ratio"].get("pi_mean")
            target_ca50__ = getattr(self.data, "ca50_target_deg", None)
            ca50_weight__ = float(
                getattr(self.data, "ca50_weight", 0.0)
                or getattr(self.settings, "ca50_weight", 0.0)
            )
            pressure_ratio_weight__ = float(getattr(self.settings, "pressure_ratio_weight", 1.0))
            workload_weight__ = float(getattr(self.settings, "workload_weight", 0.1))

            # Stage B: if TE enabled, refine with guardrail; else return Stage A
            if getattr(self.settings, "use_thermal_efficiency", False):
                try:
                    from campro.physics.thermal_efficiency_simple import SimplifiedThermalModel

                    te_model__ = SimplifiedThermalModel()
                    eps__ = float(self.settings.pressure_guard_epsilon)
                    lam__ = float(self.settings.pressure_guard_lambda)

                    def objective_te__(t: np.ndarray, x: np.ndarray, v: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
                        metrics__ = te_model__.evaluate_efficiency(x, v, a)
                        te_score__ = float(metrics__.get("total_efficiency", 0.0))
                        T = float(self.data.cycle_time)
                        theta = (2.0 * np.pi) * (np.asarray(t, dtype=float) / max(T, 1e-9))
                        theta[0] = 0.0
                        theta[-1] = 2.0 * np.pi
                        loss_p_val__, _, ratio_stats_te__, work_stats_te__ = _loss_p__(theta, x)
                        viol__ = max(0.0, loss_p_val__ - eps__)
                        penalty__ = lam__ * (viol__ ** 2)
                        jerk_term__ = float(np.trapz(u ** 2, t)) * 1e-6
                        v_theta__ = np.gradient(x, theta)
                        eval_out__ = adapter.evaluate(
                            theta,
                            x,
                            v_theta__,
                            1.0,
                            c_mid__,
                            geom,
                            thermo,
                            combustion=build_combustion_inputs(1.0),
                            cycle_time_s=self.data.cycle_time,
                        )
                        # Use cycle work from evaluation for workload-aligned pressure ratio
                        cycle_work_eval_te__ = float(eval_out__.get("cycle_work_j", 0.0))
                        workload_override_te__ = cycle_work_eval_te__ if cycle_work_eval_te__ > 0.0 else None
                        pi_eval__ = _pressure_ratio(eval_out__, workload_override=workload_override_te__)
                        ratio_penalty__ = pressure_ratio_weight__ * float(np.mean((pi_eval__ - pi_ref__) ** 2))
                        ratio_target__ = (
                            float(target_pi_mean__)
                            if target_pi_mean__ is not None
                            else float(np.mean(pi_ref__))
                        )
                        ratio_mean_penalty__ = workload_weight__ * (float(np.mean(pi_eval__)) - ratio_target__) ** 2
                        ca_penalty__ = 0.0
                        if target_ca50__ is not None and ca50_weight__ > 0.0:
                            ca_markers_eval__ = eval_out__.get("ca_markers") or {}
                            ca50_val__ = ca_markers_eval__.get("CA50")
                            if ca50_val__ is not None:
                                ca_penalty__ = ca50_weight__ * ((float(ca50_val__) - float(target_ca50__)) ** 2)
                        cycle_work_eval__ = float(eval_out__.get("cycle_work_j", 0.0))
                        work_error_eval__ = cycle_work_eval__ - float(workload_target_j__ or 0.0)
                        work_alignment_penalty__ = workload_weight__ * (work_error_eval__ ** 2)
                        work_error_trace__ = float(work_stats_te__.get("cycle_work_error_j", work_error_eval__))
                        work_trace_penalty__ = workload_weight__ * (work_error_trace__ ** 2)
                        return (
                            -te_score__
                            + penalty__
                            + jerk_term__
                            + ratio_penalty__
                            + ratio_mean_penalty__
                            + ca_penalty__
                            + work_alignment_penalty__
                            + work_trace_penalty__
                        )

                    # Stage B can use Stage A result as initial guess
                    kwargs_b = {}
                    if hasattr(result_stage_a__, "solution") and result_stage_a__.solution:
                        # Extract solution from Stage A to use as warm-start for Stage B
                        kwargs_b["initial_guess"] = result_stage_a__.solution
                    result_stage_b__ = self.primary_optimizer.solve_custom_objective(
                        objective_function=objective_te__,
                        constraints=cam_constraints,
                        distance=self.data.stroke,
                        time_horizon=self.data.cycle_time,
                        n_points=int(self.settings.universal_n_points),
                        **kwargs_b,
                    )

                    sol_b__ = result_stage_b__.solution or {}
                    t_b__ = sol_b__.get("time")
                    x_b__ = sol_b__.get("position")
                    guard_ok__ = None
                    loss_p_b__ = None
                    ratio_stats_b__ = None
                    stage_b_ratio_summary__ = None
                    stage_b_ca_markers__ = None
                    work_stats_b__ = None
                    stage_b_work_summary__ = None
                    if t_b__ is not None and x_b__ is not None:
                        # Map Stage B result to universal grid for consistent guardrail checks
                        T = float(self.data.cycle_time)
                        th_tmp__ = (2.0 * np.pi) * (np.asarray(t_b__, dtype=float) / max(T, 1e-9))
                        th_tmp__[0] = 0.0
                        th_tmp__[-1] = 2.0 * np.pi
                        ug_theta = self.data.universal_theta_rad if self.data.universal_theta_rad is not None else th_tmp__
                        if self.settings.mapper_method == "pchip":
                            x_b_u__ = GridMapper.periodic_pchip_resample(th_tmp__, x_b__, ug_theta)
                        elif self.settings.mapper_method == "barycentric":
                            x_b_u__ = GridMapper.barycentric_resample(th_tmp__, x_b__, ug_theta)
                        elif self.settings.mapper_method == "projection":
                            # For guardrail loss, prefer projection to conserve integrals
                            w_src = GridMapper.trapz_weights(th_tmp__)
                            x_b_u__ = GridMapper.l2_project(th_tmp__, x_b__, ug_theta, weights_from=w_src)
                        else:
                            x_b_u__ = GridMapper.periodic_linear_resample(th_tmp__, x_b__, ug_theta)
                        loss_p_b__, _, ratio_stats_b__, work_stats_b__ = _loss_p__(ug_theta, x_b_u__)
                        guard_ok__ = bool(loss_p_b__ <= eps__)
                        try:
                            v_b_u__ = GridMapper.fft_derivative(ug_theta, x_b_u__)
                        except Exception:
                            v_b_u__ = np.gradient(x_b_u__, ug_theta)
                        eval_stage_b__ = adapter.evaluate(
                            ug_theta,
                            x_b_u__,
                            v_b_u__,
                            1.0,
                            c_mid__,
                            geom,
                            thermo,
                            combustion=build_combustion_inputs(1.0),
                            cycle_time_s=self.data.cycle_time,
                        )
                        # Use cycle work from evaluation for workload-aligned pressure ratio
                        cycle_work_stage_b__ = float(eval_stage_b__.get("cycle_work_j", 0.0))
                        workload_override_stage_b__ = cycle_work_stage_b__ if cycle_work_stage_b__ > 0.0 else None
                        pi_stage_b__ = _pressure_ratio(eval_stage_b__, workload_override=workload_override_stage_b__)
                        cycle_work_eval__ = float(eval_stage_b__.get("cycle_work_j", 0.0))
                        work_error_eval__ = cycle_work_eval__ - float(workload_target_j__ or 0.0)
                        stage_b_work_summary__ = {
                            "cycle_work_j": cycle_work_eval__,
                            "work_error_j": work_error_eval__,
                        }
                        stage_b_ratio_summary__ = {
                            "pi_mean": float(np.mean(pi_stage_b__)),
                            "pi_peak": float(np.max(pi_stage_b__)),
                            "pi_min": float(np.min(pi_stage_b__)),
                            "pi_std": float(np.std(pi_stage_b__)),
                        }
                        ca_markers_stage_b__ = eval_stage_b__.get("ca_markers")
                        if ca_markers_stage_b__:
                            stage_b_ca_markers__ = ca_markers_stage_b__
                    else:
                        ratio_stats_b__ = {}
                        work_stats_b__ = {}
                        stage_b_work_summary__ = {}
                        stage_b_work_summary__ = {}

                    result_stage_b__.metadata["pressure_invariance_stage_a"] = result_stage_a__.metadata.get(
                        "pressure_invariance", {}
                    )
                    result_stage_b__.metadata["pressure_guardrail"] = {
                        "epsilon": eps__,
                        "lambda": lam__,
                        "loss_p": loss_p_b__,
                        "satisfied": guard_ok__,
                        "pressure_ratio": ratio_stats_b__,
                        "stage_b_base_ratio": stage_b_ratio_summary__,
                        "stage_b_work": stage_b_work_summary__,
                        "pressure_ratio_target_mean": (
                            float(target_pi_mean__) if target_pi_mean__ is not None else float(np.mean(pi_ref__))
                        ),
                        "work_target_j": float(workload_target_j__ or 0.0),
                        "workload": work_stats_b__,
                    }
                    if stage_b_ca_markers__:
                        result_stage_b__.metadata["ca_markers"] = stage_b_ca_markers__

                    primary_logger.step(4, None, "Stage B: Thermal efficiency refinement")
                    primary_logger.info("Stage B completed successfully")
                    primary_logger.complete_phase(success=True)
                    return result_stage_b__

                except Exception as exc:
                    primary_logger.warning(f"Stage B (TE refine) failed: {exc}")
                    log.error(f"Stage B (TE refine) failed: {exc}")
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    primary_logger.info("Returning Stage A result")
                    primary_logger.complete_phase(success=True)
                    return result_stage_a__
            else:
                primary_logger.step(4, None, "Stage B: Thermal efficiency refinement (disabled)")
                primary_logger.complete_phase(success=True)
                return result_stage_a__

        except Exception as exc:
            primary_logger.error(f"Always-on invariance flow failed; aborting: {exc}")
            log.error(f"Always-on invariance flow failed; aborting: {exc}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            primary_logger.complete_phase(success=False)
            raise

        # If enabled, route primary optimization through thermal-efficiency adapter
        if getattr(self.settings, "use_thermal_efficiency", False):
            try:
                # Build motion-law constraints expected by the adapter
                from campro.optimization.motion_law import (
                    MotionLawConstraints,
                    MotionType,
                )
                from campro.optimization.thermal_efficiency_adapter import (
                    ThermalEfficiencyAdapter,
                    ThermalEfficiencyConfig,
                )

                # Map unified settings to adapter config if provided
                adapter_cfg = ThermalEfficiencyConfig()
                cfg_overrides = (
                    getattr(self.settings, "thermal_efficiency_config", None) or {}
                )
                for k, v in cfg_overrides.items():
                    if hasattr(adapter_cfg, k):
                        setattr(adapter_cfg, k, v)

                # Pass analysis setting to adapter
                if hasattr(self.settings, "enable_ipopt_analysis"):
                    adapter_cfg.enable_analysis = self.settings.enable_ipopt_analysis

                adapter = ThermalEfficiencyAdapter(adapter_cfg)
                ml_constraints = MotionLawConstraints(
                    stroke=cam_constraints.stroke,
                    upstroke_duration_percent=cam_constraints.upstroke_duration_percent,
                    zero_accel_duration_percent=cam_constraints.zero_accel_duration_percent
                    or 0.0,
                    max_velocity=cam_constraints.max_velocity,
                    max_acceleration=cam_constraints.max_acceleration,
                    max_jerk=cam_constraints.max_jerk,
                )
                # Run thermal-efficiency optimization
                adapter_result = adapter.solve_motion_law(
                    ml_constraints,
                    MotionType.MINIMUM_JERK,
                )
                status = (
                    OptimizationStatus.CONVERGED
                    if adapter_result.convergence_status == "converged"
                    else OptimizationStatus.FAILED
                )
                if status == OptimizationStatus.CONVERGED:
                    # Convert to framework OptimizationResult
                    result = OptimizationResult(
                        status=status,
                        objective_value=adapter_result.objective_value,
                        solution=adapter_result.to_dict(),
                        iterations=adapter_result.iterations,
                        solve_time=adapter_result.solve_time,
                    )

                    # Attach primary-level assumptions to convergence info for downstream visibility
                    assumptions = {
                        "constant_temperature": True,
                        "ideal_fuel_load": True,
                        "angular_sampling_points": 360,
                        "independent_variable": "cam_angle_radians",
                        "constant_temperature_K": float(
                            getattr(self.settings, "constant_temperature_K", 900.0),
                        ),
                        "constant_load_value": float(
                            getattr(self.settings, "constant_load_value", 1.0),
                        ),
                    }
                    result.metadata.update({"assumptions": assumptions})

                    # Extract analysis from adapter if available
                    adapter_result_dict = adapter_result.to_dict()
                    if "ipopt_analysis" in adapter_result_dict:
                        result.metadata["ipopt_analysis"] = adapter_result_dict[
                            "ipopt_analysis"
                        ]
                        self.data.primary_ipopt_analysis = adapter_result_dict[
                            "ipopt_analysis"
                        ]

                        # Collect data for MA57 migration analysis
                        if (
                            self.settings.enable_ipopt_analysis
                            and self.data.primary_ipopt_analysis is not None
                        ):
                            problem_size = (
                                len(self.data.primary_theta)
                                if self.data.primary_theta is not None
                                else 100,
                                10,
                            )
                            self.migration_analyzer.add_ma27_run(
                                phase="primary",
                                problem_size=problem_size,
                                ma27_report=self.data.primary_ipopt_analysis,
                                metadata={
                                    "stroke": self.data.stroke,
                                    "cycle_time": self.data.cycle_time,
                                    "use_thermal_efficiency": self.settings.use_thermal_efficiency,
                                },
                            )
                    else:
                        self.data.primary_ipopt_analysis = None

                    return result
                # TE path failed; raise immediately
                raise RuntimeError(
                    "Thermal-efficiency optimization failed; install CasADi+IPOPT and retry",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Thermal-efficiency adapter path failed: {exc}. "
                    "Cannot fallback to simple optimization. Fix CasADi integration.",
                ) from exc

        # Pressure-invariance robust objective path (when enabled and TE is not enabled)
        if getattr(self.settings, "use_pressure_invariance", False):
            try:
                from campro.physics.simple_cycle_adapter import (
                    SimpleCycleAdapter,
                    CycleGeometry,
                    CycleThermo,
                    WiebeParams,
                )
                from campro.optimization.motion import MotionOptimizer

                # Initialize adapter and default configs
                adapter = SimpleCycleAdapter(
                    wiebe=WiebeParams(a=5.0, m=2.0, start_deg=-5.0, duration_deg=25.0),
                    alpha_fuel_to_base=float(self.settings.bounce_alpha),
                    beta_base=float(self.settings.bounce_beta),
                )
                geom = CycleGeometry(
                    area_mm2=float(self.settings.piston_area_mm2),
                    Vc_mm3=float(self.settings.clearance_volume_mm3),
                )
                thermo = CycleThermo(
                    gamma_bounce=1.25,
                    p_atm_kpa=101.325,
                )

                # Seed s_ref from a nominal minimum-jerk solve
                base_result = self.primary_optimizer.solve_cam_motion_law(
                    cam_constraints=cam_constraints,
                    motion_type=motion_type,
                    cycle_time=self.data.cycle_time,
                    afr=getattr(self.data, "afr", None),
                    ignition_timing=getattr(self.data, "ignition_timing", None),
                    fuel_mass=getattr(self.data, "fuel_mass", None),
                    ca50_target_deg=getattr(self.data, "ca50_target_deg", None),
                    ca50_weight=getattr(self.data, "ca50_weight", None),
                    duration_target_deg=getattr(self.data, "ca_duration_target_deg", None),
                    duration_weight=getattr(self.data, "ca_duration_weight", None),
                )
                base_sol = base_result.solution or {}
                # Fall back to data if solution missing
                theta_seed = base_sol.get(
                    "cam_angle",
                    np.linspace(np.pi / 180.0, 2 * np.pi, 360, endpoint=True),  # 1-360° range
                )
                x_seed = base_sol.get("position", np.zeros_like(theta_seed))
                v_seed = np.gradient(x_seed, theta_seed)
                # Nominal fuel=1.0, mid load
                mid_idx = max(0, (len(self.settings.load_sweep) - 1) // 2)
                c_mid = float(self.settings.load_sweep[mid_idx])

                base_afr = getattr(self.data, "afr", None)
                base_fuel_mass = getattr(self.data, "fuel_mass", None)
                ignition_time_s = getattr(self.data, "ignition_timing", None)
                ignition_theta_deg = getattr(self.data, "ignition_deg", None)
                base_temp = getattr(self.settings, "constant_temperature_K", None)
                if base_temp is None:
                    base_temp = getattr(self.data, "primary_constant_temperature_K", None)
                if base_temp is None:
                    base_temp = 900.0

                def build_combustion_inputs_local(fuel_scale: float | None) -> dict[str, float] | None:
                    if base_afr is None or base_fuel_mass is None:
                        return None
                    fuel_mass_val = float(base_fuel_mass)
                    if fuel_scale is not None:
                        fuel_mass_val = max(fuel_mass_val * float(fuel_scale), 1e-9)
                    payload: dict[str, float] = {
                        "afr": float(base_afr),
                        "fuel_mass": fuel_mass_val,
                        "cycle_time_s": float(self.data.cycle_time),
                        "initial_temperature_K": float(base_temp),
                        "initial_pressure_Pa": float(thermo.p_atm_kpa) * 1e3,
                    }
                    if ignition_time_s is not None:
                        payload["ignition_time_s"] = float(ignition_time_s)
                    if ignition_theta_deg is not None:
                        payload["ignition_theta_deg"] = float(ignition_theta_deg)
                    # Add injector delay if provided
                    if hasattr(self.data, "injector_delay_s") and self.data.injector_delay_s is not None:
                        payload["injector_delay_s"] = float(self.data.injector_delay_s)
                    if hasattr(self.data, "injector_delay_deg") and self.data.injector_delay_deg is not None:
                        payload["injector_delay_deg"] = float(self.data.injector_delay_deg)
                    return payload

                area_m2 = float(geom.area_mm2) * 1e-6
                try:
                    constant_load = float(getattr(self.settings, "constant_load_value", 0.0))
                except Exception:
                    constant_load = 0.0
                stroke_m = max(float(self.data.stroke) / 1000.0, 1e-6)
                workload_target_j = getattr(self.settings, "workload_target", None)
                if workload_target_j is None:
                    workload_target_j = constant_load * stroke_m
                else:
                    workload_target_j = float(workload_target_j)
                p_load_kpa = 0.0
                if workload_target_j and area_m2 > 0.0:
                    p_load_pa = workload_target_j / max(area_m2 * stroke_m, 1e-12)
                    p_load_kpa = p_load_pa / 1000.0
                p_cc_kpa = float(getattr(self.settings, "crankcase_pressure_kpa", 0.0))
                p_env_kpa = float(thermo.p_atm_kpa)

                combustion_mid = build_combustion_inputs_local(1.0)
                out0 = adapter.evaluate(
                    theta_seed,
                    x_seed,
                    v_seed,
                    1.0,
                    c_mid,
                    geom,
                    thermo,
                    combustion=combustion_mid,
                    cycle_time_s=self.data.cycle_time,
                )
                p_cyl_seed = out0.get("p_cyl")
                if p_cyl_seed is None:
                    p_cyl_seed = out0.get("p_comb")
                p_cyl_seed = np.asarray(p_cyl_seed, dtype=float)
                p_bounce_seed = np.asarray(out0.get("p_bounce"), dtype=float)
                denom_ref = p_load_kpa + p_env_kpa + p_bounce_seed
                pi_ref = p_cyl_seed / np.maximum(denom_ref, 1e-6)
                last_ca_markers_local: dict[str, float] | None = out0.get("ca_markers") or None

                # Build objective closure
                fuel_sweep = [float(x) for x in (self.settings.fuel_sweep or [1.0])]
                load_sweep = [float(x) for x in (self.settings.load_sweep or [0.0])]
                wj = float(self.settings.jerk_weight)
                wp = float(self.settings.dpdt_weight)
                ww = float(self.settings.imep_weight)
                wwk_local = float(getattr(self.settings, "workload_weight", 0.1))

                def _pressure_ratio_local(out: dict[str, Any]) -> np.ndarray:
                    p_cyl_val = out.get("p_cyl")
                    if p_cyl_val is None:
                        p_cyl_val = out.get("p_comb")
                    p_cyl_arr = np.asarray(p_cyl_val, dtype=float)
                    p_bounce_arr = np.asarray(out.get("p_bounce"), dtype=float)
                    denom_arr = p_load_kpa + p_cc_kpa + p_env_kpa + p_bounce_arr
                    return p_cyl_arr / np.maximum(denom_arr, 1e-6)

                def _loss_p_local(theta: np.ndarray, x: np.ndarray) -> tuple[float, float, dict[str, Any], dict[str, Any]]:
                    theta = np.asarray(theta, dtype=float)
                    x = np.asarray(x, dtype=float)
                    v_theta = np.gradient(x, theta)
                    loss_val = 0.0
                    imeps_local: list[float] = []
                    ratio_traces_local: list[np.ndarray] = []
                    ratio_case_local: list[dict[str, float]] = []
                    work_cases_local: list[dict[str, float]] = []
                    for fm in fuel_sweep:
                        for c in load_sweep:
                            out = adapter.evaluate(
                                theta,
                                x,
                                v_theta,
                                fm,
                                c,
                                geom,
                                thermo,
                                combustion=build_combustion_inputs_local(fm),
                                cycle_time_s=self.data.cycle_time,
                            )
                            pi_val = _pressure_ratio_local(out)
                            if self.settings.mapper_method == "projection":
                                w_src = GridMapper.trapz_weights(theta)
                                pi_u = GridMapper.l2_project(theta, pi_val, theta_seed, weights_from=w_src)
                            elif self.settings.mapper_method == "pchip":
                                pi_u = GridMapper.periodic_pchip_resample(theta, pi_val, theta_seed)
                            elif self.settings.mapper_method == "barycentric":
                                pi_u = GridMapper.barycentric_resample(theta, pi_val, theta_seed)
                            else:
                                pi_u = GridMapper.periodic_linear_resample(theta, pi_val, theta_seed)
                            pi_aligned = SimpleCycleAdapter.phase_align(pi_u, pi_ref)
                            w_u = GridMapper.trapz_weights(theta_seed)
                            loss_val += float(np.sum((pi_aligned - pi_ref) ** 2 * w_u))
                            imeps_local.append(float(out["imep"]))
                            ratio_traces_local.append(pi_aligned.astype(float))
                            ratio_case_local.append(
                                {
                                    "fuel_multiplier": float(fm),
                                    "load": float(c),
                                    "pi_mean": float(np.mean(pi_aligned)),
                                    "pi_peak": float(np.max(pi_aligned)),
                                    "pi_min": float(np.min(pi_aligned)),
                                }
                            )
                            cycle_work_local = float(out.get("cycle_work_j", 0.0))
                            work_cases_local.append(
                                {
                                    "fuel_multiplier": float(fm),
                                    "load": float(c),
                                    "cycle_work_j": cycle_work_local,
                                    "work_error_j": cycle_work_local - float(workload_target_j or 0.0),
                                }
                            )
                    K_local = max(1, len(fuel_sweep) * len(load_sweep))
                    if ratio_traces_local:
                        merged_local = np.concatenate(ratio_traces_local)
                        ratio_stats_local = {
                            "pi_mean": float(np.mean(merged_local)),
                            "pi_peak": float(np.max(merged_local)),
                            "pi_min": float(np.min(merged_local)),
                            "pi_std": float(np.std(merged_local)),
                            "pi_ref_mean": float(np.mean(pi_ref)),
                            "pi_ref_peak": float(np.max(pi_ref)),
                            "pi_ref_min": float(np.min(pi_ref)),
                            "pi_ref_std": float(np.std(pi_ref)),
                            "cases": ratio_case_local,
                        }
                    else:
                        ratio_stats_local = {
                            "pi_mean": 0.0,
                            "pi_peak": 0.0,
                            "pi_min": 0.0,
                            "pi_std": 0.0,
                            "pi_ref_mean": float(np.mean(pi_ref)),
                            "pi_ref_peak": float(np.max(pi_ref)),
                            "pi_ref_min": float(np.min(pi_ref)),
                            "pi_ref_std": float(np.std(pi_ref)),
                            "cases": ratio_case_local,
                        }
                    imep_avg_local = float(np.mean(imeps_local)) if imeps_local else 0.0
                    if work_cases_local:
                        work_mean_local = float(np.mean([wc["cycle_work_j"] for wc in work_cases_local]))
                    else:
                        work_mean_local = 0.0
                    work_stats_local = {
                        "target_work_j": float(workload_target_j or 0.0),
                        "cycle_work_mean_j": work_mean_local,
                        "cycle_work_error_j": work_mean_local - float(workload_target_j or 0.0),
                        "cases": work_cases_local,
                    }
                    return loss_val / K_local, imep_avg_local, ratio_stats_local, work_stats_local

                def objective(t: np.ndarray, x: np.ndarray, v: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
                    # Map time → phase angle θ ∈ [0, 2π]
                    T = float(self.data.cycle_time)
                    theta = (2.0 * np.pi) * (np.asarray(t, dtype=float) / max(T, 1e-9))
                    theta[0] = 0.0
                    theta[-1] = 2.0 * np.pi
                    loss_p_val, imep_avg_val, _, work_stats_local = _loss_p_local(theta, x)
                    jerk_term = float(np.trapz(u ** 2, t))
                    work_error_local = float(work_stats_local.get("cycle_work_error_j", 0.0))
                    work_penalty_local = wwk_local * (work_error_local ** 2)
                    return wj * jerk_term + wp * loss_p_val - ww * imep_avg_val + work_penalty_local

                # Outer EMA loop
                for _iter in range(int(max(1, self.settings.outer_iterations))):
                    kwargs = {}
                    if _iter == 0 and initial_guess is not None:
                        kwargs["initial_guess"] = initial_guess
                    result = self.primary_optimizer.solve_custom_objective(
                        objective_function=objective,
                        constraints=cam_constraints,
                        distance=self.data.stroke,
                        time_horizon=self.data.cycle_time,
                        n_points=int(self.settings.universal_n_points),
                        **kwargs,
                    )
                    sol = result.solution or {}
                    t_arr = sol.get("time")
                    x_arr = sol.get("position")
                    th = sol.get("cam_angle")
                    if th is None and t_arr is not None:
                        th = (2.0 * np.pi) * (t_arr / max(float(self.data.cycle_time), 1e-9))
                    if th is not None and x_arr is not None:
                        v_arr = np.gradient(x_arr, th)
                        out_nom = adapter.evaluate(
                            th,
                            x_arr,
                            v_arr,
                            1.0,
                            c_mid,
                            geom,
                            thermo,
                            combustion=build_combustion_inputs_local(1.0),
                            cycle_time_s=self.data.cycle_time,
                        )
                        pi_best = _pressure_ratio_local(out_nom)
                        alpha = float(self.settings.ema_alpha)
                        pi_ref = (1.0 - alpha) * pi_ref + alpha * pi_best
                        if out_nom.get("ca_markers"):
                            last_ca_markers_local = out_nom["ca_markers"]

                # Compute diagnostics on final solution
                diag_loss = 0.0
                diag_imep = 0.0
                diag_ratio_stats = {
                    "pi_mean": 0.0,
                    "pi_peak": 0.0,
                    "pi_min": 0.0,
                    "pi_std": 0.0,
                    "pi_ref_mean": float(np.mean(pi_ref)),
                    "pi_ref_peak": float(np.max(pi_ref)),
                    "pi_ref_min": float(np.min(pi_ref)),
                    "pi_ref_std": float(np.std(pi_ref)),
                    "cases": [],
                }
                diag_work_stats = {
                    "target_work_j": float(workload_target_j or 0.0),
                    "cycle_work_mean_j": 0.0,
                    "cycle_work_error_j": -float(workload_target_j or 0.0),
                    "cases": [],
                }
                if th is not None and x_arr is not None:
                    diag_loss, diag_imep, diag_ratio_stats, diag_work_stats = _loss_p_local(th, x_arr)
                result.metadata["pressure_invariance"] = {
                    "loss_p_mean": diag_loss,
                    "imep_avg": diag_imep,
                    "fuel_sweep": fuel_sweep,
                    "load_sweep": load_sweep,
                    "pressure_ratio": diag_ratio_stats,
                    "pressure_ratio_target_mean": float(np.mean(pi_ref)),
                    "pi_reference": pi_ref.tolist(),
                    "theta_deg": np.degrees(theta_seed).tolist(),
                    "denominator_base": {
                        "p_load_kpa": p_load_kpa,
                        "p_env_kpa": p_env_kpa,
                        "p_cc_kpa": p_cc_kpa,
                    },
                    "workload": diag_work_stats,
                    "work_target_j": float(workload_target_j or 0.0),
                }
                if last_ca_markers_local:
                    result.metadata["ca_markers"] = last_ca_markers_local

                # Return last result
                return result

            except Exception as exc:
                log.error(f"Pressure-invariance primary optimization failed: {exc}")
                return OptimizationResult(
                    status=OptimizationStatus.FAILED,
                    objective_value=float("inf"),
                    solution={},
                    iterations=0,
                    solve_time=0.0,
                    error_message=f"Primary (pressure-invariance) failed: {exc}",
                )

        # Regular optimization path (when thermal efficiency is not enabled)
        try:
            # Use the existing primary optimizer to solve the motion law problem
            result = self.primary_optimizer.solve_cam_motion_law(
                cam_constraints=cam_constraints,
                motion_type=motion_type,
                cycle_time=self.data.cycle_time,
                afr=getattr(self.data, "afr", None),
                ignition_timing=getattr(self.data, "ignition_timing", None),
                fuel_mass=getattr(self.data, "fuel_mass", None),
                ca50_target_deg=getattr(self.data, "ca50_target_deg", None),
                ca50_weight=getattr(self.data, "ca50_weight", None),
                duration_target_deg=getattr(self.data, "ca_duration_target_deg", None),
                duration_weight=getattr(self.data, "ca_duration_weight", None),
            )

            # The result is already an OptimizationResult from MotionOptimizer
            return result

        except Exception as exc:
            log.error(f"Primary optimization failed: {exc}")
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                objective_value=float("inf"),
                solution={},
                iterations=0,
                solve_time=0.0,
                error_message=f"Primary optimization failed: {exc}",
            )

    def _optimize_secondary(self) -> OptimizationResult:
        """Perform secondary optimization (cam-ring) with adaptive tuning."""
        # Create progress logger for secondary optimization
        secondary_logger = ProgressLogger("SECONDARY", flush_immediately=True)
        secondary_logger.start_phase()

        # Check if primary data is available
        secondary_logger.step(1, None, "Validating primary data availability")
        if self.data.primary_theta is None:
            secondary_logger.error("Primary optimization must be completed before secondary optimization")
            raise RuntimeError(
                "Primary optimization must be completed before secondary optimization",
            )
        secondary_logger.info(f"Primary data validated: {len(self.data.primary_theta)} points")

        # Ensure motion-law units are compatible with Phase-2 (millimetres)
        self._ensure_primary_motion_mm(secondary_logger)

        # Prepare primary data
        prep_start = time.time()
        secondary_logger.step(2, None, "Preparing primary data for secondary optimization")
        primary_data = {
            # Ensure secondary consumes the universal grid-aligned motion law
            # If primary outputs are not on the universal grid, map them here.
            "cam_angle": self.data.primary_theta,
            "position": self.data.primary_position,
            "velocity": self.data.primary_velocity,
            "acceleration": self.data.primary_acceleration,
            "theta_deg": self.data.primary_theta,
            "theta_rad": np.deg2rad(self.data.primary_theta)
            if self.data.primary_theta is not None
            else None,
            "time": np.linspace(0, self.data.cycle_time, len(self.data.primary_theta))
            if self.data.primary_theta is not None
            else np.array([]),
            "ca_markers": self.data.primary_ca_markers,
        }
        secondary_logger.step_complete("Primary data preparation", time.time() - prep_start)

        # Build golden radial profile for downstream collocation tracking
        golden_start = time.time()
        try:
            if (
                self.data.primary_position is not None
                and self.data.primary_velocity is not None
                and self.data.primary_acceleration is not None
                and self.data.primary_theta is not None
            ):
                N = len(self.data.primary_theta)
                t_grid = np.linspace(0.0, float(self.data.cycle_time), N)
                self.data.golden_profile = {
                    "time": t_grid,
                    "position": np.asarray(self.data.primary_position, dtype=float),
                    "velocity": np.asarray(self.data.primary_velocity, dtype=float),
                    "acceleration": np.asarray(self.data.primary_acceleration, dtype=float),
                }
                secondary_logger.step_complete("Golden profile construction", time.time() - golden_start)
        except Exception as _e:
            log.debug(f"Golden profile construction skipped: {_e}")
            secondary_logger.step_complete("Golden profile construction (skipped)", time.time() - golden_start)

        # Analyze problem characteristics
        problem_start = time.time()
        secondary_logger.step(3, None, "Analyzing problem characteristics")
        problem_chars = ProblemCharacteristics(
            n_variables=len(self.data.primary_theta)
            if self.data.primary_theta is not None
            else 100,
            n_constraints=10,  # Estimate
            problem_type="litvin",
            expected_iterations=100,
            linear_solver_ratio=0.3,  # Estimate
            has_convergence_issues=False,
        )
        secondary_logger.info(f"Problem size: {problem_chars.n_variables} variables, {problem_chars.n_constraints} constraints")
        secondary_logger.step_complete("Problem analysis", time.time() - problem_start)

        # Select optimal solver
        solver_start = time.time()
        secondary_logger.step(4, None, "Selecting optimal linear solver")
        solver_type = self.solver_selector.select_solver(problem_chars, "secondary")
        secondary_logger.info(f"Solver selected: {solver_type.value}")
        secondary_logger.step_complete("Solver selection", time.time() - solver_start)

        # Tune parameters
        tuning_start = time.time()
        secondary_logger.step(5, None, "Tuning optimization parameters")
        tuned_params = self.parameter_tuner.tune_parameters(
            "secondary",
            problem_chars,
            self.solver_selector.analysis_history.get("secondary"),
        )
        secondary_logger.info(f"Tuned parameters: max_iter={tuned_params.max_iter}, tol={tuned_params.tol:.2e}")
        secondary_logger.step_complete("Parameter tuning", time.time() - tuning_start)

        # Set initial guess based on stroke and GUI target (phase 2: cam + ring only)
        guess_start = time.time()
        secondary_logger.step(6, None, "Setting initial guess")
        initial_guess = {
            "base_radius": self.data.stroke,
        }
        secondary_logger.info(f"Initial guess: base_radius={initial_guess['base_radius']}mm")
        secondary_logger.step_complete("Initial guess setup", time.time() - guess_start)

        # A3: Compute and record simple scaling stats for secondary design variables
        scaling_start = time.time()
        try:
            bmin, bmax = (
                float(self.constraints.base_radius_min),
                float(self.constraints.base_radius_max),
            )
            sec_scales = compute_scaling_vector({"base_radius": (bmin, bmax)})
            self.data.convergence_info["scaling_secondary"] = sec_scales
            secondary_logger.step_complete("Scaling vector computation", time.time() - scaling_start)
        except Exception:
            secondary_logger.step_complete("Scaling vector computation (skipped)", time.time() - scaling_start)

        # Perform optimization
        opt_start = time.time()
        secondary_logger.step(7, None, "Running secondary optimization (cam-ring)")
        try:
            result = self.secondary_optimizer.optimize(
                primary_data=primary_data,
                initial_guess=initial_guess,
                # Provide golden tracking context for any collocation-based steps inside secondary
                golden_profile=self.data.golden_profile,
                tracking_weight=float(getattr(self.settings, "tracking_weight", 1.0)),
            )
            opt_elapsed = time.time() - opt_start
            status_str = result.status if hasattr(result, 'status') else 'unknown'
            success_str = result.is_successful() if hasattr(result, 'is_successful') else 'unknown'
            secondary_logger.info(f"Optimization completed: status={status_str}, success={success_str}, time={opt_elapsed:.3f}s")
            secondary_logger.step_complete("Secondary optimization", time.time() - opt_start)
        except Exception as e:
            import traceback
            secondary_logger.error(f"Secondary optimization failed: {e}")
            traceback.print_exc()
            raise
        log.info(
            f"Secondary optimization completed: status={result.status}, success={result.is_successful()}",
        )

        # Extract and store secondary analysis from cam ring optimizer
        secondary_logger.complete_phase(success=result.is_successful() if hasattr(result, 'is_successful') else True)
        # The cam ring optimizer uses litvin optimization which now provides analysis
        if hasattr(result, "metadata") and "ipopt_analysis" in result.metadata:
            self.data.secondary_ipopt_analysis = result.metadata["ipopt_analysis"]
            # Update analysis history for future decisions
            self.solver_selector.update_history(
                "secondary", result.metadata["ipopt_analysis"],
            )

            # Collect data for MA57 migration analysis
            if (
                self.settings.enable_ipopt_analysis
                and self.data.secondary_ipopt_analysis is not None
            ):
                problem_size = (
                    len(self.data.primary_theta)
                    if self.data.primary_theta is not None
                    else 100,
                    10,
                )
                self.migration_analyzer.add_ma27_run(
                    phase="secondary",
                    problem_size=problem_size,
                    ma27_report=self.data.secondary_ipopt_analysis,
                    metadata={
                        "base_radius": self.data.secondary_base_radius,
                        "stroke": self.data.stroke,
                        "solver_type": solver_type.value,
                    },
                )
        else:
            self.data.secondary_ipopt_analysis = None

        return result

    def _ensure_primary_motion_mm(self, phase_logger: "ProgressLogger | None" = None) -> None:
        """
        Ensure the primary motion-law data is expressed in millimetres for Phase-2.

        Converts from metres when necessary and records the conversion. Raises if units
        are missing or unsupported.
        """
        units = (self.data.primary_position_units or "").strip().lower()
        if not units:
            raise RuntimeError(
                "Primary motion units not recorded; cannot continue Phase-2 without explicit units.",
            )

        def _log(message: str) -> None:
            if phase_logger is not None:
                phase_logger.info(message)
            else:
                log.info(message)

        millimetre_aliases = {"mm", "millimetre", "millimeter"}
        metre_aliases = {"m", "meter", "metre", "meters", "metres"}

        def _to_array(value: np.ndarray | list | tuple | None) -> np.ndarray | None:
            if value is None:
                return None
            return np.asarray(value, dtype=float)

        if units in millimetre_aliases:
            self.data.primary_position = _to_array(self.data.primary_position)
            self.data.primary_velocity = _to_array(self.data.primary_velocity)
            self.data.primary_acceleration = _to_array(self.data.primary_acceleration)
            self.data.primary_jerk = _to_array(self.data.primary_jerk)
            return

        if units not in metre_aliases:
            raise ValueError(
                f"Unsupported primary motion units '{self.data.primary_position_units}'. "
                "Phase-2 expects millimetres.",
            )

        factor = 1_000.0
        self.data.primary_position = _to_array(self.data.primary_position)
        self.data.primary_velocity = _to_array(self.data.primary_velocity)
        self.data.primary_acceleration = _to_array(self.data.primary_acceleration)
        self.data.primary_jerk = _to_array(self.data.primary_jerk)

        if self.data.primary_position is not None:
            self.data.primary_position *= factor
        if self.data.primary_velocity is not None:
            self.data.primary_velocity *= factor
        if self.data.primary_acceleration is not None:
            self.data.primary_acceleration *= factor
        if self.data.primary_jerk is not None:
            self.data.primary_jerk *= factor

        self.data.primary_position_units = "mm"
        self.data.primary_velocity_units = "mm/deg"
        self.data.primary_acceleration_units = "mm/deg^2"
        self.data.primary_jerk_units = "mm/deg^3"

        _log("Converted primary motion law from metres to millimetres for Phase-2 ingestion.")

    def get_phase2_animation_inputs(self) -> dict[str, Any]:
        """Return minimal deterministic inputs for Phase-2 animation.

        Returns
        -------
        Dict[str, Any]
            A bundle containing primary and secondary outputs sufficient to
            build deterministic Phase-2 relationships without solving.
        """
        if (
            self.data.primary_theta is None
            or self.data.primary_position is None
            or self.data.secondary_base_radius is None
            or self.data.secondary_psi is None
            or self.data.secondary_R_psi is None
        ):
            raise RuntimeError(
                "Phase-2 animation inputs unavailable; ensure primary and secondary optimizations completed",
            )

        return {
            "theta_deg": self.data.primary_theta,
            "x_theta_mm": self.data.primary_position,
            "base_radius_mm": float(self.data.secondary_base_radius),
            "psi_rad": self.data.secondary_psi,
            "R_psi_mm": self.data.secondary_R_psi,
            "gear_geometry": self.data.secondary_gear_geometry or {},
        }

    def _optimize_tertiary(self) -> OptimizationResult:
        """Perform tertiary optimization (crank center optimization) with adaptive tuning."""
        # Check if primary and secondary data are available
        if self.data.primary_theta is None:
            raise RuntimeError(
                "Primary optimization must be completed before tertiary optimization",
            )
        if self.data.secondary_base_radius is None:
            raise RuntimeError(
                "Secondary optimization must be completed before tertiary optimization",
            )

        # Prepare primary data (motion law)
        if self.data.primary_theta is None:
            raise RuntimeError(
                "Primary theta data is None - primary optimization may have failed",
            )

        primary_data = {
            "theta": self.data.primary_theta,
            "crank_angle": self.data.primary_theta,
            "displacement": self.data.primary_position,
            "velocity": self.data.primary_velocity,
            "acceleration": self.data.primary_acceleration,
            "load_profile": self.data.primary_load_profile,
        }

        # Prepare secondary data (Litvin gear geometry)
        secondary_data = {
            "optimized_parameters": {
                "base_radius": self.data.secondary_base_radius,
            },
            "cam_curves": self.data.secondary_cam_curves,
            "psi": self.data.secondary_psi,
            "R_psi": self.data.secondary_R_psi,
        }

        # A3: Compute and record tertiary variable scaling based on bounds
        try:
            scales_ter = compute_scaling_vector(
                {
                    "crank_center_x": (
                        float(self.constraints.crank_center_x_min),
                        float(self.constraints.crank_center_x_max),
                    ),
                    "crank_center_y": (
                        float(self.constraints.crank_center_y_min),
                        float(self.constraints.crank_center_y_max),
                    ),
                    "crank_radius": (
                        float(self.constraints.crank_radius_min),
                        float(self.constraints.crank_radius_max),
                    ),
                    "rod_length": (
                        float(
                            getattr(
                                self.constraints,
                                "rod_length_min",
                                (self.data.secondary_base_radius or 20.0) * 4.0,
                            ),
                        ),
                        float(
                            getattr(
                                self.constraints,
                                "rod_length_max",
                                (self.data.secondary_base_radius or 20.0) * 8.0,
                            ),
                        ),
                    ),
                },
            )
            self.data.convergence_info["scaling_tertiary"] = scales_ter
        except Exception:
            pass

        # Analyze problem characteristics for tertiary optimization
        problem_chars = ProblemCharacteristics(
            n_variables=4,  # crank_center_x, crank_center_y, crank_radius, rod_length
            n_constraints=8,  # Bounds on each variable
            problem_type="crank_center",
            expected_iterations=200,
            linear_solver_ratio=0.2,  # Typically smaller problems
            has_convergence_issues=False,
        )

        # Select optimal solver (currently always MA27)
        solver_type = self.solver_selector.select_solver(problem_chars, "tertiary")
        log.info(f"Selected solver for tertiary optimization: {solver_type.value}")

        # Tune parameters
        tuned_params = self.parameter_tuner.tune_parameters(
            "tertiary",
            problem_chars,
            self.solver_selector.analysis_history.get("tertiary"),
        )
        log.info(
            f"Tuned parameters: max_iter={tuned_params.max_iter}, tol={tuned_params.tol}",
        )

        # Set initial guess based on secondary results
        initial_guess = {
            "crank_center_x": 0.0,  # Start at origin
            "crank_center_y": 0.0,  # Start at origin
            "crank_radius": self.data.secondary_base_radius
            * 2.0,  # Scale from cam base radius
            "rod_length": self.data.secondary_base_radius
            * 6.0,  # Scale from cam base radius
        }

        # Perform optimization
        result = self.tertiary_optimizer.optimize(
            primary_data=primary_data,
            secondary_data=secondary_data,
            initial_guess=initial_guess,
        )

        # Extract and store tertiary analysis from crank center optimizer
        if hasattr(result, "metadata") and "ipopt_analysis" in result.metadata:
            self.data.tertiary_ipopt_analysis = result.metadata["ipopt_analysis"]
            # Update analysis history for future decisions
            self.solver_selector.update_history(
                "tertiary", result.metadata["ipopt_analysis"],
            )

            # Collect data for MA57 migration analysis
            if (
                self.settings.enable_ipopt_analysis
                and self.data.tertiary_ipopt_analysis is not None
            ):
                problem_size = (
                    4,
                    8,
                )  # crank_center_x, crank_center_y, crank_radius, rod_length
                self.migration_analyzer.add_ma27_run(
                    phase="tertiary",
                    problem_size=problem_size,
                    ma27_report=self.data.tertiary_ipopt_analysis,
                    metadata={
                        "base_radius": self.data.secondary_base_radius,
                        "stroke": self.data.stroke,
                        "solver_type": solver_type.value,
                    },
                )
        else:
            self.data.tertiary_ipopt_analysis = None

        # Map any grid-dependent series returned by tertiary to universal grid and apply sensitivity operators
        try:
            sol_t = result.solution or {}
            th_t = sol_t.get("theta")
            if th_t is not None and self.data.universal_theta_rad is not None:
                ug_th = self.data.universal_theta_rad
                for key in ("position", "velocity", "acceleration", "control"):
                    arr = sol_t.get(key)
                    if arr is not None:
                        if self.settings.mapper_method == "pchip":
                            sol_t[key + "_universal"] = GridMapper.periodic_pchip_resample(th_t, arr, ug_th)
                        elif self.settings.mapper_method == "barycentric":
                            sol_t[key + "_universal"] = GridMapper.barycentric_resample(th_t, arr, ug_th)
                        else:
                            sol_t[key + "_universal"] = GridMapper.periodic_linear_resample(th_t, arr, ug_th)
                # Sensitivities mapping if present
                P_u2g, P_g2u = GridMapper.operators(th_t, ug_th, method=getattr(self.settings, "mapper_method", "linear"))
                grad_gi = sol_t.get("gradient")
                if grad_gi is not None:
                    sol_t["gradient_universal"] = GridMapper.pullback_gradient(np.asarray(grad_gi), P_g2u)
                jac_gi = sol_t.get("jacobian")
                if jac_gi is not None:
                    sol_t["jacobian_universal"] = GridMapper.pushforward_jacobian(np.asarray(jac_gi), P_u2g, P_g2u)
                # If tertiary metrics need an integral on U, demonstrate trapz on U for control if available
                w_u = GridMapper.trapz_weights(ug_th)
                if sol_t.get("control_universal") is not None:
                    sol_t["control_energy_universal"] = float(np.sum(sol_t["control_universal"] ** 2 * w_u))
                if sol_t.get("velocity_universal") is not None:
                    sol_t["velocity_energy_universal"] = float(np.sum(sol_t["velocity_universal"] ** 2 * w_u))
                if sol_t.get("acceleration_universal") is not None:
                    sol_t["acceleration_energy_universal"] = float(np.sum(sol_t["acceleration_universal"] ** 2 * w_u))
                if sol_t.get("position_universal") is not None:
                    # Mean position over cycle
                    sol_t["position_mean_universal"] = float(np.sum(sol_t["position_universal"] * w_u) / (2.0 * np.pi))
        except Exception:
            pass

        return result

    def _run_casadi_primary(
        self,
        *,
        cam_constraints: "CamMotionConstraints",
        max_velocity: float | None,
        max_acceleration: float | None,
        max_jerk: float | None,
        primary_logger: ProgressLogger,
    ) -> OptimizationResult | None:
        """Execute the CasADi Phase 1 flow when enabled."""
        try:
            primary_logger.step(2, None, "Running CasADi Phase 1 optimization")
            casadi_settings = self._create_casadi_settings()
            casadi_flow = CasADiUnifiedFlow(casadi_settings)

            casadi_constraints = self._build_casadi_primary_constraints(
                cam_constraints=cam_constraints,
                max_velocity=max_velocity,
                max_acceleration=max_acceleration,
                max_jerk=max_jerk,
            )
            casadi_targets = self._build_casadi_primary_targets()

            result = casadi_flow.optimize_phase1(casadi_constraints, casadi_targets)
            if result.successful:
                self._annotate_casadi_solution(
                    result,
                    cycle_time=casadi_constraints["cycle_time"],
                    duration_angle_deg=casadi_constraints.get(
                        "duration_angle_deg",
                        getattr(self.data, "duration_angle_deg", 360.0),
                    ),
                )
                result.metadata.setdefault("solver", "CasADiUnifiedFlow")
                result.metadata["source"] = "casadi_phase1"
                segments = result.metadata.get("finest_success_segments")
                if segments:
                    primary_logger.info(
                        f"CasADi optimization succeeded (segments={segments})",
                    )
                else:
                    primary_logger.info("CasADi optimization succeeded")
                primary_logger.complete_phase(success=True)
                return result

            primary_logger.warning(
                f"CasADi optimization failed: {result.error_message or result.status}",
            )
            return None

        except Exception as exc:  # pragma: no cover - safety fallback
            primary_logger.error(f"CasADi primary flow raised an exception: {exc}")
            log.error("CasADi primary flow failed", exc_info=True)
            return None

    def _build_casadi_primary_constraints(
        self,
        *,
        cam_constraints: "CamMotionConstraints",
        max_velocity: float | None,
        max_acceleration: float | None,
        max_jerk: float | None,
    ) -> dict[str, Any]:
        """
        Convert GUI constraint units to SI per-degree units for CasADi flow.
        
        Inputs are expected in per-degree units (mm/deg, mm/deg², mm/deg³) or None (unbounded).
        Outputs are converted to SI per-degree units (m/deg, m/deg², m/deg³) or None.
        
        Parameters
        ----------
        cam_constraints : CamMotionConstraints
            Cam motion constraints
        max_velocity : float | None
            Maximum velocity constraint in mm/deg, or None for unbounded
        max_acceleration : float | None
            Maximum acceleration constraint in mm/deg², or None for unbounded
        max_jerk : float | None
            Maximum jerk constraint in mm/deg³, or None for unbounded
            
        Returns
        -------
        dict[str, Any]
            Constraints dictionary with all values in SI units:
            - stroke: m
            - max_velocity: m/deg
            - max_acceleration: m/deg²
            - max_jerk: m/deg³
            
        Raises
        ------
        ValueError
            If duration_angle_deg is missing or invalid.
        """
        mm_to_m = 1e-3
        cycle_time = max(1e-6, float(self.data.cycle_time))
        stroke_m = max(1e-6, float(cam_constraints.stroke) * mm_to_m)

        # Require duration_angle_deg - no fallback
        duration_angle_deg = getattr(self.data, "duration_angle_deg", None)
        if duration_angle_deg is None:
                raise ValueError(
                    "duration_angle_deg is required for Phase 1 per-degree optimization. "
                "It must be set in data.duration_angle_deg. "
                "No fallback is allowed to prevent unit mixing."
                )
        duration_angle_deg = float(duration_angle_deg)
        if duration_angle_deg <= 0:
            raise ValueError(
                f"duration_angle_deg must be positive, got {duration_angle_deg}. "
                "Phase 1 optimization requires angle-based units, not time-based."
                )

        def _to_si_per_deg(value: float | None) -> float | None:
            """Convert per-degree constraint from mm to m (SI), or return None if value is None."""
            if value is None:
                return None
            return max(1e-6, float(value) * mm_to_m)

        # Compression ratio limits based on clearance geometry
        # CR = V_max / V_min = (stroke + clearance) / clearance
        # Default clearance: 2mm (0.002m) for typical engines
        default_clearance_m = 0.002
        max_cr_from_geometry = (stroke_m + default_clearance_m) / default_clearance_m
        min_cr_default = max(10.0, max_cr_from_geometry * 0.3)  # At least 30% of max, minimum 10
        max_cr_default = min(100.0, max_cr_from_geometry * 1.2)  # Up to 120% of max, cap at 100

        compression_limits = getattr(
            self.settings,
            "casadi_compression_ratio_limits",
            (min_cr_default, max_cr_default),
        )

        constraints_dict = {
            "stroke": stroke_m,
            "cycle_time": cycle_time,
            "duration_angle_deg": duration_angle_deg,
            "upstroke_percent": float(cam_constraints.upstroke_duration_percent),
            # Convert per-degree constraints from mm to m (SI), or None if unbounded
            "max_velocity": _to_si_per_deg(max_velocity),  # mm/deg → m/deg, or None
            "max_acceleration": _to_si_per_deg(max_acceleration),  # mm/deg² → m/deg², or None
            "max_jerk": _to_si_per_deg(max_jerk),  # mm/deg³ → m/deg³, or None
            "compression_ratio_limits": tuple(compression_limits),
        }
        
        # Pass universal grid info to CasADi flow for grid alignment
        if hasattr(self.settings, "universal_n_points") and self.settings.universal_n_points is not None:
            constraints_dict["universal_n_points"] = int(self.settings.universal_n_points)
        if self.data.universal_theta_rad is not None:
            constraints_dict["universal_theta_rad"] = self.data.universal_theta_rad
        
        return constraints_dict

    def _build_casadi_primary_targets(self) -> dict[str, Any]:
        """Create target dictionary for CasADi Phase 1 optimizer."""
        enable_te = bool(
            getattr(self.settings, "enable_thermal_efficiency", False)
            or getattr(self.settings, "use_thermal_efficiency", False),
        )
        thermal_weight = 0.1 if enable_te else 0.0

        return {
            "minimize_jerk": True,
            "maximize_thermal_efficiency": enable_te,
            "weights": {
                "jerk": max(1e-6, float(getattr(self.settings, "jerk_weight", 1.0))),
                "thermal_efficiency": thermal_weight,
                "smoothness": 0.01,
            },
        }

    def _create_casadi_settings(self) -> CasADiOptimizationSettings:
        """Map unified settings into CasADi optimization settings."""
        coarse_segments = getattr(
            self.settings,
            "casadi_coarse_segments",
            (40, 80, 160),
        )
        coarse_segments = tuple(int(max(1, seg)) for seg in coarse_segments)
        resolution_ladder = getattr(self.settings, "casadi_resolution_ladder", None)
        if resolution_ladder:
            resolution_ladder = tuple(int(max(1, seg)) for seg in resolution_ladder)

        return CasADiOptimizationSettings(
            enable_warmstart=getattr(self.settings, "enable_warmstart", True),
            poly_order=int(getattr(self.settings, "casadi_poly_order", 3)),
            collocation_method=getattr(
                self.settings,
                "casadi_collocation_method",
                "legendre",
            ),
            efficiency_target=float(
                getattr(self.settings, "thermal_efficiency_target", 0.55),
            ),
            coarse_resolution_segments=coarse_segments,
            resolution_ladder=resolution_ladder,
            target_angle_resolution_deg=float(
                getattr(self.settings, "casadi_target_angle_deg", 0.1),
            ),
            max_angle_resolution_segments=int(
                getattr(self.settings, "casadi_max_segments", 4096),
            ),
            retry_failed_level=bool(
                getattr(self.settings, "casadi_retry_failed_level", True),
            ),
        )

    def _annotate_casadi_solution(
        self,
        result: OptimizationResult,
        *,
        cycle_time: float,
        duration_angle_deg: float,
    ) -> None:
        """
        Convert CasADi SI outputs to the mm-based units expected by downstream stages.
        
        Derives cycle_time from duration_angle_deg and engine_speed_rpm if available,
        otherwise uses the provided cycle_time.
        """
        # Phase 6: Derive cycle_time from engine speed if available
        engine_speed_rpm = getattr(self.data, "engine_speed_rpm", None)
        if engine_speed_rpm is not None and engine_speed_rpm > 0:
            # cycle_time = (duration_angle_deg / 360) * (60 / rpm)
            derived_cycle_time = (duration_angle_deg / 360.0) * (60.0 / engine_speed_rpm)
            cycle_time = derived_cycle_time
            result.metadata["derived_cycle_time"] = True
            result.metadata["engine_speed_rpm"] = engine_speed_rpm
            # Phase 6: Store derived cycle_time in UnifiedOptimizationData
            self.data.cycle_time = derived_cycle_time
        else:
            result.metadata["derived_cycle_time"] = False
            # Phase 6: Store provided cycle_time in UnifiedOptimizationData
            self.data.cycle_time = cycle_time
        
        # Store cycle_time in metadata for downstream use
        result.metadata["cycle_time"] = cycle_time
        result.metadata["duration_angle_deg"] = duration_angle_deg
        solution = result.solution
        if not solution:
            return

        position = solution.get("position")
        if position is None:
            return

        position_arr = np.asarray(position, dtype=float)
        if position_arr.size == 0:
            return

        mm_scale = 1_000.0
        solution["position_m"] = position_arr
        solution["position"] = position_arr * mm_scale

        velocity = solution.get("velocity")
        if velocity is not None:
            vel_arr = np.asarray(velocity, dtype=float)
            solution["velocity_m_per_deg"] = vel_arr
            solution["velocity"] = vel_arr * mm_scale

        acceleration = solution.get("acceleration")
        if acceleration is not None:
            acc_arr = np.asarray(acceleration, dtype=float)
            solution["acceleration_m_per_deg2"] = acc_arr
            solution["acceleration"] = acc_arr * mm_scale

        jerk = solution.get("jerk")
        if jerk is not None:
            jerk_arr = np.asarray(jerk, dtype=float)
            solution["jerk_m_per_deg3"] = jerk_arr
            solution["jerk"] = jerk_arr * mm_scale

        time_grid = np.linspace(0.0, cycle_time, position_arr.size)
        solution["time"] = time_grid
        theta_deg = solution.get("theta_deg")
        if theta_deg is None:
            theta_deg = np.linspace(0.0, duration_angle_deg, position_arr.size)
        theta_rad = np.deg2rad(theta_deg)
        solution["theta_deg"] = theta_deg
        solution["theta_rad"] = theta_rad
        solution["cam_angle"] = theta_rad

        result.metadata["position_units"] = "mm"
        result.metadata["time_span"] = cycle_time
        result.metadata["duration_angle_deg"] = duration_angle_deg
        result.metadata["velocity_units"] = "mm/deg"
        result.metadata["acceleration_units"] = "mm/deg^2"
        result.metadata["jerk_units"] = "mm/deg^3"

    def _update_data_from_primary(self, result: OptimizationResult) -> None:
        """Update data structure from primary optimization results."""
        if result.status == OptimizationStatus.CONVERGED:
            solution = result.solution

            # Motion law is now generated directly in cam angle domain
            # No need for time-to-angle conversion
            cam_angle_rad = solution.get("cam_angle")
            if cam_angle_rad is not None:
                # If universal grid exists, map result to universal grid first
                if self.data.universal_theta_rad is not None:
                    ug_th = self.data.universal_theta_rad
                    pos = solution.get("position")
                    vel = solution.get("velocity")
                    acc = solution.get("acceleration")
                    if pos is not None and vel is not None and acc is not None:
                        # Check if grids already match (size and values within tolerance)
                        cam_angle_rad_arr = np.asarray(cam_angle_rad)
                        ug_th_arr = np.asarray(ug_th)
                        grids_match = (
                            cam_angle_rad_arr.shape == ug_th_arr.shape
                            and np.allclose(cam_angle_rad_arr, ug_th_arr, rtol=1e-10, atol=1e-12)
                        )
                        
                        if grids_match:
                            # Grids already match - skip remapping
                            log.info(
                                f"Grids already match ({len(cam_angle_rad_arr)} points), "
                                "skipping remapping step"
                            )
                            pos_u = pos
                            vel_u = vel
                            acc_u = acc
                        else:
                            # Grids don't match - this indicates legacy grid logic is being used
                            raise ValueError(
                                f"Grid mismatch detected: solution grid has {len(cam_angle_rad_arr)} points, "
                                f"universal grid has {len(ug_th_arr)} points. "
                                "The _update_data_from_primary process uses legacy universal grid logic. "
                                "Update CasADi optimization to produce solutions on the universal grid before optimization."
                            )
                        # Overwrite solution views to be universal-grid aligned
                        solution["cam_angle"] = ug_th
                        solution["position"] = pos_u
                        solution["velocity"] = vel_u
                        solution["acceleration"] = acc_u
                        # Also align jerk to the universal grid (keep jerk as driven value if available)
                        try:
                            jrk = solution.get("jerk")
                            if jrk is None:
                                jrk = solution.get("control")
                            if jrk is not None:
                                if grids_match:
                                    # Grids match - use jerk directly
                                    jrk_u = jrk
                                else:
                                    # This should not be reached due to earlier error, but kept for safety
                                    raise ValueError(
                                        "Grid mismatch in jerk remapping. "
                                        "The _update_data_from_primary process uses legacy universal grid logic. "
                                        "Update CasADi optimization to produce solutions on the universal grid before optimization."
                                    )
                            else:
                                # If jerk isn't provided, derive from mapped acceleration
                                dtheta_deg = np.gradient(np.degrees(ug_th))
                                jrk_u = np.gradient(acc_u, dtheta_deg, edge_order=2)
                            solution["jerk"] = jrk_u
                        except Exception:
                            # Fallback derivation from mapped acceleration on universal grid
                            dtheta_deg = np.gradient(np.degrees(ug_th))
                            solution["jerk"] = np.gradient(acc_u, dtheta_deg, edge_order=2)
                        # Sensitivities: if gradients/Jacobians exist on internal grid, pull/push to U
                        try:
                            if grids_match:
                                # Grids match - sensitivities are already on universal grid
                                grad_gi = solution.get("gradient")
                                if grad_gi is not None:
                                    solution["gradient_universal"] = np.asarray(grad_gi)
                                jac_gi = solution.get("jacobian")
                                if jac_gi is not None:
                                    solution["jacobian_universal"] = np.asarray(jac_gi)
                            else:
                                # This should not be reached due to earlier error, but kept for safety
                                raise ValueError(
                                    "Grid mismatch in sensitivity remapping. "
                                    "The _update_data_from_primary process uses legacy universal grid logic. "
                                    "Update CasADi optimization to produce solutions on the universal grid before optimization."
                                )
                        except ValueError:
                            raise
                        except Exception:
                            pass

                        # Grid diagnostics (optional)
                        try:
                            if getattr(self.settings, "enable_grid_diagnostics", False):
                                P_u2g, P_g2u = GridMapper.operators(cam_angle_rad, ug_th, method=getattr(self.settings, "mapper_method", "linear"))
                                integ_err = GridMapper.integral_conservation_error(ug_th, pos_u, cam_angle_rad, np.asarray(pos))
                                harm_err = GridMapper.harmonic_probe_error(ug_th, P_u2g, P_g2u, k=5)
                                drift_err = GridMapper.derivative_drift(ug_th, cam_angle_rad, P_u2g, P_g2u, pos_u)
                                log.info(
                                    f"Grid diagnostics (primary): integral_err={integ_err:.3e}, harmonic_err={harm_err:.3e}, drift_err={drift_err:.3e}",
                                )
                        except Exception:
                            pass
                        cam_angle_rad = ug_th
                # Convert from radians to degrees for display after mapping
                cam_angle_deg = np.degrees(cam_angle_rad)
                self.data.primary_theta = cam_angle_deg
            else:
                # No cam angle data found - this is required for per-degree optimization
                raise ValueError(
                    "Primary optimization result must include cam_angle or theta_deg. "
                    "Time-based fallback is not supported in per-degree-only contract."
                )

            # Store motion law data (per-degree units in millimetres)
            # Phase 3: Tag units in metadata so downstream stages know what they're consuming
            self.data.primary_position = solution.get("position")
            self.data.primary_velocity = solution.get("velocity")
            self.data.primary_acceleration = solution.get("acceleration")

            # Handle jerk data (may be in 'control' or 'jerk' field)
            jerk_data = solution.get("jerk")
            if jerk_data is None:
                jerk_data = solution.get("control")
            self.data.primary_jerk = jerk_data
            
            # Phase 3: Store unit metadata for downstream consumers
            if result.metadata is None:
                result.metadata = {}
            position_units = result.metadata.get("position_units", "mm")
            velocity_units = result.metadata.get("velocity_units", "mm/deg")
            acceleration_units = result.metadata.get("acceleration_units", "mm/deg^2")
            jerk_units = result.metadata.get("jerk_units", "mm/deg^3")

            self.data.primary_position_units = position_units
            self.data.primary_velocity_units = velocity_units
            self.data.primary_acceleration_units = acceleration_units
            self.data.primary_jerk_units = jerk_units

            result.metadata["primary_velocity_units"] = velocity_units
            result.metadata["primary_acceleration_units"] = acceleration_units
            result.metadata["primary_jerk_units"] = jerk_units
            result.metadata["primary_position_units"] = position_units
            if "duration_angle_deg" in result.metadata:
                result.metadata["duration_angle_deg"] = result.metadata["duration_angle_deg"]

            # Phase-1: constant load profile aligned with theta
            # Translate workload_target to load pressure if available
            if self.data.primary_theta is not None:
                n = len(self.data.primary_theta)
                load_value = None
                workload_target_j = None
                p_load_kpa = None
                
                # Check if workload-derived load pressure is available in metadata
                if result.metadata:
                    pressure_meta = result.metadata.get("pressure_invariance")
                    if pressure_meta:
                        denom_base = pressure_meta.get("denominator_base")
                        if denom_base and "p_load_kpa" in denom_base:
                            p_load_kpa = float(denom_base["p_load_kpa"])
                            # Use workload-derived load pressure as load_value
                            load_value = p_load_kpa
                        workload_target_j = pressure_meta.get("work_target_j")
                
                # Fallback to constant_load_value from settings if no workload translation
                if load_value is None:
                    try:
                        load_value = float(
                            getattr(self.settings, "constant_load_value", 1.0),
                        )
                    except Exception:
                        load_value = 1.0
                
                self.data.primary_load_profile = np.full(n, load_value, dtype=float)
                self.data.primary_constant_load_value = load_value
                
                # Store workload translation metadata
                if workload_target_j is not None or p_load_kpa is not None:
                    if not hasattr(self.data, 'primary_workload_metadata'):
                        self.data.primary_workload_metadata = {}
                    if workload_target_j is not None:
                        self.data.primary_workload_metadata["workload_target_j"] = float(workload_target_j)
                    if p_load_kpa is not None:
                        self.data.primary_workload_metadata["p_load_kpa"] = float(p_load_kpa)
            # Store constant operating temperature
            try:
                self.data.primary_constant_temperature_K = float(
                    getattr(self.settings, "constant_temperature_K", 900.0),
                )
            except Exception:
                self.data.primary_constant_temperature_K = None

            # Store convergence info
            primary_meta = {
                "status": result.status.value,
                "objective_value": result.objective_value,
                "iterations": result.iterations,
                "solve_time": result.solve_time,
            }

            if result.metadata:
                ca_markers = result.metadata.get("ca_markers")
                if ca_markers:
                    self.data.primary_ca_markers = ca_markers
                    primary_meta["ca_markers"] = ca_markers
                pressure_meta = result.metadata.get("pressure_invariance")
                if pressure_meta:
                    self.data.primary_pressure_invariance = pressure_meta
                    primary_meta["pressure_invariance"] = pressure_meta

            self.data.convergence_info["primary"] = primary_meta

    def _update_data_from_secondary(self, result: OptimizationResult) -> None:
        """Update data structure from secondary optimization results."""
        # Check for convergence (handle both enum and string status)
        is_converged = (
            result.status == OptimizationStatus.CONVERGED
            or str(result.status).lower() == "converged"
        )

        if is_converged:
            solution = result.solution

            # Extract base_radius from multiple possible locations
            base_radius = None

            # Method 1: Direct in solution
            if "base_radius" in solution:
                base_radius = solution.get("base_radius")
                log.debug(f"Extracted base_radius from solution: {base_radius}")

            # Method 2: From optimized_parameters (legacy)
            if base_radius is None:
                optimized_params = solution.get("optimized_parameters", {})
                base_radius = optimized_params.get("base_radius")
                if base_radius is not None:
                    log.debug(
                        f"Extracted base_radius from optimized_parameters: {base_radius}",
                    )

            # Method 3: From metadata (fallback)
            if base_radius is None and hasattr(result, "metadata") and result.metadata:
                gear_config = result.metadata.get("optimized_gear_config", {})
                base_radius = gear_config.get("base_center_radius")
                if base_radius is not None:
                    log.debug(f"Extracted base_radius from metadata: {base_radius}")

            self.data.secondary_base_radius = base_radius

            # Extract other data - handle nested structure from cam_ring_optimizer
            # The cam_ring_optimizer returns:
            # - "cam_profile": {"theta": ..., "profile_radius": ...}
            # - "ring_profile": {"psi": ..., "R_psi": ...}
            cam_profile = solution.get("cam_profile")
            if cam_profile is not None:
                # Extract from nested structure
                self.data.secondary_cam_curves = {
                    "theta": cam_profile.get("theta"),
                    "profile_radius": cam_profile.get("profile_radius"),
                }
            else:
                raise ValueError(
                    "Secondary optimization result must include cam_profile. "
                    "Top-level fallback keys are not supported in per-degree-only contract."
                )
            
            ring_profile = solution.get("ring_profile")
            if ring_profile is not None:
                # Extract from nested structure
                psi = ring_profile.get("psi")
                self.data.secondary_psi = psi
                
                # Store full synchronized ring profile on universal theta grid
                # This contains: theta, psi, R_planet, R_ring all aligned to the same theta domain
                theta = ring_profile.get("theta")
                R_planet = ring_profile.get("R_planet")
                R_ring = ring_profile.get("R_ring")
                R_psi = ring_profile.get("R_psi")  # For backward compatibility
                
                if theta is not None:
                    # Store complete synchronized ring profile
                    self.data.secondary_ring_profile = {
                        "theta": theta,  # Universal theta grid (degrees)
                        "psi": psi,  # Ring angle (radians)
                        "R_planet": R_planet if R_planet is not None else R_psi,  # Planet pitch radius
                        "R_ring": R_ring,  # Synchronized ring radius R_ring(θ) = ρ_target(θ) * R_planet(θ)
                    }
                    log.debug(
                        f"Stored synchronized ring profile on universal theta grid "
                        f"(len={len(theta)} points)"
                    )
                else:
                    log.warning("Ring profile missing theta grid; storing partial profile")
                    self.data.secondary_ring_profile = {
                        "psi": psi,
                        "R_planet": R_planet if R_planet is not None else R_psi,
                        "R_ring": R_ring,
                    }
                
                # Maintain backward compatibility: keep the Litvin planet trajectory in
                # secondary_R_psi when available, since legacy consumers expect that signal.
                if R_psi is not None:
                    self.data.secondary_R_psi = R_psi
                    log.debug("Stored Litvin R_psi (planet trajectory) for backward compatibility")
                elif (
                    R_ring is not None
                    and theta is not None
                    and psi is not None
                ):
                    # If R_psi is unavailable (unexpected), fall back to the synchronized ring
                    # profile resampled onto the ψ grid so legacy code still receives data.
                    from campro.optimization.grid import GridMapper

                    R_ring_on_psi = GridMapper.periodic_linear_resample(
                        from_theta=np.deg2rad(theta),
                        from_values=R_ring,
                        to_theta=psi,
                    )
                    self.data.secondary_R_psi = R_ring_on_psi
                    log.debug(
                        "Resampled R_ring from θ grid to ψ grid because R_psi was unavailable "
                        f"(θ len={len(theta)}, ψ len={len(psi)})"
                    )
                else:
                    log.warning("No R_psi or R_ring data available for legacy secondary_R_psi field")
            else:
                raise ValueError(
                    "Secondary optimization result must include ring_profile. "
                    "Top-level fallback keys are not supported in per-degree-only contract."
                )
            
            self.data.secondary_gear_geometry = solution.get("gear_geometry")
            
            # Log what was extracted for debugging
            log.debug(f"Extracted secondary data:")
            log.debug(f"  - secondary_cam_curves: {self.data.secondary_cam_curves is not None}")
            if self.data.secondary_cam_curves is not None:
                log.debug(f"    Keys: {list(self.data.secondary_cam_curves.keys())}")
            log.debug(f"  - secondary_psi: {self.data.secondary_psi is not None} ({len(self.data.secondary_psi) if self.data.secondary_psi is not None else 0} points)")
            log.debug(f"  - secondary_R_psi: {self.data.secondary_R_psi is not None} ({len(self.data.secondary_R_psi) if self.data.secondary_R_psi is not None else 0} points)")

            # Map any grid-dependent cam profile data to the universal grid for consistency
            try:
                if self.data.universal_theta_rad is not None:
                    ug_th = self.data.universal_theta_rad
                    cam_prof = solution.get("cam_profile")
                    if isinstance(cam_prof, dict):
                        th = cam_prof.get("theta")
                        rprof = cam_prof.get("profile_radius")
                        if th is not None and rprof is not None:
                            if self.settings.mapper_method == "pchip":
                                r_u = GridMapper.periodic_pchip_resample(th, rprof, ug_th)
                            elif self.settings.mapper_method == "barycentric":
                                r_u = GridMapper.barycentric_resample(th, rprof, ug_th)
                            else:
                                r_u = GridMapper.periodic_linear_resample(th, rprof, ug_th)
                            # Store mapped profile alongside original
                            if self.data.secondary_cam_curves is None:
                                self.data.secondary_cam_curves = {}
                            self.data.secondary_cam_curves["cam_profile_universal"] = {
                                "theta": ug_th,
                                "profile_radius": r_u,
                            }
                    # Transform any gradient/jacobian fields tied to theta into universal grid
                    try:
                        # If solution has theta-dependent gradients/jacobians, map them
                        th_int = None
                        if isinstance(cam_prof, dict):
                            th_int = cam_prof.get("theta")
                        if th_int is None:
                            th_int = self.data.universal_theta_rad
                        if th_int is not None and self.data.universal_theta_rad is not None:
                            P_u2g, P_g2u = GridMapper.operators(th_int, self.data.universal_theta_rad, method=getattr(self.settings, "mapper_method", "linear"))
                            grad_gi = solution.get("gradient")
                            if grad_gi is not None:
                                solution["gradient_universal"] = GridMapper.pullback_gradient(np.asarray(grad_gi), P_g2u)
                            jac_gi = solution.get("jacobian")
                            if jac_gi is not None:
                                solution["jacobian_universal"] = GridMapper.pushforward_jacobian(np.asarray(jac_gi), P_u2g, P_g2u)
                    except Exception:
                        pass
            except Exception:
                pass

            # Store convergence info
            self.data.convergence_info["secondary"] = {
                "status": str(result.status),
                "objective_value": result.objective_value,
                "iterations": result.iterations,
                "solve_time": result.solve_time,
            }

            log.info(f"Secondary optimization data updated: base_radius={base_radius}")
        else:
            log.warning(f"Secondary optimization not converged: {result.status}")

    def _update_data_from_tertiary(self, result: OptimizationResult) -> None:
        """Update data structure from tertiary optimization results."""
        if result.status == OptimizationStatus.CONVERGED:
            solution = result.solution
            optimized_params = solution.get("optimized_parameters", {})
            performance_metrics = solution.get("performance_metrics", {})

            # Store crank center optimization results
            self.data.tertiary_crank_center_x = optimized_params.get("crank_center_x")
            self.data.tertiary_crank_center_y = optimized_params.get("crank_center_y")
            self.data.tertiary_crank_radius = optimized_params.get("crank_radius")
            self.data.tertiary_rod_length = optimized_params.get("rod_length")

            # Store performance metrics
            self.data.tertiary_torque_output = performance_metrics.get(
                "cycle_average_torque",
            )
            self.data.tertiary_side_load_penalty = performance_metrics.get(
                "total_side_load_penalty",
            )
            self.data.tertiary_max_torque = performance_metrics.get("max_torque")
            self.data.tertiary_torque_ripple = performance_metrics.get("torque_ripple")
            self.data.tertiary_power_output = performance_metrics.get("power_output")
            self.data.tertiary_max_side_load = performance_metrics.get("max_side_load")

        else:
            log.error(
                "Tertiary optimization failed; clearing downstream crank-center data",
            )
            self.data.tertiary_crank_center_x = None
            self.data.tertiary_crank_center_y = None
            self.data.tertiary_crank_radius = None
            self.data.tertiary_rod_length = None
            self.data.tertiary_torque_output = None
            self.data.tertiary_side_load_penalty = None
            self.data.tertiary_max_torque = None
            self.data.tertiary_torque_ripple = None
            self.data.tertiary_power_output = None
            self.data.tertiary_max_side_load = None

        # Store convergence info regardless of status
        self.data.convergence_info["tertiary"] = {
            "status": result.status.value,
            "objective_value": result.objective_value,
            "iterations": result.iterations,
            "solve_time": result.solve_time,
            "error_message": result.metadata.get("error_message", "")
            if result.status == OptimizationStatus.FAILED
            else "",
        }

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get a summary of the complete optimization process."""
        return {
            "method": self.data.optimization_method.value
            if self.data.optimization_method
            else "unknown",
            "total_solve_time": self.data.total_solve_time,
            "convergence_info": self.data.convergence_info,
            "primary_results": {
                "stroke": self.data.stroke,
                "cycle_time": self.data.cycle_time,
                "points": len(self.data.primary_theta)
                if self.data.primary_theta is not None
                else 0,
                "constant_load_value": self.data.primary_constant_load_value,
                "constant_temperature_K": self.data.primary_constant_temperature_K,
            },
            "secondary_results": {
                "base_radius": self.data.secondary_base_radius,
                # 'rod_length': self.data.secondary_rod_length,  # Removed for phase 2
                "ring_coverage": (
                    np.max(self.data.secondary_psi) - np.min(self.data.secondary_psi)
                )
                * 180
                / np.pi
                if self.data.secondary_psi is not None
                else 0,
            },
            "tertiary_results": {
                "crank_center_x": self.data.tertiary_crank_center_x,
                "crank_center_y": self.data.tertiary_crank_center_y,
                "crank_radius": self.data.tertiary_crank_radius,
                "rod_length": self.data.tertiary_rod_length,
                "torque_output": self.data.tertiary_torque_output,
                "side_load_penalty": self.data.tertiary_side_load_penalty,
                "max_torque": self.data.tertiary_max_torque,
                "torque_ripple": self.data.tertiary_torque_ripple,
                "power_output": self.data.tertiary_power_output,
                "max_side_load": self.data.tertiary_max_side_load,
            },
        }

    def get_migration_analysis(self) -> dict[str, Any]:
        """Get MA57 migration analysis and recommendations."""
        if not self.settings.enable_ipopt_analysis:
            return {"error": "Ipopt analysis is not enabled"}

        analysis = self.migration_analyzer.analyze_migration_readiness()
        plan = self.migration_analyzer.get_migration_plan()

        return {
            "analysis": {
                "total_runs": analysis.total_runs,
                "ma57_beneficial_runs": analysis.ma57_beneficial_runs,
                "average_speedup": analysis.average_speedup,
                "convergence_improvements": analysis.convergence_improvements,
                "migration_priority": analysis.migration_priority,
                "estimated_effort": analysis.estimated_effort,
            },
            "recommendations": analysis.recommendations,
            "migration_plan": plan,
        }

    def export_migration_report(self, output_file: str) -> None:
        """Export comprehensive MA57 migration report."""
        if not self.settings.enable_ipopt_analysis:
            raise RuntimeError("Ipopt analysis is not enabled")

        self.migration_analyzer.export_analysis_report(output_file)
