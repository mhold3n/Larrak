"""
Unified optimization framework for cascaded cam-ring system optimization.

This module provides a unified framework that homogenizes all three optimization
processes (primary motion law, secondary cam-ring, tertiary sun gear) to use
shared solution methods, libraries, and data structures.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from campro.logging import get_logger
from campro.optimization.solver_analysis import MA57ReadinessReport
from campro.optimization.solver_selection import (
    AdaptiveSolverSelector,
    ProblemCharacteristics,
)
from campro.optimization.parameter_tuning import DynamicParameterTuner
from campro.optimization.ma57_migration_analyzer import MA57MigrationAnalyzer
from campro.diagnostics.feasibility import check_feasibility
from campro.diagnostics.scaling import compute_scaling_vector

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

log = get_logger(__name__)


class OptimizationMethod(Enum):
    """Available optimization methods for all optimization layers."""

    # Collocation methods
    LEGENDRE_COLLOCATION = "legendre_collocation"
    RADAU_COLLOCATION = "radau_collocation"
    HERMITE_COLLOCATION = "hermite_collocation"

    # Direct optimization methods
    SLSQP = "slsqp"
    L_BFGS_B = "l_bfgs_b"
    TNC = "tnc"
    COBYLA = "cobyla"

    # Global optimization methods
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    BASIN_HOPPING = "basin_hopping"
    DUAL_ANNEALING = "dual_annealing"

    # Advanced methods
    LAGRANGIAN = "lagrangian"
    PENALTY_METHOD = "penalty_method"
    AUGMENTED_LAGRANGIAN = "augmented_lagrangian"


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

    # Direct optimization settings
    max_iterations: int = 100
    tolerance: float = 1e-6
    step_size: float = 1e-4

    # Global optimization settings
    population_size: int = 50
    max_generations: int = 1000
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3

    # Lagrangian settings
    lagrangian_tolerance: float = 1e-8
    penalty_weight: float = 1.0
    augmented_lagrangian_iterations: int = 10

    # General settings
    parallel_processing: bool = False
    verbose: bool = True
    save_intermediate_results: bool = True

    # Ipopt analysis settings
    enable_ipopt_analysis: bool = True

    # Thermal-efficiency-focused primary optimization (complex gas optimizer)
    use_thermal_efficiency: bool = False
    thermal_efficiency_config: Optional[Dict[str, Any]] = None
    # If true, fail fast when TE is enabled but unavailable/non-converged
    require_thermal_efficiency: bool = False
    # Phase-1: constant load model (free piston against generator)
    constant_load_value: float = 1.0
    # Phase-1: constant operating temperature (free-piston idealization)
    constant_temperature_K: float = 900.0
    
    # CasADi physics validation mode settings
    enable_casadi_validation_mode: bool = False
    casadi_validation_tolerance: float = 1e-4


@dataclass
class UnifiedOptimizationConstraints:
    """Unified constraints for all optimization layers."""

    # Primary layer constraints (motion law)
    stroke_min: float = 1.0
    stroke_max: float = 100.0
    cycle_time_min: float = 0.1
    cycle_time_max: float = 10.0
    max_velocity: Optional[float] = None
    max_acceleration: Optional[float] = None
    max_jerk: Optional[float] = None

    # Secondary layer constraints (cam-ring)
    base_radius_min: float = 5.0
    base_radius_max: float = 100.0
    # connecting_rod_length_min: float = 10.0  # Removed for phase 2 simplification
    # connecting_rod_length_max: float = 200.0  # Removed for phase 2 simplification
    min_curvature_radius: float = 1.0
    max_curvature: float = 10.0

    # Tertiary layer constraints (crank center optimization)
    crank_center_x_min: float = -50.0  # mm
    crank_center_x_max: float = 50.0   # mm
    crank_center_y_min: float = -50.0  # mm
    crank_center_y_max: float = 50.0   # mm
    crank_radius_min: float = 20.0     # mm
    crank_radius_max: float = 100.0    # mm
    rod_length_min: float = 100.0      # mm
    rod_length_max: float = 300.0      # mm
    min_torque_output: float = 100.0   # N⋅m
    max_side_load_penalty: float = 500.0  # N

    # Secondary layer constraints (for backward compatibility)
    ring_gear_radius: float = 45.0  # GUI target for average ring radius

    # Physical constraints
    min_clearance: float = 2.0
    max_interference: float = 0.0
    min_ring_coverage: float = 2*np.pi  # 360° coverage required


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
    """Unified data structure for all optimization layers."""

    # Input data
    stroke: float = 20.0
    cycle_time: float = 1.0
    upstroke_duration_percent: float = 60.0
    zero_accel_duration_percent: float = 0.0
    motion_type: str = "minimum_jerk"

    # Primary results
    primary_theta: Optional[np.ndarray] = None
    primary_position: Optional[np.ndarray] = None
    primary_velocity: Optional[np.ndarray] = None
    primary_acceleration: Optional[np.ndarray] = None
    primary_jerk: Optional[np.ndarray] = None
    primary_load_profile: Optional[np.ndarray] = None
    primary_constant_load_value: Optional[float] = None
    primary_constant_temperature_K: Optional[float] = None

    # Secondary results
    secondary_base_radius: Optional[float] = None
    # secondary_rod_length: Optional[float] = None  # Removed for phase 2 simplification
    secondary_cam_curves: Optional[Dict[str, np.ndarray]] = None
    secondary_psi: Optional[np.ndarray] = None
    secondary_R_psi: Optional[np.ndarray] = None
    secondary_gear_geometry: Optional[Dict[str, Any]] = None

    # Tertiary results (crank center optimization)
    tertiary_crank_center_x: Optional[float] = None
    tertiary_crank_center_y: Optional[float] = None
    tertiary_crank_radius: Optional[float] = None
    tertiary_rod_length: Optional[float] = None
    tertiary_torque_output: Optional[float] = None
    tertiary_side_load_penalty: Optional[float] = None
    tertiary_max_torque: Optional[float] = None
    tertiary_torque_ripple: Optional[float] = None
    tertiary_power_output: Optional[float] = None
    tertiary_max_side_load: Optional[float] = None

    # Metadata
    optimization_method: Optional[OptimizationMethod] = None
    total_solve_time: float = 0.0
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    
    # Per-phase Ipopt analysis results
    primary_ipopt_analysis: Optional[MA57ReadinessReport] = None
    secondary_ipopt_analysis: Optional[MA57ReadinessReport] = None
    tertiary_ipopt_analysis: Optional[MA57ReadinessReport] = None


class UnifiedOptimizationFramework:
    """
    Unified optimization framework for cascaded cam-ring system optimization.
    
    This framework homogenizes all three optimization processes to use shared
    solution methods, libraries, and data structures for seamless cascading.
    """

    def __init__(self, name: str = "UnifiedOptimizationFramework", settings: Optional[UnifiedOptimizationSettings] = None):
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

    def _initialize_optimizers(self):
        """Initialize all optimization layers with unified settings."""
        # Create collocation settings
        collocation_settings = CollocationSettings(
            degree=self.settings.collocation_degree,
            tolerance=self.settings.collocation_tolerance,
        )

        # Initialize primary optimizer (motion law)
        # Use CasADi optimizer if enabled, otherwise fall back to MotionOptimizer
        if hasattr(self.settings, 'use_casadi') and self.settings.use_casadi:
            from campro.optimization.casadi_unified_flow import CasADiUnifiedFlow
            self.primary_optimizer = CasADiUnifiedFlow()
        else:
            self.primary_optimizer = MotionOptimizer(
                settings=collocation_settings,
            )

        # Initialize secondary optimizer (cam-ring)
        self.secondary_optimizer = CamRingOptimizer(
            name="SecondaryCamRingOptimizer",
        )

        # Initialize tertiary optimizer (crank center optimization)
        self.tertiary_optimizer = CrankCenterOptimizer(
            name="TertiaryCrankCenterOptimizer",
        )

        log.info(f"Initialized unified optimization framework with {self.settings.method.value} method")

    def configure(self, settings: Optional[UnifiedOptimizationSettings] = None,
                 constraints: Optional[UnifiedOptimizationConstraints] = None,
                 targets: Optional[UnifiedOptimizationTargets] = None,
                 **kwargs) -> None:
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

    def _configure_optimizers(self):
        """Configure all optimizers with unified settings."""
        # Configure primary optimizer
        if self.settings.method in [OptimizationMethod.LEGENDRE_COLLOCATION,
                                   OptimizationMethod.RADAU_COLLOCATION,
                                   OptimizationMethod.HERMITE_COLLOCATION]:
            collocation_settings = CollocationSettings(
                degree=self.settings.collocation_degree,
                tolerance=self.settings.collocation_tolerance,
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
            log.info(f"Validation tolerance: {self.settings.casadi_validation_tolerance}")

    def optimize_cascaded(self, input_data: Dict[str, Any]) -> UnifiedOptimizationData:
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

        log.info(f"Starting cascaded optimization with {self.settings.method.value} method")

        start_time = time.time()

        try:
            # Update data with input
            self._update_data_from_input(input_data)

            # A4: Phase-0 feasibility check for primary constraints
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
                    log.warning(
                        "Primary feasibility (NLP) failed (max_violation=%.3e): %s",
                        feas.max_violation,
                        ", ".join(feas.recommendations) if feas.recommendations else "",
                    )
            except Exception as _e:
                log.debug(f"Feasibility NLP pre-check skipped due to error: {_e}")

            # Primary optimization (motion law)
            log.info("Starting primary optimization (motion law)")
            
            # Get warm-start initial guess if using CasADi optimizer
            initial_guess = None
            if hasattr(self.primary_optimizer, 'warmstart_mgr'):
                initial_guess = self.primary_optimizer.warmstart_mgr.get_initial_guess(input_data)
                if initial_guess:
                    log.info("Using warm-start initial guess for primary optimization")
                else:
                    log.info("No suitable warm-start found, using default initial guess")
            
            primary_result = self._optimize_primary(initial_guess=initial_guess)
            self._update_data_from_primary(primary_result)
            
            # Store solution for future warm-starts if using CasADi optimizer
            if (hasattr(self.primary_optimizer, 'warmstart_mgr') and 
                primary_result.successful and 
                hasattr(primary_result, 'variables')):
                self.primary_optimizer.warmstart_mgr.store_solution(
                    input_data,
                    primary_result.variables,
                    {
                        'solve_time': primary_result.solve_time,
                        'objective_value': primary_result.objective_value,
                        'n_segments': getattr(self.primary_optimizer, 'n_segments', 50),
                        'timestamp': time.time()
                    }
                )

            # Secondary optimization (cam-ring)
            log.info("Starting secondary optimization (cam-ring)")
            # A4: Feasibility check for secondary bound ordering
            try:
                sec_pairs = {
                    "base_radius": (self.constraints.base_radius_min, self.constraints.base_radius_max),
                }
                sec_constraints = {"pairs": sec_pairs}
                _ = check_feasibility(sec_constraints, {})
                # We only record ordering issues if any
            except Exception:
                pass
            secondary_result = self._optimize_secondary()
            self._update_data_from_secondary(secondary_result)

            # Tertiary optimization (sun gear)
            log.info("Starting tertiary optimization (sun gear)")
            # A4: Feasibility check for tertiary bounds
            try:
                tert_pairs = {
                    "crank_center_x": (self.constraints.crank_center_x_min, self.constraints.crank_center_x_max),
                    "crank_center_y": (self.constraints.crank_center_y_min, self.constraints.crank_center_y_max),
                    "crank_radius": (self.constraints.crank_radius_min, self.constraints.crank_radius_max),
                }
                tert_constraints = {"pairs": tert_pairs}
                tert_feas = check_feasibility(tert_constraints, {})
                if tert_feas.violations:
                    self.data.convergence_info.setdefault("feasibility_tertiary", {})
                    self.data.convergence_info["feasibility_tertiary"].update({
                        "violations": tert_feas.violations,
                        "recommendations": tert_feas.recommendations,
                    })
            except Exception:
                pass
            tertiary_result = self._optimize_tertiary()
            self._update_data_from_tertiary(tertiary_result)

            # Finalize results
            self.data.total_solve_time = time.time() - start_time
            self.data.optimization_method = self.settings.method

            log.info(f"Cascaded optimization completed in {self.data.total_solve_time:.3f} seconds")

        except Exception as e:
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
        from campro.constants import CASADI_PHYSICS_VALIDATION_MODE, CASADI_PHYSICS_VALIDATION_TOLERANCE
        import campro.constants as constants
        
        # Note: We can't directly modify the constants, but we can log the setting
        log.info(f"CasADi validation mode enabled with tolerance {tolerance}")
        log.info("Note: Set CASADI_PHYSICS_VALIDATION_MODE=True in constants.py to activate")
        
    def disable_casadi_validation_mode(self) -> None:
        """Disable CasADi physics validation mode."""
        self.settings.enable_casadi_validation_mode = False
        log.info("CasADi validation mode disabled")

    def _update_data_from_input(self, input_data: Dict[str, Any]):
        """Update data structure from input parameters."""
        self.data.stroke = input_data.get("stroke", 20.0)
        self.data.cycle_time = input_data.get("cycle_time", 1.0)
        self.data.upstroke_duration_percent = input_data.get("upstroke_duration_percent", 60.0)
        self.data.zero_accel_duration_percent = input_data.get("zero_accel_duration_percent", 0.0)
        self.data.motion_type = input_data.get("motion_type", "minimum_jerk")
        # Optional overrides for constant load and temperature from UI/input
        if "constant_load_value" in input_data:
            try:
                self.settings.constant_load_value = float(input_data["constant_load_value"])
            except Exception:
                pass
        if "constant_temperature_K" in input_data:
            try:
                self.settings.constant_temperature_K = float(input_data["constant_temperature_K"])
            except Exception:
                pass

    def _optimize_primary(self) -> OptimizationResult:
        """Perform primary optimization (motion law)."""
        # Create cam motion constraints with user input parameters
        from campro.constraints.cam import CamMotionConstraints

        # Get constraint values from unified constraints
        max_velocity = self.constraints.max_velocity or 100.0
        max_acceleration = self.constraints.max_acceleration or 1000.0
        max_jerk = self.constraints.max_jerk or 10000.0

        # Create cam motion constraints with user input parameters
        cam_constraints = CamMotionConstraints(
            stroke=self.data.stroke,
            upstroke_duration_percent=self.data.upstroke_duration_percent,
            zero_accel_duration_percent=self.data.zero_accel_duration_percent,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            max_jerk=max_jerk,
        )

        # A3: Compute and record primary scaling vector for diagnostics
        try:
            bounds_for_scaling = {
                "position": (0.0, float(self.data.stroke)),
                "velocity": (-(max_velocity or 0.0), (max_velocity or 0.0)),
                "acceleration": (-(max_acceleration or 0.0), (max_acceleration or 0.0)),
                "jerk": (-(max_jerk or 0.0), (max_jerk or 0.0)),
            }
            scales = compute_scaling_vector(bounds_for_scaling)
            self.data.convergence_info["scaling_primary"] = scales
        except Exception:
            pass

        # Get motion type from data
        motion_type = self.data.motion_type

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
                cfg_overrides = getattr(self.settings, "thermal_efficiency_config", None) or {}
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
                    zero_accel_duration_percent=cam_constraints.zero_accel_duration_percent or 0.0,
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
                        "constant_temperature_K": float(getattr(self.settings, "constant_temperature_K", 900.0)),
                        "constant_load_value": float(getattr(self.settings, "constant_load_value", 1.0)),
                    }
                    result.metadata.update({"assumptions": assumptions})
                    
                    # Extract analysis from adapter if available
                    adapter_result_dict = adapter_result.to_dict()
                    if 'ipopt_analysis' in adapter_result_dict:
                        result.metadata['ipopt_analysis'] = adapter_result_dict['ipopt_analysis']
                        self.data.primary_ipopt_analysis = adapter_result_dict['ipopt_analysis']
                        
                        # Collect data for MA57 migration analysis
                        if self.settings.enable_ipopt_analysis and self.data.primary_ipopt_analysis is not None:
                            problem_size = (len(self.data.primary_theta) if self.data.primary_theta is not None else 100, 10)
                            self.migration_analyzer.add_ma27_run(
                                phase="primary",
                                problem_size=problem_size,
                                ma27_report=self.data.primary_ipopt_analysis,
                                metadata={
                                    "stroke": self.data.stroke,
                                    "cycle_time": self.data.cycle_time,
                                    "use_thermal_efficiency": self.settings.use_thermal_efficiency
                                }
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

        # Regular optimization path (when thermal efficiency is not enabled)
        try:
            # Use the existing primary optimizer to solve the motion law problem
            result = self.primary_optimizer.solve_cam_motion_law(
                cam_constraints=cam_constraints,
                motion_type=motion_type,
                cycle_time=self.data.cycle_time,
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
        log.info("Starting secondary optimization...")
        
        # Check if primary data is available
        if self.data.primary_theta is None:
            raise RuntimeError("Primary optimization must be completed before secondary optimization")

        # Prepare primary data
        primary_data = {
            "cam_angle": self.data.primary_theta,
            "position": self.data.primary_position,
            "velocity": self.data.primary_velocity,
            "acceleration": self.data.primary_acceleration,
            "time": np.linspace(0, self.data.cycle_time, len(self.data.primary_theta)) if self.data.primary_theta is not None else np.array([]),
        }

        # Analyze problem characteristics
        problem_chars = ProblemCharacteristics(
            n_variables=len(self.data.primary_theta) if self.data.primary_theta is not None else 100,
            n_constraints=10,  # Estimate
            problem_type="litvin",
            expected_iterations=100,
            linear_solver_ratio=0.3,  # Estimate
            has_convergence_issues=False
        )
        
        # Select optimal solver (currently always MA27)
        solver_type = self.solver_selector.select_solver(problem_chars, "secondary")
        log.info(f"Selected solver for secondary optimization: {solver_type.value}")
        
        # Tune parameters
        tuned_params = self.parameter_tuner.tune_parameters(
            "secondary", problem_chars, 
            self.solver_selector.analysis_history.get("secondary")
        )
        log.info(f"Tuned parameters: max_iter={tuned_params.max_iter}, tol={tuned_params.tol}")

        # Set initial guess based on stroke and GUI target (phase 2: cam + ring only)
        initial_guess = {
            "base_radius": self.data.stroke,
        }

        # A3: Compute and record simple scaling stats for secondary design variables
        try:
            bmin, bmax = float(self.constraints.base_radius_min), float(self.constraints.base_radius_max)
            sec_scales = compute_scaling_vector({"base_radius": (bmin, bmax)})
            self.data.convergence_info["scaling_secondary"] = sec_scales
        except Exception:
            pass

        # Perform optimization
        log.info("Calling secondary optimizer...")
        result = self.secondary_optimizer.optimize(
            primary_data=primary_data,
            initial_guess=initial_guess,
        )
        log.info(f"Secondary optimization completed: status={result.status}, success={result.is_successful()}")
        
        # Extract and store secondary analysis from cam ring optimizer
        # The cam ring optimizer uses litvin optimization which now provides analysis
        if hasattr(result, 'metadata') and 'ipopt_analysis' in result.metadata:
            self.data.secondary_ipopt_analysis = result.metadata['ipopt_analysis']
            # Update analysis history for future decisions
            self.solver_selector.update_history("secondary", result.metadata['ipopt_analysis'])
            
            # Collect data for MA57 migration analysis
            if self.settings.enable_ipopt_analysis and self.data.secondary_ipopt_analysis is not None:
                problem_size = (len(self.data.primary_theta) if self.data.primary_theta is not None else 100, 10)
                self.migration_analyzer.add_ma27_run(
                    phase="secondary",
                    problem_size=problem_size,
                    ma27_report=self.data.secondary_ipopt_analysis,
                    metadata={
                        "base_radius": self.data.secondary_base_radius,
                        "stroke": self.data.stroke,
                        "solver_type": solver_type.value
                    }
                )
        else:
            self.data.secondary_ipopt_analysis = None

        return result

    def get_phase2_animation_inputs(self) -> Dict[str, Any]:
        """Return minimal deterministic inputs for Phase-2 animation.

        Returns
        -------
        Dict[str, Any]
            A bundle containing primary and secondary outputs sufficient to
            build deterministic Phase-2 relationships without solving.
        """
        if (
            self.data.primary_theta is None or
            self.data.primary_position is None or
            self.data.secondary_base_radius is None or
            self.data.secondary_psi is None or
            self.data.secondary_R_psi is None
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
            raise RuntimeError("Primary optimization must be completed before tertiary optimization")
        if self.data.secondary_base_radius is None:
            raise RuntimeError("Secondary optimization must be completed before tertiary optimization")

        # Prepare primary data (motion law)
        if self.data.primary_theta is None:
            raise RuntimeError("Primary theta data is None - primary optimization may have failed")

        primary_data = {
            "theta": self.data.primary_theta,
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
            scales_ter = compute_scaling_vector({
                "crank_center_x": (float(self.constraints.crank_center_x_min), float(self.constraints.crank_center_x_max)),
                "crank_center_y": (float(self.constraints.crank_center_y_min), float(self.constraints.crank_center_y_max)),
                "crank_radius": (float(self.constraints.crank_radius_min), float(self.constraints.crank_radius_max)),
                "rod_length": (
                    float(getattr(self.constraints, 'rod_length_min', (self.data.secondary_base_radius or 20.0) * 4.0)),
                    float(getattr(self.constraints, 'rod_length_max', (self.data.secondary_base_radius or 20.0) * 8.0)),
                ),
            })
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
            has_convergence_issues=False
        )
        
        # Select optimal solver (currently always MA27)
        solver_type = self.solver_selector.select_solver(problem_chars, "tertiary")
        log.info(f"Selected solver for tertiary optimization: {solver_type.value}")
        
        # Tune parameters
        tuned_params = self.parameter_tuner.tune_parameters(
            "tertiary", problem_chars, 
            self.solver_selector.analysis_history.get("tertiary")
        )
        log.info(f"Tuned parameters: max_iter={tuned_params.max_iter}, tol={tuned_params.tol}")

        # Set initial guess based on secondary results
        initial_guess = {
            "crank_center_x": 0.0,  # Start at origin
            "crank_center_y": 0.0,  # Start at origin
            "crank_radius": self.data.secondary_base_radius * 2.0,  # Scale from cam base radius
            "rod_length": self.data.secondary_base_radius * 6.0,   # Scale from cam base radius
        }

        # Perform optimization
        result = self.tertiary_optimizer.optimize(
            primary_data=primary_data,
            secondary_data=secondary_data,
            initial_guess=initial_guess,
        )
        
        # Extract and store tertiary analysis from crank center optimizer
        if hasattr(result, 'metadata') and 'ipopt_analysis' in result.metadata:
            self.data.tertiary_ipopt_analysis = result.metadata['ipopt_analysis']
            # Update analysis history for future decisions
            self.solver_selector.update_history("tertiary", result.metadata['ipopt_analysis'])
            
            # Collect data for MA57 migration analysis
            if self.settings.enable_ipopt_analysis and self.data.tertiary_ipopt_analysis is not None:
                problem_size = (4, 8)  # crank_center_x, crank_center_y, crank_radius, rod_length
                self.migration_analyzer.add_ma27_run(
                    phase="tertiary",
                    problem_size=problem_size,
                    ma27_report=self.data.tertiary_ipopt_analysis,
                    metadata={
                        "base_radius": self.data.secondary_base_radius,
                        "stroke": self.data.stroke,
                        "solver_type": solver_type.value
                    }
                )
        else:
            self.data.tertiary_ipopt_analysis = None

        return result

    def _update_data_from_primary(self, result: OptimizationResult):
        """Update data structure from primary optimization results."""
        if result.status == OptimizationStatus.CONVERGED:
            solution = result.solution

            # Motion law is now generated directly in cam angle domain
            # No need for time-to-angle conversion
            cam_angle_rad = solution.get("cam_angle")
            if cam_angle_rad is not None:
                # Convert from radians to degrees for display
                cam_angle_deg = np.degrees(cam_angle_rad)
                self.data.primary_theta = cam_angle_deg
            else:
                # Fallback: try to get from time array (for backward compatibility)
                time_array = solution.get("time")
                if time_array is not None:
                    # Convert time to cam angle (assuming constant angular velocity)
                    cam_angular_velocity = 2 * np.pi / self.data.cycle_time  # rad/s
                    cam_angle_rad = time_array * cam_angular_velocity
                    cam_angle_deg = np.degrees(cam_angle_rad)
                    self.data.primary_theta = cam_angle_deg
                else:
                    log.warning("No cam angle or time data found in primary optimization result")
                    return

            # Store motion law data (already in correct units)
            self.data.primary_position = solution.get("position")
            self.data.primary_velocity = solution.get("velocity")
            self.data.primary_acceleration = solution.get("acceleration")

            # Handle jerk data (may be in 'control' or 'jerk' field)
            jerk_data = solution.get("jerk")
            if jerk_data is None:
                jerk_data = solution.get("control")
            self.data.primary_jerk = jerk_data

            # Phase-1: constant load profile aligned with theta
            if self.data.primary_theta is not None:
                n = len(self.data.primary_theta)
                try:
                    load_value = float(getattr(self.settings, "constant_load_value", 1.0))
                except Exception:
                    load_value = 1.0
                self.data.primary_load_profile = np.full(n, load_value, dtype=float)
                self.data.primary_constant_load_value = load_value
            # Store constant operating temperature
            try:
                self.data.primary_constant_temperature_K = float(getattr(self.settings, "constant_temperature_K", 900.0))
            except Exception:
                self.data.primary_constant_temperature_K = None

            # Store convergence info
            self.data.convergence_info["primary"] = {
                "status": result.status.value,
                "objective_value": result.objective_value,
                "iterations": result.iterations,
                "solve_time": result.solve_time,
            }

    def _update_data_from_secondary(self, result: OptimizationResult):
        """Update data structure from secondary optimization results."""
        # Check for convergence (handle both enum and string status)
        is_converged = (
            result.status == OptimizationStatus.CONVERGED or 
            str(result.status).lower() == "converged"
        )
        
        if is_converged:
            solution = result.solution
            
            # Extract base_radius from multiple possible locations
            base_radius = None
            
            # Method 1: Direct in solution
            if 'base_radius' in solution:
                base_radius = solution.get("base_radius")
                log.debug(f"Extracted base_radius from solution: {base_radius}")
            
            # Method 2: From optimized_parameters (legacy)
            if base_radius is None:
                optimized_params = solution.get("optimized_parameters", {})
                base_radius = optimized_params.get("base_radius")
                if base_radius is not None:
                    log.debug(f"Extracted base_radius from optimized_parameters: {base_radius}")
            
            # Method 3: From metadata (fallback)
            if base_radius is None and hasattr(result, 'metadata') and result.metadata:
                gear_config = result.metadata.get("optimized_gear_config", {})
                base_radius = gear_config.get("base_center_radius")
                if base_radius is not None:
                    log.debug(f"Extracted base_radius from metadata: {base_radius}")
            
            self.data.secondary_base_radius = base_radius
            
            # Extract other data
            self.data.secondary_cam_curves = solution.get("cam_curves")
            self.data.secondary_psi = solution.get("psi")
            self.data.secondary_R_psi = solution.get("R_psi")
            self.data.secondary_gear_geometry = solution.get("gear_geometry")

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

    def _update_data_from_tertiary(self, result: OptimizationResult):
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
            self.data.tertiary_torque_output = performance_metrics.get("cycle_average_torque")
            self.data.tertiary_side_load_penalty = performance_metrics.get("total_side_load_penalty")
            self.data.tertiary_max_torque = performance_metrics.get("max_torque")
            self.data.tertiary_torque_ripple = performance_metrics.get("torque_ripple")
            self.data.tertiary_power_output = performance_metrics.get("power_output")
            self.data.tertiary_max_side_load = performance_metrics.get("max_side_load")

        else:
            # Even if optimization failed, provide default values for display
            log.warning("Tertiary optimization failed, using default values for display")

            # Use default values based on secondary results
            self.data.tertiary_crank_center_x = 0.0  # Default to origin
            self.data.tertiary_crank_center_y = 0.0  # Default to origin
            self.data.tertiary_crank_radius = self.data.secondary_base_radius * 2.0 if self.data.secondary_base_radius else 50.0
            self.data.tertiary_rod_length = self.data.secondary_base_radius * 6.0 if self.data.secondary_base_radius else 150.0

            # Default performance metrics (placeholder values)
            self.data.tertiary_torque_output = 100.0  # N⋅m
            self.data.tertiary_side_load_penalty = 50.0  # N
            self.data.tertiary_max_torque = 120.0  # N⋅m
            self.data.tertiary_torque_ripple = 10.0  # N⋅m
            self.data.tertiary_power_output = 500.0  # W
            self.data.tertiary_max_side_load = 200.0  # N

        # Store convergence info regardless of status
        self.data.convergence_info["tertiary"] = {
            "status": result.status.value,
            "objective_value": result.objective_value,
            "iterations": result.iterations,
            "solve_time": result.solve_time,
            "error_message": result.metadata.get("error_message", "") if result.status == OptimizationStatus.FAILED else "",
        }

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of the complete optimization process."""
        return {
            "method": self.data.optimization_method.value if self.data.optimization_method else "unknown",
            "total_solve_time": self.data.total_solve_time,
            "convergence_info": self.data.convergence_info,
            "primary_results": {
                "stroke": self.data.stroke,
                "cycle_time": self.data.cycle_time,
                "points": len(self.data.primary_theta) if self.data.primary_theta is not None else 0,
                "constant_load_value": self.data.primary_constant_load_value,
                "constant_temperature_K": self.data.primary_constant_temperature_K,
            },
            "secondary_results": {
                "base_radius": self.data.secondary_base_radius,
                # 'rod_length': self.data.secondary_rod_length,  # Removed for phase 2
                "ring_coverage": (np.max(self.data.secondary_psi) - np.min(self.data.secondary_psi)) * 180 / np.pi
                                if self.data.secondary_psi is not None else 0,
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

    def get_migration_analysis(self) -> Dict[str, Any]:
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
                "estimated_effort": analysis.estimated_effort
            },
            "recommendations": analysis.recommendations,
            "migration_plan": plan
        }

    def export_migration_report(self, output_file: str):
        """Export comprehensive MA57 migration report."""
        if not self.settings.enable_ipopt_analysis:
            raise RuntimeError("Ipopt analysis is not enabled")
        
        self.migration_analyzer.export_analysis_report(output_file)
