"""
Unified optimization framework for cascaded cam-ring system optimization.

This module provides a unified framework that homogenizes all three optimization
processes (primary motion law, secondary cam-ring, tertiary sun gear) to use
shared solution methods, libraries, and data structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Type
from enum import Enum
import numpy as np
import time

from .base import BaseOptimizer, OptimizationResult, OptimizationStatus
from .collocation import CollocationOptimizer, CollocationSettings, CollocationMethod
from .motion import MotionOptimizer, MotionObjectiveType
from .cam_ring_optimizer import CamRingOptimizer, CamRingOptimizationConstraints, CamRingOptimizationTargets
from .sun_gear_optimizer import SunGearOptimizer, SunGearOptimizationConstraints, SunGearOptimizationTargets
from campro.logging import get_logger

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
    TERTIARY = "tertiary"  # Sun gear system with back rotation


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
    
    # Tertiary layer constraints (sun gear)
    sun_gear_radius_min: float = 10.0
    sun_gear_radius_max: float = 50.0
    ring_gear_radius_min: float = 30.0
    ring_gear_radius_max: float = 150.0
    ring_gear_radius: float = 45.0  # GUI target for average ring radius
    min_gear_ratio: float = 1.5
    max_gear_ratio: float = 10.0
    max_back_rotation: float = np.pi
    
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
    
    # Tertiary layer targets
    minimize_system_size: bool = True
    maximize_efficiency: bool = True
    minimize_back_rotation: bool = True
    minimize_gear_stress: bool = True
    
    # Weighting factors
    jerk_weight: float = 1.0
    time_weight: float = 0.5
    energy_weight: float = 0.3
    ring_size_weight: float = 1.0
    cam_size_weight: float = 0.5
    curvature_weight: float = 0.2
    system_size_weight: float = 1.0
    efficiency_weight: float = 0.8
    back_rotation_weight: float = 0.6
    gear_stress_weight: float = 0.4


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
    
    # Secondary results
    secondary_base_radius: Optional[float] = None
    # secondary_rod_length: Optional[float] = None  # Removed for phase 2 simplification
    secondary_cam_curves: Optional[Dict[str, np.ndarray]] = None
    secondary_psi: Optional[np.ndarray] = None
    secondary_R_psi: Optional[np.ndarray] = None
    
    # Tertiary results
    tertiary_sun_gear_radius: Optional[float] = None
    tertiary_ring_gear_radius: Optional[float] = None
    tertiary_gear_ratio: Optional[float] = None
    tertiary_journal_offset_x: Optional[float] = None
    tertiary_journal_offset_y: Optional[float] = None
    tertiary_max_back_rotation: Optional[float] = None
    tertiary_psi_complete: Optional[np.ndarray] = None
    tertiary_R_psi_complete: Optional[np.ndarray] = None
    
    # Metadata
    optimization_method: Optional[OptimizationMethod] = None
    total_solve_time: float = 0.0
    convergence_info: Dict[str, Any] = field(default_factory=dict)


class UnifiedOptimizationFramework:
    """
    Unified optimization framework for cascaded cam-ring system optimization.
    
    This framework homogenizes all three optimization processes to use shared
    solution methods, libraries, and data structures for seamless cascading.
    """
    
    def __init__(self, name: str = "UnifiedOptimizationFramework"):
        self.name = name
        self.settings = UnifiedOptimizationSettings()
        self.constraints = UnifiedOptimizationConstraints()
        self.targets = UnifiedOptimizationTargets()
        self.data = UnifiedOptimizationData()
        self._is_configured = True
        
        # Initialize optimizers
        self._initialize_optimizers()
    
    def _initialize_optimizers(self):
        """Initialize all optimization layers with unified settings."""
        # Create collocation settings
        collocation_settings = CollocationSettings(
            degree=self.settings.collocation_degree,
            tolerance=self.settings.collocation_tolerance
        )
        
        # Initialize primary optimizer (motion law)
        self.primary_optimizer = MotionOptimizer(
            settings=collocation_settings
        )
        
        # Initialize secondary optimizer (cam-ring)
        self.secondary_optimizer = CamRingOptimizer(
            name="SecondaryCamRingOptimizer"
        )
        
        # Initialize tertiary optimizer (sun gear)
        self.tertiary_optimizer = SunGearOptimizer(
            name="TertiarySunGearOptimizer"
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
        log.info(f"Configured unified optimization framework")
    
    def _configure_optimizers(self):
        """Configure all optimizers with unified settings."""
        # Configure primary optimizer
        if self.settings.method in [OptimizationMethod.LEGENDRE_COLLOCATION, 
                                   OptimizationMethod.RADAU_COLLOCATION, 
                                   OptimizationMethod.HERMITE_COLLOCATION]:
            collocation_settings = CollocationSettings(
                degree=self.settings.collocation_degree,
                tolerance=self.settings.collocation_tolerance
            )
            self.primary_optimizer.configure(settings=collocation_settings)
        
        # Configure secondary optimizer
        secondary_constraints = CamRingOptimizationConstraints(
            base_radius_min=self.constraints.base_radius_min,
            base_radius_max=self.constraints.base_radius_max,
            # connecting_rod_length_min=self.constraints.connecting_rod_length_min,  # Removed for phase 2
            # connecting_rod_length_max=self.constraints.connecting_rod_length_max,  # Removed for phase 2
            ring_base_radius_min=self.constraints.ring_gear_radius * 0.8,  # Allow variation around GUI target
            ring_base_radius_max=self.constraints.ring_gear_radius * 1.2,
            ring_amplitude_min=5.0,  # Minimum variation (increased)
            ring_amplitude_max=12.0,  # Maximum variation
            ring_frequency_min=0.5,
            ring_frequency_max=3.0,
            target_average_ring_radius=self.constraints.ring_gear_radius,  # Use GUI target
            min_curvature_radius=self.constraints.min_curvature_radius,
            max_curvature=self.constraints.max_curvature,
            max_iterations=self.settings.max_iterations,
            tolerance=self.settings.tolerance
        )
        
        secondary_targets = CamRingOptimizationTargets(
            minimize_ring_size=self.targets.minimize_ring_size,
            minimize_cam_size=self.targets.minimize_cam_size,
            minimize_curvature_variation=self.targets.minimize_curvature_variation,
            target_average_ring_radius=True,  # Enable target radius matching
            minimize_ring_radius_variation=True,  # Enable variation (inverted to encourage variation)
            maximize_ring_smoothness=True,  # Enable smoothness
            ring_size_weight=1.0,  # Balanced weight
            cam_size_weight=1.0,  # Balanced weight
            curvature_weight=self.targets.curvature_weight,
            target_radius_weight=0.5,  # Reduced weight to allow variation
            ring_smoothness_weight=0.1  # Reduced weight to allow variation
        )
        
        self.secondary_optimizer.configure(
            constraints=secondary_constraints,
            targets=secondary_targets
        )
        
        # Configure tertiary optimizer
        tertiary_constraints = SunGearOptimizationConstraints(
            sun_gear_radius_min=self.constraints.sun_gear_radius_min,
            sun_gear_radius_max=self.constraints.sun_gear_radius_max,
            ring_gear_radius_min=self.constraints.ring_gear_radius_min,
            ring_gear_radius_max=self.constraints.ring_gear_radius_max,
            min_gear_ratio=self.constraints.min_gear_ratio,
            max_gear_ratio=self.constraints.max_gear_ratio,
            max_back_rotation=self.constraints.max_back_rotation,
            max_iterations=self.settings.max_iterations,
            tolerance=self.settings.tolerance
        )
        
        tertiary_targets = SunGearOptimizationTargets(
            minimize_system_size=self.targets.minimize_system_size,
            maximize_efficiency=self.targets.maximize_efficiency,
            minimize_back_rotation=self.targets.minimize_back_rotation,
            minimize_gear_stress=self.targets.minimize_gear_stress,
            system_size_weight=self.targets.system_size_weight,
            efficiency_weight=self.targets.efficiency_weight,
            back_rotation_weight=self.targets.back_rotation_weight,
            gear_stress_weight=self.targets.gear_stress_weight
        )
        
        self.tertiary_optimizer.configure(
            constraints=tertiary_constraints,
            targets=tertiary_targets
        )
    
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
            
            # Primary optimization (motion law)
            log.info("Starting primary optimization (motion law)")
            primary_result = self._optimize_primary()
            self._update_data_from_primary(primary_result)
            
            # Secondary optimization (cam-ring)
            log.info("Starting secondary optimization (cam-ring)")
            secondary_result = self._optimize_secondary()
            self._update_data_from_secondary(secondary_result)
            
            # Tertiary optimization (sun gear)
            log.info("Starting tertiary optimization (sun gear)")
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
    
    def _update_data_from_input(self, input_data: Dict[str, Any]):
        """Update data structure from input parameters."""
        self.data.stroke = input_data.get('stroke', 20.0)
        self.data.cycle_time = input_data.get('cycle_time', 1.0)
        self.data.upstroke_duration_percent = input_data.get('upstroke_duration_percent', 60.0)
        self.data.zero_accel_duration_percent = input_data.get('zero_accel_duration_percent', 0.0)
        self.data.motion_type = input_data.get('motion_type', 'minimum_jerk')
    
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
            max_jerk=max_jerk
        )
        
        # Get motion type from data
        motion_type = self.data.motion_type
        
        # Use the cam motion law solver which properly handles upstroke duration and zero acceleration duration
        result = self.primary_optimizer.solve_cam_motion_law(
            cam_constraints=cam_constraints,
            motion_type=motion_type,
            cycle_time=self.data.cycle_time
        )
        
        return result
    
    def _optimize_secondary(self) -> OptimizationResult:
        """Perform secondary optimization (cam-ring)."""
        # Prepare primary data
        primary_data = {
            'cam_angle': self.data.primary_theta,
            'position': self.data.primary_position,
            'velocity': self.data.primary_velocity,
            'acceleration': self.data.primary_acceleration,
            'time': np.linspace(0, self.data.cycle_time, len(self.data.primary_theta))
        }
        
        # Set initial guess based on stroke and GUI target (phase 2: cam + ring only)
        initial_guess = {
            'base_radius': self.data.stroke,
            'ring_base_radius': self.constraints.ring_gear_radius,  # Use GUI target
            'ring_amplitude': 7.0,  # More variation for variable-radius ring
            'ring_frequency': 1.0  # One cycle per revolution
        }
        
        # Perform optimization
        result = self.secondary_optimizer.optimize(
            primary_data=primary_data,
            initial_guess=initial_guess
        )
        
        return result
    
    def _optimize_tertiary(self) -> OptimizationResult:
        """Perform tertiary optimization (sun gear)."""
        # Prepare primary data
        primary_data = {
            'cam_angle': self.data.primary_theta,
            'position': self.data.primary_position,
            'velocity': self.data.primary_velocity,
            'acceleration': self.data.primary_acceleration,
            'time': np.linspace(0, self.data.cycle_time, len(self.data.primary_theta))
        }
        
        # Prepare secondary data
        secondary_data = {
            'optimized_parameters': {
                'base_radius': self.data.secondary_base_radius,
                # 'connecting_rod_length': self.data.secondary_rod_length  # Removed for phase 2
            },
            'cam_curves': self.data.secondary_cam_curves,
            'psi': self.data.secondary_psi,
            'R_psi': self.data.secondary_R_psi
        }
        
        # Set initial guess based on secondary results
        initial_guess = {
            'sun_gear_radius': self.data.secondary_base_radius * 1.5,
            'ring_gear_radius': self.data.secondary_base_radius * 4.5,
            'gear_ratio': 3.0,
            'journal_offset_x': 0.0,
            'journal_offset_y': 0.0,
            'max_back_rotation': np.pi / 4
        }
        
        # Perform optimization
        result = self.tertiary_optimizer.optimize(
            primary_data=primary_data,
            secondary_data=secondary_data,
            initial_guess=initial_guess
        )
        
        return result
    
    def _update_data_from_primary(self, result: OptimizationResult):
        """Update data structure from primary optimization results."""
        if result.status == OptimizationStatus.CONVERGED:
            solution = result.solution
            
            # Motion law is now generated directly in cam angle domain
            # No need for time-to-angle conversion
            cam_angle_rad = solution.get('cam_angle')
            if cam_angle_rad is not None:
                # Convert from radians to degrees for display
                cam_angle_deg = np.degrees(cam_angle_rad)
                self.data.primary_theta = cam_angle_deg
            else:
                # Fallback: try to get from time array (for backward compatibility)
                time_array = solution.get('time')
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
            self.data.primary_position = solution.get('position')
            self.data.primary_velocity = solution.get('velocity')
            self.data.primary_acceleration = solution.get('acceleration')
            
            # Handle jerk data (may be in 'control' or 'jerk' field)
            jerk_data = solution.get('jerk')
            if jerk_data is None:
                jerk_data = solution.get('control')
            self.data.primary_jerk = jerk_data
            
            # Store convergence info
            self.data.convergence_info['primary'] = {
                'status': result.status.value,
                'objective_value': result.objective_value,
                'iterations': result.iterations,
                'solve_time': result.solve_time
            }
    
    def _update_data_from_secondary(self, result: OptimizationResult):
        """Update data structure from secondary optimization results."""
        if result.status == OptimizationStatus.CONVERGED:
            solution = result.solution
            optimized_params = solution.get('optimized_parameters', {})
            
            self.data.secondary_base_radius = optimized_params.get('base_radius')
            # self.data.secondary_rod_length = optimized_params.get('connecting_rod_length')  # Removed for phase 2
            self.data.secondary_cam_curves = solution.get('cam_curves')
            self.data.secondary_psi = solution.get('psi')
            self.data.secondary_R_psi = solution.get('R_psi')
            
            # Store convergence info
            self.data.convergence_info['secondary'] = {
                'status': result.status.value,
                'objective_value': result.objective_value,
                'iterations': result.iterations,
                'solve_time': result.solve_time
            }
    
    def _update_data_from_tertiary(self, result: OptimizationResult):
        """Update data structure from tertiary optimization results."""
        if result.status == OptimizationStatus.CONVERGED:
            solution = result.solution
            optimized_params = solution.get('optimized_parameters', {})
            
            self.data.tertiary_sun_gear_radius = optimized_params.get('sun_gear_radius')
            self.data.tertiary_ring_gear_radius = optimized_params.get('ring_gear_radius')
            self.data.tertiary_gear_ratio = optimized_params.get('gear_ratio')
            self.data.tertiary_journal_offset_x = optimized_params.get('journal_offset_x')
            self.data.tertiary_journal_offset_y = optimized_params.get('journal_offset_y')
            self.data.tertiary_max_back_rotation = optimized_params.get('max_back_rotation')
            self.data.tertiary_psi_complete = solution.get('psi')
            self.data.tertiary_R_psi_complete = solution.get('R_psi')
            
            # Store convergence info
            self.data.convergence_info['tertiary'] = {
                'status': result.status.value,
                'objective_value': result.objective_value,
                'iterations': result.iterations,
                'solve_time': result.solve_time
            }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of the complete optimization process."""
        return {
            'method': self.data.optimization_method.value if self.data.optimization_method else 'unknown',
            'total_solve_time': self.data.total_solve_time,
            'convergence_info': self.data.convergence_info,
            'primary_results': {
                'stroke': self.data.stroke,
                'cycle_time': self.data.cycle_time,
                'points': len(self.data.primary_theta) if self.data.primary_theta is not None else 0
            },
            'secondary_results': {
                'base_radius': self.data.secondary_base_radius,
                # 'rod_length': self.data.secondary_rod_length,  # Removed for phase 2
                'ring_coverage': (np.max(self.data.secondary_psi) - np.min(self.data.secondary_psi)) * 180 / np.pi 
                                if self.data.secondary_psi is not None else 0
            },
            'tertiary_results': {
                'sun_gear_radius': self.data.tertiary_sun_gear_radius,
                'ring_gear_radius': self.data.tertiary_ring_gear_radius,
                'gear_ratio': self.data.tertiary_gear_ratio,
                'back_rotation': self.data.tertiary_max_back_rotation * 180 / np.pi 
                               if self.data.tertiary_max_back_rotation is not None else 0,
                'complete_360_coverage': True if self.data.tertiary_psi_complete is not None else False
            }
        }
