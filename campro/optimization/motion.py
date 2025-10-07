"""
Motion law optimization routines.

This module implements optimization for motion law problems using
various objective functions and constraint systems.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np

from .base import BaseOptimizer, OptimizationResult, OptimizationStatus
from .collocation import CollocationOptimizer, CollocationSettings
from campro.constraints.motion import MotionConstraints
from campro.constraints.cam import CamMotionConstraints
from campro.storage import OptimizationRegistry
from campro.logging import get_logger

log = get_logger(__name__)


class MotionObjectiveType(Enum):
    """Types of motion law objectives."""
    
    MINIMUM_TIME = "minimum_time"
    MINIMUM_ENERGY = "minimum_energy"
    MINIMUM_JERK = "minimum_jerk"
    CUSTOM = "custom"


class MotionOptimizer(BaseOptimizer):
    """
    Optimizer for motion law problems.
    
    This optimizer provides high-level interfaces for solving motion law
    problems with various objective functions and constraint systems.
    """
    
    def __init__(self, settings: Optional[CollocationSettings] = None, 
                 registry: Optional[OptimizationRegistry] = None):
        super().__init__("MotionOptimizer")
        self.collocation_optimizer = CollocationOptimizer(settings)
        self.registry = registry or OptimizationRegistry()
        self._is_configured = True
    
    def configure(self, **kwargs) -> None:
        """
        Configure the motion optimizer.
        
        Args:
            **kwargs: Configuration parameters passed to collocation optimizer
        """
        self.collocation_optimizer.configure(**kwargs)
        self._is_configured = True
    
    def optimize(self, objective: Callable, constraints: Any, 
                initial_guess: Optional[Dict[str, np.ndarray]] = None,
                **kwargs) -> OptimizationResult:
        """
        Solve a motion law optimization problem.
        
        Args:
            objective: Objective function to minimize
            constraints: Motion or cam constraints
            initial_guess: Initial guess for optimization variables
            **kwargs: Additional optimization parameters
                
        Returns:
            OptimizationResult object
        """
        self._validate_inputs(objective, constraints)
        
        # Attach objective type to constraints if provided
        if 'objective_type' in kwargs:
            constraints.objective_type = kwargs['objective_type']
        
        # Delegate to collocation optimizer
        return self.collocation_optimizer.optimize(objective, constraints, initial_guess, **kwargs)
    
    def solve_minimum_time(self, constraints: Union[MotionConstraints, CamMotionConstraints],
                          distance: float, max_velocity: float, max_acceleration: float,
                          max_jerk: float, time_horizon: Optional[float] = None) -> OptimizationResult:
        """
        Solve minimum time motion law problem.
        
        Args:
            constraints: Motion or cam constraints
            distance: Total distance to travel
            max_velocity: Maximum allowed velocity
            max_acceleration: Maximum allowed acceleration
            max_jerk: Maximum allowed jerk
            time_horizon: Optional time horizon
            
        Returns:
            OptimizationResult object
        """
        log.info("Solving minimum time motion law problem")
        
        # Create objective function
        def objective(t, x, v, a, u):
            return t[-1]  # Minimize final time
        
        # Configure optimization parameters
        opt_params = {
            'distance': distance,
            'max_velocity': max_velocity,
            'max_acceleration': max_acceleration,
            'max_jerk': max_jerk,
            'objective_type': MotionObjectiveType.MINIMUM_TIME.value
        }
        
        if time_horizon is not None:
            opt_params['time_horizon'] = time_horizon
        
        return self.optimize(objective, constraints, **opt_params)
    
    def solve_minimum_energy(self, constraints: Union[MotionConstraints, CamMotionConstraints],
                           distance: float, max_velocity: float, max_acceleration: float,
                           max_jerk: float, time_horizon: float) -> OptimizationResult:
        """
        Solve minimum energy motion law problem.
        
        Args:
            constraints: Motion or cam constraints
            distance: Total distance to travel
            max_velocity: Maximum allowed velocity
            max_acceleration: Maximum allowed acceleration
            max_jerk: Maximum allowed jerk
            time_horizon: Fixed time horizon
            
        Returns:
            OptimizationResult object
        """
        log.info("Solving minimum energy motion law problem")
        
        # Create objective function
        def objective(t, x, v, a, u):
            return np.trapz(u**2, t)  # Minimize energy (integral of jerk squared)
        
        # Configure optimization parameters
        opt_params = {
            'distance': distance,
            'max_velocity': max_velocity,
            'max_acceleration': max_acceleration,
            'max_jerk': max_jerk,
            'time_horizon': time_horizon,
            'objective_type': MotionObjectiveType.MINIMUM_ENERGY.value
        }
        
        return self.optimize(objective, constraints, **opt_params)
    
    def solve_minimum_jerk(self, constraints: Union[MotionConstraints, CamMotionConstraints],
                          distance: float, max_velocity: float, max_acceleration: float,
                          max_jerk: float, time_horizon: Optional[float] = None) -> OptimizationResult:
        """
        Solve minimum jerk motion law problem.
        
        Args:
            constraints: Motion or cam constraints
            distance: Total distance to travel
            max_velocity: Maximum allowed velocity
            max_acceleration: Maximum allowed acceleration
            max_jerk: Maximum allowed jerk
            time_horizon: Optional time horizon
            
        Returns:
            OptimizationResult object
        """
        log.info("Solving minimum jerk motion law problem")
        
        # Create objective function
        def objective(t, x, v, a, u):
            return np.trapz(u**2, t)  # Minimize jerk (integral of jerk squared)
        
        # Configure optimization parameters
        opt_params = {
            'distance': distance,
            'max_velocity': max_velocity,
            'max_acceleration': max_acceleration,
            'max_jerk': max_jerk,
            'objective_type': MotionObjectiveType.MINIMUM_JERK.value
        }
        
        if time_horizon is not None:
            opt_params['time_horizon'] = time_horizon
        
        return self.optimize(objective, constraints, **opt_params)
    
    def solve_custom_objective(self, objective_function: Callable,
                              constraints: Union[MotionConstraints, CamMotionConstraints],
                              distance: float, time_horizon: Optional[float] = None,
                              **kwargs) -> OptimizationResult:
        """
        Solve motion law problem with custom objective function.
        
        Args:
            objective_function: Custom objective function
            constraints: Motion or cam constraints
            distance: Total distance to travel
            time_horizon: Optional time horizon
            **kwargs: Additional parameters
            
        Returns:
            OptimizationResult object
        """
        log.info("Solving custom objective motion law problem")
        
        # Configure optimization parameters
        opt_params = {
            'distance': distance,
            'objective_type': MotionObjectiveType.CUSTOM.value,
            **kwargs
        }
        
        if time_horizon is not None:
            opt_params['time_horizon'] = time_horizon
        
        return self.optimize(objective_function, constraints, **opt_params)
    
    def solve_cam_motion_law(self, cam_constraints: CamMotionConstraints,
                           motion_type: str = "minimum_jerk",
                           cycle_time: float = 1.0) -> OptimizationResult:
        """
        Solve cam motion law problem.
        
        Args:
            cam_constraints: Cam-specific constraints
            motion_type: Type of motion law ("minimum_time", "minimum_energy", "minimum_jerk")
            cycle_time: Total cycle time
            
        Returns:
            OptimizationResult object
        """
        log.info(f"Solving cam motion law: {motion_type}")
        
        # Convert cam constraints to motion constraints
        motion_constraints = cam_constraints.to_motion_constraints(cycle_time)
        
        # Calculate time segments
        upstroke_time = cycle_time * cam_constraints.upstroke_duration_percent / 100.0
        downstroke_time = cycle_time - upstroke_time
        
        # Use the new motion law optimizer directly
        try:
            from .motion_law_optimizer import MotionLawOptimizer
            from .motion_law import MotionLawConstraints, MotionType
            
            # Convert cam constraints to motion law constraints
            motion_law_constraints = MotionLawConstraints(
                stroke=cam_constraints.stroke,
                upstroke_duration_percent=cam_constraints.upstroke_duration_percent,
                zero_accel_duration_percent=cam_constraints.zero_accel_duration_percent or 0.0,
                max_velocity=cam_constraints.max_velocity,
                max_acceleration=cam_constraints.max_acceleration,
                max_jerk=cam_constraints.max_jerk
            )
            
            # Convert motion type string to enum
            motion_type_enum = MotionType(motion_type)
            
            # Create and configure motion law optimizer
            motion_optimizer = MotionLawOptimizer()
            motion_optimizer.n_points = 100
            
            # Solve motion law optimization
            result = motion_optimizer.solve_motion_law(motion_law_constraints, motion_type_enum)
            
            # Convert result to OptimizationResult format
            from .base import OptimizationResult, OptimizationStatus
            return OptimizationResult(
                status=OptimizationStatus.CONVERGED if result.convergence_status == "converged" else OptimizationStatus.FAILED,
                objective_value=result.objective_value,
                solution=result.to_dict(),
                iterations=result.iterations,
                solve_time=result.solve_time
            )
            
        except Exception as e:
            log.error(f"Motion law optimization failed: {e}")
            # Fall back to old method
            return self.solve_minimum_jerk(
                motion_constraints,
                distance=cam_constraints.stroke,
                max_velocity=cam_constraints.max_velocity or 100.0,
                max_acceleration=cam_constraints.max_acceleration or 50.0,
                max_jerk=cam_constraints.max_jerk or 10.0,
                time_horizon=upstroke_time
            )
    
    def store_result(self, result: OptimizationResult, optimizer_id: str = "motion_optimizer",
                    metadata: Optional[Dict[str, Any]] = None,
                    constraints: Optional[Dict[str, Any]] = None,
                    optimization_rules: Optional[Dict[str, Any]] = None) -> None:
        """
        Store optimization result in the registry with complete context.
        
        Args:
            result: Optimization result to store
            optimizer_id: Identifier for the optimizer
            metadata: Additional metadata to store
            constraints: Constraints used in optimization
            optimization_rules: Optimization rules and parameters
        """
        if not result.is_successful():
            log.warning(f"Cannot store failed optimization result for {optimizer_id}")
            return
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            'objective_value': result.objective_value,
            'solve_time': result.solve_time,
            'iterations': result.iterations,
            'convergence_info': result.convergence_info
        })
        
        # Prepare optimization rules
        if optimization_rules is None:
            optimization_rules = {
                'motion_type': getattr(result, 'motion_type', 'unknown'),
                'collocation_method': self.collocation_optimizer.settings.method,
                'collocation_degree': self.collocation_optimizer.settings.degree,
                'tolerance': self.collocation_optimizer.settings.tolerance,
                'max_iterations': self.collocation_optimizer.settings.max_iterations
            }
        
        # Prepare solver settings
        solver_settings = {
            'collocation_settings': {
                'method': self.collocation_optimizer.settings.method,
                'degree': self.collocation_optimizer.settings.degree,
                'tolerance': self.collocation_optimizer.settings.tolerance,
                'max_iterations': self.collocation_optimizer.settings.max_iterations,
                'verbose': self.collocation_optimizer.settings.verbose
            }
        }
        
        # Store in registry with complete context
        self.registry.store_result(
            optimizer_id=optimizer_id,
            result_data=result.solution,
            metadata=metadata,
            constraints=constraints,
            optimization_rules=optimization_rules,
            solver_settings=solver_settings,
            expires_in=3600  # Expire in 1 hour
        )
        
        log.info(f"Stored optimization result with complete context for {optimizer_id}")
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """Get information about the motion optimizer."""
        return {
            'name': self.name,
            'configured': self._is_configured,
            'collocation_info': self.collocation_optimizer.get_collocation_info(),
            'registry_stats': self.registry.get_registry_stats()
        }
