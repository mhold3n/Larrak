"""
Base optimization classes and interfaces.

This module defines the fundamental optimization framework that all
optimization methods inherit from, providing a consistent interface
for optimization problems, results, and status tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import time

from campro.logging import get_logger

log = get_logger(__name__)


class OptimizationStatus(Enum):
    """Status of optimization process."""
    
    PENDING = "pending"
    RUNNING = "running"
    CONVERGED = "converged"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"


@dataclass
class OptimizationResult:
    """Result of an optimization process."""
    
    # Solution data
    solution: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Optimization status
    status: OptimizationStatus = OptimizationStatus.PENDING
    
    # Performance metrics
    objective_value: Optional[float] = None
    solve_time: Optional[float] = None
    iterations: Optional[int] = None
    
    # Convergence information
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    error_message: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_successful(self) -> bool:
        """Check if optimization was successful."""
        return self.status == OptimizationStatus.CONVERGED
    
    def has_solution(self) -> bool:
        """Check if solution data is available."""
        return len(self.solution) > 0
    
    def get_solution_summary(self) -> Dict[str, Any]:
        """Get a summary of the solution."""
        if not self.has_solution():
            return {}
        
        summary = {}
        for key, values in self.solution.items():
            if isinstance(values, np.ndarray):
                summary[key] = {
                    'shape': values.shape,
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        
        return summary


class BaseOptimizer(ABC):
    """
    Base class for all optimization methods.
    
    Provides a common interface for optimization problems across different
    domains (motion, cam, physics) and methods (collocation, direct methods).
    """
    
    def __init__(self, name: str = "BaseOptimizer"):
        self.name = name
        self._is_configured = False
        self._current_result: Optional[OptimizationResult] = None
        self._optimization_history: List[OptimizationResult] = []
    
    @abstractmethod
    def configure(self, **kwargs) -> None:
        """
        Configure the optimizer with problem-specific parameters.
        
        Args:
            **kwargs: Configuration parameters
        """
        pass
    
    @abstractmethod
    def optimize(self, objective: Callable, constraints: Any, 
                initial_guess: Optional[Dict[str, np.ndarray]] = None,
                **kwargs) -> OptimizationResult:
        """
        Solve an optimization problem.
        
        Args:
            objective: Objective function to minimize
            constraints: Constraint system
            initial_guess: Initial guess for optimization variables
            **kwargs: Additional optimization parameters
            
        Returns:
            OptimizationResult object
        """
        pass
    
    def is_configured(self) -> bool:
        """Check if optimizer is configured."""
        return self._is_configured
    
    def get_current_result(self) -> Optional[OptimizationResult]:
        """Get the most recent optimization result."""
        return self._current_result
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get all optimization results."""
        return self._optimization_history.copy()
    
    def clear_history(self) -> None:
        """Clear optimization history."""
        self._optimization_history.clear()
    
    def _start_optimization(self) -> OptimizationResult:
        """Start a new optimization process."""
        result = OptimizationResult()
        result.status = OptimizationStatus.RUNNING
        self._current_result = result
        return result
    
    def _finish_optimization(self, result: OptimizationResult, 
                           solution: Dict[str, np.ndarray],
                           objective_value: Optional[float] = None,
                           convergence_info: Optional[Dict[str, Any]] = None,
                           error_message: Optional[str] = None) -> OptimizationResult:
        """Finish an optimization process."""
        result.solve_time = time.time() - result.solve_time if result.solve_time else None
        result.solution = solution
        result.objective_value = objective_value
        result.convergence_info = convergence_info or {}
        result.error_message = error_message
        
        # Determine final status
        if error_message:
            result.status = OptimizationStatus.FAILED
        elif len(solution) > 0:
            result.status = OptimizationStatus.CONVERGED
        else:
            result.status = OptimizationStatus.FAILED
        
        # Store in history
        self._optimization_history.append(result)
        
        log.info(f"Optimization {result.status.value}: {self.name}")
        if result.solve_time:
            log.info(f"Solve time: {result.solve_time:.3f} seconds")
        if result.objective_value is not None:
            log.info(f"Objective value: {result.objective_value:.6f}")
        
        return result
    
    def _validate_inputs(self, objective: Callable, constraints: Any) -> None:
        """Validate optimization inputs."""
        if not callable(objective):
            raise ValueError("Objective must be callable")
        
        if constraints is None:
            raise ValueError("Constraints cannot be None")
        
        if not self._is_configured:
            raise RuntimeError(f"Optimizer {self.name} is not configured. Call configure() first.")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all optimizations."""
        if not self._optimization_history:
            return {}
        
        successful_results = [r for r in self._optimization_history if r.is_successful()]
        
        summary = {
            'total_optimizations': len(self._optimization_history),
            'successful_optimizations': len(successful_results),
            'success_rate': len(successful_results) / len(self._optimization_history),
        }
        
        if successful_results:
            solve_times = [r.solve_time for r in successful_results if r.solve_time is not None]
            if solve_times:
                summary['avg_solve_time'] = np.mean(solve_times)
                summary['min_solve_time'] = np.min(solve_times)
                summary['max_solve_time'] = np.max(solve_times)
            
            objective_values = [r.objective_value for r in successful_results if r.objective_value is not None]
            if objective_values:
                summary['avg_objective_value'] = np.mean(objective_values)
                summary['min_objective_value'] = np.min(objective_values)
                summary['max_objective_value'] = np.max(objective_values)
        
        return summary


