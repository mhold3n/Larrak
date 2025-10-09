"""
Secondary collocation optimizer for cascaded optimization.

This module implements a secondary collocation optimizer that can use
results from primary motion law optimization to perform additional
optimization tasks.
"""

import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from campro.logging import get_logger
from campro.storage import OptimizationRegistry, StorageResult

from .base import BaseOptimizer, OptimizationResult
from .collocation import CollocationOptimizer, CollocationSettings

log = get_logger(__name__)


class SecondaryOptimizer(BaseOptimizer):
    """
    Secondary collocation optimizer shell for cascaded optimization.
    
    This is a generic shell that can perform secondary optimization tasks
    based on externally provided constraints, relationships, and optimization targets.
    The specific implementation details are passed in during optimization.
    """

    def __init__(self, name: str = "SecondaryOptimizer",
                 registry: Optional[OptimizationRegistry] = None,
                 settings: Optional[CollocationSettings] = None):
        super().__init__(name)
        self.registry = registry or OptimizationRegistry()
        self.collocation_optimizer = CollocationOptimizer(settings)
        self._is_configured = True

    def configure(self, **kwargs) -> None:
        """
        Configure the secondary optimizer.
        
        Args:
            **kwargs: Configuration parameters
                - registry: Optimization registry
                - collocation_settings: Collocation settings
                - primary_optimizer_id: ID of primary optimizer to use
        """
        if "registry" in kwargs:
            self.registry = kwargs["registry"]

        if "collocation_settings" in kwargs:
            self.collocation_optimizer.configure(**kwargs["collocation_settings"])

        self._is_configured = True
        log.info(f"Configured secondary optimizer: {self.name}")

    def optimize(self, objective: Callable, constraints: Any,
                initial_guess: Optional[Dict[str, np.ndarray]] = None,
                **kwargs) -> OptimizationResult:
        """
        Solve a secondary optimization problem using primary results and external specifications.
        
        Args:
            objective: Objective function to minimize
            constraints: Constraint system
            initial_guess: Initial guess for optimization variables
            **kwargs: Additional optimization parameters
                - primary_optimizer_id: ID of primary optimizer to use
                - secondary_constraints: Specific constraints for secondary optimization
                - secondary_relationships: Relationships between primary and secondary optimization
                - optimization_targets: Specific targets for secondary optimization
                - processing_function: Function to process primary results
                
        Returns:
            OptimizationResult object
        """
        self._validate_inputs(objective, constraints)

        result = self._start_optimization()
        result.solve_time = time.time()

        try:
            # Get primary optimization result
            primary_optimizer_id = kwargs.get("primary_optimizer_id", "motion_optimizer")
            primary_result = self.registry.get_result(primary_optimizer_id)

            if primary_result is None:
                raise ValueError(f"No primary result found for optimizer '{primary_optimizer_id}'")

            log.info(f"Using primary result from '{primary_optimizer_id}' for secondary optimization")

            # Extract external specifications
            secondary_constraints = kwargs.get("secondary_constraints", {})
            secondary_relationships = kwargs.get("secondary_relationships", {})
            optimization_targets = kwargs.get("optimization_targets", {})
            processing_function = kwargs.get("processing_function")

            # Process primary solution using external specifications
            if processing_function is not None:
                processed_solution = processing_function(
                    primary_result.data,
                    secondary_constraints,
                    secondary_relationships,
                    optimization_targets,
                )
            else:
                # Default: return primary solution unchanged (no processing)
                processed_solution = primary_result.data.copy()
                log.warning("No processing function provided - returning primary solution unchanged")

            # Calculate objective value
            objective_value = self._calculate_objective_value(objective, processed_solution)

            # Finish optimization
            result = self._finish_optimization(
                result, processed_solution, objective_value,
                convergence_info={
                    "primary_optimizer_id": primary_optimizer_id,
                    "secondary_constraints": secondary_constraints,
                    "secondary_relationships": secondary_relationships,
                    "optimization_targets": optimization_targets,
                    "processing_function": processing_function.__name__ if processing_function else None,
                    "primary_objective_value": primary_result.metadata.get("objective_value"),
                },
            )

        except Exception as e:
            error_message = f"Secondary optimization failed: {e!s}"
            log.error(error_message)
            result = self._finish_optimization(result, {}, error_message=error_message)

        return result

    def _validate_inputs(self, objective: Callable, constraints: Any) -> None:
        """Validate optimization inputs for secondary optimization."""
        if not callable(objective):
            raise ValueError("Objective must be callable")

        # Constraints can be None for secondary optimization as we use primary results
        # if constraints is None:
        #     raise ValueError("Constraints cannot be None")

        if not self._is_configured:
            raise RuntimeError(f"Optimizer {self.name} is not configured. Call configure() first.")

    def process_primary_result(self, primary_optimizer_id: str = "motion_optimizer",
                              secondary_constraints: Optional[Dict[str, Any]] = None,
                              secondary_relationships: Optional[Dict[str, Any]] = None,
                              optimization_targets: Optional[Dict[str, Any]] = None,
                              processing_function: Optional[Callable] = None,
                              objective_function: Optional[Callable] = None) -> OptimizationResult:
        """
        Process primary optimization result using external specifications.
        
        Args:
            primary_optimizer_id: ID of primary optimizer
            secondary_constraints: Specific constraints for secondary optimization
            secondary_relationships: Relationships between primary and secondary optimization
            optimization_targets: Specific targets for secondary optimization
            processing_function: Function to process primary results
            objective_function: Objective function for secondary optimization
            
        Returns:
            OptimizationResult object
        """
        log.info(f"Processing primary result from '{primary_optimizer_id}' with external specifications")

        # Use default objective if none provided
        if objective_function is None:
            def objective_function(t, x, v, a, u):
                return np.trapz(u**2, t)  # Default: minimize jerk

        return self.optimize(
            objective=objective_function,
            constraints=None,  # Will use external constraints
            primary_optimizer_id=primary_optimizer_id,
            secondary_constraints=secondary_constraints or {},
            secondary_relationships=secondary_relationships or {},
            optimization_targets=optimization_targets or {},
            processing_function=processing_function,
        )


    def _calculate_objective_value(self, objective: Callable,
                                 solution: Dict[str, np.ndarray]) -> Optional[float]:
        """Calculate the objective value for the solution."""
        try:
            # Extract arrays from solution
            t = solution.get("time", np.arange(len(solution.get("position", []))))
            x = solution.get("position", np.zeros_like(t))
            v = solution.get("velocity", np.zeros_like(t))
            a = solution.get("acceleration", np.zeros_like(t))
            u = solution.get("control", np.zeros_like(t))

            # Calculate objective value
            return float(objective(t, x, v, a, u))
        except Exception as e:
            log.warning(f"Could not calculate objective value: {e}")
            return None

    def get_available_primary_results(self) -> Dict[str, StorageResult]:
        """Get all available primary optimization results."""
        return self.registry.get_available_results(self.name)

    def get_optimizer_info(self) -> Dict[str, Any]:
        """Get information about the secondary optimizer."""
        return {
            "name": self.name,
            "configured": self._is_configured,
            "registry_stats": self.registry.get_registry_stats(),
            "collocation_info": self.collocation_optimizer.get_collocation_info(),
        }
