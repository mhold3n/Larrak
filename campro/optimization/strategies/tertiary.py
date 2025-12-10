"""
Tertiary collocation optimizer for motion law and linkage tuning.

This module implements a third optimization layer that can access the complete
optimization context from previous layers, including results, constraints,
and optimization rules. This enables robust tuning of initial motion laws
and follower linkage placement relative to cam center.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from campro.logging import get_logger
from campro.storage import OptimizationRegistry

from campro.optimization.numerical.base import BaseOptimizer, OptimizationResult
from campro.optimization.numerical.collocation import CollocationOptimizer, CollocationSettings
from campro.optimization.utils.grid import GridSpec

log = get_logger(__name__)


@dataclass
class LinkageParameters:
    """Parameters for follower linkage placement and geometry."""

    # Cam center position
    cam_center_x: float = 0.0
    cam_center_y: float = 0.0

    # Follower linkage center position
    follower_center_x: float = 0.0
    follower_center_y: float = 0.0

    # Linkage geometry
    linkage_length: float = 50.0
    linkage_angle: float = 0.0

    # Follower geometry
    follower_radius: float = 5.0
    follower_offset: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cam_center_x": self.cam_center_x,
            "cam_center_y": self.cam_center_y,
            "follower_center_x": self.follower_center_x,
            "follower_center_y": self.follower_center_y,
            "linkage_length": self.linkage_length,
            "linkage_angle": self.linkage_angle,
            "follower_radius": self.follower_radius,
            "follower_offset": self.follower_offset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinkageParameters:
        """Create from dictionary."""
        return cls(**data)


class TertiaryOptimizer(BaseOptimizer):
    """
    Tertiary collocation optimizer shell for advanced optimization.

    This is a generic shell that can perform tertiary optimization tasks
    based on externally provided constraints, relationships, and optimization targets.
    The specific implementation details are passed in during optimization.
    It has full visibility into the optimization chain, accessing results,
    constraints, and optimization rules from previous layers.
    """

    def __init__(
        self,
        name: str = "TertiaryOptimizer",
        registry: OptimizationRegistry | None = None,
        settings: CollocationSettings | None = None,
    ):
        super().__init__(name)
        self.registry = registry or OptimizationRegistry()
        self.collocation_optimizer = CollocationOptimizer(settings)
        self._is_configured = True

    # Mapping contract (universal grid exchange domain)
    def inputs_from_universal(
        self,
        universal_theta: list[float] | None = None,
        primary_data: dict | None = None,
        secondary_data: dict | None = None,
        grid_spec: GridSpec | None = None,
    ) -> dict:
        """Prepare inputs from universal grid (default: passthrough).

        grid_spec provides hp metadata for internal solvers if needed later.
        """
        return {
            "theta": universal_theta,
            "primary": primary_data or {},
            "secondary": secondary_data or {},
        }

    def outputs_to_universal(self, result: dict, grid_spec: GridSpec | None = None) -> dict:
        """Return results (tertiary is largely parametric; passthrough)."""
        return result

    def configure(self, **kwargs) -> None:
        """
        Configure the tertiary optimizer.

        Args:
            **kwargs: Configuration parameters
                - registry: Optimization registry
                - collocation_settings: Collocation settings
                - linkage_parameters: Initial linkage parameters
        """
        if "registry" in kwargs:
            self.registry = kwargs["registry"]

        if "collocation_settings" in kwargs:
            self.collocation_optimizer.configure(**kwargs["collocation_settings"])

        if "linkage_parameters" in kwargs:
            self.linkage_parameters = kwargs["linkage_parameters"]
        else:
            self.linkage_parameters = LinkageParameters()

        self._is_configured = True
        log.info(f"Configured tertiary optimizer: {self.name}")

    def optimize(
        self,
        objective: Callable,
        constraints: Any,
        initial_guess: dict[str, np.ndarray] | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Solve a tertiary optimization problem with full context visibility and external specifications.

        Args:
            objective: Objective function to minimize
            constraints: Constraint system
            initial_guess: Initial guess for optimization variables
            **kwargs: Additional optimization parameters
                - primary_optimizer_id: ID of primary optimizer to use
                - secondary_optimizer_id: ID of secondary optimizer to use
                - tertiary_constraints: Specific constraints for tertiary optimization
                - tertiary_relationships: Relationships between all optimization layers
                - optimization_targets: Specific targets for tertiary optimization
                - processing_function: Function to process optimization context

        Returns:
            OptimizationResult object
        """
        self._validate_inputs(objective, constraints)

        result = self._start_optimization()
        result.solve_time = time.time()

        try:
            # Get complete optimization context
            optimization_context = self._get_complete_optimization_context(**kwargs)

            log.info("Using complete optimization context for tertiary optimization")
            log.info(
                f"  - Primary results: {list(optimization_context['primary_results'].keys())}",
            )
            log.info(
                f"  - Secondary results: {list(optimization_context['secondary_results'].keys())}",
            )

            # Extract external specifications
            tertiary_constraints = kwargs.get("tertiary_constraints", {})
            tertiary_relationships = kwargs.get("tertiary_relationships", {})
            optimization_targets = kwargs.get("optimization_targets", {})
            processing_function = kwargs.get("processing_function")

            # Process optimization context using external specifications
            if processing_function is not None:
                processed_solution = processing_function(
                    optimization_context,
                    tertiary_constraints,
                    tertiary_relationships,
                    optimization_targets,
                    **kwargs,
                )
            else:
                # Default: return primary solution unchanged (no processing)
                primary_results = optimization_context["primary_results"]
                if primary_results:
                    processed_solution = list(primary_results.values())[0].data.copy()
                else:
                    processed_solution = {}
                log.warning(
                    "No processing function provided - returning primary solution unchanged",
                )

            # Calculate objective value
            objective_value = self._calculate_objective_value(
                objective,
                processed_solution,
            )

            # Finish optimization
            result = self._finish_optimization(
                result,
                processed_solution,
                objective_value,
                convergence_info={
                    "tertiary_constraints": tertiary_constraints,
                    "tertiary_relationships": tertiary_relationships,
                    "optimization_targets": optimization_targets,
                    "processing_function": processing_function.__name__
                    if processing_function
                    else None,
                    "context_optimizers": list(
                        optimization_context["primary_results"].keys(),
                    )
                    + list(optimization_context["secondary_results"].keys()),
                    "linkage_parameters": getattr(
                        self,
                        "linkage_parameters",
                        {},
                    ).to_dict()
                    if hasattr(self, "linkage_parameters")
                    else {},
                },
            )

        except Exception as e:
            error_message = f"Tertiary optimization failed: {e!s}"
            log.error(error_message)
            result = self._finish_optimization(result, {}, error_message=error_message)

        return result

    def process_optimization_context(
        self,
        primary_optimizer_id: str = "motion_optimizer",
        secondary_optimizer_id: str = "secondary_optimizer",
        tertiary_constraints: dict[str, Any] | None = None,
        tertiary_relationships: dict[str, Any] | None = None,
        optimization_targets: dict[str, Any] | None = None,
        processing_function: Callable | None = None,
        objective_function: Callable | None = None,
    ) -> OptimizationResult:
        """
        Process optimization context using external specifications.

        Args:
            primary_optimizer_id: ID of primary optimizer
            secondary_optimizer_id: ID of secondary optimizer
            tertiary_constraints: Specific constraints for tertiary optimization
            tertiary_relationships: Relationships between all optimization layers
            optimization_targets: Specific targets for tertiary optimization
            processing_function: Function to process optimization context
            objective_function: Objective function for tertiary optimization

        Returns:
            OptimizationResult object
        """
        log.info(
            f"Processing optimization context from '{primary_optimizer_id}' and '{secondary_optimizer_id}' with external specifications",
        )

        # Use default objective if none provided
        if objective_function is None:

            def objective_function(t, x, v, a, u):
                return np.trapz(u**2, t)  # Default: minimize jerk

        return self.optimize(
            objective=objective_function,
            constraints=None,  # Will use external constraints
            primary_optimizer_id=primary_optimizer_id,
            secondary_optimizer_id=secondary_optimizer_id,
            tertiary_constraints=tertiary_constraints or {},
            tertiary_relationships=tertiary_relationships or {},
            optimization_targets=optimization_targets or {},
            processing_function=processing_function,
        )

    def _get_complete_optimization_context(self, **kwargs) -> dict[str, Any]:
        """
        Get complete optimization context including results, constraints, and rules.

        Args:
            **kwargs: Optimization parameters

        Returns:
            Dictionary containing complete optimization context
        """
        primary_optimizer_id = kwargs.get("primary_optimizer_id", "motion_optimizer")
        secondary_optimizer_id = kwargs.get(
            "secondary_optimizer_id",
            "secondary_optimizer",
        )

        # Get primary optimization context
        primary_result = self.registry.get_result(primary_optimizer_id)
        if primary_result is None:
            raise ValueError(
                f"No primary result found for optimizer '{primary_optimizer_id}'",
            )

        # Get secondary optimization context
        secondary_result = self.registry.get_result(secondary_optimizer_id)

        # Build complete context
        context = {
            "primary_results": {primary_optimizer_id: primary_result},
            "secondary_results": {secondary_optimizer_id: secondary_result}
            if secondary_result
            else {},
            "primary_constraints": primary_result.constraints,
            "primary_rules": primary_result.optimization_rules,
            "primary_settings": primary_result.solver_settings,
            "secondary_constraints": secondary_result.constraints if secondary_result else None,
            "secondary_rules": secondary_result.optimization_rules if secondary_result else None,
            "secondary_settings": secondary_result.solver_settings if secondary_result else None,
        }

        # Add linkage parameters if available
        if hasattr(self, "linkage_parameters"):
            context["linkage_parameters"] = self.linkage_parameters.to_dict()
        else:
            context["linkage_parameters"] = {}

        return context

    def _calculate_objective_value(
        self,
        objective: Callable,
        solution: dict[str, np.ndarray],
    ) -> float | None:
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

    def _validate_inputs(self, objective: Callable, constraints: Any) -> None:
        """Validate optimization inputs for tertiary optimization."""
        if not callable(objective):
            raise TypeError("Objective must be callable")

        # Constraints can be None for tertiary optimization as we use context
        # if constraints is None:
        #     raise ValueError("Constraints cannot be None")

        if not self._is_configured:
            raise RuntimeError(
                f"Optimizer {self.name} is not configured. Call configure() first.",
            )

    def get_complete_context(
        self,
        primary_optimizer_id: str = "motion_optimizer",
        secondary_optimizer_id: str = "secondary_optimizer",
    ) -> dict[str, Any]:
        """Get complete optimization context for analysis."""
        return self._get_complete_optimization_context(
            primary_optimizer_id=primary_optimizer_id,
            secondary_optimizer_id=secondary_optimizer_id,
        )

    def get_optimizer_info(self) -> dict[str, Any]:
        """Get information about the tertiary optimizer."""
        return {
            "name": self.name,
            "configured": self._is_configured,
            "registry_stats": self.registry.get_registry_stats(),
            "collocation_info": self.collocation_optimizer.get_collocation_info(),
            "linkage_parameters": self.linkage_parameters.to_dict(),
        }
