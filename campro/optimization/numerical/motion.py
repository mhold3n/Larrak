"""
Motion law optimization routines.

This module implements optimization for motion law problems using
various objective functions and constraint systems.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from campro.constraints.cam import CamMotionConstraints
from campro.constraints.motion import MotionConstraints
from campro.logging import get_logger
from campro.storage import OptimizationRegistry

from .base import BaseOptimizer, OptimizationResult, OptimizationStatus
from .collocation import CollocationOptimizer, CollocationSettings
from campro.optimization.utils.grid import GridSpec

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

    def __init__(
        self,
        settings: CollocationSettings | None = None,
        registry: OptimizationRegistry | None = None,
    ):
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

    # Mapping contract (universal grid exchange domain)
    def inputs_from_universal(
        self,
        universal_theta: np.ndarray,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        acceleration: np.ndarray | None = None,
        grid_spec: GridSpec | None = None,
    ) -> dict[str, np.ndarray]:
        """Map inputs from universal grid to the internal grid (default: passthrough).

        Motion law typically generates, not consumes, states; this is a placeholder
        to satisfy the stage contract and allow future non-uniform collocation.
        """
        # Future: if grid_spec.family != "uniform", prefer barycentric mapping here.
        return {
            "theta": universal_theta,
            "position": position if position is not None else np.zeros_like(universal_theta),
            "velocity": velocity if velocity is not None else np.zeros_like(universal_theta),
            "acceleration": acceleration
            if acceleration is not None
            else np.zeros_like(universal_theta),
        }

    def outputs_to_universal(
        self,
        theta_internal: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        universal_theta: np.ndarray,
        grid_spec: GridSpec | None = None,
    ) -> dict[str, np.ndarray]:
        """Map solver outputs to universal grid (default: passthrough when grids match)."""
        if theta_internal.shape == universal_theta.shape and np.allclose(
            theta_internal, universal_theta
        ):
            return {
                "theta": theta_internal,
                "position": position,
                "velocity": velocity,
                "acceleration": acceleration,
            }
        # Fallback simple resample via numpy interp for now; advanced mappers chosen from grid_spec later
        import numpy as _np

        def _per_resample(th, vals, tgt):
            two_pi = 2.0 * _np.pi
            th = _np.mod(th, two_pi)
            order = _np.argsort(th)
            th = th[order]
            vals = _np.asarray(vals)[order]
            th_ext = _np.concatenate([th, th[:1] + two_pi])
            vals_ext = _np.concatenate([vals, vals[:1]])
            tgt = _np.mod(tgt, two_pi)
            return _np.interp(tgt, th_ext, vals_ext)

        return {
            "theta": universal_theta,
            "position": _per_resample(theta_internal, position, universal_theta),
            "velocity": _per_resample(theta_internal, velocity, universal_theta),
            "acceleration": _per_resample(theta_internal, acceleration, universal_theta),
        }

    def optimize(
        self,
        objective: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
        constraints: MotionConstraints | CamMotionConstraints,
        initial_guess: dict[str, NDArray[np.float64]] | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
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

        # Do NOT stamp umbrella types like 'custom' onto constraints; this field is
        # read by downstream motion-law paths expecting concrete MotionType values.
        # Keep concrete types only; pass umbrella/custom via kwargs without mutation.
        if "objective_type" in kwargs and kwargs["objective_type"] not in {"custom"}:
            constraints.objective_type = kwargs["objective_type"]

        # Delegate to collocation optimizer
        return self.collocation_optimizer.optimize(
            objective,
            constraints,
            initial_guess,
            **kwargs,
        )

    def solve_minimum_time(
        self,
        constraints: MotionConstraints | CamMotionConstraints,
        distance: float,
        max_velocity: float,
        max_acceleration: float,
        max_jerk: float,
        time_horizon: float | None = None,
    ) -> OptimizationResult:
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
        def objective(
            t: np.ndarray,
            x: np.ndarray,
            v: np.ndarray,
            a: np.ndarray,
            u: np.ndarray,
        ) -> float:
            return float(t[-1])  # Minimize final time

        # Configure optimization parameters
        opt_params = {
            "distance": distance,
            "max_velocity": max_velocity,
            "max_acceleration": max_acceleration,
            "max_jerk": max_jerk,
            "objective_type": MotionObjectiveType.MINIMUM_TIME.value,
        }

        if time_horizon is not None:
            opt_params["time_horizon"] = time_horizon

        return self.optimize(objective, constraints, **opt_params)

    def solve_minimum_energy(
        self,
        constraints: MotionConstraints | CamMotionConstraints,
        distance: float,
        max_velocity: float,
        max_acceleration: float,
        max_jerk: float,
        time_horizon: float,
    ) -> OptimizationResult:
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
        def objective(
            t: np.ndarray,
            x: np.ndarray,
            v: np.ndarray,
            a: np.ndarray,
            u: np.ndarray,
        ) -> float:
            return float(np.trapz(u**2, t))  # Minimize energy (integral of jerk squared)

        # Configure optimization parameters
        opt_params = {
            "distance": distance,
            "max_velocity": max_velocity,
            "max_acceleration": max_acceleration,
            "max_jerk": max_jerk,
            "time_horizon": time_horizon,
            "objective_type": MotionObjectiveType.MINIMUM_ENERGY.value,
        }

        return self.optimize(objective, constraints, **opt_params)

    def solve_minimum_jerk(
        self,
        constraints: MotionConstraints | CamMotionConstraints,
        distance: float,
        max_velocity: float,
        max_acceleration: float,
        max_jerk: float,
        time_horizon: float | None = None,
    ) -> OptimizationResult:
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
        def objective(
            t: np.ndarray,
            x: np.ndarray,
            v: np.ndarray,
            a: np.ndarray,
            u: np.ndarray,
        ) -> float:
            return float(np.trapz(u**2, t))  # Minimize jerk (integral of jerk squared)

        # Configure optimization parameters
        opt_params = {
            "distance": distance,
            "max_velocity": max_velocity,
            "max_acceleration": max_acceleration,
            "max_jerk": max_jerk,
            "objective_type": MotionObjectiveType.MINIMUM_JERK.value,
        }

        if time_horizon is not None:
            opt_params["time_horizon"] = time_horizon

        return self.optimize(objective, constraints, **opt_params)

    def solve_custom_objective(
        self,
        objective_function: Callable[
            [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], float
        ],
        constraints: MotionConstraints | CamMotionConstraints,
        distance: float,
        time_horizon: float | None = None,
        **kwargs,
    ) -> OptimizationResult:
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
            "distance": distance,
            "objective_type": MotionObjectiveType.CUSTOM.value,
            **kwargs,
        }

        if time_horizon is not None:
            opt_params["time_horizon"] = time_horizon

        return self.optimize(objective_function, constraints, **opt_params)

    def solve_cam_motion_law(
        self,
        cam_constraints: CamMotionConstraints,
        motion_type: str = "minimum_jerk",
        cycle_time: float = 1.0,
        *,
        n_points: int | None = None,
    ) -> OptimizationResult:
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
            from .motion_law import MotionLawConstraints, MotionType
            from .motion_law_optimizer import MotionLawOptimizer

            # Convert cam constraints to motion law constraints
            motion_law_constraints = MotionLawConstraints(
                stroke=cam_constraints.stroke,
                upstroke_duration_percent=cam_constraints.upstroke_duration_percent,
                zero_accel_duration_percent=cam_constraints.zero_accel_duration_percent or 0.0,
                max_velocity=cam_constraints.max_velocity,
                max_acceleration=cam_constraints.max_acceleration,
                max_jerk=cam_constraints.max_jerk,
            )

            # Convert motion type string to enum
            motion_type_enum = MotionType(motion_type)

            # Create and configure motion law optimizer
            motion_optimizer = MotionLawOptimizer()
            # Set sampling points over 0..2π (default to universal or 360)
            motion_optimizer.n_points = int(n_points) if n_points is not None else 360

            # Solve motion law optimization
            result = motion_optimizer.solve_motion_law(
                motion_law_constraints,
                motion_type_enum,
            )

            # Log explicit free-piston idealization assumptions
            log.info(
                "Assumptions: constant temperature; ideal fuel load; 360-point angular sampling",
            )

            # Convert result to OptimizationResult format, attaching assumptions via metadata
            return OptimizationResult(
                status=OptimizationStatus.CONVERGED
                if result.convergence_status == "converged"
                else OptimizationStatus.FAILED,
                objective_value=result.objective_value,
                solution=result.to_dict(),
                iterations=result.iterations,
                solve_time=result.solve_time,
                metadata={
                    "assumptions": {
                        "constant_temperature": True,
                        "ideal_fuel_load": True,
                        "angular_sampling_points": 360,
                        "independent_variable": "cam_angle_radians",
                    },
                },
            )

        except Exception as e:
            log.error(f"Motion law optimization failed: {e}")
            raise RuntimeError("Motion law optimization failed") from e

    def store_result(
        self,
        result: OptimizationResult,
        optimizer_id: str = "motion_optimizer",
        metadata: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        optimization_rules: dict[str, Any] | None = None,
    ) -> None:
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
        metadata.update(
            {
                "objective_value": result.objective_value,
                "solve_time": result.solve_time,
                "iterations": result.iterations,
                "convergence_info": result.convergence_info,
            },
        )

        # Prepare optimization rules
        if optimization_rules is None:
            optimization_rules = {
                "motion_type": getattr(result, "motion_type", "unknown"),
                "collocation_method": self.collocation_optimizer.settings.method,
                "collocation_degree": self.collocation_optimizer.settings.degree,
                "tolerance": self.collocation_optimizer.settings.tolerance,
                "max_iterations": self.collocation_optimizer.settings.max_iterations,
            }

        # Prepare solver settings
        solver_settings = {
            "collocation_settings": {
                "method": self.collocation_optimizer.settings.method,
                "degree": self.collocation_optimizer.settings.degree,
                "tolerance": self.collocation_optimizer.settings.tolerance,
                "max_iterations": self.collocation_optimizer.settings.max_iterations,
                "verbose": self.collocation_optimizer.settings.verbose,
            },
        }

        # Store in registry with complete context
        self.registry.store_result(
            optimizer_id=optimizer_id,
            result_data=result.solution,
            metadata=metadata,
            constraints=constraints,
            optimization_rules=optimization_rules,
            solver_settings=solver_settings,
            expires_in=3600,  # Expire in 1 hour
        )

        log.info(f"Stored optimization result with complete context for {optimizer_id}")

    def get_optimizer_info(self) -> dict[str, Any]:
        """Get information about the motion optimizer."""
        return {
            "name": self.name,
            "configured": self._is_configured,
            "collocation_info": self.collocation_optimizer.get_collocation_info(),
            "registry_stats": self.registry.get_registry_stats(),
        }
