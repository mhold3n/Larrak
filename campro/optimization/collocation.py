"""
Collocation-based optimization methods.

This module implements optimization using direct collocation methods
with CasADi and Ipopt, supporting various collocation schemes.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from campro.constants import (
    COLLOCATION_METHODS,
    COLLOCATION_TOLERANCE,
    DEFAULT_COLLOCATION_DEGREE,
    DEFAULT_MAX_ITERATIONS,
)
from campro.logging import get_logger

from .base import BaseOptimizer, OptimizationResult


# Lightweight grid utilities to support litvin optimization without importing heavy deps
@dataclass(frozen=True)
class CollocationGrid:
    theta: Sequence[float]


def make_uniform_grid(n: int) -> CollocationGrid:
    import math

    n = max(8, int(n))
    theta = [2.0 * math.pi * i / n for i in range(n)]
    return CollocationGrid(theta=theta)


def central_diff(
    values: Sequence[float], h: float,
) -> tuple[Sequence[float], Sequence[float]]:
    n = len(values)
    d = [0.0] * n
    d2 = [0.0] * n
    for i in range(n):
        ip = (i + 1) % n
        im = (i - 1) % n
        d[i] = (values[ip] - values[im]) / (2.0 * h)
        d2[i] = (values[ip] - 2.0 * values[i] + values[im]) / (h * h)
    return d, d2


log = get_logger(__name__)


class CollocationMethod(Enum):
    """Available collocation methods."""

    LEGENDRE = "legendre"
    RADAU = "radau"
    LOBATTO = "lobatto"


@dataclass
class CollocationSettings:
    """Settings for collocation method."""

    degree: int = DEFAULT_COLLOCATION_DEGREE
    method: str = "legendre"
    tolerance: float = COLLOCATION_TOLERANCE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    verbose: bool = False

    def __post_init__(self):
        """Validate collocation settings."""
        if self.degree < 1:
            raise ValueError("Collocation degree must be at least 1")
        if self.method not in COLLOCATION_METHODS:
            raise ValueError(f"Unknown collocation method: {self.method}")
        if self.tolerance <= 0:
            raise ValueError("Tolerance must be positive")
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")


class CollocationOptimizer(BaseOptimizer):
    """
    Optimizer using direct collocation methods with CasADi and Ipopt.

    This optimizer implements various collocation schemes for solving
    optimal control problems with high accuracy.
    """

    def __init__(self, settings: CollocationSettings | None = None):
        super().__init__("CollocationOptimizer")
        self.settings = settings or CollocationSettings()
        self._is_configured = True

    def configure(self, **kwargs) -> None:
        """
        Configure the collocation optimizer.

        Args:
            **kwargs: Configuration parameters
                - degree: Collocation degree
                - method: Collocation method
                - tolerance: Solver tolerance
                - max_iterations: Maximum iterations
                - verbose: Verbose output
        """
        if "degree" in kwargs:
            self.settings.degree = kwargs["degree"]
        if "method" in kwargs:
            self.settings.method = kwargs["method"]
        if "tolerance" in kwargs:
            self.settings.tolerance = kwargs["tolerance"]
        if "max_iterations" in kwargs:
            self.settings.max_iterations = kwargs["max_iterations"]
        if "verbose" in kwargs:
            self.settings.verbose = kwargs["verbose"]

        # Validate settings
        self.settings.__post_init__()
        self._is_configured = True

        log.info(
            f"Configured collocation optimizer: {self.settings.method}, degree {self.settings.degree}",
        )

    def optimize(
        self,
        objective: Callable,
        constraints: Any,
        initial_guess: dict[str, np.ndarray] | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Solve an optimization problem using collocation.

        Args:
            objective: Objective function to minimize
            constraints: Constraint system
            initial_guess: Initial guess for optimization variables
            **kwargs: Additional optimization parameters
                - time_horizon: Time horizon for optimization
                - n_points: Number of collocation points

        Returns:
            OptimizationResult object
        """
        self._validate_inputs(objective, constraints)

        result = self._start_optimization()
        result.solve_time = time.time()

        try:
            # Extract optimization parameters
            time_horizon = kwargs.get("time_horizon", 1.0)
            n_points = kwargs.get("n_points", 100)

            # Create collocation problem (avoid passing duplicates in kwargs)
            forwarding_kwargs = dict(kwargs)
            forwarding_kwargs.pop("time_horizon", None)
            forwarding_kwargs.pop("n_points", None)
            solution = self._solve_collocation_problem(
                objective,
                constraints,
                time_horizon,
                n_points,
                initial_guess,
                **forwarding_kwargs,
            )

            # Calculate objective value
            objective_value = self._calculate_objective_value(objective, solution)

            # Finish optimization
            result = self._finish_optimization(
                result,
                solution,
                objective_value,
                convergence_info={
                    "method": self.settings.method,
                    "degree": self.settings.degree,
                },
            )

        except Exception as e:
            error_message = f"Collocation optimization failed: {e!s}"
            log.error(error_message)
            result = self._finish_optimization(result, {}, error_message=error_message)

        return result

    def _solve_collocation_problem(
        self,
        objective: Callable,
        constraints: Any,
        time_horizon: float,
        n_points: int,
        initial_guess: dict[str, np.ndarray] | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """
        Solve the collocation problem.

        This implementation uses the new motion law optimizer for motion law problems
        and falls back to analytical solutions for other problems.
        """
        log.info(
            f"Solving collocation problem: {self.settings.method}, degree {self.settings.degree}",
        )

        # Allow generic constraints to supply their own collocation problem
        if hasattr(constraints, "build_collocation_problem") and callable(
            constraints.build_collocation_problem,
        ):
            return constraints.build_collocation_problem(
                objective, time_horizon, n_points, initial_guess,
            )
        # Otherwise, check if this is a motion law problem
        if hasattr(constraints, "stroke") and hasattr(
            constraints, "upstroke_duration_percent",
        ):
            # This is a motion law problem - use the new motion law optimizer
            return self._solve_motion_law_problem(constraints, time_horizon, n_points, **kwargs)
        # Detect generic MotionConstraints (hard feasibility + optional golden tracking)
        try:
            from campro.constraints.motion import MotionConstraints as _MC
        except Exception:  # pragma: no cover - defensive import
            _MC = None  # type: ignore

        if _MC is not None and isinstance(constraints, _MC):
            return self._solve_motion_constraints_problem(
                constraints,
                time_horizon,
                n_points,
                initial_guess,
                golden_profile=kwargs.get("golden_profile") if isinstance(kwargs, dict) else None,  # type: ignore[name-defined]
                tracking_weight=kwargs.get("tracking_weight", 1.0) if isinstance(kwargs, dict) else 1.0,  # type: ignore[name-defined]
            )
        # For non-motion law problems, raise an error to force proper implementation
        raise NotImplementedError(
            f"Collocation optimization for problem type {type(constraints).__name__} "
            f"is not yet implemented. Only motion law problems are supported.",
        )

    def _solve_motion_law_problem(
        self, constraints: Any, time_horizon: float, n_points: int, **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        """
        Solve motion law problem using the new motion law optimizer.

        Args:
            constraints: Motion law constraints
            time_horizon: Time horizon (cycle time)
            n_points: Number of points

        Returns:
            Dictionary containing motion law solution
        """
        try:
            # Import the new motion law optimizer
            from .motion_law import MotionLawConstraints, MotionType
            from .motion_law_optimizer import MotionLawOptimizer

            # Convert constraints to MotionLawConstraints
            motion_constraints = MotionLawConstraints(
                stroke=constraints.stroke,
                upstroke_duration_percent=constraints.upstroke_duration_percent,
                zero_accel_duration_percent=constraints.zero_accel_duration_percent,
                max_velocity=getattr(constraints, "max_velocity", None),
                max_acceleration=getattr(constraints, "max_acceleration", None),
                max_jerk=getattr(constraints, "max_jerk", None),
            )

            # Get motion type preference: prefer explicit kwarg (from GUI/settings),
            # otherwise fall back to any legacy field on constraints.
            motion_type_str = (
                kwargs.get("motion_type")
                if isinstance(kwargs, dict) and "motion_type" in kwargs
                else getattr(constraints, "objective_type", "minimum_jerk")
            )
            motion_type = MotionType(str(motion_type_str))

            # Create and configure motion law optimizer
            motion_optimizer = MotionLawOptimizer()
            motion_optimizer.n_points = n_points

            # Solve motion law optimization
            result = motion_optimizer.solve_motion_law(motion_constraints, motion_type)

            # Convert result to expected format
            solution = {
                "time": np.linspace(
                    0, time_horizon, n_points,
                ),  # Keep time for compatibility
                "position": result.position,
                "velocity": result.velocity,
                "acceleration": result.acceleration,
                "control": result.jerk,  # 'control' is jerk in collocation
                "cam_angle": result.cam_angle,  # Add cam angle for new format
                "jerk": result.jerk,  # Add jerk for new format
            }

            log.info(f"Motion law optimization completed: {result.convergence_status}")
            return solution

        except Exception as e:
            log.error(f"Motion law optimization failed: {e}")
            # Re-raise the exception instead of falling back to fake solutions
            raise RuntimeError(f"Motion law optimization failed: {e}") from e

    def _solve_motion_constraints_problem(
        self,
        constraints: Any,
        time_horizon: float,
        n_points: int,
        initial_guess: dict[str, np.ndarray] | None = None,
        *,
        golden_profile: dict[str, np.ndarray] | None = None,
        tracking_weight: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """
        Solve generic MotionConstraints with hard feasibility and soft tracking to a golden profile.

        Notes:
        - This uses direct transcription with piecewise-constant jerk over a uniform grid.
        - GUI-selected method (Lobatto/Radau) is accepted via settings and will control
          future higher-degree collocation; for now, dynamics use constant-jerk segments.
        - Golden profile is expected in radial follower coordinates (position/velocity/accel arrays).
        """
        import casadi as ca  # Local import to avoid global dependency at module import time

        dt = float(time_horizon) / max(1, int(n_points) - 1)
        N = max(2, int(n_points))

        opti = ca.Opti()

        # Decision variables (vectors length N)
        p = opti.variable(N)  # position
        v = opti.variable(N)  # velocity
        a = opti.variable(N)  # acceleration
        j = opti.variable(N - 1)  # piecewise-constant jerk per interval

        # Initial guess
        if initial_guess:
            if "position" in initial_guess:
                opti.set_initial(p, np.asarray(initial_guess["position"]))
            if "velocity" in initial_guess:
                opti.set_initial(v, np.asarray(initial_guess["velocity"]))
            if "acceleration" in initial_guess:
                opti.set_initial(a, np.asarray(initial_guess["acceleration"]))

        # Hard path bounds
        if getattr(constraints, "position_bounds", None) is not None:
            lb, ub = constraints.position_bounds
            opti.subject_to(p >= lb)
            opti.subject_to(p <= ub)
        if getattr(constraints, "velocity_bounds", None) is not None:
            lb, ub = constraints.velocity_bounds
            opti.subject_to(v >= lb)
            opti.subject_to(v <= ub)
        if getattr(constraints, "acceleration_bounds", None) is not None:
            lb, ub = constraints.acceleration_bounds
            opti.subject_to(a >= lb)
            opti.subject_to(a <= ub)
        if getattr(constraints, "jerk_bounds", None) is not None:
            lb, ub = constraints.jerk_bounds
            opti.subject_to(j >= lb)
            opti.subject_to(j <= ub)

        # Boundary conditions (if provided)
        if getattr(constraints, "initial_position", None) is not None:
            opti.subject_to(p[0] == float(constraints.initial_position))
        if getattr(constraints, "initial_velocity", None) is not None:
            opti.subject_to(v[0] == float(constraints.initial_velocity))
        if getattr(constraints, "initial_acceleration", None) is not None:
            opti.subject_to(a[0] == float(constraints.initial_acceleration))
        if getattr(constraints, "final_position", None) is not None:
            opti.subject_to(p[-1] == float(constraints.final_position))
        if getattr(constraints, "final_velocity", None) is not None:
            opti.subject_to(v[-1] == float(constraints.final_velocity))
        if getattr(constraints, "final_acceleration", None) is not None:
            opti.subject_to(a[-1] == float(constraints.final_acceleration))

        # Discrete dynamics with piecewise-constant jerk
        # v[k+1] = v[k] + dt * a[k]
        # a[k+1] = a[k] + dt * j[k]
        # p[k+1] = p[k] + dt * v[k] + 0.5 * dt^2 * a[k]
        for k in range(N - 1):
            opti.subject_to(v[k + 1] == v[k] + dt * a[k])
            opti.subject_to(a[k + 1] == a[k] + dt * j[k])
            opti.subject_to(p[k + 1] == p[k] + dt * v[k] + 0.5 * (dt * dt) * a[k])

        # Tracking objective to golden profile (position/velocity/acceleration)
        obj = 0
        if golden_profile is not None:
            gp_pos = np.asarray(golden_profile.get("position")) if "position" in golden_profile else None
            gp_vel = np.asarray(golden_profile.get("velocity")) if "velocity" in golden_profile else None
            gp_acc = np.asarray(golden_profile.get("acceleration")) if "acceleration" in golden_profile else None

            # Ensure shapes match N; if not, interpolate to our grid
            t = np.linspace(0.0, float(time_horizon), N)
            if gp_pos is not None and gp_pos.shape[0] != N:
                # Interpolate using numpy (piecewise linear is acceptable for tracking)
                gp_t = np.linspace(0.0, float(time_horizon), gp_pos.shape[0])
                gp_pos = np.interp(t, gp_t, gp_pos)
            if gp_vel is not None and gp_vel.shape[0] != N:
                gp_t = np.linspace(0.0, float(time_horizon), gp_vel.shape[0])
                gp_vel = np.interp(t, gp_t, gp_vel)
            if gp_acc is not None and gp_acc.shape[0] != N:
                gp_t = np.linspace(0.0, float(time_horizon), gp_acc.shape[0])
                gp_acc = np.interp(t, gp_t, gp_acc)

            if gp_pos is not None:
                obj = obj + tracking_weight * ca.sumsqr(p - gp_pos)
            if gp_vel is not None:
                # Smaller weight for velocity tracking unless specified by caller
                obj = obj + 0.2 * tracking_weight * ca.sumsqr(v - gp_vel)
            if gp_acc is not None:
                obj = obj + 0.1 * tracking_weight * ca.sumsqr(a - gp_acc)

        # Mild regularization on jerk to keep smoothness if no golden is provided
        obj = obj + 1e-6 * ca.sumsqr(j)

        opti.minimize(obj)

        # Solver options
        p_opts = {"expand": True}
        s_opts = {
            "print_time": False,
            "ipopt": {
                "tol": float(self.settings.tolerance),
                "max_iter": int(self.settings.max_iterations),
                # Keep linear solver selection consistent with environment; ma27 will be used if available
                # The GUI method (Lobatto/Radau) is currently a global toggle and will govern future higher-degree schemes.
            },
        }
        opti.solver("ipopt", p_opts, s_opts)

        sol = opti.solve()

        p_val = np.asarray(sol.value(p)).reshape(-1)
        v_val = np.asarray(sol.value(v)).reshape(-1)
        a_val = np.asarray(sol.value(a)).reshape(-1)

        return {
            "time": np.linspace(0.0, float(time_horizon), N),
            "position": p_val,
            "velocity": v_val,
            "acceleration": a_val,
            "control": np.diff(a_val) / dt,
        }

    def _generate_analytical_solution(
        self, t: np.ndarray, constraints: Any, time_horizon: float,
    ) -> dict[str, np.ndarray]:
        """
        Generate analytical solutions based on motion type.

        This creates different motion profiles based on the objective type.
        """
        n_points = len(t)

        # Get motion parameters
        if hasattr(constraints, "stroke"):
            distance = constraints.stroke
        else:
            distance = 20.0  # Default stroke

        # Get timing parameters from constraints
        upstroke_duration_percent = getattr(
            constraints, "upstroke_duration_percent", 60.0,
        )
        zero_accel_duration_percent = getattr(
            constraints, "zero_accel_duration_percent", 0.0,
        )

        # Get objective type from constraints or use default
        objective_type = getattr(constraints, "objective_type", "minimum_jerk")

        if objective_type == "minimum_jerk":
            return self._generate_minimum_jerk_profile(
                t,
                distance,
                time_horizon,
                upstroke_duration_percent,
                zero_accel_duration_percent,
            )
        if objective_type == "minimum_time":
            return self._generate_minimum_time_profile(
                t,
                distance,
                time_horizon,
                upstroke_duration_percent,
                zero_accel_duration_percent,
            )
        if objective_type == "minimum_energy":
            return self._generate_minimum_energy_profile(
                t,
                distance,
                time_horizon,
                upstroke_duration_percent,
                zero_accel_duration_percent,
            )
        # Default to minimum jerk
        return self._generate_minimum_jerk_profile(
            t,
            distance,
            time_horizon,
            upstroke_duration_percent,
            zero_accel_duration_percent,
        )

    def _generate_minimum_jerk_profile(
        self,
        t: np.ndarray,
        distance: float,
        time_horizon: float,
        upstroke_duration_percent: float,
        zero_accel_duration_percent: float,
    ) -> dict[str, np.ndarray]:
        """Generate a minimum jerk motion profile (smooth S-curve) with user-specified timing."""
        n_points = len(t)

        # Calculate timing segments
        upstroke_time = time_horizon * upstroke_duration_percent / 100.0
        downstroke_time = time_horizon - upstroke_time
        zero_accel_time = time_horizon * zero_accel_duration_percent / 100.0

        # For minimum jerk, we'll create a smooth profile that respects the upstroke timing
        # Normalize time to [0, 1] for the upstroke portion
        position = np.zeros_like(t)
        velocity = np.zeros_like(t)
        acceleration = np.zeros_like(t)
        jerk = np.zeros_like(t)

        for i, time in enumerate(t):
            if time <= upstroke_time:
                # Upstroke phase - use minimum jerk profile
                tau = time / upstroke_time  # Normalize to [0, 1] for upstroke

                # Minimum jerk profile: smooth S-curve
                position[i] = distance * (6 * tau**5 - 15 * tau**4 + 10 * tau**3)
                velocity[i] = (distance / upstroke_time) * (
                    30 * tau**4 - 60 * tau**3 + 30 * tau**2
                )
                acceleration[i] = (distance / upstroke_time**2) * (
                    120 * tau**3 - 180 * tau**2 + 60 * tau
                )
                jerk[i] = (distance / upstroke_time**3) * (
                    360 * tau**2 - 360 * tau + 60
                )
            else:
                # Downstroke phase - mirror the upstroke
                downstroke_tau = (
                    time - upstroke_time
                ) / downstroke_time  # Normalize to [0, 1] for downstroke

                # Mirror the upstroke profile
                position[i] = distance * (
                    1
                    - (
                        6 * downstroke_tau**5
                        - 15 * downstroke_tau**4
                        + 10 * downstroke_tau**3
                    )
                )
                velocity[i] = -(distance / downstroke_time) * (
                    30 * downstroke_tau**4
                    - 60 * downstroke_tau**3
                    + 30 * downstroke_tau**2
                )
                acceleration[i] = -(distance / downstroke_time**2) * (
                    120 * downstroke_tau**3
                    - 180 * downstroke_tau**2
                    + 60 * downstroke_tau
                )
                jerk[i] = -(distance / downstroke_time**3) * (
                    360 * downstroke_tau**2 - 360 * downstroke_tau + 60
                )

        return {
            "time": t,
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "control": jerk,
        }

    def _generate_minimum_time_profile(
        self,
        t: np.ndarray,
        distance: float,
        time_horizon: float,
        upstroke_duration_percent: float,
        zero_accel_duration_percent: float,
    ) -> dict[str, np.ndarray]:
        """Generate a minimum time motion profile (bang-bang control) with user-specified timing."""
        n_points = len(t)

        # Calculate timing segments based on user inputs
        upstroke_time = time_horizon * upstroke_duration_percent / 100.0
        downstroke_time = time_horizon - upstroke_time
        zero_accel_time = time_horizon * zero_accel_duration_percent / 100.0

        # For minimum time, use trapezoidal velocity profile
        # Split upstroke into acceleration and constant velocity phases
        t_accel_up = upstroke_time * 0.4  # 40% of upstroke for acceleration
        t_const_up = upstroke_time * 0.6  # 60% of upstroke for constant velocity

        # Split downstroke into constant velocity and deceleration phases
        t_const_down = downstroke_time * 0.6  # 60% of downstroke for constant velocity
        t_decel_down = downstroke_time * 0.4  # 40% of downstroke for deceleration

        # Calculate maximum velocity to achieve the stroke
        total_distance = (
            0.5 * t_accel_up + t_const_up + t_const_down + 0.5 * t_decel_down
        )
        max_velocity = distance / total_distance

        # Generate position, velocity, acceleration
        position = np.zeros_like(t)
        velocity = np.zeros_like(t)
        acceleration = np.zeros_like(t)

        for i, time in enumerate(t):
            if time <= t_accel_up:
                # Upstroke acceleration phase
                velocity[i] = max_velocity * (time / t_accel_up)
                position[i] = 0.5 * max_velocity * time**2 / t_accel_up
                acceleration[i] = max_velocity / t_accel_up
            elif time <= t_accel_up + t_const_up:
                # Upstroke constant velocity phase
                velocity[i] = max_velocity
                position[i] = 0.5 * max_velocity * t_accel_up + max_velocity * (
                    time - t_accel_up
                )
                acceleration[i] = 0
            elif time <= t_accel_up + t_const_up + t_const_down:
                # Downstroke constant velocity phase
                velocity[i] = max_velocity
                position[i] = (
                    0.5 * max_velocity * t_accel_up
                    + max_velocity * t_const_up
                    + max_velocity * (time - t_accel_up - t_const_up)
                )
                acceleration[i] = 0
            else:
                # Downstroke deceleration phase
                decel_time = time - t_accel_up - t_const_up - t_const_down
                velocity[i] = max_velocity * (1 - decel_time / t_decel_down)
                position[i] = (
                    0.5 * max_velocity * t_accel_up
                    + max_velocity * t_const_up
                    + max_velocity * t_const_down
                    + max_velocity * decel_time
                    - 0.5 * max_velocity * decel_time**2 / t_decel_down
                )
                acceleration[i] = -max_velocity / t_decel_down

        # Calculate jerk (derivative of acceleration)
        jerk = np.gradient(acceleration, t)

        return {
            "time": t,
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "control": jerk,
        }

    def _generate_minimum_energy_profile(
        self,
        t: np.ndarray,
        distance: float,
        time_horizon: float,
        upstroke_duration_percent: float,
        zero_accel_duration_percent: float,
    ) -> dict[str, np.ndarray]:
        """Generate a minimum energy motion profile (smooth acceleration) with user-specified timing."""
        n_points = len(t)

        # Calculate timing segments
        upstroke_time = time_horizon * upstroke_duration_percent / 100.0
        downstroke_time = time_horizon - upstroke_time
        zero_accel_time = time_horizon * zero_accel_duration_percent / 100.0

        # For minimum energy, we'll create a smooth profile that respects the upstroke timing
        position = np.zeros_like(t)
        velocity = np.zeros_like(t)
        acceleration = np.zeros_like(t)
        jerk = np.zeros_like(t)

        for i, time in enumerate(t):
            if time <= upstroke_time:
                # Upstroke phase - use minimum energy profile
                tau = time / upstroke_time  # Normalize to [0, 1] for upstroke

                # Minimum energy profile: smooth acceleration curve
                position[i] = distance * (3 * tau**2 - 2 * tau**3)
                velocity[i] = (distance / upstroke_time) * (6 * tau - 6 * tau**2)
                acceleration[i] = (distance / upstroke_time**2) * (6 - 12 * tau)
                jerk[i] = -12 * distance / upstroke_time**3
            else:
                # Downstroke phase - mirror the upstroke
                downstroke_tau = (
                    time - upstroke_time
                ) / downstroke_time  # Normalize to [0, 1] for downstroke

                # Mirror the upstroke profile
                position[i] = distance * (
                    1 - (3 * downstroke_tau**2 - 2 * downstroke_tau**3)
                )
                velocity[i] = -(distance / downstroke_time) * (
                    6 * downstroke_tau - 6 * downstroke_tau**2
                )
                acceleration[i] = -(distance / downstroke_time**2) * (
                    6 - 12 * downstroke_tau
                )
                jerk[i] = 12 * distance / downstroke_time**3

        return {
            "time": t,
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "control": jerk,
        }

    def _calculate_objective_value(
        self, objective: Callable, solution: dict[str, np.ndarray],
    ) -> float | None:
        """Calculate the objective value for the solution."""
        try:
            # Prefer jerk-squared integral when available (common for smoothness minimization)
            if "jerk" in solution:
                theta = solution.get("cam_angle")
                if theta is None:
                    # Fall back to time if cam angle missing
                    theta = solution.get("time")
                if theta is not None:
                    j = np.asarray(solution["jerk"])
                    return float(np.trapz(j * j, theta))
            # If custom objective callable is provided, evaluate it
            if callable(objective):
                return float(objective(solution))
            # As a last resort, return None
            return None
        except Exception as e:
            log.warning(f"Could not calculate objective value: {e}")
            return None

    def get_collocation_info(self) -> dict[str, Any]:
        """Get information about the collocation method."""
        return {
            "method": self.settings.method,
            "degree": self.settings.degree,
            "tolerance": self.settings.tolerance,
            "max_iterations": self.settings.max_iterations,
            "verbose": self.settings.verbose,
        }
