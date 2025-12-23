"""
Motion law optimizer using proper collocation methods.

This module implements real optimization for motion law generation using
collocation methods with proper constraint handling.

NOTE: This module uses scipy.optimize.minimize and may be legacy code.
The main optimization flow (phases 1, 2, 3) uses CasADi/IPOPT via:
- Phase 1: FreePistonPhase1Adapter
- Phase 2: CamRingOptimizer (Litvin optimization with CasADi/IPOPT)
- Phase 3: CrankCenterOptimizer (currently uses scipy, may need conversion)

This optimizer is still referenced in motion.py and collocation.py but may
not be part of the primary optimization flow.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import casadi as ca
import numpy as np
from casadi import MX, Opti

from campro.logging import get_logger
from campro.utils import format_duration

from .base import BaseOptimizer, OptimizationResult, OptimizationStatus


def _filter_duplicate_points(
    x: np.ndarray, y: np.ndarray, min_spacing: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter duplicate or very close points from control point arrays.

    This prevents divide-by-zero warnings in scipy.interpolate.CubicSpline
    which uses finite differences for derivative computation.

    Args:
        x: Control point x-coordinates (must be monotonic)
        y: Control point y-coordinates
        min_spacing: Minimum spacing between consecutive x values

    Returns:
        Tuple of (filtered_x, filtered_y)
    """
    if len(x) < 2:
        return x, y

    # Find points that are sufficiently spaced
    unique_mask = np.concatenate([[True], np.diff(x) > min_spacing])

    # Always keep the last point to preserve boundary conditions
    if not unique_mask[-1]:
        unique_mask[-1] = True

    return x[unique_mask], y[unique_mask]


from .motion_law import (
    MotionLawConstraints,
    MotionLawResult,
    MotionLawValidator,
    MotionType,
)

log = get_logger(__name__)


class MotionLawOptimizer(BaseOptimizer):
    """
    Optimizer for motion law problems using collocation methods.

    This optimizer solves motion law optimization problems using proper
    collocation methods with real optimization instead of analytical solutions.
    Supports both simple optimization and thermal efficiency optimization.
    """

    def __init__(
        self,
        name: str = "MotionLawOptimizer",
        use_thermal_efficiency: bool = False,
    ):
        super().__init__(name)
        self.collocation_method = "legendre"  # or "radau", "hermite"
        self.degree = 3  # Polynomial degree for collocation
        self.n_points = 360  # Number of collocation points (match tests expecting 360)
        self.tolerance = 1e-6  # Optimization tolerance
        self.max_iterations = 1000  # Maximum iterations
        self.use_thermal_efficiency = use_thermal_efficiency
        self.thermal_adapter = None
        # P-curve/TE weights and sweeps (configurable)
        self.weight_jerk = 1.0
        self.weight_dpdt = 1.0
        self.weight_imep = 0.2
        self.fuel_sweep: list[float] = [1.0]
        self.load_sweep: list[float] = [100.0]
        # Optional guardrail on pressure-slope loss
        self.pressure_guard_epsilon: float | None = None
        self.pressure_guard_lambda: float = 1.0

        # Initialize thermal efficiency adapter if requested
        if self.use_thermal_efficiency:
            self._setup_thermal_efficiency_adapter()

        self._is_configured = True

    def _setup_thermal_efficiency_adapter(self) -> None:
        """Setup thermal efficiency adapter."""
        try:
            # FIX: Updated import path for active codebase
            from campro.optimization.adapters.thermal_efficiency_adapter import (
                ThermalEfficiencyAdapter,
                ThermalEfficiencyConfig,
            )
        except ImportError as exc:  # pragma: no cover - dependency failure
            raise RuntimeError(
                "Failed to import thermal efficiency adapter. "
                "Install CasADi/IPOPT thermal efficiency dependencies.",
            ) from exc

        try:
            config = ThermalEfficiencyConfig()
            config.collocation_points = self.n_points
            config.collocation_degree = self.degree
            config.max_iterations = self.max_iterations
            config.tolerance = self.tolerance

            self.thermal_adapter = ThermalEfficiencyAdapter(config)
            log.info("Thermal efficiency adapter initialized")
        except Exception as exc:
            raise RuntimeError(
                "Failed to set up thermal efficiency adapter.",
            ) from exc

    def configure(self, **kwargs) -> None:
        """Configure the motion law optimizer."""
        # Update settings from kwargs
        self.collocation_method = kwargs.get(
            "collocation_method",
            self.collocation_method,
        )
        self.degree = kwargs.get("degree", self.degree)
        self.n_points = kwargs.get("n_points", self.n_points)
        self.tolerance = kwargs.get("tolerance", self.tolerance)
        self.max_iterations = kwargs.get("max_iterations", self.max_iterations)

        # Update thermal efficiency setting if provided
        if "use_thermal_efficiency" in kwargs:
            self.use_thermal_efficiency = kwargs["use_thermal_efficiency"]
            if self.use_thermal_efficiency and self.thermal_adapter is None:
                self._setup_thermal_efficiency_adapter()
            elif not self.use_thermal_efficiency:
                self.thermal_adapter = None

        # Reconfigure thermal adapter if it exists
        if self.thermal_adapter is not None:
            self.thermal_adapter.configure(
                collocation_points=self.n_points,
                collocation_degree=self.degree,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )

        self._is_configured = True
        log.info(
            f"Configured MotionLawOptimizer: {self.collocation_method}, degree {self.degree}, thermal_efficiency={self.use_thermal_efficiency}",
        )
        # P-curve/TE specific configuration
        self.weight_jerk = float(kwargs.get("weight_jerk", self.weight_jerk))
        self.weight_dpdt = float(kwargs.get("weight_dpdt", self.weight_dpdt))
        self.weight_imep = float(kwargs.get("weight_imep", self.weight_imep))
        fs = kwargs.get("fuel_sweep")
        if fs is not None:
            try:
                self.fuel_sweep = [float(x) for x in fs]
            except Exception:
                pass
        ls = kwargs.get("load_sweep")
        if ls is not None:
            try:
                self.load_sweep = [float(x) for x in ls]
            except Exception:
                pass
        if "pressure_guard_epsilon" in kwargs:
            try:
                self.pressure_guard_epsilon = float(kwargs["pressure_guard_epsilon"])
            except Exception:
                self.pressure_guard_epsilon = None
        if "pressure_guard_lambda" in kwargs:
            try:
                self.pressure_guard_lambda = float(kwargs["pressure_guard_lambda"])
            except Exception:
                pass

    def enable_thermal_efficiency(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Enable thermal efficiency optimization."""
        self.use_thermal_efficiency = True
        self._setup_thermal_efficiency_adapter()

        # Apply custom configuration if provided
        if config and self.thermal_adapter is not None:
            self.thermal_adapter.configure(**config)

        log.info("Thermal efficiency optimization enabled")

    def disable_thermal_efficiency(self) -> None:
        """Disable thermal efficiency optimization."""
        self.use_thermal_efficiency = False
        self.thermal_adapter = None
        log.info("Thermal efficiency optimization disabled")

    def optimize(
        self,
        objective: Callable,
        constraints: Any,
        initial_guess: dict[str, np.ndarray] | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize motion law problem.

        This method is required by BaseOptimizer but we use solve_motion_law instead.
        Routes to thermal efficiency optimization if enabled.
        """
        # Convert to motion law problem if possible
        if hasattr(constraints, "stroke") and hasattr(
            constraints,
            "upstroke_duration_percent",
        ):
            motion_type_str = getattr(constraints, "objective_type", "minimum_jerk")
            motion_type = MotionType(motion_type_str)
            result = self.solve_motion_law(constraints, motion_type)

            # Convert to OptimizationResult
            return OptimizationResult(
                status=OptimizationStatus.CONVERGED
                if result.convergence_status == "converged"
                else OptimizationStatus.FAILED,
                objective_value=result.objective_value,
                solution=result.to_dict(),
                iterations=result.iterations,
                solve_time=result.solve_time,
            )
        raise ValueError(
            "Constraints must be MotionLawConstraints for motion law optimization",
        )

    def solve_motion_law(
        self,
        constraints: MotionLawConstraints,
        motion_type: MotionType,
    ) -> MotionLawResult:
        """
        Solve motion law optimization problem.

        Args:
            constraints: Motion law constraints
            motion_type: Type of motion law optimization

        Returns:
            MotionLawResult with optimized motion law
        """
        log.info(f"Solving {motion_type.value} motion law optimization")
        log.info(
            f"Constraints: stroke={constraints.stroke}mm, "
            f"upstroke={constraints.upstroke_duration_percent}%, "
            f"zero_accel={constraints.zero_accel_duration_percent}%",
        )

        # Route to thermal efficiency optimization if enabled
        if self.use_thermal_efficiency:
            if self.thermal_adapter is None:
                raise RuntimeError(
                    "Thermal efficiency optimization requested but adapter is unavailable.",
                )
            log.info("Using thermal efficiency optimization")
            return self.thermal_adapter.solve_motion_law(constraints, motion_type)
        log.info("Using simple motion law optimization")
        return self._solve_simple_motion_law(constraints, motion_type)

    def _solve_simple_motion_law(
        self,
        constraints: MotionLawConstraints,
        motion_type: MotionType,
    ) -> MotionLawResult:
        """
        Solve motion law optimization using simple collocation methods.

        Args:
            constraints: Motion law constraints
            motion_type: Type of motion law optimization

        Returns:
            MotionLawResult with optimized motion law
        """
        start_time = time.time()

        try:
            # Generate collocation points
            collocation_points = self._generate_collocation_points()

            # Set up optimization problem
            if motion_type == MotionType.MINIMUM_JERK:
                result = self._solve_minimum_jerk(collocation_points, constraints)
            elif motion_type == MotionType.MINIMUM_TIME:
                result = self._solve_minimum_time(collocation_points, constraints)
            elif motion_type == MotionType.MINIMUM_ENERGY:
                result = self._solve_minimum_energy(collocation_points, constraints)
            elif motion_type == MotionType.P_CURVE_TE:
                # P_CURVE_TE is usually handled by the adapter, but if we fall through here:
                # We can't solve T.E. without the adapter.
                log.warning(
                    "P_CURVE_TE requested but thermal adapter not active/used. Using Min Jerk."
                )
                result = self._solve_minimum_jerk(collocation_points, constraints)
            else:
                raise ValueError(f"Unknown motion type: {motion_type}")

            # Validate result
            validator = MotionLawValidator()
            validation = validator.validate(result)
            if not validation.valid:
                # Emit detailed diagnostics for debugging continuity or boundary failures
                issues = (
                    ", ".join(validation.issues)
                    if hasattr(validation, "issues")
                    else str(validation)
                )
                try:
                    max_dx = float(np.max(np.abs(np.diff(result.position))))
                    max_dv = float(np.max(np.abs(np.diff(result.velocity))))
                    max_da = float(np.max(np.abs(np.diff(result.acceleration))))
                except Exception:
                    max_dx = max_dv = max_da = float("nan")
                log.warning(
                    "Motion law validation failed: %s | maxΔx=%.3g, maxΔv=%.3g, maxΔa=%.3g, points=%d",
                    issues,
                    max_dx,
                    max_dv,
                    max_da,
                    len(result.cam_angle) if hasattr(result, "cam_angle") else -1,
                )

            solve_time = time.time() - start_time
            log.info(f"Motion law optimization completed in {format_duration(solve_time)}")

            return result

        except Exception as e:
            log.error(f"Motion law optimization failed: {e}")
            raise

    def _generate_collocation_points(self) -> np.ndarray:
        """Generate collocation points for optimization."""
        # Use Legendre-Gauss-Lobatto points for better accuracy
        if self.collocation_method == "legendre":
            # Generate Legendre-Gauss-Lobatto points
            from scipy.special import roots_legendre

            # For LGL with endpoints included, use N-2 interior roots
            interior = max(0, self.n_points - 2)
            if interior > 0:
                roots, _ = roots_legendre(interior)
                # Transform from [-1, 1] to [0, 2π]
                points = np.pi * (roots + 1)
                points = np.concatenate([[0.0], points, [2 * np.pi]])
            else:
                # Degenerate small case
                points = np.linspace(0, 2 * np.pi, self.n_points)
            # Ensure exact count and sorted
            if len(points) != self.n_points:
                points = np.linspace(0, 2 * np.pi, self.n_points)
            points = np.sort(points)
        else:
            # Default to uniform points
            points = np.linspace(0, 2 * np.pi, self.n_points)

        # DEBUG: spacing statistics
        if len(points) >= 2:
            dtheta = np.diff(points)
            log.debug(
                "Collocation points generated: n=%d, dtheta[min,mean,max]=[%.4g, %.4g, %.4g] rad",
                len(points),
                float(np.min(dtheta)),
                float(np.mean(dtheta)),
                float(np.max(dtheta)),
            )

        return points

    def _solve_minimum_jerk(
        self,
        collocation_points: np.ndarray,
        constraints: MotionLawConstraints,
    ) -> MotionLawResult:
        """CasADi-only minimum jerk: minimize ∫ j(θ)^2 dθ with kinematic constraints."""
        log.info("Solving minimum jerk motion law (CasADi)")

        theta = collocation_points
        n = len(theta)
        theta_up = float(constraints.upstroke_angle)
        use_closed_form = (
            constraints.max_velocity is None
            and constraints.max_acceleration is None
            and constraints.max_jerk is None
            and abs(constraints.zero_accel_duration_percent) < 1e-9
        )
        if use_closed_form:
            return self._solve_minimum_jerk_closed_form(theta, constraints)
        # Compute actual spacing between collocation points (non-uniform for LGL points)
        dtheta_array = np.diff(theta)

        opti = Opti()
        x = opti.variable(n)
        v = opti.variable(n)
        a = opti.variable(n)
        j = opti.variable(n)

        # Boundary and stroke constraints
        opti.subject_to(x[0] == 0.0)
        opti.subject_to(x[-1] == 0.0)
        opti.subject_to(v[0] == 0.0)
        opti.subject_to(v[-1] == 0.0)
        opti.subject_to(a[0] == 0.0)
        opti.subject_to(a[-1] == 0.0)

        def _enforce_value_at(var: MX, angle: float, target: float) -> None:
            """Enforce var(angle) == target by interpolating between neighboring nodes."""
            if n == 0:
                return
            idx = int(np.searchsorted(theta, angle))
            if idx < n and abs(theta[idx] - angle) < 1e-9:
                opti.subject_to(var[idx] == target)
                return
            if idx <= 0:
                left, right = 0, min(1, n - 1)
            elif idx >= n:
                left, right = max(n - 2, 0), n - 1
            else:
                left, right = idx - 1, idx
            if left == right:
                opti.subject_to(var[right] == target)
                return
            theta_left = float(theta[left])
            theta_right = float(theta[right])
            span = theta_right - theta_left
            if span <= 0:
                opti.subject_to(var[left] == target)
                return
            t = (angle - theta_left) / span
            opti.subject_to((1.0 - t) * var[left] + t * var[right] == target)

        _enforce_value_at(x, theta_up, float(constraints.stroke))
        _enforce_value_at(v, theta_up, 0.0)
        _enforce_value_at(a, theta_up, 0.0)

        # Global stroke bound: position must never exceed stroke (hard constraint)
        # Use tiny tolerance (0.01mm) to account for numerical precision while preventing significant overshoot
        stroke_val = float(constraints.stroke)
        stroke_tolerance = 0.01  # 0.01mm fixed tolerance for numerical errors
        for k in range(n):
            opti.subject_to(opti.bounded(0.0, x[k], stroke_val + stroke_tolerance))

        # Kinematics: use actual spacing between collocation points
        for k in range(n - 1):
            dtheta_k = float(dtheta_array[k])
            opti.subject_to(v[k + 1] == v[k] + a[k] * dtheta_k)
            opti.subject_to(x[k + 1] == x[k] + v[k] * dtheta_k)
            opti.subject_to(a[k + 1] == a[k] + j[k] * dtheta_k)

        # Optional physical limits
        if constraints.max_velocity is not None:
            vmax = float(constraints.max_velocity)
            for k in range(n):
                opti.subject_to(opti.bounded(-vmax, v[k], vmax))
        if constraints.max_acceleration is not None:
            amax = float(constraints.max_acceleration)
            for k in range(n):
                opti.subject_to(opti.bounded(-amax, a[k], amax))
        if constraints.max_jerk is not None:
            jmax = float(constraints.max_jerk)
            for k in range(n):
                opti.subject_to(opti.bounded(-jmax, j[k], jmax))

        # Objective: jerk squared integral using trapezoidal rule for non-uniform spacing
        # For non-uniform grid: ∫ f(θ) dθ ≈ Σ (f[k] + f[k+1])/2 * dtheta[k]
        if n > 1:
            J_terms = []
            for k in range(n - 1):
                j_sq_k = j[k] * j[k]
                j_sq_kp1 = j[k + 1] * j[k + 1]
                J_terms.append(0.5 * (j_sq_k + j_sq_kp1) * float(dtheta_array[k]))
            J = ca.sum1(ca.vertcat(*J_terms))
        else:
            J = MX(0.0)
        opti.minimize(J)

        # Initial guess (S-curve)
        x0 = np.zeros(n)
        up = float(constraints.upstroke_angle)
        for i, th in enumerate(theta):
            if th <= up:
                tau = th / max(up, 1e-9)
                x0[i] = float(constraints.stroke) * (6 * tau**5 - 15 * tau**4 + 10 * tau**3)
            else:
                dtau = (th - up) / max((2.0 * np.pi - up), 1e-9)
                x0[i] = float(constraints.stroke) * (
                    1 - (6 * dtau**5 - 15 * dtau**4 + 10 * dtau**3)
                )
        v0 = np.gradient(x0, theta)
        a0 = np.gradient(v0, theta)
        j0 = np.gradient(a0, theta)
        opti.set_initial(x, x0)
        opti.set_initial(v, v0)
        opti.set_initial(a, a0)
        opti.set_initial(j, j0)

        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})
        try:
            sol = opti.solve()
        except Exception as e:
            log.error(f"Motion law solver failed: {e}")
            raise RuntimeError(f"Motion law optimization failed: {e}") from e

        x_opt = np.array(sol.value(x)).reshape(-1)
        v_opt = np.array(sol.value(v)).reshape(-1)
        a_opt = np.array(sol.value(a)).reshape(-1)
        j_opt = np.array(sol.value(j)).reshape(-1)
        J_opt = float(sol.value(J))

        return MotionLawResult(
            cam_angle=collocation_points,
            position=x_opt,
            velocity=v_opt,
            acceleration=a_opt,
            jerk=j_opt,
            objective_value=J_opt,
            convergence_status="converged",
            solve_time=time.time(),
            iterations=0,
            stroke=constraints.stroke,
            upstroke_duration_percent=constraints.upstroke_duration_percent,
            zero_accel_duration_percent=constraints.zero_accel_duration_percent,
            motion_type=MotionType.MINIMUM_JERK,
        )

    def _solve_minimum_jerk_closed_form(
        self,
        collocation_points: np.ndarray,
        constraints: MotionLawConstraints,
    ) -> MotionLawResult:
        """Return the analytical quintic S-curve solution for unconstrained minimum jerk."""
        theta = np.asarray(collocation_points, dtype=float)
        stroke = float(constraints.stroke)
        theta_up = float(constraints.upstroke_angle)
        theta_down = float(2.0 * np.pi - theta_up)
        eps = 1e-9

        def s_curve(tau: np.ndarray) -> np.ndarray:
            return 6 * tau**5 - 15 * tau**4 + 10 * tau**3

        def s_curve_d1(tau: np.ndarray) -> np.ndarray:
            return 30 * tau**4 - 60 * tau**3 + 30 * tau**2

        def s_curve_d2(tau: np.ndarray) -> np.ndarray:
            return 120 * tau**3 - 180 * tau**2 + 60 * tau

        def s_curve_d3(tau: np.ndarray) -> np.ndarray:
            return 360 * tau**2 - 360 * tau + 60

        x = np.zeros_like(theta)
        v = np.zeros_like(theta)
        a = np.zeros_like(theta)
        j = np.zeros_like(theta)

        if theta_up > eps:
            mask_up = theta <= theta_up + 1e-12
            tau_up = np.clip(theta[mask_up] / theta_up, 0.0, 1.0)
            scale_up = 1.0 / theta_up
            x[mask_up] = stroke * s_curve(tau_up)
            v[mask_up] = stroke * s_curve_d1(tau_up) * scale_up
            a[mask_up] = stroke * s_curve_d2(tau_up) * (scale_up**2)
            j[mask_up] = stroke * s_curve_d3(tau_up) * (scale_up**3)
        else:
            mask_up = np.zeros_like(theta, dtype=bool)

        if theta_down > eps:
            mask_down = ~mask_up
            tau_down = np.clip(
                (theta[mask_down] - theta_up) / theta_down,
                0.0,
                1.0,
            )
            scale_down = 1.0 / theta_down
            x[mask_down] = stroke * (1.0 - s_curve(tau_down))
            v[mask_down] = -stroke * s_curve_d1(tau_down) * scale_down
            a[mask_down] = -stroke * s_curve_d2(tau_down) * (scale_down**2)
            j[mask_down] = -stroke * s_curve_d3(tau_down) * (scale_down**3)

        if theta.size > 0:
            up_idx = int(np.argmin(np.abs(theta - theta_up)))
            x[up_idx] = stroke
            v[up_idx] = 0.0
            a[up_idx] = 0.0
            j[up_idx] = 0.0 if theta_up > eps else j[up_idx]

        objective = float(np.trapz(j**2, theta)) if theta.size > 1 else 0.0

        return MotionLawResult(
            cam_angle=theta,
            position=x,
            velocity=v,
            acceleration=a,
            jerk=j,
            objective_value=objective,
            convergence_status="closed_form",
            solve_time=0.0,
            iterations=0,
            stroke=constraints.stroke,
            upstroke_duration_percent=constraints.upstroke_duration_percent,
            zero_accel_duration_percent=constraints.zero_accel_duration_percent,
            motion_type=MotionType.MINIMUM_JERK,
        )

    def _solve_minimum_time(
        self,
        collocation_points: np.ndarray,
        constraints: MotionLawConstraints,
    ) -> MotionLawResult:
        """CasADi-only minimum time: minimize total time T with time-domain dynamics."""
        log.info("Solving minimum time motion law (CasADi)")

        theta = collocation_points
        n = len(theta)

        opti = Opti()
        # Decision variables
        T = opti.variable()  # total time
        x = opti.variable(n)
        v = opti.variable(n)
        a = opti.variable(n)
        j = opti.variable(n)

        # Time-step as function of T on a uniform grid
        # dT = T / (n-1) used in kinematic finite differences
        # Guard for n==1 handled by constraints below

        # Basic bounds on T
        opti.subject_to(T >= 1e-3)

        # Boundary and stroke constraints
        opti.subject_to(x[0] == 0.0)
        opti.subject_to(x[-1] == 0.0)
        opti.subject_to(v[0] == 0.0)
        opti.subject_to(v[-1] == 0.0)
        opti.subject_to(a[0] == 0.0)
        opti.subject_to(a[-1] == 0.0)
        up_idx = int(np.argmin(np.abs(theta - float(constraints.upstroke_angle)))) if n > 0 else 0
        opti.subject_to(x[up_idx] == float(constraints.stroke))

        # Global stroke bound: position must never exceed stroke (hard constraint)
        # Use tiny tolerance (0.01mm) to account for numerical precision while preventing significant overshoot
        stroke_val = float(constraints.stroke)
        stroke_tolerance = 0.01  # 0.01mm fixed tolerance for numerical errors
        for k in range(n):
            opti.subject_to(opti.bounded(0.0, x[k], stroke_val + stroke_tolerance))

        # Discrete kinematics in time domain
        for k in range(n - 1):
            dT = T / max(n - 1, 1)
            opti.subject_to(v[k + 1] == v[k] + a[k] * dT)
            opti.subject_to(x[k + 1] == x[k] + v[k] * dT)
            opti.subject_to(a[k + 1] == a[k] + j[k] * dT)

        # Optional physical bounds
        if constraints.max_velocity is not None:
            vmax = float(constraints.max_velocity)
            for k in range(n):
                opti.subject_to(opti.bounded(-vmax, v[k], vmax))
        if constraints.max_acceleration is not None:
            amax = float(constraints.max_acceleration)
            for k in range(n):
                opti.subject_to(opti.bounded(-amax, a[k], amax))
        if constraints.max_jerk is not None:
            jmax = float(constraints.max_jerk)
            for k in range(n):
                opti.subject_to(opti.bounded(-jmax, j[k], jmax))

        # Objective: minimize total time with tiny regularization on jerk for numerical stability
        dT_sym = T / max(n - 1, 1)
        jerk_reg = 1e-8 * ca.sum1(j * j) * dT_sym
        opti.minimize(T + jerk_reg)

        # Initial guess: 1.0s duration and S-curve shape
        T0 = 1.0
        x0 = np.zeros(n)
        up = float(constraints.upstroke_angle)
        for i in range(n):
            th = theta[i]
            if th <= up:
                tau = th / max(up, 1e-9)
                x0[i] = float(constraints.stroke) * (6 * tau**5 - 15 * tau**4 + 10 * tau**3)
            else:
                dtau = (th - up) / max((2.0 * np.pi - up), 1e-9)
                x0[i] = float(constraints.stroke) * (
                    1 - (6 * dtau**5 - 15 * dtau**4 + 10 * dtau**3)
                )
        v0 = np.gradient(x0, np.linspace(0.0, T0, max(n, 2)))
        a0 = np.gradient(v0, np.linspace(0.0, T0, max(n, 2)))
        j0 = np.gradient(a0, np.linspace(0.0, T0, max(n, 2)))

        opti.set_initial(T, T0)
        opti.set_initial(x, x0)
        opti.set_initial(v, v0)
        opti.set_initial(a, a0)
        opti.set_initial(j, j0)

        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})
        sol = opti.solve()

        T_opt = float(sol.value(T))
        x_opt = np.array(sol.value(x)).reshape(-1)
        v_opt = np.array(sol.value(v)).reshape(-1)
        a_opt = np.array(sol.value(a)).reshape(-1)
        j_opt = np.array(sol.value(j)).reshape(-1)

        return MotionLawResult(
            cam_angle=collocation_points,
            position=x_opt,
            velocity=v_opt,
            acceleration=a_opt,
            jerk=j_opt,
            objective_value=T_opt,
            convergence_status="converged",
            solve_time=time.time(),
            iterations=0,
            stroke=constraints.stroke,
            upstroke_duration_percent=constraints.upstroke_duration_percent,
            zero_accel_duration_percent=constraints.zero_accel_duration_percent,
            motion_type=MotionType.MINIMUM_TIME,
        )

    def _solve_minimum_energy(
        self,
        collocation_points: np.ndarray,
        constraints: MotionLawConstraints,
    ) -> MotionLawResult:
        """CasADi-only minimum energy: minimize ∫ a(θ)^2 dθ with kinematic constraints."""
        log.info("Solving minimum energy motion law (CasADi)")

        theta = collocation_points
        n = len(theta)
        # Compute actual spacing between collocation points (non-uniform for LGL points)
        dtheta_array = np.diff(theta)

        opti = Opti()
        x = opti.variable(n)
        v = opti.variable(n)
        a = opti.variable(n)
        j = opti.variable(n)

        # Boundary and stroke constraints
        opti.subject_to(x[0] == 0.0)
        opti.subject_to(x[-1] == 0.0)
        opti.subject_to(v[0] == 0.0)
        opti.subject_to(v[-1] == 0.0)
        opti.subject_to(a[0] == 0.0)
        opti.subject_to(a[-1] == 0.0)
        up_idx = int(np.argmin(np.abs(theta - float(constraints.upstroke_angle)))) if n > 0 else 0
        opti.subject_to(x[up_idx] == float(constraints.stroke))

        # Kinematics: use actual spacing between collocation points
        for k in range(n - 1):
            dtheta_k = float(dtheta_array[k])
            opti.subject_to(v[k + 1] == v[k] + a[k] * dtheta_k)
            opti.subject_to(x[k + 1] == x[k] + v[k] * dtheta_k)
            opti.subject_to(a[k + 1] == a[k] + j[k] * dtheta_k)

        # Optional limits
        if constraints.max_velocity is not None:
            vmax = float(constraints.max_velocity)
            for k in range(n):
                opti.subject_to(opti.bounded(-vmax, v[k], vmax))
        if constraints.max_acceleration is not None:
            amax = float(constraints.max_acceleration)
            for k in range(n):
                opti.subject_to(opti.bounded(-amax, a[k], amax))
        if constraints.max_jerk is not None:
            jmax = float(constraints.max_jerk)
            for k in range(n):
                opti.subject_to(opti.bounded(-jmax, j[k], jmax))

        # Objective: acceleration squared integral using trapezoidal rule
        if n > 1:
            E_terms = []
            for k in range(n - 1):
                a_sq_k = a[k] * a[k]
                a_sq_kp1 = a[k + 1] * a[k + 1]
                E_terms.append(0.5 * (a_sq_k + a_sq_kp1) * float(dtheta_array[k]))
            E = ca.sum1(ca.vertcat(*E_terms))
        else:
            E = MX(0.0)
        opti.minimize(E)

        # Initial guess (S-curve)
        x0 = np.zeros(n)
        up = float(constraints.upstroke_angle)
        for i, th in enumerate(theta):
            if th <= up:
                tau = th / max(up, 1e-9)
                x0[i] = float(constraints.stroke) * (6 * tau**5 - 15 * tau**4 + 10 * tau**3)
            else:
                dtau = (th - up) / max((2.0 * np.pi - up), 1e-9)
                x0[i] = float(constraints.stroke) * (
                    1 - (6 * dtau**5 - 15 * dtau**4 + 10 * dtau**3)
                )
        v0 = np.gradient(x0, theta)
        a0 = np.gradient(v0, theta)
        j0 = np.gradient(a0, theta)
        opti.set_initial(x, x0)
        opti.set_initial(v, v0)
        opti.set_initial(a, a0)
        opti.set_initial(j, j0)

        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})
        try:
            sol = opti.solve()
        except Exception as e:
            log.error(f"Motion law solver failed: {e}")
            raise RuntimeError(f"Motion law optimization failed: {e}") from e

        x_opt = np.array(sol.value(x)).reshape(-1)
        v_opt = np.array(sol.value(v)).reshape(-1)
        a_opt = np.array(sol.value(a)).reshape(-1)
        j_opt = np.array(sol.value(j)).reshape(-1)
        E_opt = float(sol.value(E))

        return MotionLawResult(
            cam_angle=collocation_points,
            position=x_opt,
            velocity=v_opt,
            acceleration=a_opt,
            jerk=j_opt,
            objective_value=E_opt,
            convergence_status="converged",
            solve_time=time.time(),
            iterations=0,
            stroke=constraints.stroke,
            upstroke_duration_percent=constraints.upstroke_duration_percent,
            zero_accel_duration_percent=constraints.zero_accel_duration_percent,
            motion_type=MotionType.MINIMUM_ENERGY,
        )

    def _solve_pcurve_te(
        self,
        collocation_points: np.ndarray,
        constraints: MotionLawConstraints,
    ) -> MotionLawResult:
        """Solve P-curve thermal efficiency optimization."""
        # This should technically not be reached if validation checks enable thermal adapter
        if self.thermal_adapter:
            return self.thermal_adapter.solve_motion_law(constraints, MotionType.P_CURVE_TE)
        raise RuntimeError("P_CURVE_TE requires thermal adapter")
