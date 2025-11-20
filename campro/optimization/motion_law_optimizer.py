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
from typing import Any, Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from casadi import Opti, MX
import casadi as ca

from campro.logging import get_logger
from campro.utils import format_duration

from .base import BaseOptimizer, OptimizationResult, OptimizationStatus
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
        self, name: str = "MotionLawOptimizer", use_thermal_efficiency: bool = False,
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
            from .thermal_efficiency_adapter import (
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
            "collocation_method", self.collocation_method,
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
        self, config: dict[str, Any] | None = None,
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
            constraints, "upstroke_duration_percent",
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
        self, constraints: MotionLawConstraints, motion_type: MotionType,
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
        self, constraints: MotionLawConstraints, motion_type: MotionType,
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
                result = self._solve_pcurve_te(collocation_points, constraints)
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
        self, collocation_points: np.ndarray, constraints: MotionLawConstraints,
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
                j_sq_kp1 = j[k+1] * j[k+1]
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
                x0[i] = float(constraints.stroke) * (1 - (6 * dtau**5 - 15 * dtau**4 + 10 * dtau**3))
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
        self, collocation_points: np.ndarray, constraints: MotionLawConstraints,
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
                x0[i] = float(constraints.stroke) * (1 - (6 * dtau**5 - 15 * dtau**4 + 10 * dtau**3))
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
        self, collocation_points: np.ndarray, constraints: MotionLawConstraints,
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

        # Objective: acceleration squared integral using trapezoidal rule for non-uniform spacing
        # For non-uniform grid: ∫ f(θ) dθ ≈ Σ (f[k] + f[k+1])/2 * dtheta[k]
        if n > 1:
            J_terms = []
            for k in range(n - 1):
                a_sq_k = a[k] * a[k]
                a_sq_kp1 = a[k+1] * a[k+1]
                J_terms.append(0.5 * (a_sq_k + a_sq_kp1) * float(dtheta_array[k]))
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
                x0[i] = float(constraints.stroke) * (1 - (6 * dtau**5 - 15 * dtau**4 + 10 * dtau**3))
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
            motion_type=MotionType.MINIMUM_ENERGY,
        )

    def _solve_pcurve_te(
        self, collocation_points: np.ndarray, constraints: MotionLawConstraints,
    ) -> MotionLawResult:
        """
        Solve P-curve/TE objective using CasADi-only formulation.

        Objective J = wj * ∫ j(θ)^2 dθ + wp * loss_p(slope) - wimep * iMEP

        Notes:
            - All constraints and objectives are symbolic CasADi expressions
            - Kinematics enforced via finite-difference equalities on θ-grid
            - Pressure model mirrors SimpleCycleAdapter in a symbolic form
        """
        # Grid
        theta = collocation_points
        n = len(theta)
        # Compute actual spacing between collocation points (non-uniform for LGL points)
        dtheta_array = np.diff(theta)
        # For periodic boundary conditions, also compute spacing at boundaries
        # Wrap-around spacing: from last to first point (for periodic)
        dtheta_wrap = float(2.0 * np.pi - theta[-1] + theta[0]) if n > 1 else 2.0 * np.pi

        # Weights and sweeps from configuration
        wj = float(self.weight_jerk)
        wp = float(self.weight_dpdt)
        wimep = float(self.weight_imep)
        fuel_sweep = list(self.fuel_sweep)
        load_sweep = list(self.load_sweep)

        # Geometry and thermo constants (toy model consistent with adapter)
        area_mm2 = 1.0
        Vc_mm3 = 1000.0
        gamma_bounce = 1.25
        alpha_fuel_to_base = 30.0
        beta_base = 100.0
        p_atm_kpa = 101.325

        # CasADi Opti variables
        opti = Opti()
        x = opti.variable(n)  # position [mm]
        v = opti.variable(n)  # velocity dx/dθ [mm/rad]
        a = opti.variable(n)  # acceleration d²x/dθ²
        j = opti.variable(n)  # jerk d³x/dθ³

        # Boundary conditions (periodicity and stroke)
        opti.subject_to(x[0] == 0.0)
        opti.subject_to(x[-1] == 0.0)
        opti.subject_to(v[0] == 0.0)
        opti.subject_to(v[-1] == 0.0)
        opti.subject_to(a[0] == 0.0)
        opti.subject_to(a[-1] == 0.0)

        # Upstroke stroke target at θ = upstroke_angle
        upstroke_angle = float(constraints.upstroke_angle)
        # Find closest grid index to upstroke end
        up_idx = int(np.argmin(np.abs(theta - upstroke_angle))) if n > 0 else 0
        opti.subject_to(x[up_idx] == float(constraints.stroke))

        # Global stroke bound: position must never exceed stroke (hard constraint)
        # Use tiny tolerance (0.01mm) to account for numerical precision while preventing significant overshoot
        stroke_val = float(constraints.stroke)
        stroke_tolerance = 0.01  # 0.01mm fixed tolerance for numerical errors
        for k in range(n):
            opti.subject_to(opti.bounded(0.0, x[k], stroke_val + stroke_tolerance))

        # Kinematics constraints: use actual spacing between collocation points
        for k in range(n - 1):
            dtheta_k = float(dtheta_array[k])
            opti.subject_to(v[k + 1] == v[k] + a[k] * dtheta_k)
            opti.subject_to(x[k + 1] == x[k] + v[k] * dtheta_k)
            opti.subject_to(a[k + 1] == a[k] + j[k] * dtheta_k)

        # Optional limits if provided
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

        # Pressure model and slope objective
        # Prepare constant arrays as MX for CasADi ops
        theta_mx = MX(theta.tolist())

        # Wiebe burn fraction xb(θ)
        # Parameters chosen to be consistent with a simple profile
        a_wiebe = 5.0
        m_wiebe = 2.0
        start_deg = -5.0
        duration_deg = 25.0
        th_deg = theta_mx * (180.0 / np.pi)
        z = (th_deg - start_deg) / max(duration_deg, 1e-9)
        # clip z in CasADi: approximate with smooth clamps
        z0 = 0.5 * ((z + MX.fabs(z)) - (z - duration_deg + MX.fabs(z - duration_deg))) / max(duration_deg, 1e-9)
        xb = 1.0 - ca.exp(-a_wiebe * ca.power(ca.fmax(0, ca.fmin(1, z0)), (m_wiebe + 1.0)))

        # Volumes and pressures
        V = Vc_mm3 + area_mm2 * x
        V0 = V[0]
        p_comb = p_atm_kpa * (1.0 + 0.3 * fuel_sweep[0] * xb)
        p0_bounce = alpha_fuel_to_base * fuel_sweep[0] + beta_base
        p_bounce = p0_bounce * ca.power(V0 / V, gamma_bounce)
        p = p_comb - p_bounce

        # Periodic derivative dp/dθ with central differencing on non-uniform grid
        dp = opti.variable(n)
        for k in range(n):
            kp = (k + 1) % n
            km = (k - 1) % n
            # Compute actual spacing for central differencing
            if k == 0:
                # Wrap-around: forward spacing from n-1 to 0, backward from n-1 to 0
                dtheta_forward = dtheta_wrap
                dtheta_backward = float(dtheta_array[n-2]) if n > 2 else dtheta_wrap
            elif k == n - 1:
                # Wrap-around: forward spacing from n-1 to 0, backward from n-2 to n-1
                dtheta_forward = dtheta_wrap
                dtheta_backward = float(dtheta_array[n-2]) if n > 2 else dtheta_wrap
            else:
                # Regular interior points
                dtheta_forward = float(dtheta_array[k])
                dtheta_backward = float(dtheta_array[k-1])
            # Central difference: (p[kp] - p[km]) / (dtheta_forward + dtheta_backward)
            dtheta_total = dtheta_forward + dtheta_backward
            opti.subject_to(dp[k] == (p[kp] - p[km]) / dtheta_total)

        # Normalized slope: zero-mean, unit L2
        mean_dp = (1.0 / n) * ca.sum1(dp)
        s = dp - mean_dp
        # L2 norm using trapezoidal rule for non-uniform spacing with periodic boundary
        if n > 1:
            s_sq = s * s
            # Regular segments
            s_sq_mid = 0.5 * (s_sq[:-1] + s_sq[1:])
            dtheta_ca = ca.DM(dtheta_array)
            norm_s_regular = ca.sum1(s_sq_mid * dtheta_ca)
            # Wrap-around segment for periodic boundary
            norm_s_wrap = 0.5 * (s_sq[-1] + s_sq[0]) * dtheta_wrap
            norm_s = ca.sqrt(norm_s_regular + norm_s_wrap + 1e-12)
        else:
            norm_s = ca.sqrt(s[0]**2 + 1e-12)
        s_norm = s / norm_s

        # Reference slope: compute once from a seed profile (flat x), unit vector
        # Here we use s_norm itself for invariance target of zero-change across conditions
        s_ref = s_norm

        # Loss over sweeps (single-entry defaults)
        loss_p = MX(0)
        K = len(fuel_sweep) * len(load_sweep)
        for fm in fuel_sweep:
            for _c in load_sweep:
                # reuse same p model: slope aligned to reference
                # Trapezoidal rule for non-uniform spacing with periodic boundary
                if n > 1:
                    diff = s_norm - s_ref
                    diff_sq = diff * diff
                    # Regular segments
                    diff_sq_mid = 0.5 * (diff_sq[:-1] + diff_sq[1:])
                    loss_regular = ca.sum1(diff_sq_mid * dtheta_ca)
                    # Wrap-around segment
                    loss_wrap = 0.5 * (diff_sq[-1] + diff_sq[0]) * dtheta_wrap
                    loss_p += loss_regular + loss_wrap
                else:
                    loss_p += (s_norm[0] - s_ref[0])**2
        loss_p = loss_p / max(K, 1)

        # iMEP approximation: ∮ p dV = ∮ p * A * v dθ (since dV = A dx = A v dθ)
        # Trapezoidal rule for non-uniform spacing with periodic boundary
        if n > 1:
            pv = p * v
            # Regular segments
            pv_mid = 0.5 * (pv[:-1] + pv[1:])
            imep_regular = ca.sum1(pv_mid * dtheta_ca)
            # Wrap-around segment
            imep_wrap = 0.5 * (pv[-1] + pv[0]) * dtheta_wrap
            imep = (area_mm2) * (imep_regular + imep_wrap)
        else:
            imep = (area_mm2) * p[0] * v[0] * dtheta_wrap

        # Jerk integral using trapezoidal rule for non-uniform spacing with periodic boundary
        if n > 1:
            j_sq = j * j
            # Regular segments
            j_sq_mid = 0.5 * (j_sq[:-1] + j_sq[1:])
            jerk_regular = ca.sum1(j_sq_mid * dtheta_ca)
            # Wrap-around segment
            jerk_wrap = 0.5 * (j_sq[-1] + j_sq[0]) * dtheta_wrap
            jerk_cost = jerk_regular + jerk_wrap
        else:
            jerk_cost = j[0]**2 * dtheta_wrap

        # Optional guardrail penalty on pressure slope mismatch
        penalty = MX(0)
        if self.pressure_guard_epsilon is not None:
            eps = float(self.pressure_guard_epsilon)
            lam = float(self.pressure_guard_lambda)
            viol = MX.fmax(0, loss_p - eps)
            penalty = lam * (viol * viol)

        J = wj * jerk_cost + wp * loss_p + penalty - wimep * imep
        opti.minimize(J)

        # Solver
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})

        # Initial guess: simple S-curve on upstroke
        x0 = np.zeros(n)
        for i, th in enumerate(theta):
            if th <= upstroke_angle:
                tau = th / max(upstroke_angle, 1e-9)
                x0[i] = float(constraints.stroke) * (6 * tau**5 - 15 * tau**4 + 10 * tau**3)
            else:
                dtau = (th - upstroke_angle) / max((2.0 * np.pi - upstroke_angle), 1e-9)
                x0[i] = float(constraints.stroke) * (1 - (6 * dtau**5 - 15 * dtau**4 + 10 * dtau**3))
        v0 = np.gradient(x0, theta)
        a0 = np.gradient(v0, theta)
        j0 = np.gradient(a0, theta)
        opti.set_initial(x, x0)
        opti.set_initial(v, v0)
        opti.set_initial(a, a0)
        opti.set_initial(j, j0)

        sol = opti.solve()
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
            motion_type=MotionType.P_CURVE_TE,
        )

    def _generate_initial_guess_minimum_jerk(
        self, control_points: np.ndarray, constraints: MotionLawConstraints,
    ) -> np.ndarray:
        """Generate initial guess for minimum jerk optimization."""
        # Create a smooth S-curve initial guess
        n_control = len(control_points)
        initial_guess = np.zeros(n_control)

        # Create S-curve profile
        for i, theta in enumerate(control_points):
            if theta <= constraints.upstroke_angle:
                # Upstroke phase
                tau = theta / constraints.upstroke_angle
                # S-curve: 6t^5 - 15t^4 + 10t^3
                initial_guess[i] = constraints.stroke * (
                    6 * tau**5 - 15 * tau**4 + 10 * tau**3
                )
            else:
                # Downstroke phase
                downstroke_tau = (
                    theta - constraints.upstroke_angle
                ) / constraints.downstroke_angle
                # Mirror the upstroke
                initial_guess[i] = constraints.stroke * (
                    1
                    - (
                        6 * downstroke_tau**5
                        - 15 * downstroke_tau**4
                        + 10 * downstroke_tau**3
                    )
                )

        return initial_guess

    def _minimum_jerk_objective(
        self,
        params: np.ndarray,
        control_points: np.ndarray,
        collocation_points: np.ndarray,
    ) -> float:
        """Calculate minimum jerk objective function."""
        # Interpolate control points to collocation points
        cs = CubicSpline(control_points, params, bc_type="natural")
        position = cs(collocation_points)

        # Calculate derivatives
        velocity = cs.derivative()(collocation_points)
        acceleration = cs.derivative(2)(collocation_points)
        jerk = cs.derivative(3)(collocation_points)

        # Objective: minimize jerk squared
        objective = np.trapz(jerk**2, collocation_points)

        return objective

    def _define_motion_law_constraints(
        self,
        params: np.ndarray,
        control_points: np.ndarray,
        collocation_points: np.ndarray,
        constraints: MotionLawConstraints,
    ) -> list[dict]:
        """Define constraints for motion law optimization."""
        constraint_list = []

        # Boundary conditions
        def boundary_position_start(params):
            cs = CubicSpline(control_points, params, bc_type="natural")
            return cs(0.0)  # Should be 0

        def boundary_position_end(params):
            cs = CubicSpline(control_points, params, bc_type="natural")
            return cs(2 * np.pi)  # Should be 0

        def boundary_velocity_start(params):
            cs = CubicSpline(control_points, params, bc_type="natural")
            return cs.derivative()(0.0)  # Should be 0

        def boundary_velocity_end(params):
            cs = CubicSpline(control_points, params, bc_type="natural")
            return cs.derivative()(2 * np.pi)  # Should be 0

        def boundary_acceleration_start(params):
            cs = CubicSpline(control_points, params, bc_type="natural")
            return cs.derivative(2)(0.0)  # Should be 0

        def boundary_acceleration_end(params):
            cs = CubicSpline(control_points, params, bc_type="natural")
            return cs.derivative(2)(2 * np.pi)  # Should be 0

        # Stroke constraint
        def stroke_constraint(params):
            cs = CubicSpline(control_points, params, bc_type="natural")
            return cs(constraints.upstroke_angle) - constraints.stroke  # Should be 0

        # Add constraints
        constraint_list.append(
            {
                "type": "eq",
                "fun": boundary_position_start,
            },
        )
        constraint_list.append(
            {
                "type": "eq",
                "fun": boundary_position_end,
            },
        )
        constraint_list.append(
            {
                "type": "eq",
                "fun": boundary_velocity_start,
            },
        )
        constraint_list.append(
            {
                "type": "eq",
                "fun": boundary_velocity_end,
            },
        )
        constraint_list.append(
            {
                "type": "eq",
                "fun": boundary_acceleration_start,
            },
        )
        constraint_list.append(
            {
                "type": "eq",
                "fun": boundary_acceleration_end,
            },
        )
        constraint_list.append(
            {
                "type": "eq",
                "fun": stroke_constraint,
            },
        )

        return constraint_list

    def _extract_solution_minimum_jerk(
        self,
        params: np.ndarray,
        control_points: np.ndarray,
        collocation_points: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Extract solution from optimization result."""
        cs = CubicSpline(control_points, params, bc_type="natural")

        position = cs(collocation_points)
        velocity = cs.derivative()(collocation_points)
        acceleration = cs.derivative(2)(collocation_points)
        jerk = cs.derivative(3)(collocation_points)

        return {
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "jerk": jerk,
        }

    def _solve_trapezoidal_velocity_profile(
        self, collocation_points: np.ndarray, constraints: MotionLawConstraints,
    ) -> dict[str, np.ndarray]:
        """Solve trapezoidal velocity profile for minimum time approximation."""
        n_points = len(collocation_points)
        position = np.zeros(n_points)
        velocity = np.zeros(n_points)
        acceleration = np.zeros(n_points)
        jerk = np.zeros(n_points)

        upstroke_angle = constraints.upstroke_angle
        downstroke_angle = constraints.downstroke_angle

        # Calculate maximum velocity to achieve stroke
        # Use trapezoidal profile: accel -> constant -> decel
        accel_angle = upstroke_angle * 0.3  # 30% for acceleration
        const_angle = upstroke_angle * 0.4  # 40% for constant velocity
        decel_angle = upstroke_angle * 0.3  # 30% for deceleration

        max_velocity = constraints.stroke / (
            0.5 * accel_angle + const_angle + 0.5 * decel_angle
        )

        for i, theta in enumerate(collocation_points):
            if theta <= accel_angle:
                # Acceleration phase
                velocity[i] = max_velocity * (theta / accel_angle)
                position[i] = 0.5 * max_velocity * theta**2 / accel_angle
                acceleration[i] = max_velocity / accel_angle
            elif theta <= accel_angle + const_angle:
                # Constant velocity phase
                velocity[i] = max_velocity
                position[i] = 0.5 * max_velocity * accel_angle + max_velocity * (
                    theta - accel_angle
                )
                acceleration[i] = 0
            elif theta <= upstroke_angle:
                # Deceleration phase
                decel_theta = theta - accel_angle - const_angle
                velocity[i] = max_velocity * (1 - decel_theta / decel_angle)
                position[i] = (
                    0.5 * max_velocity * accel_angle
                    + max_velocity * const_angle
                    + max_velocity * decel_theta
                    - 0.5 * max_velocity * decel_theta**2 / decel_angle
                )
                acceleration[i] = -max_velocity / decel_angle
            else:
                # Downstroke phase (mirror upstroke)
                downstroke_theta = theta - upstroke_angle
                if downstroke_theta <= accel_angle:
                    # Downstroke acceleration
                    velocity[i] = -max_velocity * (downstroke_theta / accel_angle)
                    position[i] = (
                        constraints.stroke
                        - 0.5 * max_velocity * downstroke_theta**2 / accel_angle
                    )
                    acceleration[i] = -max_velocity / accel_angle
                elif downstroke_theta <= accel_angle + const_angle:
                    # Downstroke constant velocity
                    velocity[i] = -max_velocity
                    position[i] = (
                        constraints.stroke
                        - 0.5 * max_velocity * accel_angle
                        - max_velocity * (downstroke_theta - accel_angle)
                    )
                    acceleration[i] = 0
                else:
                    # Downstroke deceleration
                    decel_theta = downstroke_theta - accel_angle - const_angle
                    velocity[i] = -max_velocity * (1 - decel_theta / decel_angle)
                    position[i] = (
                        constraints.stroke
                        - 0.5 * max_velocity * accel_angle
                        - max_velocity * const_angle
                        - max_velocity * decel_theta
                        + 0.5 * max_velocity * decel_theta**2 / decel_angle
                    )
                    acceleration[i] = max_velocity / decel_angle

        # Calculate jerk as derivative of acceleration
        jerk = np.gradient(acceleration, collocation_points)

        # Ensure proper boundary conditions and stroke
        position[0] = 0.0
        # Scale downstroke to exactly return to 0
        position[-1] = 0.0
        velocity[0] = 0.0
        velocity[-1] = 0.0

        # Normalize position so that maximum equals stroke during upstroke
        max_pos = np.max(position)
        if max_pos > 1e-9:
            scale = constraints.stroke / max_pos
            position = position * scale
            velocity = velocity * scale
            acceleration = acceleration * scale
            jerk = jerk * scale

        return {
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "jerk": jerk,
            "objective_value": float(
                np.trapz(np.abs(acceleration), collocation_points),
            ),
            "iterations": 1,
        }

    def _solve_smooth_acceleration_profile(
        self, collocation_points: np.ndarray, constraints: MotionLawConstraints,
    ) -> dict[str, np.ndarray]:
        """Solve smooth acceleration profile for minimum energy approximation."""
        n_points = len(collocation_points)
        position = np.zeros(n_points)
        velocity = np.zeros(n_points)
        acceleration = np.zeros(n_points)
        jerk = np.zeros(n_points)

        upstroke_angle = constraints.upstroke_angle
        downstroke_angle = constraints.downstroke_angle

        for i, theta in enumerate(collocation_points):
            if theta <= upstroke_angle:
                # Upstroke phase - smooth acceleration
                tau = theta / upstroke_angle
                # Use cubic profile for smooth acceleration
                position[i] = constraints.stroke * (3 * tau**2 - 2 * tau**3)
                velocity[i] = (constraints.stroke / upstroke_angle) * (
                    6 * tau - 6 * tau**2
                )
                acceleration[i] = (constraints.stroke / upstroke_angle**2) * (
                    6 - 12 * tau
                )
                jerk[i] = -12 * constraints.stroke / upstroke_angle**3
            else:
                # Downstroke phase - mirror upstroke
                downstroke_tau = (theta - upstroke_angle) / downstroke_angle
                position[i] = constraints.stroke * (
                    1 - (3 * downstroke_tau**2 - 2 * downstroke_tau**3)
                )
                velocity[i] = -(constraints.stroke / downstroke_angle) * (
                    6 * downstroke_tau - 6 * downstroke_tau**2
                )
                acceleration[i] = -(constraints.stroke / downstroke_angle**2) * (
                    6 - 12 * downstroke_tau
                )
                jerk[i] = 12 * constraints.stroke / downstroke_angle**3

        # Ensure proper boundary conditions
        position[0] = 0.0
        position[-1] = 0.0
        velocity[0] = 0.0
        velocity[-1] = 0.0

        return {
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "jerk": jerk,
        }

    def _solve_bang_bang_control(
        self, collocation_points: np.ndarray, constraints: MotionLawConstraints,
    ) -> dict[str, Any]:
        """
        Solve minimum time motion law using proper bang-bang control.

        Bang-bang control uses maximum acceleration/deceleration to minimize time.
        """
        log.info("Implementing bang-bang control for minimum time")

        n_points = len(collocation_points)
        position = np.zeros(n_points)
        velocity = np.zeros(n_points)
        acceleration = np.zeros(n_points)
        jerk = np.zeros(n_points)

        upstroke_angle = constraints.upstroke_angle
        downstroke_angle = constraints.downstroke_angle

        # Get maximum acceleration (use constraint or calculate from stroke)
        max_accel = constraints.max_acceleration
        if max_accel is None:
            # Calculate maximum acceleration needed to achieve stroke
            max_accel = 4 * constraints.stroke / upstroke_angle**2

        # Bang-bang control: maximum acceleration until halfway, then maximum deceleration
        upstroke_midpoint = upstroke_angle / 2
        downstroke_midpoint = upstroke_angle + downstroke_angle / 2

        for i, theta in enumerate(collocation_points):
            if theta <= upstroke_midpoint:
                # Upstroke acceleration phase (bang-bang: max acceleration)
                acceleration[i] = max_accel
                velocity[i] = max_accel * theta
                position[i] = 0.5 * max_accel * theta**2
            elif theta <= upstroke_angle:
                # Upstroke deceleration phase (bang-bang: max deceleration)
                decel_theta = theta - upstroke_midpoint
                acceleration[i] = -max_accel
                velocity[i] = max_accel * upstroke_midpoint - max_accel * decel_theta
                position[i] = (
                    0.5 * max_accel * upstroke_midpoint**2
                    + max_accel * upstroke_midpoint * decel_theta
                    - 0.5 * max_accel * decel_theta**2
                )
            elif theta <= downstroke_midpoint:
                # Downstroke acceleration phase (bang-bang: max deceleration)
                downstroke_theta = theta - upstroke_angle
                acceleration[i] = -max_accel
                velocity[i] = -max_accel * downstroke_theta
                position[i] = constraints.stroke - 0.5 * max_accel * downstroke_theta**2
            else:
                # Downstroke deceleration phase (bang-bang: max acceleration)
                downstroke_theta = theta - upstroke_angle
                decel_theta = downstroke_theta - downstroke_angle / 2
                acceleration[i] = max_accel
                velocity[i] = (
                    -max_accel * downstroke_angle / 2 + max_accel * decel_theta
                )
                position[i] = (
                    constraints.stroke
                    - 0.5 * max_accel * (downstroke_angle / 2) ** 2
                    - max_accel * (downstroke_angle / 2) * decel_theta
                    + 0.5 * max_accel * decel_theta**2
                )

        # Calculate jerk as derivative of acceleration
        jerk = np.gradient(acceleration, collocation_points)

        # Calculate objective value (total time is minimized by bang-bang control)
        # For bang-bang control, the objective is the total cycle time
        total_time = 2 * np.sqrt(2 * constraints.stroke / max_accel)
        objective_value = total_time

        return {
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "jerk": jerk,
            "objective_value": objective_value,
            "iterations": 1,  # Bang-bang is analytical, no iterations needed
        }

    def _solve_minimum_energy_optimization(
        self, collocation_points: np.ndarray, constraints: MotionLawConstraints,
    ) -> dict[str, Any]:
        """
        Solve minimum energy motion law optimization.

        Objective: Minimize ∫[0 to 2π] (x''(θ))² dθ
        """
        log.info("Implementing minimum energy optimization")

        n_points = len(collocation_points)

        # Use B-spline parameterization for smoothness
        n_control_points = min(20, n_points // 2)
        control_points = np.linspace(0, 2 * np.pi, n_control_points)

        # Initial guess: smooth S-curve (physical units)
        initial_guess_phys = self._generate_initial_guess_minimum_energy(
            control_points,
            constraints,
        )

        # Scaling: normalize control-point positions by stroke
        pos_scale = 1.0 / max(float(constraints.stroke), 1e-6)
        initial_guess_norm = initial_guess_phys * pos_scale

        # Define objective function (minimize acceleration^2) operating on normalized parameters by converting to physical
        def objective(params_norm):
            params_phys = params_norm / pos_scale
            return self._minimum_energy_objective(
                params_phys, control_points, collocation_points,
            )

        # Define constraints and wrap to accept normalized parameters
        base_constraints = self._define_motion_law_constraints(
            initial_guess_phys,
            control_points,
            collocation_points,
            constraints,
        )
        constraint_list = [
            {"type": c["type"], "fun": (lambda p, f=c["fun"]: f(p / pos_scale))}
            for c in base_constraints
        ]

        # Solve optimization in normalized parameter space
        result = minimize(
            objective,
            initial_guess_norm,
            method="SLSQP",
            constraints=constraint_list,
            options={
                "maxiter": self.max_iterations,
                "ftol": self.tolerance,
                "disp": False,
            },
        )

        if not result.success:
            log.warning(
                f"Minimum energy optimization did not converge: {result.message}",
            )

        # Extract solution (convert params back to physical units)
        params_phys = result.x / pos_scale
        solution = self._extract_solution_minimum_energy(
            params_phys,
            control_points,
            collocation_points,
        )

        return {
            "position": solution["position"],
            "velocity": solution["velocity"],
            "acceleration": solution["acceleration"],
            "jerk": solution["jerk"],
            "objective_value": result.fun,
            "iterations": result.nit,
        }

    def _generate_initial_guess_minimum_energy(
        self, control_points: np.ndarray, constraints: MotionLawConstraints,
    ) -> np.ndarray:
        """Generate initial guess for minimum energy optimization."""
        # Create a smooth S-curve initial guess (similar to minimum jerk)
        n_control = len(control_points)
        initial_guess = np.zeros(n_control)

        # Create S-curve profile
        for i, theta in enumerate(control_points):
            if theta <= constraints.upstroke_angle:
                # Upstroke phase
                tau = theta / constraints.upstroke_angle
                # S-curve: 6t^5 - 15t^4 + 10t^3
                initial_guess[i] = constraints.stroke * (
                    6 * tau**5 - 15 * tau**4 + 10 * tau**3
                )
            else:
                # Downstroke phase
                downstroke_tau = (
                    theta - constraints.upstroke_angle
                ) / constraints.downstroke_angle
                # Mirror the upstroke
                initial_guess[i] = constraints.stroke * (
                    1
                    - (
                        6 * downstroke_tau**5
                        - 15 * downstroke_tau**4
                        + 10 * downstroke_tau**3
                    )
                )

        return initial_guess

    def _minimum_energy_objective(
        self,
        params: np.ndarray,
        control_points: np.ndarray,
        collocation_points: np.ndarray,
    ) -> float:
        """Calculate minimum energy objective function."""
        # Interpolate control points to collocation points
        cs = CubicSpline(control_points, params, bc_type="natural")
        position = cs(collocation_points)

        # Calculate derivatives
        velocity = cs.derivative()(collocation_points)
        acceleration = cs.derivative(2)(collocation_points)

        # Objective: minimize acceleration squared (energy)
        objective = np.trapz(acceleration**2, collocation_points)

        return objective

    def _extract_solution_minimum_energy(
        self,
        params: np.ndarray,
        control_points: np.ndarray,
        collocation_points: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Extract solution from minimum energy optimization result."""
        cs = CubicSpline(control_points, params, bc_type="natural")

        position = cs(collocation_points)
        velocity = cs.derivative()(collocation_points)
        acceleration = cs.derivative(2)(collocation_points)
        jerk = cs.derivative(3)(collocation_points)

        return {
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "jerk": jerk,
        }
