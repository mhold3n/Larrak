"""
CasADi-based motion law optimizer using Opti stack with direct collocation.

This module implements Phase 1 optimization using CasADi's Opti stack for
motion law optimization with thermal efficiency objectives and warm-starting
capabilities.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from casadi import *

from campro.logging import get_logger
from campro.optimization.base import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationStatus,
)

log = get_logger(__name__)


@dataclass
class CasADiMotionProblem:
    """Problem specification for CasADi motion law optimization."""

    # Boundary conditions
    stroke: float
    cycle_time: float
    upstroke_percent: float

    # Physical constraints
    max_velocity: float
    max_acceleration: float
    max_jerk: float
    compression_ratio_limits: Tuple[float, float] = (20.0, 70.0)

    # Objectives
    minimize_jerk: bool = True
    maximize_thermal_efficiency: bool = True
    weights: Dict[str, float] = None

    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                "jerk": 1.0,
                "thermal_efficiency": 0.1,
                "smoothness": 0.01,
            }


class CasADiMotionOptimizer(BaseOptimizer):
    """
    CasADi-based motion law optimizer using Opti stack with direct collocation.

    Implements Phase 1 optimization with:
    - Direct collocation with Legendre/Radau polynomials
    - Thermal efficiency objectives from FPE literature
    - Warm-starting capabilities
    - Physics-based constraints
    """

    def __init__(
        self,
        n_segments: int = 50,
        poly_order: int = 3,
        collocation_method: str = "legendre",
    ):
        """
        Initialize CasADi motion optimizer.

        Parameters
        ----------
        n_segments : int
            Number of finite elements for collocation
        poly_order : int
            Polynomial order for collocation (3 = cubic)
        collocation_method : str
            Collocation method: "legendre" or "radau"
        """
        super().__init__()
        self.n_segments = n_segments
        self.poly_order = poly_order
        self.collocation_method = collocation_method

        # Initialize Opti stack
        self.opti = Opti()

        # Set default solver options
        self.opti.solver(
            "ipopt",
            {
                "ipopt.linear_solver": "ma57",
                "ipopt.max_iter": 1000,
                "ipopt.tol": 1e-6,
                "ipopt.print_level": 0,
                "ipopt.warm_start_init_point": "yes",
            },
        )

        log.info(
            f"Initialized CasADiMotionOptimizer: {n_segments} segments, "
            f"order {poly_order}, method {collocation_method}",
        )

    def setup_collocation(self, problem: CasADiMotionProblem) -> Dict[str, Any]:
        """
        Setup direct collocation discretization.

        Parameters
        ----------
        problem : CasADiMotionProblem
            Problem specification

        Returns
        -------
        Dict[str, Any]
            Collocation variables and parameters
        """
        # Time discretization
        T = problem.cycle_time
        dt = T / self.n_segments

        # State variables: position, velocity, acceleration
        x = self.opti.variable(self.n_segments + 1)  # position
        v = self.opti.variable(self.n_segments + 1)  # velocity
        a = self.opti.variable(self.n_segments + 1)  # acceleration

        # Control variable: jerk
        j = self.opti.variable(self.n_segments)  # jerk

        # Collocation points (Legendre-Gauss-Radau)
        if self.collocation_method == "legendre":
            tau_root = np.polynomial.legendre.leggauss(self.poly_order)[0]
        else:  # radau
            tau_root = np.polynomial.legendre.leggauss(self.poly_order)[0]
            tau_root = np.concatenate([[-1], tau_root])

        # Collocation matrices
        C = np.zeros((self.poly_order + 1, self.poly_order + 1))
        D = np.zeros((self.poly_order + 1, 1))

        # Construct collocation matrices
        for j in range(self.poly_order + 1):
            # Construct Lagrange polynomial
            coeffs = np.zeros(self.poly_order + 1)
            coeffs[j] = 1.0
            poly = np.polynomial.Polynomial(coeffs)

            # Evaluate at collocation points
            for k in range(self.poly_order + 1):
                C[j, k] = poly(tau_root[k])
                D[j, 0] = poly(1.0)  # End point

        return {
            "x": x,
            "v": v,
            "a": a,
            "j": j,
            "dt": dt,
            "C": C,
            "D": D,
            "tau_root": tau_root,
        }

    def add_boundary_conditions(
        self, problem: CasADiMotionProblem, collocation_vars: Dict[str, Any],
    ) -> None:
        """Add boundary conditions to the optimization problem."""
        x, v = collocation_vars["x"], collocation_vars["v"]

        # Position boundary conditions
        self.opti.subject_to(x[0] == 0)  # Start at zero
        self.opti.subject_to(x[-1] == problem.stroke)  # End at stroke

        # Velocity boundary conditions
        self.opti.subject_to(v[0] == 0)  # Start at rest
        self.opti.subject_to(v[-1] == 0)  # End at rest

        # Upstroke constraint
        upstroke_time = problem.cycle_time * problem.upstroke_percent / 100.0
        upstroke_index = int(upstroke_time / (problem.cycle_time / self.n_segments))
        upstroke_index = min(upstroke_index, self.n_segments)

        # Velocity should be positive during upstroke
        for i in range(upstroke_index):
            self.opti.subject_to(v[i] >= 0)

        # Velocity should be negative during downstroke
        for i in range(upstroke_index, self.n_segments + 1):
            self.opti.subject_to(v[i] <= 0)

    def add_motion_constraints(
        self, problem: CasADiMotionProblem, collocation_vars: Dict[str, Any],
    ) -> None:
        """Add motion constraints (velocity, acceleration, jerk limits)."""
        v, a, j = collocation_vars["v"], collocation_vars["a"], collocation_vars["j"]

        # Velocity constraints
        for i in range(self.n_segments + 1):
            self.opti.subject_to(
                self.opti.bounded(-problem.max_velocity, v[i], problem.max_velocity),
            )

        # Acceleration constraints
        for i in range(self.n_segments + 1):
            self.opti.subject_to(
                self.opti.bounded(
                    -problem.max_acceleration, a[i], problem.max_acceleration,
                ),
            )

        # Jerk constraints
        for i in range(self.n_segments):
            self.opti.subject_to(
                self.opti.bounded(-problem.max_jerk, j[i], problem.max_jerk),
            )

    def add_physics_constraints(
        self, problem: CasADiMotionProblem, collocation_vars: Dict[str, Any],
    ) -> None:
        """Add physics-based constraints from FPE literature."""
        x, v, a = collocation_vars["x"], collocation_vars["v"], collocation_vars["a"]
        dt = collocation_vars["dt"]

        # Compression ratio constraints
        # CR = V_max / V_min = (stroke + clearance) / clearance
        # Simplified: CR proportional to stroke
        clearance = 0.002  # 2mm clearance
        max_cr = (problem.stroke + clearance) / clearance
        min_cr = problem.compression_ratio_limits[0]

        # Ensure compression ratio is within limits
        for i in range(self.n_segments + 1):
            cr = (x[i] + clearance) / clearance
            self.opti.subject_to(cr >= min_cr)
            self.opti.subject_to(cr <= min(max_cr, problem.compression_ratio_limits[1]))

        # Pressure rate constraint (avoid diesel knock)
        # Simplified: limit acceleration rate
        max_pressure_rate = 1000.0  # Pa/ms
        for i in range(self.n_segments - 1):
            pressure_rate = abs(a[i + 1] - a[i]) / dt
            self.opti.subject_to(pressure_rate <= max_pressure_rate)

    def add_collocation_constraints(self, collocation_vars: Dict[str, Any]) -> None:
        """Add collocation constraints for state continuity."""
        x, v, a, j = (
            collocation_vars["x"],
            collocation_vars["v"],
            collocation_vars["a"],
            collocation_vars["j"],
        )
        dt, C, D = collocation_vars["dt"], collocation_vars["C"], collocation_vars["D"]

        # State continuity constraints
        for k in range(self.n_segments):
            # Position continuity: x[k+1] = x[k] + integral(v)
            # Velocity continuity: v[k+1] = v[k] + integral(a)
            # Acceleration continuity: a[k+1] = a[k] + integral(j)

            # Simplified collocation (for now, use trapezoidal rule)
            self.opti.subject_to(x[k + 1] == x[k] + 0.5 * dt * (v[k] + v[k + 1]))
            self.opti.subject_to(v[k + 1] == v[k] + 0.5 * dt * (a[k] + a[k + 1]))
            self.opti.subject_to(a[k + 1] == a[k] + 0.5 * dt * (j[k] + j[k + 1]))

    def add_thermal_efficiency_objective(
        self, problem: CasADiMotionProblem, collocation_vars: Dict[str, Any],
    ) -> None:
        """Add thermal efficiency objective from FPE literature."""
        x, v, a = collocation_vars["x"], collocation_vars["v"], collocation_vars["a"]
        dt = collocation_vars["dt"]

        # Thermal efficiency objective (simplified Otto cycle)
        # η_thermal ≈ 1 - 1/CR^(γ-1) + heat_loss_penalty + mech_loss_penalty

        gamma = 1.4  # Specific heat ratio
        clearance = 0.002  # 2mm clearance

        # Compression ratio at each point
        cr = (x + clearance) / clearance

        # Otto cycle efficiency: 1 - 1/CR^(γ-1)
        otto_efficiency = 1 - 1 / (cr ** (gamma - 1))

        # Heat loss penalty (proportional to velocity - Woschni correlation simplified)
        heat_loss_penalty = 0.1 * v**2

        # Mechanical loss penalty (proportional to acceleration)
        mech_loss_penalty = 0.01 * a**2

        # Total thermal efficiency (negative for minimization)
        thermal_efficiency = otto_efficiency - heat_loss_penalty - mech_loss_penalty

        # Average thermal efficiency over cycle
        avg_thermal_efficiency = sum(thermal_efficiency) / len(thermal_efficiency)

        # Add to objective (minimize negative efficiency)
        self.opti.minimize(
            -problem.weights["thermal_efficiency"] * avg_thermal_efficiency,
        )

    def add_jerk_objective(
        self, problem: CasADiMotionProblem, collocation_vars: Dict[str, Any],
    ) -> None:
        """Add jerk minimization objective for smoothness."""
        j = collocation_vars["j"]
        dt = collocation_vars["dt"]

        # Jerk squared integral (minimize for smoothness)
        jerk_squared = sum(j**2) * dt

        # Add to objective
        self.opti.minimize(problem.weights["jerk"] * jerk_squared)

    def solve(
        self,
        problem: CasADiMotionProblem,
        initial_guess: Optional[Dict[str, np.ndarray]] = None,
    ) -> OptimizationResult:
        """
        Solve the motion law optimization problem.

        Parameters
        ----------
        problem : CasADiMotionProblem
            Problem specification
        initial_guess : Optional[Dict[str, np.ndarray]]
            Initial guess for variables

        Returns
        -------
        OptimizationResult
            Optimization results
        """
        try:
            # Setup collocation
            collocation_vars = self.setup_collocation(problem)

            # Add constraints
            self.add_boundary_conditions(problem, collocation_vars)
            self.add_motion_constraints(problem, collocation_vars)
            self.add_physics_constraints(problem, collocation_vars)
            self.add_collocation_constraints(collocation_vars)

            # Add objectives
            if problem.minimize_jerk:
                self.add_jerk_objective(problem, collocation_vars)

            if problem.maximize_thermal_efficiency:
                self.add_thermal_efficiency_objective(problem, collocation_vars)

            # Set initial guess if provided
            if initial_guess:
                self.opti.set_initial(collocation_vars["x"], initial_guess.get("x"))
                self.opti.set_initial(collocation_vars["v"], initial_guess.get("v"))
                self.opti.set_initial(collocation_vars["a"], initial_guess.get("a"))
                self.opti.set_initial(collocation_vars["j"], initial_guess.get("j"))

            # Solve
            log.info("Solving CasADi motion law optimization...")
            sol = self.opti.solve()

            # Extract results
            x_opt = sol.value(collocation_vars["x"])
            v_opt = sol.value(collocation_vars["v"])
            a_opt = sol.value(collocation_vars["a"])
            j_opt = sol.value(collocation_vars["j"])

            # Compute objective value
            objective_value = sol.value(self.opti.f)

            # Create result
            result = OptimizationResult(
                status=OptimizationStatus.CONVERGED,
                successful=True,
                objective_value=objective_value,
                solve_time=sol.stats()["t_wall_total"],
                variables={
                    "position": x_opt,
                    "velocity": v_opt,
                    "acceleration": a_opt,
                    "jerk": j_opt,
                },
                metadata={
                    "n_segments": self.n_segments,
                    "poly_order": self.poly_order,
                    "collocation_method": self.collocation_method,
                    "solver_stats": sol.stats(),
                },
            )

            log.info(
                f"CasADi optimization completed successfully in {result.solve_time:.3f}s",
            )
            return result

        except Exception as e:
            log.error(f"CasADi optimization failed: {e}")
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                successful=False,
                objective_value=float("inf"),
                solve_time=0.0,
                variables={},
                metadata={"error": str(e)},
            )

    def optimize(self, constraints, targets, **kwargs) -> OptimizationResult:
        """
        Optimize motion law with given constraints and targets.

        This method provides compatibility with the existing optimization interface.
        """
        # Convert constraints and targets to CasADiMotionProblem
        problem = CasADiMotionProblem(
            stroke=constraints.get("stroke", 0.1),
            cycle_time=constraints.get("cycle_time", 0.0385),
            upstroke_percent=constraints.get("upstroke_percent", 50.0),
            max_velocity=constraints.get("max_velocity", 5.0),
            max_acceleration=constraints.get("max_acceleration", 500.0),
            max_jerk=constraints.get("max_jerk", 50000.0),
            compression_ratio_limits=constraints.get(
                "compression_ratio_limits", (20.0, 70.0),
            ),
            minimize_jerk=targets.get("minimize_jerk", True),
            maximize_thermal_efficiency=targets.get(
                "maximize_thermal_efficiency", True,
            ),
            weights=targets.get("weights", {}),
        )

        # Get initial guess if provided
        initial_guess = kwargs.get("initial_guess")

        return self.solve(problem, initial_guess)
