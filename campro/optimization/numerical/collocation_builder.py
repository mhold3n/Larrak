"""
Collocation Builder DSL.

This module provides a high-level builder for constructing direct collocation
problems with CasADi and IPOPT. It abstracts the details of mesh generation,
polynomial approximation, and defect constraints.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import casadi as ca
import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class CollocationVariable:
    """Metadata for a decision variable."""

    name: str
    symbol: ca.SX
    is_state: bool
    bounds: tuple[float, float] = (-np.inf, np.inf)
    initial: float | np.ndarray = 0.0


class CollocationBuilder:
    """
    Builder for direct collocation problems.

    This class manages the construction of the NLP, including:
    - Time grid and intervals
    - State and control variables
    - Collocation constraints (defects)
    - Continuity constraints
    - Boundary conditions
    """

    def __init__(
        self,
        time_horizon: float,
        n_points: int,
        method: Literal["legendre", "radau"] = "radau",
        degree: int = 3,
    ):
        """
        Initialize the builder.

        Args:
            time_horizon: Total duration of the phase.
            n_points: Number of control intervals (mesh points).
            method: Collocation method ("legendre" or "radau").
            degree: Degree of interpolating polynomial.
        """
        self.T = time_horizon
        self.N = n_points
        self.method = method
        self.degree = degree
        self.h = self.T / self.N  # Time step

        self.states: dict[str, CollocationVariable] = {}
        self.controls: dict[str, CollocationVariable] = {}
        self.parameters: dict[str, ca.SX] = {}

        self.dynamics: Callable | None = None
        self.path_constraints: list[tuple[Any, tuple[float, float]]] = []
        self._boundary_conditions: list[tuple[Callable, float, str]] = []

        # Collocation coefficients
        self.B: np.ndarray | None = None
        self.C: np.ndarray | None = None
        self.D: np.ndarray | None = None
        self._setup_collocation_coeffs()

        # NLP lists
        self.w: list[ca.SX] = []
        self.w0: list[float] = []
        self.lbw: list[float] = []
        self.ubw: list[float] = []
        self.g: list[ca.SX] = []
        self.lbg: list[float] = []
        self.ubg: list[float] = []
        self.J: ca.SX = 0

        # Storage for built variables (symbolic)
        # _X[name][k] -> state at start of interval k (and final point N)
        self._X: dict[str, list[ca.SX]] = {}
        # _U[name][k] -> control at interval k
        self._U: dict[str, list[ca.SX]] = {}
        # _XC[name][k][j] -> state at collocation point j of interval k
        self._XC: dict[str, list[list[ca.SX]]] = {}

        # Map from variable name to indices in w (for result extraction)
        self._var_indices: dict[str, list[int]] = {}

        self.constraint_groups: dict[str, list[int]] = {}

    def _add_constraint_indices(self, group: str, indices: list[int]):
        """Helper to add indices to a constraint group."""
        if group not in self.constraint_groups:
            self.constraint_groups[group] = []
        self.constraint_groups[group].extend(indices)

    def _setup_collocation_coeffs(self):
        """Calculate collocation coefficients (B, C, D matrices)."""
        # Get collocation points
        tau_root = ca.collocation_points(self.degree, self.method)

        # Add 0 to the beginning
        tau = [0] + tau_root

        # Coefficients of the collocation equation
        self.C = np.zeros((self.degree + 1, self.degree + 1))

        # Coefficients of the continuity equation
        self.D = np.zeros(self.degree + 1)

        # Coefficients of the quadrature function
        self.B = np.zeros(self.degree + 1)

        # Construct Lagrange polynomials
        for j in range(self.degree + 1):
            p = np.poly1d([1])
            for r in range(self.degree + 1):
                if r != j:
                    p *= np.poly1d([1, -tau[r]]) / (tau[j] - tau[r])

            self.D[j] = p(1.0)
            pder = np.polyder(p)
            for r in range(self.degree + 1):
                self.C[j, r] = pder(tau[r])
            pint = np.polyint(p)
            self.B[j] = pint(1.0)

        # Snap small values to zero to prevent numerical noise in Jacobian
        self.C[np.abs(self.C) < 1e-14] = 0.0
        self.D[np.abs(self.D) < 1e-14] = 0.0
        self.B[np.abs(self.B) < 1e-14] = 0.0

    def add_state(
        self,
        name: str,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        initial: float | np.ndarray = 0.0,
    ) -> ca.SX:
        """Add a state variable."""
        if name in self.states:
            raise ValueError(f"State '{name}' already exists.")

        sym = ca.SX.sym(name)
        self.states[name] = CollocationVariable(
            name=name, symbol=sym, is_state=True, bounds=bounds, initial=initial
        )
        return sym

    def add_control(
        self,
        name: str,
        bounds: tuple[float, float] = (-np.inf, np.inf),
        initial: float | np.ndarray = 0.0,
    ) -> ca.SX:
        """Add a control variable."""
        if name in self.controls:
            raise ValueError(f"Control '{name}' already exists.")

        sym = ca.SX.sym(name)
        self.controls[name] = CollocationVariable(
            name=name, symbol=sym, is_state=False, bounds=bounds, initial=initial
        )
        return sym

    def add_parameter(self, name: str, value: float) -> ca.SX:
        """Add a fixed parameter (treated as a constant in expressions)."""
        # For SX, parameters are just constants or symbols substituted later.
        # Here we'll return a constant SX for simplicity in expressions.
        # If we wanted to optimize over it, we'd add it to w.
        # If we want it to be changeable without rebuilding, we'd use ca.SX.sym and pass as param.
        # But for this builder, let's treat it as a constant value injected into expressions.
        p = ca.SX(value)
        self.parameters[name] = p
        return p

    def set_dynamics(self, f: Callable[[Any, Any], Any]):
        """Set the system dynamics x_dot = f(x, u)."""
        self.dynamics = f

    def add_path_constraint(
        self, expr: Any, bounds: tuple[float, float] = (-np.inf, np.inf)
    ):
        """Add a path constraint enforced at all collocation points."""
        self.path_constraints.append((expr, bounds))

    def add_boundary_condition(
        self,
        expr_fn: Callable[[dict[str, Any], dict[str, Any]], Any],
        val: float,
        loc: Literal["initial", "final"] = "initial",
    ):
        """Add a boundary condition."""
        self._boundary_conditions.append((expr_fn, val, loc))

    def set_objective(self, expr: ca.SX):
        """Set the objective function to minimize."""
        self.J = expr

    def build(self):
        """Construct the NLP variables and constraints."""
        if self.dynamics is None:
            raise ValueError("Dynamics must be set before building.")

        # Initialize storage
        for name in self.states:
            self._X[name] = []
            self._XC[name] = []
            self._var_indices[name] = []
        for name in self.controls:
            self._U[name] = []
            self._var_indices[name] = []

        # 1. Create variables for each interval
        # Initial point (k=0)
        for name, var in self.states.items():
            Xk = ca.SX.sym(f"{name}_0")
            self.w.append(Xk)
            self.lbw.append(var.bounds[0])
            self.ubw.append(var.bounds[1])
            self.w0.append(
                float(var.initial) if np.isscalar(var.initial) else var.initial[0]
            )  # Handle array initial
            self._X[name].append(Xk)
            self._var_indices[name].append(len(self.w) - 1)

        for k in range(self.N):
            # Controls for this interval
            for name, var in self.controls.items():
                Uk = ca.SX.sym(f"{name}_{k}")
                self.w.append(Uk)
                self.lbw.append(var.bounds[0])
                self.ubw.append(var.bounds[1])
                self.w0.append(
                    float(var.initial) if np.isscalar(var.initial) else var.initial[0]
                )
                self._U[name].append(Uk)
                self._var_indices[name].append(len(self.w) - 1)

            # Collocation points
            for name, var in self.states.items():
                Xc_k = []
                for j in range(self.degree):
                    Xkj = ca.SX.sym(f"{name}_{k}_{j + 1}")
                    self.w.append(Xkj)
                    self.lbw.append(var.bounds[0])
                    self.ubw.append(var.bounds[1])
                    self.w0.append(
                        float(var.initial)
                        if np.isscalar(var.initial)
                        else var.initial[0]
                    )
                    Xc_k.append(Xkj)
                    self._var_indices[name].append(
                        len(self.w) - 1
                    )  # Note: this mixes grid and colloc points in indices
                self._XC[name].append(Xc_k)

            # End point of interval (next state)
            for name, var in self.states.items():
                Xnext = ca.SX.sym(f"{name}_{k + 1}")
                self.w.append(Xnext)
                self.lbw.append(var.bounds[0])
                self.ubw.append(var.bounds[1])
                self.w0.append(
                    float(var.initial) if np.isscalar(var.initial) else var.initial[0]
                )
                self._X[name].append(Xnext)
                self._var_indices[name].append(len(self.w) - 1)

        # 2. Loop over intervals and apply collocation equations
        for k in range(self.N):
            # Gather states/controls for this interval
            X_k = {name: self._X[name][k] for name in self.states}
            U_k = {name: self._U[name][k] for name in self.controls}
            XC_k = [
                {name: self._XC[name][k][j] for name in self.states}
                for j in range(self.degree)
            ]
            X_next = {name: self._X[name][k + 1] for name in self.states}

            # Loop over collocation points
            for j in range(1, self.degree + 1):
                # Evaluate dynamics once per collocation point
                state_at_j = XC_k[j - 1]
                dynamics_eval = self.dynamics(state_at_j, U_k)

                # Expression for the state derivative at the collocation point
                for name in self.states:
                    approx_deriv = self.C[0, j] * X_k[name]
                    for r in range(self.degree):
                        approx_deriv += self.C[r + 1, j] * XC_k[r][name]

                    dx_dt = dynamics_eval[name]

                    # Collocation constraint
                    self.g.append(approx_deriv - dx_dt * self.h)
                    self.lbg.append(0.0)
                    self.ubg.append(0.0)
                    self._add_constraint_indices(
                        f"collocation_{name}", [len(self.g) - 1]
                    )

            # Continuity constraint
            for name in self.states:
                continuity_expr = self.D[0] * X_k[name]
                for r in range(self.degree):
                    continuity_expr += self.D[r + 1] * XC_k[r][name]

                self.g.append(X_next[name] - continuity_expr)
                self.lbg.append(0.0)
                self.ubg.append(0.0)
                self._add_constraint_indices(f"continuity_{name}", [len(self.g) - 1])

        # 3. Apply Boundary Conditions
        for i, (expr_fn, val, loc) in enumerate(self._boundary_conditions):
            idx = 0 if loc == "initial" else -1
            X_b = {name: self._X[name][idx] for name in self.states}
            U_idx = 0 if loc == "initial" else -1
            U_b = {name: self._U[name][U_idx] for name in self.controls}

            expr = expr_fn(X_b, U_b)
            self.g.append(expr - val)
            self.lbg.append(0.0)
            self.ubg.append(0.0)
            self._add_constraint_indices(f"boundary_{loc}_{i}", [len(self.g) - 1])

    def export_nlp(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Export the NLP in standard CasADi format.

        Returns:
            nlp: Dictionary with 'x', 'f', 'g'.
            meta: Dictionary with 'w0', 'lbw', 'ubw', 'lbg', 'ubg', 'n_vars', 'n_constraints'.
        """
        nlp = {"x": ca.vertcat(*self.w), "f": self.J, "g": ca.vertcat(*self.g)}

        meta = {
            "w0": np.array(self.w0),
            "lbw": np.array(self.lbw),
            "ubw": np.array(self.ubw),
            "lbg": np.array(self.lbg),
            "ubg": np.array(self.ubg),
            "n_vars": len(self.w),
            "n_constraints": len(self.g),
            "K": self.N,
            "C": self.degree,  # Assuming degree corresponds to C collocation points
            "constraint_groups": self.constraint_groups,
        }
        return nlp, meta

    def solve(self, solver_opts: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        """
        Solve the NLP using CasADi's nlpsol.

        Args:
            solver_opts: Options passed to ca.nlpsol.

        Returns:
            Dictionary of results {var_name: array_of_values}.
        """
        nlp, meta = self.export_nlp()

        opts = solver_opts or {}
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        sol = solver(
            x0=meta["w0"],
            lbx=meta["lbw"],
            ubx=meta["ubw"],
            lbg=meta["lbg"],
            ubg=meta["ubg"],
        )

        w_opt = sol["x"].full().flatten()

        # Extract results
        results = {}

        # Helper to extract variable values
        # We need to filter indices to get only the grid points (start of intervals + final point)
        # _X[name] contains [X0, X1, ..., XN] (N+1 points)
        # But self.w contains them mixed with collocation points.
        # However, _X[name] stores the symbolic variables.
        # We can find their indices in self.w?
        # No, self.w contains the symbols. We can match by identity?
        # Or just use the structure we built.

        # Actually, we can just use ca.Function to evaluate the symbols given w_opt
        # But that's slow.
        # Better: we know the structure.

        # Let's use the symbols in _X and _U to evaluate.
        # Create a function that outputs all X grid points given w.

        outputs = []
        output_names = []

        for name in self.states:
            # Stack all grid points
            outputs.append(ca.vertcat(*self._X[name]))
            output_names.append(name)

        for name in self.controls:
            outputs.append(ca.vertcat(*self._U[name]))
            output_names.append(name)

        # Create extraction function
        extractor = ca.Function("extractor", [nlp["x"]], outputs)
        res_vals = extractor(w_opt)

        for i, name in enumerate(output_names):
            results[name] = res_vals[i].full().flatten()

        results["_w_opt"] = w_opt

        return results

    def get_time_grid(self) -> np.ndarray:
        """Get the time grid."""
        return np.linspace(0, self.T, self.N + 1)
