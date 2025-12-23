"""
IPOPT Solver Adapter: Wraps solve_cycle as SolverInterface.

Adapts the existing IPOPT-based solver to work with the orchestrator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from campro.logging import get_logger
from campro.orchestration.orchestrator import SolverInterface

log = get_logger(__name__)


class IPOPTSolverAdapter:
    """
    Adapter wrapping solve_cycle as SolverInterface.

    Converts between candidate dict format and params format expected
    by the existing NLP solver.
    """

    def __init__(
        self,
        base_params: dict[str, Any] | None = None,
        use_warm_start: bool = True,
    ):
        """
        Initialize adapter.

        Args:
            base_params: Base parameters to merge with candidate
            use_warm_start: Whether to use warm start when refining
        """
        self.base_params = base_params or {}
        self.use_warm_start = use_warm_start
        self._last_solution: dict[str, Any] | None = None

    def refine(
        self,
        candidate: dict[str, Any],
        objective_fn: Any,
        max_step: np.ndarray,
    ) -> dict[str, Any]:
        """
        Refine candidate using IPOPT solver.

        Args:
            candidate: Candidate design as parameter dict
            objective_fn: Objective function (from surrogate)
            max_step: Maximum step sizes (from trust region)

        Returns:
            Refined candidate dict
        """
        from campro.optimization.driver import solve_cycle

        # Merge base params with candidate
        params = {**self.base_params, **candidate}

        # Apply step limits as tighter bounds
        if "bounds" in params and max_step is not None:
            params = self._apply_step_limits(params, max_step)

        # Warm start from last solution
        if self.use_warm_start and self._last_solution is not None:
            params["warm_start"] = self._last_solution

        try:
            result = solve_cycle(params)

            # Store solution for warm start
            if hasattr(result, "data") and "x" in result.data:
                self._last_solution = {"x": result.data["x"]}

            # Convert result back to candidate format
            return self._result_to_candidate(result, candidate)

        except Exception as e:
            log.warning(f"Solver refinement failed: {e}")
            return candidate  # Return original on failure

    def _apply_step_limits(
        self,
        params: dict[str, Any],
        max_step: np.ndarray,
    ) -> dict[str, Any]:
        """Tighten bounds based on trust region step limits."""
        params = params.copy()

        if "x0" in params:
            x0 = np.array(params["x0"])

            # Tighten bounds around current point
            if "lb" in params:
                lb = np.array(params["lb"])
                params["lb"] = np.maximum(lb, x0 - max_step[: len(x0)])

            if "ub" in params:
                ub = np.array(params["ub"])
                params["ub"] = np.minimum(ub, x0 + max_step[: len(x0)])

        return params

    def _result_to_candidate(
        self,
        result: Any,
        original: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert solver result to candidate dict format."""
        candidate = original.copy()

        # Extract key outputs
        if hasattr(result, "success"):
            candidate["__solver_success"] = result.success

        if hasattr(result, "objective"):
            candidate["__objective"] = result.objective

        if hasattr(result, "x") or (hasattr(result, "data") and "x" in result.data):
            candidate["__x_solution"] = result.data["x"] if hasattr(result, "data") else result.x

        # Update design variables
        if hasattr(result, "design_vars"):
            candidate.update(result.design_vars)

        return candidate


class SimpleSolverAdapter:
    """
    Simple adapter for local refinement without full NLP.

    Uses gradient-free local search for quick refinement.
    """

    def __init__(self, step_scale: float = 0.1, n_evals: int = 10):
        self.step_scale = step_scale
        self.n_evals = n_evals

    def refine(
        self,
        candidate: dict[str, Any],
        objective_fn: Any,
        max_step: np.ndarray,
    ) -> dict[str, Any]:
        """
        Refine using simple local search.

        Evaluates objective_fn at nearby points and returns best.
        """
        # Extract numeric values
        numeric_keys = [
            k for k, v in candidate.items() if isinstance(v, (int, float)) and not k.startswith("_")
        ]

        if not numeric_keys:
            return candidate

        best_candidate = candidate.copy()
        try:
            best_obj = objective_fn(candidate)
        except Exception:
            return candidate

        # Random local search
        rng = np.random.default_rng()
        for _ in range(self.n_evals):
            trial = candidate.copy()

            for i, key in enumerate(numeric_keys):
                step = max_step[i % len(max_step)] if max_step is not None else self.step_scale
                delta = rng.uniform(-step, step)
                trial[key] = candidate[key] + delta

            try:
                obj = objective_fn(trial)
                if obj > best_obj:  # Maximizing
                    best_obj = obj
                    best_candidate = trial
            except Exception:
                continue

        return best_candidate


__all__ = [
    "IPOPTSolverAdapter",
    "SimpleSolverAdapter",
]
