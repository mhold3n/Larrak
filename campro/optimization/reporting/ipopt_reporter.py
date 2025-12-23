"""IPOPT solver reporting and diagnostics.

This module provides structured reporting for IPOPT optimization runs,
including iteration summaries, convergence tracking, and diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class IterationData:
    """Data for a single IPOPT iteration."""

    iteration: int
    objective: float
    inf_pr: float  # Primal infeasibility
    inf_du: float  # Dual infeasibility
    mu: float  # Barrier parameter
    step_type: str = ""
    alpha_pr: float = 0.0  # Primal step size
    alpha_du: float = 0.0  # Dual step size


@dataclass
class SolveSummary:
    """Summary of an IPOPT solve."""

    status: str
    iterations: int
    objective: float
    primal_infeasibility: float
    dual_infeasibility: float
    total_time_s: float
    restoration_steps: int = 0
    termination_reason: str = ""
    iteration_history: list[IterationData] = field(default_factory=list)


# =============================================================================
# Reporter Class
# =============================================================================


class IPOPTReporter:
    """Reporter for IPOPT solver diagnostics.

    Provides methods for logging iteration progress, summarizing results,
    and generating diagnostic reports.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        verbose: bool = True,
        show_iterations: bool = False,
    ) -> None:
        """Initialize reporter.

        Args:
            logger: Logger instance (uses module logger if None)
            verbose: Whether to output verbose messages
            show_iterations: Whether to log each iteration
        """
        self.logger = logger or log
        self.verbose = verbose
        self.show_iterations = show_iterations
        self._iterations: list[IterationData] = []

    def log_iteration(self, data: IterationData) -> None:
        """Log a single iteration."""
        self._iterations.append(data)
        if self.show_iterations:
            self.logger.debug(
                f"Iter {data.iteration:4d}: obj={data.objective:.6e} "
                f"inf_pr={data.inf_pr:.2e} inf_du={data.inf_du:.2e} "
                f"mu={data.mu:.2e} {data.step_type}"
            )

    def summarize_solve(
        self,
        stats: dict[str, Any],
        solve_time: float,
    ) -> SolveSummary:
        """Generate solve summary from IPOPT stats.

        Args:
            stats: IPOPT solver statistics dictionary
            solve_time: Total solve time in seconds

        Returns:
            SolveSummary with key metrics
        """
        # Extract iterations from stats
        iterations = stats.get("iterations", {})
        if isinstance(iterations, dict):
            k = _to_array(iterations.get("k", []))
            obj = _to_array(iterations.get("obj") or iterations.get("f", []))
            inf_pr = _to_array(iterations.get("inf_pr", []))
            inf_du = _to_array(iterations.get("inf_du", []))
            mu = _to_array(iterations.get("mu", []))
            step_types = iterations.get("type", []) or iterations.get("step_type", [])
            if hasattr(step_types, "tolist"):
                step_types = step_types.tolist()
            step_types = [str(s) for s in step_types]

            n_iters = len(k)
            restoration_steps = sum(1 for s in step_types if "r" in s.lower())

            # Build iteration history
            history = []
            for i in range(n_iters):
                history.append(
                    IterationData(
                        iteration=int(k[i]) if i < len(k) else i,
                        objective=float(obj[i]) if i < len(obj) else float("nan"),
                        inf_pr=float(inf_pr[i]) if i < len(inf_pr) else float("nan"),
                        inf_du=float(inf_du[i]) if i < len(inf_du) else float("nan"),
                        mu=float(mu[i]) if i < len(mu) else float("nan"),
                        step_type=step_types[i] if i < len(step_types) else "",
                    )
                )
        else:
            n_iters = 0
            restoration_steps = 0
            obj = np.array([])
            inf_pr = np.array([])
            inf_du = np.array([])
            history = []

        # Extract final values
        final_obj = float(obj[-1]) if len(obj) > 0 else float(stats.get("f", float("nan")))
        final_inf_pr = float(inf_pr[-1]) if len(inf_pr) > 0 else 0.0
        final_inf_du = float(inf_du[-1]) if len(inf_du) > 0 else 0.0

        # Get status
        status = stats.get("return_status", "unknown")
        if isinstance(status, bytes):
            status = status.decode("utf-8")

        return SolveSummary(
            status=str(status),
            iterations=n_iters,
            objective=final_obj,
            primal_infeasibility=final_inf_pr,
            dual_infeasibility=final_inf_du,
            total_time_s=solve_time,
            restoration_steps=restoration_steps,
            iteration_history=history,
        )

    def log_summary(self, summary: SolveSummary) -> None:
        """Log solve summary."""
        self.logger.info(
            f"IPOPT Solve Complete: status={summary.status}, "
            f"iterations={summary.iterations}, "
            f"objective={summary.objective:.6e}"
        )
        self.logger.info(
            f"Final residuals: inf_pr={summary.primal_infeasibility:.2e}, "
            f"inf_du={summary.dual_infeasibility:.2e}"
        )
        self.logger.info(
            f"Time: {summary.total_time_s:.2f}s, restoration_steps={summary.restoration_steps}"
        )

    def log_convergence_trend(self, summary: SolveSummary, n_recent: int = 10) -> None:
        """Log convergence trend from recent iterations.

        Args:
            summary: Solve summary with iteration history
            n_recent: Number of recent iterations to show
        """
        history = summary.iteration_history
        if len(history) < 2:
            return

        # Show recent iterations
        recent = history[-n_recent:]
        self.logger.debug(f"Recent {len(recent)} iterations:")
        for data in recent:
            self.logger.debug(
                f"  k={data.iteration:4d} obj={data.objective:.4e} "
                f"inf_pr={data.inf_pr:.2e} inf_du={data.inf_du:.2e} "
                f"mu={data.mu:.2e} {data.step_type}"
            )

        # Compute convergence rate
        if len(history) >= 5:
            inf_pr_vals = [h.inf_pr for h in history[-5:]]
            if all(v > 0 for v in inf_pr_vals):
                log_rates = np.diff(np.log10(inf_pr_vals))
                avg_rate = np.mean(log_rates)
                self.logger.info(f"Convergence rate (log10 inf_pr/iter): {avg_rate:.3f}")


# =============================================================================
# Utilities
# =============================================================================


def _to_array(data: Any) -> np.ndarray:
    """Convert data to numpy array."""
    if data is None:
        return np.array([])
    try:
        arr = np.asarray(data, dtype=float)
        return arr.flatten()
    except Exception:
        try:
            return np.array([float(x) for x in data], dtype=float)
        except Exception:
            return np.array([])


def format_solve_result(summary: SolveSummary) -> str:
    """Format solve result as human-readable string.

    Args:
        summary: Solve summary

    Returns:
        Formatted multi-line string
    """
    lines = [
        "=" * 60,
        "IPOPT Solve Summary",
        "=" * 60,
        f"Status:              {summary.status}",
        f"Iterations:          {summary.iterations}",
        f"Objective:           {summary.objective:.6e}",
        f"Primal Infeasibility: {summary.primal_infeasibility:.2e}",
        f"Dual Infeasibility:   {summary.dual_infeasibility:.2e}",
        f"Total Time:          {summary.total_time_s:.2f}s",
        f"Restoration Steps:   {summary.restoration_steps}",
        "=" * 60,
    ]
    return "\n".join(lines)


def check_convergence_status(summary: SolveSummary) -> tuple[bool, str]:
    """Check if solve converged successfully.

    Args:
        summary: Solve summary

    Returns:
        Tuple of (converged, reason)
    """
    status_lower = summary.status.lower()

    if "solved" in status_lower or "optimal" in status_lower:
        return True, "Converged to optimal solution"

    if "acceptable" in status_lower:
        return True, "Converged to acceptable solution"

    if "infeas" in status_lower:
        return False, f"Infeasible problem: {summary.status}"

    if "max_iter" in status_lower:
        return False, "Maximum iterations exceeded"

    if "restoration" in status_lower:
        return False, "Failed in restoration phase"

    return False, f"Unknown status: {summary.status}"
