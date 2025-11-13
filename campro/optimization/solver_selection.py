"""Adaptive solver selection stub for MA27-only configuration."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from campro.logging import get_logger
from campro.optimization.solver_analysis import IpoptAnalysisReport

log = get_logger(__name__)


class SolverType(Enum):
    """Supported linear solvers."""

    MA27 = "ma27"


@dataclass
class ProblemCharacteristics:
    """Characteristics of the optimization problem."""

    n_variables: int
    n_constraints: int
    problem_type: str
    expected_iterations: int
    linear_solver_ratio: float
    has_convergence_issues: bool


@dataclass
class AnalysisHistory:
    """Historical analysis data for decision making."""

    avg_grade: str = "low"
    avg_linear_solver_ratio: float = 0.0
    avg_iterations: int = 0
    convergence_issues_count: int = 0


class AdaptiveSolverSelector:
    """Select the (single) supported solver and track basic analysis history."""

    def __init__(self) -> None:
        self.analysis_history: Dict[str, AnalysisHistory] = {}

    def select_solver(
        self,
        problem_chars: ProblemCharacteristics,
        phase: str,
    ) -> SolverType:
        """Return the only supported solver (MA27)."""
        log.debug(
            "Solver selection requested for phase '%s' with %d variables; MA27 enforced.",
            phase,
            problem_chars.n_variables,
        )
        return SolverType.MA27

    def update_history(
        self, phase: str, analysis: Dict[str, Any] | IpoptAnalysisReport | None
    ) -> None:
        """Record basic statistics from Ipopt analysis metadata.
        
        Accepts either a dict (for backward compatibility) or an IpoptAnalysisReport object.
        """
        if not analysis:
            return

        # Handle both dict and IpoptAnalysisReport object types
        if isinstance(analysis, IpoptAnalysisReport):
            # IpoptAnalysisReport object: access attributes directly
            stats = analysis.stats
            grade = analysis.grade
        elif isinstance(analysis, dict):
            # Dict: use .get() method for backward compatibility
            stats = analysis.get("stats", {})
            grade = analysis.get("grade", "low")
        else:
            # Unknown type: skip processing
            log.warning(
                f"update_history received unknown analysis type: {type(analysis)}. Skipping."
            )
            return

        history = self.analysis_history.setdefault(phase, AnalysisHistory())

        # stats is always a dict (IpoptAnalysisReport.stats is dict[str, Any])
        ls_time_ratio = float(stats.get("ls_time_ratio") or 0.0)
        iter_count = int(stats.get("iter_count") or 0)

        history.avg_linear_solver_ratio = ls_time_ratio
        history.avg_iterations = iter_count
        history.avg_grade = str(grade)
        if history.avg_grade in {"medium", "high"}:
            history.convergence_issues_count += 1

    def clear_history(self, phase: str | None = None) -> None:
        """Clear stored analysis data."""
        if phase is None:
            self.analysis_history.clear()
        else:
            self.analysis_history.pop(phase, None)

    def get_recommendation(self, phase: str) -> str:
        """Return a human-readable recommendation for the phase."""
        history = self.analysis_history.get(phase)
        if history and history.avg_grade in {"medium", "high"}:
            return "Monitor MA27 performance (consider scaling or tuning)"
        return "MA27 is sufficient"


__all__ = [
    "AdaptiveSolverSelector",
    "AnalysisHistory",
    "ProblemCharacteristics",
    "SolverType",
]
