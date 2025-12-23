"""Adaptive solver selection based on problem characteristics and availability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from campro.logging import get_logger
from campro.optimization.solvers.solver_analysis import IpoptAnalysisReport

log = get_logger(__name__)


class SolverType(Enum):
    """Supported linear solvers."""

    MA27 = "ma27"
    MA57 = "ma57"
    MA77 = "ma77"
    MA86 = "ma86"
    MA97 = "ma97"


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
    """Select optimal solver based on problem characteristics and availability."""

    def __init__(self) -> None:
        self.analysis_history: dict[str, AnalysisHistory] = {}
        self._available_solvers: list[str] | None = None
        self._detect_available_solvers()

    def _detect_available_solvers(self) -> None:
        """Detect which HSL solvers are available."""
        try:
            from campro.environment.hsl_detector import detect_available_solvers

            self._available_solvers = detect_available_solvers(test_runtime=True)
            if not self._available_solvers:
                # Fallback to MA27 if no solvers detected
                log.warning("No HSL solvers detected; defaulting to MA27")
                self._available_solvers = ["ma27"]
        except ImportError:
            log.warning("hsl_detector not available; defaulting to MA27")
            self._available_solvers = ["ma27"]
        except Exception as e:
            log.warning(f"Error detecting available solvers: {e}; defaulting to MA27")
            self._available_solvers = ["ma27"]

    def _is_solver_available(self, solver: SolverType) -> bool:
        """Check if a solver is available."""
        if self._available_solvers is None:
            self._detect_available_solvers()
        return solver.value in (self._available_solvers or [])

    def select_solver(
        self,
        problem_chars: ProblemCharacteristics,
        phase: str,
    ) -> SolverType:
        """
        Select optimal solver based on problem size and availability.

        Selection logic:
        - Small problems (< 5,000 vars): MA27
        - Medium problems (5,000-50,000 vars): MA57 if available, else MA27
        - Large problems (> 50,000 vars): MA97/MA86 if available and multi-core,
          else MA77 if available, else MA27
        """
        n_vars = problem_chars.n_variables

        # Small problems: prefer MA27
        if n_vars < 5000:
            if self._is_solver_available(SolverType.MA27):
                log.debug(
                    f"Solver selection for phase '{phase}': MA27 (small problem: {n_vars} vars)"
                )
                return SolverType.MA27
            # Fallback if MA27 not available (shouldn't happen)
            log.warning("MA27 not available; this should not happen")
            return SolverType.MA27

        # Medium problems: prefer MA57
        elif n_vars < 50000:
            if self._is_solver_available(SolverType.MA57):
                log.debug(
                    f"Solver selection for phase '{phase}': MA57 (medium problem: {n_vars} vars)"
                )
                return SolverType.MA57
            else:
                # Fallback to MA27
                log.debug(
                    f"Solver selection for phase '{phase}': MA27 (MA57 not available, {n_vars} vars)"
                )
                return SolverType.MA27

        # Large problems: prefer parallel solvers (MA97/MA86) or MA77
        else:
            # Try MA97 first (best for very large problems)
            if self._is_solver_available(SolverType.MA97):
                log.debug(
                    f"Solver selection for phase '{phase}': MA97 (large problem: {n_vars} vars)"
                )
                return SolverType.MA97

            # Try MA86 (parallel solver for large problems)
            if self._is_solver_available(SolverType.MA86):
                log.debug(
                    f"Solver selection for phase '{phase}': MA86 (large problem: {n_vars} vars)"
                )
                return SolverType.MA86

            # Try MA77 (out-of-core solver)
            if self._is_solver_available(SolverType.MA77):
                log.debug(
                    f"Solver selection for phase '{phase}': MA77 (large problem: {n_vars} vars)"
                )
                return SolverType.MA77

            # Fallback to MA27
            log.debug(
                f"Solver selection for phase '{phase}': MA27 (no large-problem solvers available, {n_vars} vars)"
            )
            return SolverType.MA27

    def update_history(
        self, phase: str, analysis: dict[str, Any] | IpoptAnalysisReport | None
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
            available = ", ".join(self._available_solvers or ["ma27"])
            return (
                f"Monitor solver performance (available: {available}); consider scaling or tuning"
            )
        available = ", ".join(self._available_solvers or ["ma27"])
        return f"Current solver is sufficient (available: {available})"

    def get_available_solvers(self) -> list[str]:
        """Get list of available solver names."""
        if self._available_solvers is None:
            self._detect_available_solvers()
        return self._available_solvers or ["ma27"]


__all__ = [
    "AdaptiveSolverSelector",
    "AnalysisHistory",
    "ProblemCharacteristics",
    "SolverType",
]
