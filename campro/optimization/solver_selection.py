"""Adaptive solver selection based on problem characteristics and analysis history."""
from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import Enum

from campro.logging import get_logger
from campro.optimization.solver_analysis import MA57ReadinessReport

log = get_logger(__name__)

# Detect macOS platform - MA97 has known crash bug on macOS
IS_MACOS = platform.system().lower() == "darwin"


class SolverType(Enum):
    """Available HSL linear solvers."""

    MA27 = "ma27"  # Classic sparse symmetric solver - best for small to medium problems
    MA57 = "ma57"  # Modern sparse symmetric solver - best for medium to large problems
    MA77 = "ma77"  # Out-of-core solver - best for very large problems with limited RAM
    MA86 = (
        "ma86"  # Parallel solver (CPU) - best for large problems on multi-core systems
    )
    MA97 = "ma97"  # Advanced parallel solver - best for very large problems on multi-core systems
    # Non-HSL solvers are not permitted in this project


@dataclass
class ProblemCharacteristics:
    """Characteristics of the optimization problem."""

    n_variables: int
    n_constraints: int
    problem_type: str  # "thermal", "litvin", "crank_center"
    expected_iterations: int
    linear_solver_ratio: float
    has_convergence_issues: bool


@dataclass
class AnalysisHistory:
    """Historical analysis data for decision making."""

    avg_grade: str
    avg_linear_solver_ratio: float
    avg_iterations: int
    convergence_issues_count: int
    ma57_benefits: list[bool]


class AdaptiveSolverSelector:
    """Select optimal solver based on problem characteristics and history."""

    def __init__(self):
        self.analysis_history: dict[str, AnalysisHistory] = {}

    def select_solver(
        self, problem_chars: ProblemCharacteristics, phase: str,
    ) -> SolverType:
        """
        Select optimal HSL solver for given problem characteristics.

        Selection criteria:
        - MA27: Small to medium problems (< 5,000 variables)
        - MA57: Medium to large problems (5,000-50,000 variables)
        - MA77: Very large problems (> 50,000 variables) or limited RAM
        - MA86: Large problems on multi-core systems
        - MA97: Very large problems on multi-core systems (most advanced)
        """
        # Get historical data for this phase
        history = self.analysis_history.get(phase)

        # Determine optimal solver based on problem characteristics
        n_vars = problem_chars.n_variables

        # Check solver availability
        import sys
        print("DEBUG: About to call _get_available_solvers()", file=sys.stderr, flush=True)
        available_solvers = self._get_available_solvers()
        print(f"DEBUG: _get_available_solvers() returned: {[s.value for s in available_solvers]}", file=sys.stderr, flush=True)

        # Selection logic based on problem size and characteristics
        print(f"DEBUG: n_vars={n_vars}, selecting solver...", file=sys.stderr, flush=True)
        if n_vars < 5000:
            # Small to medium problems - prefer MA27
            if SolverType.MA27 in available_solvers:
                chosen = SolverType.MA27
            else:
                chosen = self._fallback_solver(available_solvers)
        elif n_vars < 50000:
            # Medium to large problems - prefer MA57
            if SolverType.MA57 in available_solvers:
                chosen = SolverType.MA57
            elif SolverType.MA27 in available_solvers:
                chosen = SolverType.MA27
            else:
                chosen = self._fallback_solver(available_solvers)
        # Very large problems - prefer MA97 or MA77
        # CRITICAL: Skip MA97 on macOS due to known crash bug
        elif not IS_MACOS and SolverType.MA97 in available_solvers:
            chosen = SolverType.MA97
        elif SolverType.MA77 in available_solvers:
            chosen = SolverType.MA77
        elif SolverType.MA86 in available_solvers:
            chosen = SolverType.MA86
        elif SolverType.MA57 in available_solvers:
            chosen = SolverType.MA57
        else:
            chosen = self._fallback_solver(available_solvers)

        print(f"DEBUG: Solver chosen: {chosen.value}", file=sys.stderr, flush=True)
        log.debug(
            "Selected solver for %s phase: %s (problem size: %d variables, available: %s)",
            phase,
            chosen.value,
            n_vars,
            [s.value for s in available_solvers],
        )

        print("DEBUG: About to return from select_solver()", file=sys.stderr, flush=True)
        return chosen

    def _get_available_solvers(self) -> list[SolverType]:
        """Check which HSL solvers are available."""
        available = []

        try:
            import casadi as ca

            # Test each solver
            for solver_type in SolverType:
                # CRITICAL: Skip MA97 on macOS due to known segmentation fault bug
                # MA97 destructor crashes on macOS during cleanup
                if IS_MACOS and solver_type == SolverType.MA97:
                    log.debug("Skipping MA97 on macOS due to known crash bug")
                    continue
                try:
                    # Create a simple test problem
                    x = ca.SX.sym("x")
                    f = x**2
                    g = x - 1
                    nlp = {"x": x, "f": f, "g": g}

                    # Try to create a solver with this HSL linear solver
                    solver = ca.nlpsol(
                        f"test_{solver_type.value}",
                        "ipopt",
                        nlp,
                        {
                            "ipopt.linear_solver": solver_type.value,
                            "ipopt.print_level": 0,
                            "ipopt.sb": "yes",
                        },
                    )

                    # Test the solver - if we can create it and run it, it's available
                    # Don't require success - just that the solver can execute
                    result = solver(x0=0, lbg=0, ubg=0)
                    # If we get here without exception, the solver is available
                    # The solver might not succeed on the test problem, but that's okay
                    available.append(solver_type)

                except Exception:
                    # Solver not available
                    pass

        except ImportError:
            # CasADi not available
            pass

        import sys
        print(f"DEBUG: _get_available_solvers() returning {len(available)} solvers: {[s.value for s in available]}", file=sys.stderr, flush=True)
        return available

    def _fallback_solver(self, available_solvers: list[SolverType]) -> SolverType:
        """Select a fallback solver from available options."""
        # Prefer MA27 as the most robust fallback
        if SolverType.MA27 in available_solvers:
            return SolverType.MA27
        if SolverType.MA57 in available_solvers:
            return SolverType.MA57
        if available_solvers:
            return available_solvers[0]  # Return first available
        # This should not happen if HSL is properly installed
        raise RuntimeError("No HSL solvers are available")

    def update_history(self, phase: str, analysis: MA57ReadinessReport):
        """Update analysis history for future decisions."""
        # Handle case where analysis is None (optimization failed)
        if analysis is None:
            log.warning(
                f"No analysis available for phase {phase}, skipping history update",
            )
            return

        if phase not in self.analysis_history:
            # Handle None values from stats
            ls_time_ratio = analysis.stats.get("ls_time_ratio")
            if ls_time_ratio is None:
                ls_time_ratio = 0.0
            iter_count = analysis.stats.get("iter_count")
            if iter_count is None:
                iter_count = 0
            
            self.analysis_history[phase] = AnalysisHistory(
                avg_grade=analysis.grade,
                avg_linear_solver_ratio=ls_time_ratio,
                avg_iterations=iter_count,
                convergence_issues_count=1
                if analysis.grade in ["medium", "high"]
                else 0,
                ma57_benefits=[analysis.grade in ["medium", "high"]],
            )
        else:
            # Update running averages
            history = self.analysis_history[phase]
            n = len(history.ma57_benefits)

            # Moving average for numerical metrics
            # Handle None values from stats
            ls_time_ratio = analysis.stats.get("ls_time_ratio")
            if ls_time_ratio is None:
                ls_time_ratio = 0.0
            iter_count = analysis.stats.get("iter_count")
            if iter_count is None:
                iter_count = 0
            
            history.avg_linear_solver_ratio = (
                history.avg_linear_solver_ratio * n + ls_time_ratio
            ) / (n + 1)
            history.avg_iterations = int(
                (history.avg_iterations * n + iter_count) / (n + 1),
            )

            # Update counts
            if analysis.grade in ["medium", "high"]:
                history.convergence_issues_count += 1
                history.ma57_benefits.append(True)
            else:
                history.ma57_benefits.append(False)

            # Update grade (most recent)
            history.avg_grade = analysis.grade

        # Handle case where analysis.grade might be None
        grade_str = analysis.grade if analysis.grade is not None else "unknown"
        # Handle case where ls_time_ratio might be None
        ls_ratio = analysis.stats.get("ls_time_ratio")
        if ls_ratio is None:
            ls_ratio = 0.0
        log.debug(
            f"Updated analysis history for {phase} phase: grade={grade_str}, "
            f"ls_ratio={ls_ratio:.3f}",
        )

    def get_history_summary(self, phase: str) -> dict | None:
        """Get summary of analysis history for a phase."""
        if phase not in self.analysis_history:
            return None

        history = self.analysis_history[phase]
        return {
            "phase": phase,
            "avg_grade": history.avg_grade,
            "avg_linear_solver_ratio": history.avg_linear_solver_ratio,
            "avg_iterations": history.avg_iterations,
            "convergence_issues_count": history.convergence_issues_count,
            "ma57_benefit_percentage": sum(history.ma57_benefits)
            / len(history.ma57_benefits)
            if history.ma57_benefits
            else 0.0,
            "total_analyses": len(history.ma57_benefits),
        }

    def get_all_history_summaries(self) -> dict[str, dict]:
        """Get summaries for all phases."""
        return {
            phase: self.get_history_summary(phase)
            for phase in self.analysis_history
        }

    def clear_history(self, phase: str | None = None):
        """Clear analysis history for a phase or all phases."""
        if phase is None:
            self.analysis_history.clear()
            log.info("Cleared all analysis history")
        elif phase in self.analysis_history:
            del self.analysis_history[phase]
            log.info(f"Cleared analysis history for {phase} phase")
        else:
            log.warning(f"No analysis history found for {phase} phase")

    def should_consider_ma57(self, phase: str) -> bool:
        """
        Determine if MA57 should be considered for future optimizations.

        This is a placeholder for future MA57 availability logic.
        """
        if phase not in self.analysis_history:
            return False

        history = self.analysis_history[phase]

        # Criteria for considering MA57:
        # 1. High linear solver time ratio
        # 2. Frequent convergence issues
        # 3. Large problem sizes (would need to be passed in)

        ma57_benefit_percentage = (
            sum(history.ma57_benefits) / len(history.ma57_benefits)
            if history.ma57_benefits
            else 0.0
        )

        return (
            history.avg_linear_solver_ratio > 0.4
            or ma57_benefit_percentage > 0.5
            or history.convergence_issues_count > 3
        )

    def get_recommendation(self, phase: str) -> str:
        """Get solver recommendation for a phase."""
        if self.should_consider_ma57(phase):
            return "Consider MA57 when available"
        return "MA27 is sufficient"
