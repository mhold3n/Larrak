from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from campro.optimization.solver_selection import (
    AdaptiveSolverSelector,
    AnalysisHistory,
    ProblemCharacteristics,
    SolverType,
)


def test_solver_selection_always_returns_ma27():
    selector = AdaptiveSolverSelector()
    pc = ProblemCharacteristics(
        n_variables=1000,
        n_constraints=500,
        problem_type="test",
        expected_iterations=2000,
        linear_solver_ratio=0.8,
        has_convergence_issues=True,
    )

    solver = selector.select_solver(pc, "primary")
    assert solver is SolverType.MA27


def test_update_history_records_basic_stats():
    selector = AdaptiveSolverSelector()
    analysis = {
        "grade": "medium",
        "stats": {"ls_time_ratio": 0.45, "iter_count": 150},
    }

    selector.update_history("primary", analysis)
    history = selector.analysis_history["primary"]

    assert isinstance(history, AnalysisHistory)
    assert history.avg_grade == "medium"
    assert history.avg_linear_solver_ratio == 0.45
    assert history.avg_iterations == 150
    assert history.convergence_issues_count == 1
