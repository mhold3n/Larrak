from campro.optimization.solver_selection import (
    AdaptiveSolverSelector,
    ProblemCharacteristics,
    SolverType,
)
from campro.optimization.solver_analysis import MA57ReadinessReport

# Stub readiness report class if not detailed in codebase
class _StubReport:
    def __init__(self, grade: str, ls_ratio: float, iter_count: int):
        self.grade = grade
        self.stats = {"ls_time_ratio": ls_ratio, "iter_count": iter_count}


def test_solver_selection_defaults_to_ma27(monkeypatch):
    selector = AdaptiveSolverSelector()

    pc = ProblemCharacteristics(
        n_variables=100,
        n_constraints=80,
        problem_type="thermal",
        expected_iterations=40,
        linear_solver_ratio=0.3,
        has_convergence_issues=False,
    )

    # Patch detection to False
    monkeypatch.setattr(
        "campro.optimization.solver_detection.is_ma57_available", lambda: False
    )

    assert selector.select_solver(pc, "phase1") is SolverType.MA27


def test_solver_selection_chooses_ma57_when_available(monkeypatch):
    selector = AdaptiveSolverSelector()

    # Update history with high-ratio readiness suggesting MA57 benefit
    report = _StubReport("high", 0.6, 100)
    selector.update_history("phase1", report)  # type: ignore[arg-type]

    pc = ProblemCharacteristics(
        n_variables=600,
        n_constraints=500,
        problem_type="thermal",
        expected_iterations=100,
        linear_solver_ratio=0.6,
        has_convergence_issues=True,
    )

    monkeypatch.setattr(
        "campro.optimization.solver_detection.is_ma57_available", lambda: True
    )

    assert selector.select_solver(pc, "phase1") is SolverType.MA57
