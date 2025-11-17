from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from campro.optimization.parameter_tuning import (
    DynamicParameterTuner,
    SolverParameters,
)
from campro.optimization.solver_selection import (
    AdaptiveSolverSelector,
    AnalysisHistory,
    ProblemCharacteristics,
    SolverType,
)


def _problem_chars(
    n_vars: int = 100,
    n_constraints: int = 50,
    problem_type: str = "litvin",
    has_issues: bool = False,
) -> ProblemCharacteristics:
    """Create test problem characteristics."""
    return ProblemCharacteristics(
        n_variables=n_vars,
        n_constraints=n_constraints,
        problem_type=problem_type,
        expected_iterations=500,
        linear_solver_ratio=0.3,
        has_convergence_issues=has_issues,
    )


def test_solver_selector_initialization() -> None:
    """Test solver selector initialization."""
    selector = AdaptiveSolverSelector()
    assert selector.analysis_history == {}


def test_solver_selection_always_returns_ma27() -> None:
    """Test solver selection always returns MA27."""
    selector = AdaptiveSolverSelector()
    problem_chars = _problem_chars(250, 120, has_issues=True)
    solver = selector.select_solver(problem_chars, "secondary")
    assert solver == SolverType.MA27


def test_analysis_history_update() -> None:
    """Test analysis history update records stats."""
    selector = AdaptiveSolverSelector()
    analysis = {
        "grade": "medium",
        "stats": {"ls_time_ratio": 0.35, "iter_count": 900},
    }
    
    selector.update_history("secondary", analysis)
    history = selector.analysis_history["secondary"]
    
    assert history.avg_grade == "medium"
    assert history.avg_linear_solver_ratio == 0.35
    assert history.avg_iterations == 900
    assert history.convergence_issues_count == 1


def test_history_clearing() -> None:
    """Test history clearing."""
    selector = AdaptiveSolverSelector()
    analysis = {
        "grade": "low",
        "stats": {"ls_time_ratio": 0.2, "iter_count": 500},
    }
    
    selector.update_history("primary", analysis)
    selector.update_history("secondary", analysis)
    
    selector.clear_history("primary")
    assert "primary" not in selector.analysis_history
    assert "secondary" in selector.analysis_history
    
    selector.clear_history()
    assert selector.analysis_history == {}


def test_parameter_tuner_initialization() -> None:
    """Test parameter tuner initialization."""
    tuner = DynamicParameterTuner()
    assert isinstance(tuner.default_params, SolverParameters)
    assert tuner.default_params.linear_solver == "ma27"


def test_parameter_tuning_for_different_phases() -> None:
    """Test parameter tuning for different phases."""
    tuner = DynamicParameterTuner()
    problem_chars = _problem_chars()
    
    primary_params = tuner.tune_parameters("primary", problem_chars, None)
    assert isinstance(primary_params, SolverParameters)
    
    secondary_params = tuner.tune_parameters("secondary", problem_chars, None)
    assert isinstance(secondary_params, SolverParameters)
    
    tertiary_chars = _problem_chars(4, 8, "crank_center")
    tertiary_params = tuner.tune_parameters("tertiary", tertiary_chars, None)
    assert isinstance(tertiary_params, SolverParameters)


def test_parameter_tuning_with_convergence_issues() -> None:
    """Test parameter tuning with convergence issues."""
    tuner = DynamicParameterTuner()
    problem_chars = _problem_chars(500, 200, has_issues=True)
    problem_chars.expected_iterations = 2000
    problem_chars.linear_solver_ratio = 0.6
    
    params = tuner.tune_parameters("secondary", problem_chars, None)
    assert params.mu_strategy == "monotone"
    assert params.max_iter > 1000
    assert params.tol < 1e-6


def test_parameter_tuning_with_analysis_history() -> None:
    """Test parameter tuning with analysis history."""
    tuner = DynamicParameterTuner()
    history = AnalysisHistory(
        avg_grade="high",
        avg_linear_solver_ratio=0.5,
        avg_iterations=2000,
        convergence_issues_count=3,
    )
    
    problem_chars = _problem_chars(200, 100, "tertiary")
    problem_chars.expected_iterations = 1000
    
    params = tuner.tune_parameters("tertiary", problem_chars, history)
    assert params.mu_strategy == "monotone"
    assert params.max_iter > 1000


def test_ipopt_and_casadi_options_creation() -> None:
    """Test IPOPT and CasADi options creation."""
    tuner = DynamicParameterTuner()
    params = tuner.get_default_parameters()
    
    ipopt_options = tuner.create_ipopt_options(params)
    assert ipopt_options["linear_solver"] == "ma27"
    
    casadi_options = tuner.create_casadi_options(params)
    assert "ipopt.max_iter" in casadi_options
    assert "ipopt.mu_strategy" in casadi_options


def test_unified_framework_integration() -> None:
    """Test unified framework uses adaptive tuning."""
    from campro.optimization.unified_framework import UnifiedOptimizationFramework
    
    framework = UnifiedOptimizationFramework()
    assert isinstance(framework.solver_selector, AdaptiveSolverSelector)
    assert isinstance(framework.parameter_tuner, DynamicParameterTuner)
    
    problem_chars = _problem_chars()
    solver = framework.solver_selector.select_solver(problem_chars, "secondary")
    assert solver == SolverType.MA27
    
    params = framework.parameter_tuner.tune_parameters("secondary", problem_chars, None)
    assert isinstance(params, SolverParameters)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
