"""
Tests for simplified adaptive solver selection and parameter tuning.
"""

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


class TestAdaptiveSolverSelection:
    def test_solver_selector_initialization(self):
        selector = AdaptiveSolverSelector()
        assert selector.analysis_history == {}

    def test_solver_selection_always_returns_ma27(self):
        selector = AdaptiveSolverSelector()
        problem_chars = ProblemCharacteristics(
            n_variables=250,
            n_constraints=120,
            problem_type="litvin",
            expected_iterations=1200,
            linear_solver_ratio=0.4,
            has_convergence_issues=True,
        )

        solver = selector.select_solver(problem_chars, "secondary")
        assert solver == SolverType.MA27

    def test_analysis_history_update_records_stats(self):
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

    def test_history_clearing(self):
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


class TestDynamicParameterTuning:
    def test_parameter_tuner_initialization(self):
        tuner = DynamicParameterTuner()
        assert isinstance(tuner.default_params, SolverParameters)
        assert tuner.default_params.linear_solver == "ma27"

    def test_parameter_tuning_for_different_phases(self):
        tuner = DynamicParameterTuner()
        problem_chars = ProblemCharacteristics(
            n_variables=100,
            n_constraints=50,
            problem_type="litvin",
            expected_iterations=500,
            linear_solver_ratio=0.3,
            has_convergence_issues=False,
        )

        primary_params = tuner.tune_parameters("primary", problem_chars, None)
        assert isinstance(primary_params, SolverParameters)

        secondary_params = tuner.tune_parameters("secondary", problem_chars, None)
        assert isinstance(secondary_params, SolverParameters)

        tertiary_chars = ProblemCharacteristics(
            n_variables=4,
            n_constraints=8,
            problem_type="crank_center",
            expected_iterations=200,
            linear_solver_ratio=0.2,
            has_convergence_issues=False,
        )
        tertiary_params = tuner.tune_parameters("tertiary", tertiary_chars, None)
        assert isinstance(tertiary_params, SolverParameters)

    def test_parameter_tuning_with_convergence_issues(self):
        tuner = DynamicParameterTuner()
        problem_chars = ProblemCharacteristics(
            n_variables=500,
            n_constraints=200,
            problem_type="litvin",
            expected_iterations=2000,
            linear_solver_ratio=0.6,
            has_convergence_issues=True,
        )

        params = tuner.tune_parameters("secondary", problem_chars, None)
        assert params.mu_strategy == "monotone"
        assert params.max_iter > 1000
        assert params.tol < 1e-6

    def test_parameter_tuning_with_large_problems(self):
        tuner = DynamicParameterTuner()
        problem_chars = ProblemCharacteristics(
            n_variables=1000,
            n_constraints=500,
            problem_type="litvin",
            expected_iterations=3000,
            linear_solver_ratio=0.4,
            has_convergence_issues=False,
        )

        params = tuner.tune_parameters("secondary", problem_chars, None)
        assert params.max_iter > 1000
        assert params.tol < 1e-6

    def test_parameter_tuning_with_analysis_history(self):
        tuner = DynamicParameterTuner()
        history = AnalysisHistory(
            avg_grade="high",
            avg_linear_solver_ratio=0.5,
            avg_iterations=2000,
            convergence_issues_count=3,
        )

        problem_chars = ProblemCharacteristics(
            n_variables=200,
            n_constraints=100,
            problem_type="tertiary",
            expected_iterations=1000,
            linear_solver_ratio=0.3,
            has_convergence_issues=False,
        )

        params = tuner.tune_parameters("tertiary", problem_chars, history)
        assert params.mu_strategy == "monotone"
        assert params.max_iter > 1000

    def test_ipopt_and_casadi_options_creation(self):
        tuner = DynamicParameterTuner()
        params = tuner.get_default_parameters()

        ipopt_options = tuner.create_ipopt_options(params)
        assert ipopt_options["linear_solver"] == "ma27"

        casadi_options = tuner.create_casadi_options(params)
        assert "ipopt.max_iter" in casadi_options
        assert "ipopt.mu_strategy" in casadi_options

    def test_problem_characteristics_estimation(self):
        tuner = DynamicParameterTuner()

        primary_chars = tuner.estimate_problem_characteristics(100, 50, "primary")
        assert primary_chars.problem_type == "primary"

        secondary_chars = tuner.estimate_problem_characteristics(200, 100, "secondary")
        assert secondary_chars.problem_type == "secondary"

        tertiary_chars = tuner.estimate_problem_characteristics(4, 8, "tertiary")
        assert tertiary_chars.problem_type == "tertiary"

    def test_tuning_summary_generation(self):
        tuner = DynamicParameterTuner()
        problem_chars = ProblemCharacteristics(
            n_variables=100,
            n_constraints=50,
            problem_type="litvin",
            expected_iterations=500,
            linear_solver_ratio=0.3,
            has_convergence_issues=False,
        )
        params = tuner.tune_parameters("secondary", problem_chars, None)
        summary = tuner.get_tuning_summary("secondary", problem_chars, params)

        assert summary["phase"] == "secondary"
        assert "problem_characteristics" in summary
        assert "tuned_parameters" in summary


class TestUnifiedFrameworkIntegration:
    def test_unified_framework_uses_adaptive_tuning(self):
        from campro.optimization.unified_framework import UnifiedOptimizationFramework

        framework = UnifiedOptimizationFramework()
        assert isinstance(framework.solver_selector, AdaptiveSolverSelector)
        assert isinstance(framework.parameter_tuner, DynamicParameterTuner)

    def test_adaptive_tuning_components_are_accessible(self):
        from campro.optimization.unified_framework import UnifiedOptimizationFramework

        framework = UnifiedOptimizationFramework()
        problem_chars = ProblemCharacteristics(
            n_variables=100,
            n_constraints=50,
            problem_type="litvin",
            expected_iterations=500,
            linear_solver_ratio=0.3,
            has_convergence_issues=False,
        )

        solver = framework.solver_selector.select_solver(problem_chars, "secondary")
        assert solver == SolverType.MA27

        params = framework.parameter_tuner.tune_parameters("secondary", problem_chars, None)
        assert isinstance(params, SolverParameters)

        framework.solver_selector.clear_history()
        assert framework.solver_selector.analysis_history == {}
