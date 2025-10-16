"""
Tests for adaptive solver selection and parameter tuning.

This module tests the adaptive solver selection and dynamic parameter tuning
functionality integrated with the unified optimization framework.
"""

import pytest
from unittest.mock import Mock, patch

from campro.optimization.solver_selection import (
    AdaptiveSolverSelector,
    ProblemCharacteristics,
    AnalysisHistory,
    SolverType,
)
from campro.optimization.parameter_tuning import (
    DynamicParameterTuner,
    SolverParameters,
)
from campro.optimization.solver_analysis import MA57ReadinessReport


class TestAdaptiveSolverSelection:
    """Test adaptive solver selection functionality."""

    def test_solver_selector_initialization(self):
        """Test adaptive solver selector initialization."""
        selector = AdaptiveSolverSelector()
        
        assert selector.analysis_history == {}
        assert isinstance(selector, AdaptiveSolverSelector)

    def test_solver_selection_based_on_problem_characteristics(self):
        """Test solver selection logic."""
        selector = AdaptiveSolverSelector()
        
        # Test problem characteristics
        problem_chars = ProblemCharacteristics(
            n_variables=100,
            n_constraints=50,
            problem_type="litvin",
            expected_iterations=500,
            linear_solver_ratio=0.3,
            has_convergence_issues=False
        )
        
        # Test solver selection
        solver = selector.select_solver(problem_chars, "secondary")
        
        # Currently always returns MA27
        assert solver == SolverType.MA27
        assert solver.value == "ma27"

    def test_analysis_history_update(self):
        """Test analysis history updates after optimization."""
        selector = AdaptiveSolverSelector()
        
        # Create mock analysis
        analysis = MA57ReadinessReport(
            grade="medium",
            reasons=["High iteration count (1500)"],
            suggested_action="Consider MA57 if available",
            stats={
                "success": True,
                "iter_count": 1500,
                "ls_time_ratio": 0.4,
                "primal_inf": 1e-6,
                "dual_inf": 1e-6,
            }
        )
        
        # Update history
        selector.update_history("secondary", analysis)
        
        # Verify history was updated
        assert "secondary" in selector.analysis_history
        history = selector.analysis_history["secondary"]
        assert history.avg_grade == "medium"
        assert history.avg_linear_solver_ratio == 0.4
        assert history.avg_iterations == 1500
        assert history.convergence_issues_count == 1
        assert history.ma57_benefits == [True]

    def test_analysis_history_running_averages(self):
        """Test running averages in analysis history."""
        selector = AdaptiveSolverSelector()
        
        # First analysis
        analysis1 = MA57ReadinessReport(
            grade="low",
            reasons=["No adverse indicators detected"],
            suggested_action="Stick with MA27",
            stats={
                "success": True,
                "iter_count": 500,
                "ls_time_ratio": 0.2,
                "primal_inf": 1e-6,
                "dual_inf": 1e-6,
            }
        )
        
        # Second analysis
        analysis2 = MA57ReadinessReport(
            grade="high",
            reasons=["High iteration count (2000)", "Linear solver dominates runtime"],
            suggested_action="Strongly consider MA57",
            stats={
                "success": True,
                "iter_count": 2000,
                "ls_time_ratio": 0.6,
                "primal_inf": 1e-6,
                "dual_inf": 1e-6,
            }
        )
        
        # Update history twice
        selector.update_history("primary", analysis1)
        selector.update_history("primary", analysis2)
        
        # Verify running averages
        history = selector.analysis_history["primary"]
        assert history.avg_grade == "high"  # Most recent
        assert history.avg_linear_solver_ratio == 0.4  # (0.2 + 0.6) / 2
        assert history.avg_iterations == 1250  # (500 + 2000) / 2
        assert history.convergence_issues_count == 1  # Only second analysis
        assert history.ma57_benefits == [False, True]

    def test_history_summary_generation(self):
        """Test history summary generation."""
        selector = AdaptiveSolverSelector()
        
        # Add some history
        analysis = MA57ReadinessReport(
            grade="medium",
            reasons=["High iteration count (1500)"],
            suggested_action="Consider MA57 if available",
            stats={
                "success": True,
                "iter_count": 1500,
                "ls_time_ratio": 0.4,
                "primal_inf": 1e-6,
                "dual_inf": 1e-6,
            }
        )
        selector.update_history("secondary", analysis)
        
        # Get summary
        summary = selector.get_history_summary("secondary")
        
        assert summary is not None
        assert summary["phase"] == "secondary"
        assert summary["avg_grade"] == "medium"
        assert summary["avg_linear_solver_ratio"] == 0.4
        assert summary["avg_iterations"] == 1500
        assert summary["convergence_issues_count"] == 1
        assert summary["ma57_benefit_percentage"] == 1.0
        assert summary["total_analyses"] == 1

    def test_ma57_consideration_logic(self):
        """Test MA57 consideration logic."""
        selector = AdaptiveSolverSelector()
        
        # Test with no history
        assert not selector.should_consider_ma57("primary")
        
        # Add history that suggests MA57 benefit
        analysis = MA57ReadinessReport(
            grade="high",
            reasons=["High iteration count (2000)", "Linear solver dominates runtime"],
            suggested_action="Strongly consider MA57",
            stats={
                "success": True,
                "iter_count": 2000,
                "ls_time_ratio": 0.6,  # High LS ratio
                "primal_inf": 1e-6,
                "dual_inf": 1e-6,
            }
        )
        selector.update_history("secondary", analysis)
        
        # Should consider MA57 due to high LS ratio
        assert selector.should_consider_ma57("secondary")
        
        # Test recommendation
        recommendation = selector.get_recommendation("secondary")
        assert "MA57" in recommendation

    def test_history_clearing(self):
        """Test history clearing functionality."""
        selector = AdaptiveSolverSelector()
        
        # Add some history
        analysis = MA57ReadinessReport(
            grade="low",
            reasons=["No adverse indicators detected"],
            suggested_action="Stick with MA27",
            stats={
                "success": True,
                "iter_count": 500,
                "ls_time_ratio": 0.2,
                "primal_inf": 1e-6,
                "dual_inf": 1e-6,
            }
        )
        selector.update_history("primary", analysis)
        selector.update_history("secondary", analysis)
        
        # Clear specific phase
        selector.clear_history("primary")
        assert "primary" not in selector.analysis_history
        assert "secondary" in selector.analysis_history
        
        # Clear all history
        selector.clear_history()
        assert len(selector.analysis_history) == 0


class TestDynamicParameterTuning:
    """Test dynamic parameter tuning functionality."""

    def test_parameter_tuner_initialization(self):
        """Test dynamic parameter tuner initialization."""
        tuner = DynamicParameterTuner()
        
        assert tuner.default_params is not None
        assert isinstance(tuner.default_params, SolverParameters)
        assert tuner.default_params.max_iter == 1000
        assert tuner.default_params.tol == 1e-6
        assert tuner.default_params.linear_solver == "ma27"

    def test_parameter_tuning_for_different_phases(self):
        """Test dynamic parameter tuning for primary/secondary/tertiary."""
        tuner = DynamicParameterTuner()
        
        # Test problem characteristics
        problem_chars = ProblemCharacteristics(
            n_variables=100,
            n_constraints=50,
            problem_type="litvin",
            expected_iterations=500,
            linear_solver_ratio=0.3,
            has_convergence_issues=False
        )
        
        # Test primary phase tuning
        primary_params = tuner.tune_parameters("primary", problem_chars, None)
        assert isinstance(primary_params, SolverParameters)
        assert primary_params.max_iter >= 500  # Should be tuned based on expected iterations
        
        # Test secondary phase tuning
        secondary_params = tuner.tune_parameters("secondary", problem_chars, None)
        assert isinstance(secondary_params, SolverParameters)
        
        # Test tertiary phase tuning
        tertiary_chars = ProblemCharacteristics(
            n_variables=4,
            n_constraints=8,
            problem_type="crank_center",
            expected_iterations=200,
            linear_solver_ratio=0.2,
            has_convergence_issues=False
        )
        tertiary_params = tuner.tune_parameters("tertiary", tertiary_chars, None)
        assert isinstance(tertiary_params, SolverParameters)

    def test_parameter_tuning_with_convergence_issues(self):
        """Test parameter tuning when convergence issues are detected."""
        tuner = DynamicParameterTuner()
        
        # Problem with convergence issues
        problem_chars = ProblemCharacteristics(
            n_variables=500,
            n_constraints=200,
            problem_type="litvin",
            expected_iterations=2000,
            linear_solver_ratio=0.6,
            has_convergence_issues=True
        )
        
        params = tuner.tune_parameters("secondary", problem_chars, None)
        
        # Should use monotone strategy for convergence issues
        assert params.mu_strategy == "monotone"
        assert params.max_iter > 1000  # Should increase max iterations
        assert params.tol < 1e-6  # Should decrease tolerance

    def test_parameter_tuning_with_large_problems(self):
        """Test parameter tuning for large problems."""
        tuner = DynamicParameterTuner()
        
        # Large problem
        problem_chars = ProblemCharacteristics(
            n_variables=1000,
            n_constraints=500,
            problem_type="litvin",
            expected_iterations=3000,
            linear_solver_ratio=0.4,
            has_convergence_issues=False
        )
        
        params = tuner.tune_parameters("secondary", problem_chars, None)
        
        # Should use more conservative settings for large problems
        assert params.max_iter > 1000
        assert params.tol < 1e-6

    def test_parameter_tuning_with_analysis_history(self):
        """Test parameter tuning with analysis history."""
        tuner = DynamicParameterTuner()
        
        # Create analysis history
        history = AnalysisHistory(
            avg_grade="high",
            avg_linear_solver_ratio=0.5,
            avg_iterations=2000,
            convergence_issues_count=3,
            ma57_benefits=[True, True, False]
        )
        
        problem_chars = ProblemCharacteristics(
            n_variables=200,
            n_constraints=100,
            problem_type="tertiary",
            expected_iterations=1000,
            linear_solver_ratio=0.3,
            has_convergence_issues=False
        )
        
        params = tuner.tune_parameters("tertiary", problem_chars, history)
        
        # Should use more conservative settings due to high grade
        assert params.max_iter > 1000
        assert params.mu_strategy == "monotone"

    def test_ipopt_options_creation(self):
        """Test creation of Ipopt options from SolverParameters."""
        tuner = DynamicParameterTuner()
        params = tuner.get_default_parameters()
        
        ipopt_options = tuner.create_ipopt_options(params)
        
        assert isinstance(ipopt_options, dict)
        assert "max_iter" in ipopt_options
        assert "tol" in ipopt_options
        assert "linear_solver" in ipopt_options
        assert "mu_strategy" in ipopt_options
        assert "print_level" in ipopt_options
        assert "hessian_approximation" in ipopt_options

    def test_casadi_options_creation(self):
        """Test creation of CasADi options from SolverParameters."""
        tuner = DynamicParameterTuner()
        params = tuner.get_default_parameters()
        
        casadi_options = tuner.create_casadi_options(params)
        
        assert isinstance(casadi_options, dict)
        assert "ipopt.max_iter" in casadi_options
        assert "ipopt.tol" in casadi_options
        assert "ipopt.linear_solver" in casadi_options
        assert "ipopt.mu_strategy" in casadi_options
        assert "ipopt.print_level" in casadi_options
        assert "ipopt.hessian_approximation" in casadi_options

    def test_problem_characteristics_estimation(self):
        """Test problem characteristics estimation."""
        tuner = DynamicParameterTuner()
        
        # Test estimation for different phases
        primary_chars = tuner.estimate_problem_characteristics(100, 50, "primary")
        assert primary_chars.n_variables == 100
        assert primary_chars.n_constraints == 50
        assert primary_chars.problem_type == "primary"
        assert primary_chars.expected_iterations >= 500
        
        secondary_chars = tuner.estimate_problem_characteristics(200, 100, "secondary")
        assert secondary_chars.expected_iterations >= 1000
        
        tertiary_chars = tuner.estimate_problem_characteristics(4, 8, "tertiary")
        assert tertiary_chars.expected_iterations >= 200

    def test_tuning_summary_generation(self):
        """Test tuning summary generation."""
        tuner = DynamicParameterTuner()
        
        problem_chars = ProblemCharacteristics(
            n_variables=100,
            n_constraints=50,
            problem_type="litvin",
            expected_iterations=500,
            linear_solver_ratio=0.3,
            has_convergence_issues=False
        )
        
        params = tuner.tune_parameters("secondary", problem_chars, None)
        summary = tuner.get_tuning_summary("secondary", problem_chars, params)
        
        assert isinstance(summary, dict)
        assert "phase" in summary
        assert "problem_characteristics" in summary
        assert "tuned_parameters" in summary
        assert "tuning_rationale" in summary
        assert summary["phase"] == "secondary"


class TestUnifiedFrameworkIntegration:
    """Test unified framework integration with adaptive tuning."""

    def test_unified_framework_uses_adaptive_tuning(self):
        """Test unified framework integration with adaptive tuning."""
        from campro.optimization.unified_framework import UnifiedOptimizationFramework
        
        # Create framework
        framework = UnifiedOptimizationFramework()
        
        # Verify adaptive tuning components are initialized
        assert hasattr(framework, 'solver_selector')
        assert hasattr(framework, 'parameter_tuner')
        assert isinstance(framework.solver_selector, AdaptiveSolverSelector)
        assert isinstance(framework.parameter_tuner, DynamicParameterTuner)

    def test_adaptive_tuning_components_are_accessible(self):
        """Test that adaptive tuning components are accessible."""
        from campro.optimization.unified_framework import UnifiedOptimizationFramework
        
        framework = UnifiedOptimizationFramework()
        
        # Test solver selector methods
        problem_chars = ProblemCharacteristics(
            n_variables=100,
            n_constraints=50,
            problem_type="litvin",
            expected_iterations=500,
            linear_solver_ratio=0.3,
            has_convergence_issues=False
        )
        
        solver = framework.solver_selector.select_solver(problem_chars, "secondary")
        assert solver == SolverType.MA27
        
        # Test parameter tuner methods
        params = framework.parameter_tuner.tune_parameters("secondary", problem_chars, None)
        assert isinstance(params, SolverParameters)
        
        # Test history management
        framework.solver_selector.clear_history()
        assert len(framework.solver_selector.analysis_history) == 0
