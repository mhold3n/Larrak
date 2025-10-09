"""
Tests for the Motion Law Optimization Library

This module tests the refactored optimization library components including
the core optimizer, configuration factory, and result processing.
"""

from unittest.mock import Mock, patch

import pytest

from campro.freepiston.opt.config_factory import (
    ConfigFactory,
    create_engine_config,
    create_optimization_scenario,
    get_preset_config,
)
from campro.freepiston.opt.optimization_lib import (
    AdaptiveBackend,
    IPOPTBackend,
    MotionLawOptimizer,
    OptimizationConfig,
    OptimizationResult,
    ProblemBuilder,
    ResultProcessor,
    RobustIPOPTBackend,
    create_adaptive_optimizer,
    create_robust_optimizer,
    create_standard_optimizer,
    quick_optimize,
)
from campro.freepiston.opt.solution import Solution


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = OptimizationConfig()

        assert config.geometry == {}
        assert config.thermodynamics == {}
        assert config.bounds == {}
        assert config.constraints == {}
        assert config.num == {"K": 20, "C": 3}
        assert config.solver == {}
        assert config.objective == {"method": "indicated_work"}
        assert config.model_type == "0d"
        assert config.use_1d_gas is False
        assert config.n_cells == 50
        assert config.validation == {}
        assert config.output == {}
        assert config.warm_start is None
        assert config.refinement_strategy == "adaptive"
        assert config.max_refinements == 3

    def test_custom_config(self):
        """Test custom configuration creation."""
        config = OptimizationConfig(
            geometry={"bore": 0.1, "stroke": 0.08},
            thermodynamics={"gamma": 1.4, "R": 287.0},
            num={"K": 30, "C": 4},
            model_type="1d",
            use_1d_gas=True,
            n_cells=100,
        )

        assert config.geometry["bore"] == 0.1
        assert config.geometry["stroke"] == 0.08
        assert config.thermodynamics["gamma"] == 1.4
        assert config.thermodynamics["R"] == 287.0
        assert config.num["K"] == 30
        assert config.num["C"] == 4
        assert config.model_type == "1d"
        assert config.use_1d_gas is True
        assert config.n_cells == 100


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_default_result(self):
        """Test default result creation."""
        result = OptimizationResult(success=True)

        assert result.success is True
        assert result.solution is None
        assert result.objective_value == float("inf")
        assert result.iterations == 0
        assert result.cpu_time == 0.0
        assert result.kkt_error == float("inf")
        assert result.feasibility_error == float("inf")
        assert result.message == ""
        assert result.status == -1
        assert result.validation_metrics == {}
        assert result.physics_validation == {}
        assert result.performance_metrics == {}
        assert result.config is None
        assert result.metadata == {}
        assert result.warnings == []
        assert result.errors == []

    def test_custom_result(self):
        """Test custom result creation."""
        solution = Mock(spec=Solution)
        config = OptimizationConfig()

        result = OptimizationResult(
            success=True,
            solution=solution,
            objective_value=1.5e6,
            iterations=150,
            cpu_time=25.3,
            kkt_error=1e-6,
            feasibility_error=1e-8,
            message="Optimization successful",
            status=0,
            config=config,
            warnings=["Minor warning"],
            errors=[],
        )

        assert result.success is True
        assert result.solution is solution
        assert result.objective_value == 1.5e6
        assert result.iterations == 150
        assert result.cpu_time == 25.3
        assert result.kkt_error == 1e-6
        assert result.feasibility_error == 1e-8
        assert result.message == "Optimization successful"
        assert result.status == 0
        assert result.config is config
        assert result.warnings == ["Minor warning"]
        assert result.errors == []


class TestSolverBackends:
    """Test solver backend implementations."""

    def test_ipopt_backend_creation(self):
        """Test IPOPT backend creation."""
        backend = IPOPTBackend()
        assert backend.options == {}

        options = {"max_iter": 1000, "tol": 1e-6}
        backend = IPOPTBackend(options)
        assert backend.options == options

    def test_robust_ipopt_backend_creation(self):
        """Test robust IPOPT backend creation."""
        backend = RobustIPOPTBackend()
        assert backend.options == {}

        options = {"max_iter": 5000}
        backend = RobustIPOPTBackend(options)
        assert backend.options == options

    def test_adaptive_backend_creation(self):
        """Test adaptive backend creation."""
        backend = AdaptiveBackend()
        assert backend.max_refinements == 3
        assert backend.options == {}

        backend = AdaptiveBackend(max_refinements=5, options={"tol": 1e-4})
        assert backend.max_refinements == 5
        assert backend.options == {"tol": 1e-4}

    @patch("campro.freepiston.opt.optimization_lib.solve_cycle")
    def test_ipopt_backend_solve_success(self, mock_solve_cycle):
        """Test IPOPT backend solve with success."""
        # Mock successful solve
        mock_solution = Mock(spec=Solution)
        mock_solution.meta = {
            "optimization": {
                "success": True,
                "f_opt": 1.5e6,
                "iterations": 150,
                "kkt_error": 1e-6,
                "feasibility_error": 1e-8,
                "message": "Success",
                "status": 0,
            },
        }
        mock_solve_cycle.return_value = mock_solution

        backend = IPOPTBackend()
        problem = {"geometry": {}, "num": {"K": 10, "C": 2}}

        result = backend.solve(problem)

        assert result.success is True
        assert result.objective_value == 1.5e6
        assert result.iterations == 150
        assert result.kkt_error == 1e-6
        assert result.feasibility_error == 1e-8
        assert result.message == "Success"
        assert result.status == 0
        assert result.solution is mock_solution

    @patch("campro.freepiston.opt.optimization_lib.solve_cycle")
    def test_ipopt_backend_solve_failure(self, mock_solve_cycle):
        """Test IPOPT backend solve with failure."""
        # Mock failed solve
        mock_solve_cycle.side_effect = Exception("Solver failed")

        backend = IPOPTBackend()
        problem = {"geometry": {}, "num": {"K": 10, "C": 2}}

        result = backend.solve(problem)

        assert result.success is False
        assert "Solver failed" in result.errors[0]
        assert result.cpu_time > 0


class TestProblemBuilder:
    """Test ProblemBuilder class."""

    def test_problem_builder_creation(self):
        """Test problem builder creation."""
        config = OptimizationConfig()
        builder = ProblemBuilder(config)

        assert builder.config is config
        assert builder._problem == {}

    def test_build_default_problem(self):
        """Test building default problem."""
        config = OptimizationConfig()
        builder = ProblemBuilder(config)

        problem = builder.build()

        assert "geometry" in problem
        assert "thermodynamics" in problem
        assert "bounds" in problem
        assert "constraints" in problem
        assert "num" in problem
        assert "solver" in problem
        assert "obj" in problem
        assert "model_type" in problem
        assert "flow" in problem
        assert "validation" in problem
        assert "output" in problem

        assert problem["num"] == {"K": 20, "C": 3}
        assert problem["model_type"] == "0d"
        assert problem["flow"]["use_1d_gas"] is False
        assert problem["flow"]["mesh_cells"] == 50

    def test_build_with_warm_start(self):
        """Test building problem with warm start."""
        config = OptimizationConfig()
        config.warm_start = {"x0": [1.0, 2.0, 3.0]}

        builder = ProblemBuilder(config)
        problem = builder.build()

        assert "warm_start" in problem
        assert problem["warm_start"] == {"x0": [1.0, 2.0, 3.0]}

    def test_fluent_interface(self):
        """Test fluent interface for problem builder."""
        config = OptimizationConfig()
        builder = ProblemBuilder(config)

        # Test method chaining
        result = (builder
                 .with_geometry({"bore": 0.12})
                 .with_thermodynamics({"gamma": 1.4})
                 .with_bounds({"v_max": 30.0})
                 .with_constraints({"short_circuit_max": 0.1})
                 .with_objective({"method": "thermal_efficiency"})
                 .with_solver_options({"max_iter": 1000})
                 .with_1d_model(n_cells=40))

        assert result is builder  # Should return self for chaining

        # Verify configuration was updated
        assert config.geometry["bore"] == 0.12
        assert config.thermodynamics["gamma"] == 1.4
        assert config.bounds["v_max"] == 30.0
        assert config.constraints["short_circuit_max"] == 0.1
        assert config.objective["method"] == "thermal_efficiency"
        assert config.solver["max_iter"] == 1000
        assert config.model_type == "1d"
        assert config.use_1d_gas is True
        assert config.n_cells == 40

    def test_with_0d_model(self):
        """Test switching to 0D model."""
        config = OptimizationConfig()
        config.model_type = "1d"
        config.use_1d_gas = True

        builder = ProblemBuilder(config)
        builder.with_0d_model()

        assert config.model_type == "0d"
        assert config.use_1d_gas is False


class TestResultProcessor:
    """Test ResultProcessor class."""

    def test_result_processor_creation(self):
        """Test result processor creation."""
        config = OptimizationConfig()
        processor = ResultProcessor(config)

        assert processor.config is config
        assert processor.solution_validator is not None
        assert processor.physics_validator is not None

    def test_process_unsuccessful_result(self):
        """Test processing unsuccessful result."""
        config = OptimizationConfig()
        processor = ResultProcessor(config)

        result = OptimizationResult(success=False)
        processed = processor.process(result)

        assert processed is result  # Should return same object
        assert processed.validation_metrics == {}
        assert processed.physics_validation == {}
        assert processed.performance_metrics == {}

    def test_process_result_without_solution(self):
        """Test processing result without solution."""
        config = OptimizationConfig()
        processor = ResultProcessor(config)

        result = OptimizationResult(success=True, solution=None)
        processed = processor.process(result)

        assert processed is result
        assert processed.validation_metrics == {}
        assert processed.physics_validation == {}
        assert processed.performance_metrics == {}

    @patch("campro.freepiston.opt.optimization_lib.SolutionValidator")
    @patch("campro.freepiston.opt.optimization_lib.PhysicsValidator")
    def test_process_successful_result(self, mock_physics_validator, mock_solution_validator):
        """Test processing successful result."""
        # Mock validators
        mock_sol_validator = Mock()
        mock_sol_validator.validate.return_value = {"success": True, "warnings": []}
        mock_solution_validator.return_value = mock_sol_validator

        mock_phys_validator = Mock()
        mock_phys_validator.validate.return_value = {"success": True, "warnings": []}
        mock_physics_validator.return_value = mock_phys_validator

        config = OptimizationConfig()
        processor = ResultProcessor(config)

        # Mock solution with state data
        mock_solution = Mock(spec=Solution)
        mock_solution.data = {
            "states": {
                "pressure": [1e5, 2e5, 1.5e5],
                "temperature": [300, 400, 350],
                "x_L": [0.0, 0.01, 0.02],
                "x_R": [0.1, 0.11, 0.12],
            },
        }

        result = OptimizationResult(success=True, solution=mock_solution)
        processed = processor.process(result)

        assert processed.validation_metrics == {"success": True, "warnings": []}
        assert processed.physics_validation == {"success": True, "warnings": []}
        assert "max_pressure" in processed.performance_metrics
        assert "min_pressure" in processed.performance_metrics
        assert "max_temperature" in processed.performance_metrics
        assert "min_temperature" in processed.performance_metrics
        assert "min_piston_gap" in processed.performance_metrics


class TestMotionLawOptimizer:
    """Test MotionLawOptimizer class."""

    def test_optimizer_creation_with_config(self):
        """Test optimizer creation with OptimizationConfig."""
        config = OptimizationConfig()
        optimizer = MotionLawOptimizer(config)

        assert optimizer.config is config
        assert isinstance(optimizer.solver_backend, IPOPTBackend)
        assert isinstance(optimizer.problem_builder, ProblemBuilder)
        assert isinstance(optimizer.result_processor, ResultProcessor)

    def test_optimizer_creation_with_dict(self):
        """Test optimizer creation with dictionary config."""
        config_dict = {
            "geometry": {"bore": 0.1},
            "thermodynamics": {"gamma": 1.4},
            "num": {"K": 15, "C": 2},
        }
        optimizer = MotionLawOptimizer(config_dict)

        assert isinstance(optimizer.config, OptimizationConfig)
        assert optimizer.config.geometry["bore"] == 0.1
        assert optimizer.config.thermodynamics["gamma"] == 1.4
        assert optimizer.config.num["K"] == 15
        assert optimizer.config.num["C"] == 2

    def test_optimizer_creation_with_custom_backend(self):
        """Test optimizer creation with custom backend."""
        config = OptimizationConfig()
        custom_backend = Mock()

        optimizer = MotionLawOptimizer(config, custom_backend)

        assert optimizer.solver_backend is custom_backend

    def test_get_problem_builder(self):
        """Test getting problem builder."""
        config = OptimizationConfig()
        optimizer = MotionLawOptimizer(config)

        builder = optimizer.get_problem_builder()

        assert isinstance(builder, ProblemBuilder)
        assert builder.config is config

    def test_set_solver_backend(self):
        """Test setting solver backend."""
        config = OptimizationConfig()
        optimizer = MotionLawOptimizer(config)

        custom_backend = Mock()
        optimizer.set_solver_backend(custom_backend)

        assert optimizer.solver_backend is custom_backend

    @patch("campro.freepiston.opt.optimization_lib.IPOPTBackend")
    def test_optimize_success(self, mock_backend_class):
        """Test successful optimization."""
        # Mock backend
        mock_backend = Mock()
        mock_backend.solve.return_value = OptimizationResult(
            success=True,
            objective_value=1.5e6,
            iterations=150,
            cpu_time=25.3,
            message="Success",
        )
        mock_backend_class.return_value = mock_backend

        config = OptimizationConfig()
        optimizer = MotionLawOptimizer(config, mock_backend)

        result = optimizer.optimize()

        assert result.success is True
        assert result.objective_value == 1.5e6
        assert result.iterations == 150
        assert result.cpu_time == 25.3
        assert result.message == "Success"
        assert result.config is config

        # Verify backend was called
        mock_backend.solve.assert_called_once()

    @patch("campro.freepiston.opt.optimization_lib.IPOPTBackend")
    def test_optimize_failure(self, mock_backend_class):
        """Test failed optimization."""
        # Mock backend
        mock_backend = Mock()
        mock_backend.solve.return_value = OptimizationResult(
            success=False,
            errors=["Solver failed"],
        )
        mock_backend_class.return_value = mock_backend

        config = OptimizationConfig()
        optimizer = MotionLawOptimizer(config, mock_backend)

        result = optimizer.optimize()

        assert result.success is False
        assert "Solver failed" in result.errors
        assert result.config is config

    @patch("campro.freepiston.opt.optimization_lib.IPOPTBackend")
    def test_optimize_with_exception(self, mock_backend_class):
        """Test optimization with exception."""
        # Mock backend that raises exception
        mock_backend = Mock()
        mock_backend.solve.side_effect = Exception("Unexpected error")
        mock_backend_class.return_value = mock_backend

        config = OptimizationConfig()
        optimizer = MotionLawOptimizer(config, mock_backend)

        result = optimizer.optimize()

        assert result.success is False
        assert "Unexpected error" in result.errors[0]
        assert result.config is config


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_standard_optimizer(self):
        """Test create_standard_optimizer function."""
        config = OptimizationConfig()
        optimizer = create_standard_optimizer(config)

        assert isinstance(optimizer, MotionLawOptimizer)
        assert isinstance(optimizer.solver_backend, IPOPTBackend)

    def test_create_robust_optimizer(self):
        """Test create_robust_optimizer function."""
        config = OptimizationConfig()
        optimizer = create_robust_optimizer(config)

        assert isinstance(optimizer, MotionLawOptimizer)
        assert isinstance(optimizer.solver_backend, RobustIPOPTBackend)

    def test_create_adaptive_optimizer(self):
        """Test create_adaptive_optimizer function."""
        config = OptimizationConfig()
        optimizer = create_adaptive_optimizer(config, max_refinements=5)

        assert isinstance(optimizer, MotionLawOptimizer)
        assert isinstance(optimizer.solver_backend, AdaptiveBackend)
        assert optimizer.solver_backend.max_refinements == 5

    @patch("campro.freepiston.opt.optimization_lib.IPOPTBackend")
    def test_quick_optimize_standard(self, mock_backend_class):
        """Test quick_optimize with standard backend."""
        # Mock backend
        mock_backend = Mock()
        mock_backend.solve.return_value = OptimizationResult(success=True)
        mock_backend_class.return_value = mock_backend

        config = OptimizationConfig()
        result = quick_optimize(config, backend="standard")

        assert result.success is True

    @patch("campro.freepiston.opt.optimization_lib.RobustIPOPTBackend")
    def test_quick_optimize_robust(self, mock_backend_class):
        """Test quick_optimize with robust backend."""
        # Mock backend
        mock_backend = Mock()
        mock_backend.solve.return_value = OptimizationResult(success=True)
        mock_backend_class.return_value = mock_backend

        config = OptimizationConfig()
        result = quick_optimize(config, backend="robust")

        assert result.success is True

    @patch("campro.freepiston.opt.optimization_lib.AdaptiveBackend")
    def test_quick_optimize_adaptive(self, mock_backend_class):
        """Test quick_optimize with adaptive backend."""
        # Mock backend
        mock_backend = Mock()
        mock_backend.solve.return_value = OptimizationResult(success=True)
        mock_backend_class.return_value = mock_backend

        config = OptimizationConfig()
        result = quick_optimize(config, backend="adaptive")

        assert result.success is True


class TestConfigFactory:
    """Test ConfigFactory class."""

    def test_create_default_config(self):
        """Test creating default configuration."""
        config = ConfigFactory.create_default_config()

        assert isinstance(config, OptimizationConfig)
        assert "bore" in config.geometry
        assert "gamma" in config.thermodynamics
        assert "xL_min" in config.bounds
        assert config.num["K"] == 20
        assert config.num["C"] == 3

    def test_create_high_performance_config(self):
        """Test creating high performance configuration."""
        config = ConfigFactory.create_high_performance_config()

        assert config.num["K"] == 50
        assert config.num["C"] == 4
        assert config.solver["ipopt"]["max_iter"] == 5000
        assert config.solver["ipopt"]["tol"] == 1e-8

    def test_create_quick_test_config(self):
        """Test creating quick test configuration."""
        config = ConfigFactory.create_quick_test_config()

        assert config.num["K"] == 10
        assert config.num["C"] == 2
        assert config.solver["ipopt"]["max_iter"] == 500
        assert config.solver["ipopt"]["tol"] == 1e-4

    def test_create_1d_config(self):
        """Test creating 1D configuration."""
        config = ConfigFactory.create_1d_config(n_cells=40)

        assert config.model_type == "1d"
        assert config.use_1d_gas is True
        assert config.n_cells == 40
        assert config.num["K"] == 30

    def test_create_robust_config(self):
        """Test creating robust configuration."""
        config = ConfigFactory.create_robust_config()

        assert config.solver["ipopt"]["max_iter"] == 10000
        assert config.solver["ipopt"]["mu_strategy"] == "adaptive"
        assert config.refinement_strategy == "adaptive"
        assert config.max_refinements == 3

    def test_create_custom_config(self):
        """Test creating custom configuration."""
        config = ConfigFactory.create_custom_config(
            geometry={"bore": 0.15},
            num={"K": 25, "C": 4},
            model_type="1d",
        )

        assert config.geometry["bore"] == 0.15
        assert config.num["K"] == 25
        assert config.num["C"] == 4
        assert config.model_type == "1d"

    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "geometry": {"bore": 0.12, "stroke": 0.08},
            "thermodynamics": {"gamma": 1.4, "R": 287.0},
            "num": {"K": 30, "C": 3},
            "model_type": "1d",
            "use_1d_gas": True,
            "n_cells": 60,
        }

        config = ConfigFactory.from_dict(config_dict)

        assert config.geometry["bore"] == 0.12
        assert config.geometry["stroke"] == 0.08
        assert config.thermodynamics["gamma"] == 1.4
        assert config.thermodynamics["R"] == 287.0
        assert config.num["K"] == 30
        assert config.num["C"] == 3
        assert config.model_type == "1d"
        assert config.use_1d_gas is True
        assert config.n_cells == 60

    def test_to_yaml_and_from_yaml(self, tmp_path):
        """Test saving and loading configuration to/from YAML."""
        config = ConfigFactory.create_custom_config(
            geometry={"bore": 0.12},
            num={"K": 25, "C": 3},
        )

        yaml_file = tmp_path / "test_config.yaml"

        # Save to YAML
        ConfigFactory.to_yaml(config, yaml_file)
        assert yaml_file.exists()

        # Load from YAML
        loaded_config = ConfigFactory.from_yaml(yaml_file)

        assert loaded_config.geometry["bore"] == 0.12
        assert loaded_config.num["K"] == 25
        assert loaded_config.num["C"] == 3


class TestPresetFunctions:
    """Test preset configuration functions."""

    def test_get_preset_config_default(self):
        """Test getting default preset."""
        config = get_preset_config("default")

        assert isinstance(config, OptimizationConfig)
        assert config.num["K"] == 20
        assert config.num["C"] == 3

    def test_get_preset_config_high_performance(self):
        """Test getting high performance preset."""
        config = get_preset_config("high_performance")

        assert config.num["K"] == 50
        assert config.num["C"] == 4

    def test_get_preset_config_invalid(self):
        """Test getting invalid preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("invalid_preset")

    def test_create_engine_config_opposed_piston(self):
        """Test creating opposed piston engine configuration."""
        config = create_engine_config("opposed_piston")

        assert config.geometry["bore"] == 0.12
        assert config.geometry["stroke"] == 0.08
        assert config.geometry["compression_ratio"] == 12.0
        assert config.geometry["mass"] == 1.2

    def test_create_engine_config_free_piston(self):
        """Test creating free piston engine configuration."""
        config = create_engine_config("free_piston")

        assert config.geometry["bore"] == 0.08
        assert config.geometry["stroke"] == 0.06
        assert config.geometry["compression_ratio"] == 15.0
        assert config.geometry["mass"] == 0.8

    def test_create_engine_config_conventional(self):
        """Test creating conventional engine configuration."""
        config = create_engine_config("conventional")

        assert config.geometry["bore"] == 0.1
        assert config.geometry["stroke"] == 0.1
        assert config.geometry["compression_ratio"] == 10.0
        assert config.geometry["mass"] == 1.0

    def test_create_engine_config_invalid(self):
        """Test creating invalid engine configuration."""
        with pytest.raises(ValueError, match="Unknown engine type"):
            create_engine_config("invalid_engine")

    def test_create_optimization_scenario_efficiency(self):
        """Test creating efficiency scenario configuration."""
        config = create_optimization_scenario("efficiency")

        assert config.objective["method"] == "thermal_efficiency"
        assert config.objective["w"]["eta_th"] == 1.0
        assert config.constraints["short_circuit_max"] == 0.05
        assert config.constraints["scavenging_min"] == 0.9

    def test_create_optimization_scenario_power(self):
        """Test creating power scenario configuration."""
        config = create_optimization_scenario("power")

        assert config.objective["method"] == "indicated_work"
        assert config.bounds["Q_comb_max"] == 20000.0
        assert config.bounds["p_max"] == 2e7

    def test_create_optimization_scenario_emissions(self):
        """Test creating emissions scenario configuration."""
        config = create_optimization_scenario("emissions")

        assert config.objective["method"] == "scavenging"
        assert config.objective["w"]["short_circuit"] == 3.0
        assert config.constraints["short_circuit_max"] == 0.02
        assert config.constraints["scavenging_min"] == 0.95

    def test_create_optimization_scenario_durability(self):
        """Test creating durability scenario configuration."""
        config = create_optimization_scenario("durability")

        assert config.objective["method"] == "smoothness"
        assert config.objective["w"]["smooth"] == 1.0
        assert config.bounds["v_max"] == 30.0
        assert config.bounds["a_max"] == 500.0

    def test_create_optimization_scenario_invalid(self):
        """Test creating invalid scenario configuration."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            create_optimization_scenario("invalid_scenario")


if __name__ == "__main__":
    pytest.main([__file__])
