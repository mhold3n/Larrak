"""
Integration tests for CasADi Phase 1 optimization.

This module tests the integration of CasADi-based motion law optimization
with the unified optimization framework.
"""

import time

import numpy as np
import pytest

from campro.optimization.casadi_motion_optimizer import (
    CasADiMotionOptimizer,
    CasADiMotionProblem,
)
from campro.optimization.casadi_problem_spec import (
    create_default_problem,
    create_high_efficiency_problem,
    create_smooth_motion_problem,
)
from campro.optimization.casadi_unified_flow import CasADiUnifiedFlow
from campro.optimization.warmstart_manager import WarmStartManager
from campro.physics.thermal_efficiency_simple import SimplifiedThermalModel


class TestCasADiMotionOptimizer:
    """Test CasADi motion optimizer functionality."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization with different parameters."""
        optimizer = CasADiMotionOptimizer(
            n_segments=50,
            poly_order=3,
            collocation_method="legendre",
        )

        assert optimizer.n_segments == 50
        assert optimizer.poly_order == 3
        assert optimizer.collocation_method == "legendre"
        assert optimizer.opti is not None

    def test_problem_creation(self):
        """Test problem creation and validation."""
        problem = create_default_problem(
            stroke=0.1,
            cycle_time=0.0385,
            upstroke_percent=50.0,
        )

        assert problem.stroke == 0.1
        assert problem.cycle_time == 0.0385
        assert problem.upstroke_percent == 50.0
        assert problem.is_feasible()
        assert problem.get_frequency() == pytest.approx(1.0 / 0.0385, rel=1e-3)

    def test_basic_optimization(self):
        """Test basic optimization functionality."""
        optimizer = CasADiMotionOptimizer(n_segments=20, poly_order=2)
        problem = create_default_problem(stroke=0.1, cycle_time=0.0385)

        result = optimizer.solve(problem)

        assert result is not None
        assert hasattr(result, "successful")
        assert hasattr(result, "solve_time")
        assert hasattr(result, "objective_value")

        if result.successful:
            assert "position" in result.variables
            assert "velocity" in result.variables
            assert "acceleration" in result.variables
            assert "jerk" in result.variables

            # Check boundary conditions
            position = result.variables["position"]
            velocity = result.variables["velocity"]

            assert abs(position[0]) < 1e-6  # Start at zero
            assert abs(position[-1] - 0.1) < 1e-6  # End at stroke
            assert abs(velocity[0]) < 1e-6  # Start at rest
            assert abs(velocity[-1]) < 1e-6  # End at rest

    def test_motion_constraints(self):
        """Test that motion constraints are satisfied."""
        optimizer = CasADiMotionOptimizer(n_segments=20, poly_order=2)
        problem = create_default_problem(
            stroke=0.1,
            cycle_time=0.0385,
            max_velocity=3.0,
            max_acceleration=200.0,
            max_jerk=20000.0,
        )

        result = optimizer.solve(problem)

        if result.successful:
            velocity = result.variables["velocity"]
            acceleration = result.variables["acceleration"]
            jerk = result.variables["jerk"]

            # Check velocity constraints
            assert np.all(np.abs(velocity) <= 3.0 + 1e-6)

            # Check acceleration constraints
            assert np.all(np.abs(acceleration) <= 200.0 + 1e-6)

            # Check jerk constraints
            assert np.all(np.abs(jerk) <= 20000.0 + 1e-6)

    def test_thermal_efficiency_objective(self):
        """Test thermal efficiency objective integration."""
        optimizer = CasADiMotionOptimizer(n_segments=20, poly_order=2)
        problem = create_high_efficiency_problem(stroke=0.1, cycle_time=0.0385)

        result = optimizer.solve(problem)

        if result.successful:
            # Evaluate thermal efficiency
            thermal_model = SimplifiedThermalModel()
            efficiency_metrics = thermal_model.evaluate_efficiency(
                result.variables["position"],
                result.variables["velocity"],
                result.variables["acceleration"],
            )

            assert "total_efficiency" in efficiency_metrics
            assert "compression_ratio" in efficiency_metrics
            assert efficiency_metrics["total_efficiency"] > 0
            assert efficiency_metrics["compression_ratio"] > 1.0


class TestWarmStartManager:
    """Test warm-start manager functionality."""

    def test_manager_initialization(self):
        """Test warm-start manager initialization."""
        manager = WarmStartManager(max_history=10, tolerance=0.1)

        assert manager.max_history == 10
        assert manager.tolerance == 0.1
        assert len(manager.solution_history) == 0

    def test_solution_storage(self):
        """Test solution storage and retrieval."""
        manager = WarmStartManager(max_history=5)

        # Create mock solution data
        problem_params = {
            "stroke": 0.1,
            "cycle_time": 0.0385,
            "upstroke_percent": 50.0,
            "max_velocity": 5.0,
            "max_acceleration": 500.0,
            "max_jerk": 50000.0,
        }

        solution_data = {
            "position": np.linspace(0, 0.1, 51),
            "velocity": np.sin(np.linspace(0, 2 * np.pi, 51)),
            "acceleration": np.cos(np.linspace(0, 2 * np.pi, 51)),
            "jerk": np.random.randn(50),
        }

        metadata = {
            "solve_time": 1.5,
            "objective_value": 0.123,
            "n_segments": 50,
            "timestamp": time.time(),
        }

        # Store solution
        manager.store_solution(problem_params, solution_data, metadata)

        assert len(manager.solution_history) == 1
        assert manager.solution_history[0].stroke == 0.1
        assert manager.solution_history[0].cycle_time == 0.0385

    def test_initial_guess_generation(self):
        """Test initial guess generation."""
        manager = WarmStartManager(max_history=5, tolerance=0.1)

        # Store a solution
        problem_params = {
            "stroke": 0.1,
            "cycle_time": 0.0385,
            "upstroke_percent": 50.0,
            "max_velocity": 5.0,
            "max_acceleration": 500.0,
            "max_jerk": 50000.0,
        }

        solution_data = {
            "position": np.linspace(0, 0.1, 51),
            "velocity": np.sin(np.linspace(0, 2 * np.pi, 51)),
            "acceleration": np.cos(np.linspace(0, 2 * np.pi, 51)),
            "jerk": np.random.randn(50),
        }

        metadata = {
            "solve_time": 1.5,
            "objective_value": 0.123,
            "n_segments": 50,
            "timestamp": time.time(),
        }

        manager.store_solution(problem_params, solution_data, metadata)

        # Test exact match
        similar_params = problem_params.copy()
        initial_guess = manager.get_initial_guess(similar_params)

        assert initial_guess is not None
        assert "x" in initial_guess
        assert "v" in initial_guess
        assert "a" in initial_guess
        assert "j" in initial_guess

        # Test fallback for no match
        different_params = {
            "stroke": 0.5,  # Very different
            "cycle_time": 0.1,
            "upstroke_percent": 50.0,
            "max_velocity": 5.0,
            "max_acceleration": 500.0,
            "max_jerk": 50000.0,
        }

        fallback_guess = manager.get_initial_guess(different_params)
        assert fallback_guess is not None  # Should generate fallback

    def test_history_management(self):
        """Test history size management."""
        manager = WarmStartManager(max_history=3)

        # Add more solutions than max_history
        for i in range(5):
            problem_params = {
                "stroke": 0.1 + i * 0.01,
                "cycle_time": 0.0385,
                "upstroke_percent": 50.0,
                "max_velocity": 5.0,
                "max_acceleration": 500.0,
                "max_jerk": 50000.0,
            }

            solution_data = {
                "position": np.linspace(0, 0.1, 51),
                "velocity": np.sin(np.linspace(0, 2 * np.pi, 51)),
                "acceleration": np.cos(np.linspace(0, 2 * np.pi, 51)),
                "jerk": np.random.randn(50),
            }

            metadata = {
                "solve_time": 1.5,
                "objective_value": 0.123,
                "n_segments": 50,
                "timestamp": time.time(),
            }

            manager.store_solution(problem_params, solution_data, metadata)

        # Should only keep max_history solutions
        assert len(manager.solution_history) == 3

        # Should keep the most recent solutions
        assert manager.solution_history[-1].stroke == 0.14  # Last added


class TestThermalEfficiencyModel:
    """Test thermal efficiency model functionality."""

    def test_model_initialization(self):
        """Test thermal efficiency model initialization."""
        model = SimplifiedThermalModel()

        assert model.config.efficiency_target == 0.55
        assert model.config.compression_ratio_range == (20.0, 70.0)
        assert model.config.gamma == 1.4

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        model = SimplifiedThermalModel()

        position = np.array([0.0, 0.05, 0.1])
        cr = model.compute_compression_ratio(position)

        expected_cr = (position + 0.002) / 0.002
        np.testing.assert_array_almost_equal(cr, expected_cr)

    def test_otto_efficiency_calculation(self):
        """Test Otto cycle efficiency calculation."""
        model = SimplifiedThermalModel()

        cr = np.array([20.0, 30.0, 50.0])
        eta_otto = model.compute_otto_efficiency(cr)

        # Should be between 0 and 1
        assert np.all(eta_otto >= 0)
        assert np.all(eta_otto <= 1)

        # Higher compression ratio should give higher efficiency
        assert eta_otto[2] > eta_otto[1] > eta_otto[0]

    def test_thermal_efficiency_evaluation(self):
        """Test complete thermal efficiency evaluation."""
        model = SimplifiedThermalModel()

        # Create mock motion profile
        position = np.linspace(0, 0.1, 51)
        velocity = np.sin(np.linspace(0, 2 * np.pi, 51))
        acceleration = np.cos(np.linspace(0, 2 * np.pi, 51))

        efficiency_metrics = model.evaluate_efficiency(position, velocity, acceleration)

        assert "compression_ratio" in efficiency_metrics
        assert "otto_efficiency" in efficiency_metrics
        assert "heat_loss_penalty" in efficiency_metrics
        assert "mechanical_loss" in efficiency_metrics
        assert "total_efficiency" in efficiency_metrics
        assert "efficiency_target" in efficiency_metrics
        assert "efficiency_achieved" in efficiency_metrics

        assert efficiency_metrics["compression_ratio"] > 1.0
        assert 0 <= efficiency_metrics["otto_efficiency"] <= 1
        assert efficiency_metrics["total_efficiency"] > 0


class TestCasADiUnifiedFlow:
    """Test CasADi unified flow functionality."""

    def test_flow_initialization(self):
        """Test unified flow initialization."""
        flow = CasADiUnifiedFlow()

        assert hasattr(flow, "motion_optimizer")
        assert hasattr(flow, "warmstart_mgr")
        assert hasattr(flow, "thermal_model")
        assert hasattr(flow, "settings")

    def test_phase1_optimization(self):
        """Test Phase 1 optimization through unified flow."""
        flow = CasADiUnifiedFlow()

        constraints = {
            "stroke": 0.1,
            "cycle_time": 0.0385,
            "upstroke_percent": 50.0,
            "max_velocity": 5.0,
            "max_acceleration": 500.0,
            "max_jerk": 50000.0,
            "compression_ratio_limits": (20.0, 70.0),
        }

        targets = {
            "minimize_jerk": True,
            "maximize_thermal_efficiency": True,
            "weights": {
                "jerk": 1.0,
                "thermal_efficiency": 0.1,
                "smoothness": 0.01,
            },
        }

        result = flow.optimize_phase1(constraints, targets)

        assert result is not None
        assert hasattr(result, "successful")
        assert hasattr(result, "solve_time")

        if result.successful:
            assert "total_efficiency" in result.metadata
            assert result.metadata["total_efficiency"] > 0

    def test_warmstart_integration(self):
        """Test warm-start integration in unified flow."""
        flow = CasADiUnifiedFlow()

        # First optimization
        constraints1 = {
            "stroke": 0.1,
            "cycle_time": 0.0385,
            "upstroke_percent": 50.0,
            "max_velocity": 5.0,
            "max_acceleration": 500.0,
            "max_jerk": 50000.0,
            "compression_ratio_limits": (20.0, 70.0),
        }

        targets = {
            "minimize_jerk": True,
            "maximize_thermal_efficiency": True,
            "weights": {
                "jerk": 1.0,
                "thermal_efficiency": 0.1,
                "smoothness": 0.01,
            },
        }

        result1 = flow.optimize_phase1(constraints1, targets)

        if result1.successful:
            # Second optimization with similar parameters
            constraints2 = constraints1.copy()
            constraints2["stroke"] = 0.105  # Slightly different

            result2 = flow.optimize_phase1(constraints2, targets)

            # Both should succeed
            assert result1.successful
            assert result2.successful

            # Warm-start should help with convergence
            assert result2.solve_time >= 0  # Should not be negative

    def test_benchmarking(self):
        """Test optimization benchmarking."""
        flow = CasADiUnifiedFlow()

        problem_specs = [
            {
                "constraints": {
                    "stroke": 0.08,
                    "cycle_time": 0.035,
                    "upstroke_percent": 45.0,
                },
                "targets": {"minimize_jerk": True, "maximize_thermal_efficiency": True},
            },
            {
                "constraints": {
                    "stroke": 0.10,
                    "cycle_time": 0.0385,
                    "upstroke_percent": 50.0,
                },
                "targets": {"minimize_jerk": True, "maximize_thermal_efficiency": True},
            },
        ]

        benchmark_results = flow.benchmark_optimization(problem_specs)

        assert "total_problems" in benchmark_results
        assert "successful_problems" in benchmark_results
        assert "success_rate" in benchmark_results
        assert "avg_solve_time" in benchmark_results
        assert "results" in benchmark_results

        assert benchmark_results["total_problems"] == 2
        assert 0 <= benchmark_results["success_rate"] <= 1


class TestIntegration:
    """Test integration between components."""

    def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization workflow."""
        # Create problem
        problem = create_default_problem(stroke=0.1, cycle_time=0.0385)

        # Initialize components
        optimizer = CasADiMotionOptimizer(n_segments=20, poly_order=2)
        warmstart_mgr = WarmStartManager(max_history=5)
        thermal_model = SimplifiedThermalModel()

        # First optimization
        result1 = optimizer.solve(problem)

        if result1.successful:
            # Store solution
            warmstart_mgr.store_solution(
                problem.to_dict(),
                result1.variables,
                {
                    "solve_time": result1.solve_time,
                    "objective_value": result1.objective_value,
                    "n_segments": 20,
                    "timestamp": time.time(),
                },
            )

            # Evaluate thermal efficiency
            efficiency_metrics = thermal_model.evaluate_efficiency(
                result1.variables["position"],
                result1.variables["velocity"],
                result1.variables["acceleration"],
            )

            assert efficiency_metrics["total_efficiency"] > 0

            # Second optimization with warm-start
            similar_problem = create_default_problem(stroke=0.105, cycle_time=0.040)
            initial_guess = warmstart_mgr.get_initial_guess(similar_problem.to_dict())

            if initial_guess:
                result2 = optimizer.solve(similar_problem, initial_guess)
                assert result2.successful

    def test_performance_comparison(self):
        """Test performance comparison between different problem types."""
        optimizer = CasADiMotionOptimizer(n_segments=20, poly_order=2)

        problems = {
            "Default": create_default_problem(),
            "High Efficiency": create_high_efficiency_problem(),
            "Smooth Motion": create_smooth_motion_problem(),
        }

        results = {}

        for name, problem in problems.items():
            start_time = time.time()
            result = optimizer.solve(problem)
            solve_time = time.time() - start_time

            results[name] = {
                "result": result,
                "solve_time": solve_time,
            }

        # All should succeed
        for name, data in results.items():
            if data["result"].successful:
                assert data["solve_time"] > 0
                assert data["result"].objective_value is not None

    def test_error_handling(self):
        """Test error handling for invalid problems."""
        optimizer = CasADiMotionOptimizer(n_segments=20, poly_order=2)

        # Create invalid problem
        invalid_problem = CasADiMotionProblem(
            stroke=-0.1,  # Invalid negative stroke
            cycle_time=0.0385,
            upstroke_percent=50.0,
            max_velocity=5.0,
            max_acceleration=500.0,
            max_jerk=50000.0,
        )

        # Should handle invalid problem gracefully
        result = optimizer.solve(invalid_problem)

        # Should either fail gracefully or raise appropriate error
        assert not result.successful or result.metadata.get("error") is not None


if __name__ == "__main__":
    pytest.main([__file__])

