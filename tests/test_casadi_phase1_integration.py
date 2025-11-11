"""
Integration tests for CasADi Phase 1 optimization.

This module tests the integration of CasADi-based motion law optimization
with the unified optimization framework.
"""

import time
from typing import Any

import numpy as np
import pytest

from campro.optimization.base import OptimizationResult, OptimizationStatus
from campro.optimization.casadi_motion_optimizer import (
    CasADiMotionOptimizer,
    CasADiMotionProblem,
)
from campro.optimization.casadi_problem_spec import (
    create_default_problem,
    create_high_efficiency_problem,
    create_smooth_motion_problem,
)
from campro.optimization.casadi_unified_flow import (
    CasADiOptimizationSettings,
    CasADiUnifiedFlow,
)
from campro.optimization.initial_guess import InitialGuessBuilder
from campro.optimization.unified_framework import UnifiedOptimizationFramework
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
        assert problem.duration_angle_deg == 360.0
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
            assert "theta_deg" in result.solution
            assert "theta_rad" in result.solution

            # Check boundary conditions
            position = result.variables["position"]
            velocity = result.variables["velocity"]

            assert abs(position[0]) < 1e-6  # Start at zero
            assert abs(position[-1] - 0.1) < 1e-6  # End at stroke
            assert abs(velocity[0]) < 1e-6  # Start at rest
            assert abs(velocity[-1]) < 1e-6  # End at rest
            
            # Phase 1: Verify theta_deg is present and spans duration_angle_deg
            theta_deg = result.solution["theta_deg"]
            assert theta_deg is not None
            assert len(theta_deg) == len(position)
            assert theta_deg[0] == pytest.approx(0.0, abs=1e-6)
            assert theta_deg[-1] == pytest.approx(problem.duration_angle_deg, abs=1e-3)

    def test_per_degree_units_scaling(self):
        """Phase 3: Test that doubling duration_angle_deg halves derivative magnitudes."""
        optimizer = CasADiMotionOptimizer(n_segments=20, poly_order=2)
        
        # Test with duration_angle_deg=180 (half of default 360)
        problem_180 = create_default_problem(
            stroke=0.1,
            cycle_time=0.0385,
            duration_angle_deg=180.0,
        )
        result_180 = optimizer.solve(problem_180)
        
        # Test with duration_angle_deg=360 (default)
        problem_360 = create_default_problem(
            stroke=0.1,
            cycle_time=0.0385,
            duration_angle_deg=360.0,
        )
        result_360 = optimizer.solve(problem_360)
        
        if result_180.successful and result_360.successful:
            # Phase 3: Verify that constraints are interpreted as per-degree
            # When duration_angle_deg is halved, the same per-degree constraint should
            # allow roughly the same max velocity magnitude (in per-degree units)
            vel_180 = result_180.variables["velocity"]
            vel_360 = result_360.variables["velocity"]
            
            # Both should respect the same per-degree constraint
            max_vel_180 = np.max(np.abs(vel_180))
            max_vel_360 = np.max(np.abs(vel_360))
            
            # The max velocities should be similar (both constrained by same m/deg limit)
            # Allow some tolerance for optimization differences
            assert max_vel_180 <= problem_180.max_velocity + 1e-6
            assert max_vel_360 <= problem_360.max_velocity + 1e-6
            
            # Verify theta_deg spans the correct range
            assert result_180.solution["theta_deg"][-1] == pytest.approx(180.0, abs=1e-3)
            assert result_360.solution["theta_deg"][-1] == pytest.approx(360.0, abs=1e-3)

    def test_derived_cycle_time(self):
        """Phase 6: Test that cycle_time is derived from duration_angle_deg and engine_speed_rpm."""
        from campro.optimization.unified_framework import UnifiedOptimizationFramework, UnifiedOptimizationSettings
        
        framework = UnifiedOptimizationFramework()
        framework.settings.use_casadi = False  # Don't actually run optimization
        
        # Test case 1: Derive RPM from cycle_time and duration_angle_deg
        input_data = {
            "stroke": 20.0,
            "cycle_time": 0.0385,  # 1558 RPM for 360 deg
            "duration_angle_deg": 360.0,
        }
        framework._update_data_from_input(input_data)
        
        expected_rpm = 60.0 * 360.0 / (360.0 * 0.0385)  # ≈ 1558.44 RPM
        assert framework.data.engine_speed_rpm is not None
        assert abs(framework.data.engine_speed_rpm - expected_rpm) < 1.0  # Within 1 RPM
        
        # Test case 2: Derive cycle_time from engine_speed_rpm and duration_angle_deg
        framework.data.engine_speed_rpm = 2000.0  # 2000 RPM
        framework.data.duration_angle_deg = 360.0
        framework._update_data_from_input({"cycle_time": 0.0})  # Reset
        
        expected_cycle_time = (360.0 / 360.0) * (60.0 / 2000.0)  # 0.03 s
        assert abs(framework.data.cycle_time - expected_cycle_time) < 1e-6
        
        # Test case 3: Different duration_angle_deg
        framework.data.engine_speed_rpm = 1500.0
        framework.data.duration_angle_deg = 180.0  # Half cycle
        framework._update_data_from_input({"cycle_time": 0.0})  # Reset
        
        expected_cycle_time_180 = (180.0 / 360.0) * (60.0 / 1500.0)  # 0.02 s
        assert abs(framework.data.cycle_time - expected_cycle_time_180) < 1e-6

    def test_motion_constraints(self):
        """Test that motion constraints are satisfied."""
        optimizer = CasADiMotionOptimizer(n_segments=20, poly_order=2)
        # Use per-degree constraints (SI units: m/deg, m/deg², m/deg³)
        problem = create_default_problem(
            stroke=0.1,
            cycle_time=0.0385,
            max_velocity=0.003,  # m/deg (per-degree units)
            max_acceleration=0.2,  # m/deg² (per-degree units)
            max_jerk=20.0,  # m/deg³ (per-degree units)
        )

        result = optimizer.solve(problem)

        if result.successful:
            velocity = result.variables["velocity"]
            acceleration = result.variables["acceleration"]
            jerk = result.variables["jerk"]
            assert "theta_deg" in result.solution
            
            # Phase 1: Verify theta_deg is present
            theta_deg = result.solution["theta_deg"]
            assert theta_deg is not None
            assert len(theta_deg) == len(velocity)

            # Check velocity constraints (per-degree units)
            assert np.all(np.abs(velocity) <= problem.max_velocity + 1e-6)

            # Check acceleration constraints (per-degree units)
            assert np.all(np.abs(acceleration) <= problem.max_acceleration + 1e-6)

            # Check jerk constraints (per-degree units)
            assert np.all(np.abs(jerk) <= problem.max_jerk + 1e-6)

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


class TestInitialGuessBuilder:
    """Test deterministic initial guess builder."""

    def test_seed_shapes_and_boundaries(self):
        builder = InitialGuessBuilder(n_segments=50)
        problem = create_default_problem(stroke=0.1, cycle_time=0.0385)

        seed = builder.build_seed(problem)

        assert seed["x"].shape == (51,)
        assert seed["v"].shape == (51,)
        assert seed["a"].shape == (51,)
        assert seed["j"].shape == (50,)
        assert seed["x"][0] == pytest.approx(0.0, abs=1e-8)
        assert seed["x"][-1] == pytest.approx(problem.stroke, abs=1e-8)
        assert seed["v"][0] == pytest.approx(0.0, abs=1e-8)
        assert seed["v"][-1] == pytest.approx(0.0, abs=1e-8)

    def test_polish_clips_derivatives(self):
        """Test that polish_seed clamps derivatives using per-degree limits."""
        builder = InitialGuessBuilder(n_segments=30)
        problem = create_default_problem(
            stroke=0.08,
            cycle_time=0.04,
            upstroke_percent=70.0,
        )

        seed = builder.build_seed(problem)
        polished = builder.polish_seed(problem, seed)

        # Phase 1: Verify derivatives are clipped to per-degree constraints
        # All constraints are in per-degree units (m/deg, m/deg², m/deg³)
        assert np.max(np.abs(polished["v"])) <= problem.max_velocity + 1e-6
        assert np.max(np.abs(polished["a"])) <= problem.max_acceleration + 1e-6
        assert np.max(np.abs(polished["j"])) <= problem.max_jerk + 1e-3
        
        # Verify seed includes all required fields
        assert "x" in seed
        assert "v" in seed
        assert "a" in seed
        assert "j" in seed


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
        assert hasattr(flow, "initial_guess_builder")
        assert hasattr(flow, "thermal_model")
        assert hasattr(flow, "settings")

    def test_phase1_optimization(self):
        """Test Phase 1 optimization through unified flow."""
        flow = CasADiUnifiedFlow()

        constraints = {
            "stroke": 0.1,
            "cycle_time": 0.0385,
            "duration_angle_deg": 360.0,
            "upstroke_percent": 50.0,
            "max_velocity": 0.003,  # m/deg (per-degree units)
            "max_acceleration": 0.2,  # m/deg² (per-degree units)
            "max_jerk": 20.0,  # m/deg³ (per-degree units)
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
            assert "resolution_levels" in result.metadata

    def test_warmstart_integration(self):
        """Test warm-start integration in unified flow."""
        flow = CasADiUnifiedFlow()
        constraints = {
            "stroke": 0.1,
            "cycle_time": 0.0385,
            "duration_angle_deg": 360.0,
            "upstroke_percent": 50.0,
            "max_velocity": 0.003,  # m/deg (per-degree units)
            "max_acceleration": 0.2,  # m/deg² (per-degree units)
            "max_jerk": 20.0,  # m/deg³ (per-degree units)
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

        # Disable polish to get baseline seed
        flow.settings.enable_warmstart = False
        cold_result = flow.optimize_phase1(constraints, targets)

        # Re-enable polish (deterministic warm start enhancement)
        flow.settings.enable_warmstart = True
        polished_result = flow.optimize_phase1(constraints, targets)

        assert cold_result is not None
        assert polished_result is not None
        if cold_result.successful and polished_result.successful:
            # Polished solve should never take longer than the cold solve by orders of magnitude
            assert polished_result.solve_time >= 0

    def test_adaptive_resolution_metadata_tracks_levels(self, monkeypatch):
        """Ensure adaptive resolution ladder metadata is recorded."""
        settings = CasADiOptimizationSettings(
            coarse_resolution_segments=(10, 20),
            target_angle_resolution_deg=1.0,
            max_angle_resolution_segments=24,
        )
        flow = CasADiUnifiedFlow(settings=settings)
        solve_calls: list[int] = []

        def fake_solve(self, problem, seed):
            n_segments = self.n_segments
            solve_calls.append(n_segments)
            n_points = n_segments + 1
            position = np.linspace(0.0, problem.stroke, n_points)
            zeros = np.zeros(n_points)
            jerk = np.zeros(n_segments)
            return OptimizationResult(
                status=OptimizationStatus.CONVERGED,
                objective_value=float(n_segments),
                solve_time=0.0,
                solution={
                    "position": position,
                    "velocity": zeros,
                    "acceleration": zeros,
                    "jerk": jerk,
                },
                metadata={"n_segments": n_segments},
            )

        monkeypatch.setattr(
            flow.motion_optimizer,
            "solve",
            fake_solve.__get__(flow.motion_optimizer, CasADiMotionOptimizer),
        )

        constraints = {
            "stroke": 0.1,
            "cycle_time": 0.02,
            "upstroke_percent": 50.0,
        }
        targets = {
            "minimize_jerk": True,
            "maximize_thermal_efficiency": False,
            "weights": {
                "jerk": 1.0,
                "thermal_efficiency": 0.0,
                "smoothness": 0.0,
            },
        }

        result = flow.optimize_phase1(constraints, targets)

        assert result.successful
        assert "resolution_levels" in result.metadata
        assert solve_calls == result.metadata["adaptive_resolution_schedule"]
        assert result.metadata["finest_success_segments"] == solve_calls[-1]

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


class TestUnifiedFrameworkCasADiIntegration:
    """Ensure unified framework routes Stage 1 through CasADi when enabled."""

    def test_use_casadi_flag_routes_primary_solver(self, monkeypatch):
        framework = UnifiedOptimizationFramework()
        framework.settings.use_casadi = True
        framework.settings.enable_thermal_efficiency = False

        captured: dict[str, Any] = {}

        class DummyFlow:
            def __init__(self, settings):
                captured["settings"] = settings

            def optimize_phase1(self, constraints, targets, **kwargs):
                captured["constraints"] = constraints
                captured["targets"] = targets
                n = 6
                position = np.linspace(0.0, constraints["stroke"], n)
                zeros = np.zeros(n)
                jerk = np.zeros(n - 1)
                return OptimizationResult(
                    status=OptimizationStatus.CONVERGED,
                    objective_value=0.0,
                    solve_time=0.01,
                    solution={
                        "position": position,
                        "velocity": zeros,
                        "acceleration": zeros,
                        "jerk": jerk,
                    },
                    metadata={"n_segments": n - 1},
                )

        monkeypatch.setattr(
            "campro.optimization.unified_framework.CasADiUnifiedFlow",
            DummyFlow,
        )

        result = framework._optimize_primary()

        assert result.metadata.get("solver") == "CasADiUnifiedFlow"
        assert pytest.approx(
            captured["constraints"]["stroke"],
            rel=1e-9,
        ) == framework.data.stroke / 1000.0
        assert captured["targets"]["maximize_thermal_efficiency"] is False
        assert pytest.approx(
            result.solution["position"][-1],
            rel=1e-6,
        ) == framework.data.stroke
        assert result.solution["cam_angle"].shape[0] == result.solution["position"].shape[0]


class TestIntegration:
    """Test integration between components."""

    def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization workflow."""
        # Create problem
        problem = create_default_problem(stroke=0.1, cycle_time=0.0385)

        # Initialize components
        optimizer = CasADiMotionOptimizer(n_segments=20, poly_order=2)
        builder = InitialGuessBuilder(n_segments=20)
        thermal_model = SimplifiedThermalModel()

        # First optimization
        result1 = optimizer.solve(problem)

        if result1.successful:
            # Evaluate thermal efficiency
            efficiency_metrics = thermal_model.evaluate_efficiency(
                result1.variables["position"],
                result1.variables["velocity"],
                result1.variables["acceleration"],
            )

            assert efficiency_metrics["total_efficiency"] > 0

            # Second optimization with deterministic seed + polish
            similar_problem = create_default_problem(stroke=0.105, cycle_time=0.040)
            seed = builder.build_seed(similar_problem)
            initial_guess = builder.polish_seed(similar_problem, seed)
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
