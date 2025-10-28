"""
Integration tests for three-run optimization system.

Tests the complete three-run optimization pipeline with Phase 3 integration:
- Run 1: Motion law optimization (primary)
- Run 2: Litvin profile synthesis (secondary)
- Run 3: Crank center optimization (tertiary)
"""

from unittest.mock import Mock

import numpy as np
import pytest

from campro.optimization.base import OptimizationStatus
from campro.optimization.unified_framework import (
    UnifiedOptimizationFramework,
)


class TestThreeRunOptimizationIntegration:
    """Integration tests for complete three-run optimization system."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create unified framework
        self.framework = UnifiedOptimizationFramework()

        # Create test input data
        self.input_data = {
            "stroke": 20.0,
            "cycle_time": 1.0,
            "upstroke_duration_percent": 60.0,
            "zero_accel_duration_percent": 0.0,
            "motion_type": "minimum_jerk",
        }

        # Configure framework
        self.framework.configure()

    def test_framework_initialization(self):
        """Test that framework initializes with correct optimizers."""
        # Verify optimizer types
        assert hasattr(self.framework, "primary_optimizer")
        assert hasattr(self.framework, "secondary_optimizer")
        assert hasattr(self.framework, "tertiary_optimizer")

        # Verify tertiary optimizer is CrankCenterOptimizer
        from campro.optimization.crank_center_optimizer import CrankCenterOptimizer

        assert isinstance(self.framework.tertiary_optimizer, CrankCenterOptimizer)

        # Verify framework is configured
        assert self.framework._is_configured

    def test_constraints_and_targets(self):
        """Test that constraints and targets are properly configured."""
        constraints = self.framework.constraints
        targets = self.framework.targets

        # Test crank center constraints
        assert hasattr(constraints, "crank_center_x_min")
        assert hasattr(constraints, "crank_center_x_max")
        assert hasattr(constraints, "crank_center_y_min")
        assert hasattr(constraints, "crank_center_y_max")
        assert hasattr(constraints, "crank_radius_min")
        assert hasattr(constraints, "crank_radius_max")
        assert hasattr(constraints, "rod_length_min")
        assert hasattr(constraints, "rod_length_max")
        assert hasattr(constraints, "min_torque_output")
        assert hasattr(constraints, "max_side_load_penalty")

        # Test crank center targets
        assert hasattr(targets, "maximize_torque")
        assert hasattr(targets, "minimize_side_loading")
        assert hasattr(targets, "minimize_side_loading_during_compression")
        assert hasattr(targets, "minimize_side_loading_during_combustion")
        assert hasattr(targets, "minimize_torque_ripple")
        assert hasattr(targets, "maximize_power_output")

        # Test weighting factors
        assert hasattr(targets, "torque_weight")
        assert hasattr(targets, "side_load_weight")
        assert hasattr(targets, "compression_side_load_weight")
        assert hasattr(targets, "combustion_side_load_weight")

    def test_data_structure(self):
        """Test that data structure includes crank center optimization fields."""
        data = self.framework.data

        # Test tertiary results fields
        assert hasattr(data, "tertiary_crank_center_x")
        assert hasattr(data, "tertiary_crank_center_y")
        assert hasattr(data, "tertiary_crank_radius")
        assert hasattr(data, "tertiary_rod_length")
        assert hasattr(data, "tertiary_torque_output")
        assert hasattr(data, "tertiary_side_load_penalty")
        assert hasattr(data, "tertiary_max_torque")
        assert hasattr(data, "tertiary_torque_ripple")
        assert hasattr(data, "tertiary_power_output")
        assert hasattr(data, "tertiary_max_side_load")

        # Initial values should be None
        assert data.tertiary_crank_center_x is None
        assert data.tertiary_crank_center_y is None
        assert data.tertiary_torque_output is None

    def test_optimization_summary(self):
        """Test that optimization summary includes crank center results."""
        summary = self.framework.get_optimization_summary()

        # Test that summary includes tertiary results
        assert "tertiary_results" in summary

        tertiary_results = summary["tertiary_results"]
        assert "crank_center_x" in tertiary_results
        assert "crank_center_y" in tertiary_results
        assert "crank_radius" in tertiary_results
        assert "rod_length" in tertiary_results
        assert "torque_output" in tertiary_results
        assert "side_load_penalty" in tertiary_results
        assert "max_torque" in tertiary_results
        assert "torque_ripple" in tertiary_results
        assert "power_output" in tertiary_results
        assert "max_side_load" in tertiary_results

    def test_tertiary_optimizer_configuration(self):
        """Test that tertiary optimizer is properly configured."""
        tertiary_optimizer = self.framework.tertiary_optimizer

        # Test that optimizer is configured
        assert tertiary_optimizer._is_configured

        # Test that constraints are set
        assert hasattr(tertiary_optimizer, "constraints")
        assert hasattr(tertiary_optimizer, "targets")

        # Test constraint values
        constraints = tertiary_optimizer.constraints
        assert constraints.crank_center_x_min == -50.0
        assert constraints.crank_center_x_max == 50.0
        assert constraints.crank_center_y_min == -50.0
        assert constraints.crank_center_y_max == 50.0
        assert constraints.min_torque_output == 100.0
        assert constraints.max_side_load_penalty == 500.0

        # Test target values
        targets = tertiary_optimizer.targets
        assert targets.maximize_torque is True
        assert targets.minimize_side_loading is True
        assert targets.minimize_side_loading_during_compression is True
        assert targets.minimize_side_loading_during_combustion is True

    def test_physics_models_integration(self):
        """Test that physics models are properly integrated."""
        tertiary_optimizer = self.framework.tertiary_optimizer

        # Test that physics models are initialized
        assert hasattr(tertiary_optimizer, "_torque_calculator")
        assert hasattr(tertiary_optimizer, "_side_load_analyzer")
        assert hasattr(tertiary_optimizer, "_kinematics")

        # Test model types
        from campro.physics.kinematics.crank_kinematics import CrankKinematics
        from campro.physics.mechanics.side_loading import SideLoadAnalyzer
        from campro.physics.mechanics.torque_analysis import PistonTorqueCalculator

        assert isinstance(tertiary_optimizer._torque_calculator, PistonTorqueCalculator)
        assert isinstance(tertiary_optimizer._side_load_analyzer, SideLoadAnalyzer)
        assert isinstance(tertiary_optimizer._kinematics, CrankKinematics)

    def test_mock_three_run_optimization(self):
        """Test three-run optimization with mocked results."""
        # Create mock results
        mock_primary = Mock()
        mock_primary.status = OptimizationStatus.CONVERGED
        mock_primary.solution = {
            "cam_angle": np.linspace(0, 2 * np.pi, 100),
            "position": np.sin(np.linspace(0, 2 * np.pi, 100)),
            "velocity": np.cos(np.linspace(0, 2 * np.pi, 100)),
            "acceleration": -np.sin(np.linspace(0, 2 * np.pi, 100)),
        }
        mock_primary.objective_value = 0.1
        mock_primary.iterations = 10
        mock_primary.solve_time = 0.5

        mock_secondary = Mock()
        mock_secondary.status = OptimizationStatus.CONVERGED
        mock_secondary.solution = {
            "optimized_parameters": {"base_radius": 25.0},
            "psi": np.linspace(0, 2 * np.pi, 100),
            "R_psi": 25.0 * np.ones(100),
        }
        mock_secondary.objective_value = 0.2
        mock_secondary.iterations = 15
        mock_secondary.solve_time = 0.8

        mock_tertiary = Mock()
        mock_tertiary.status = OptimizationStatus.CONVERGED
        mock_tertiary.solution = {
            "optimized_parameters": {
                "crank_center_x": 5.0,
                "crank_center_y": -2.0,
                "crank_radius": 50.0,
                "rod_length": 150.0,
            },
            "performance_metrics": {
                "cycle_average_torque": 250.0,
                "total_side_load_penalty": 150.0,
                "max_torque": 300.0,
                "torque_ripple": 0.1,
                "power_output": 25000.0,
                "max_side_load": 200.0,
            },
        }
        mock_tertiary.objective_value = 0.3
        mock_tertiary.iterations = 20
        mock_tertiary.solve_time = 1.2

        # Mock the optimization methods
        self.framework._optimize_primary = lambda: mock_primary
        self.framework._optimize_secondary = lambda: mock_secondary
        self.framework._optimize_tertiary = lambda: mock_tertiary

        # Run cascaded optimization
        result = self.framework.optimize_cascaded(self.input_data)

        # Verify results
        assert result.tertiary_crank_center_x == 5.0
        assert result.tertiary_crank_center_y == -2.0
        assert result.tertiary_crank_radius == 50.0
        assert result.tertiary_rod_length == 150.0
        assert result.tertiary_torque_output == 250.0
        assert result.tertiary_side_load_penalty == 150.0
        assert result.tertiary_max_torque == 300.0
        assert result.tertiary_torque_ripple == 0.1
        assert result.tertiary_power_output == 25000.0
        assert result.tertiary_max_side_load == 200.0

        # Verify convergence info
        assert "tertiary" in result.convergence_info
        tertiary_info = result.convergence_info["tertiary"]
        assert tertiary_info["status"] == "converged"
        assert tertiary_info["objective_value"] == 0.3
        assert tertiary_info["iterations"] == 20
        assert tertiary_info["solve_time"] == 1.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
