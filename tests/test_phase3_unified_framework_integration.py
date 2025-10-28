"""
Integration tests for Phase 3 unified framework integration.

Tests the integration of CrankCenterOptimizer into the unified optimization framework,
ensuring proper data flow between all three optimization stages and correct handling
of crank center optimization results.
"""

from unittest.mock import Mock, patch

import numpy as np

from campro.optimization.base import OptimizationStatus
from campro.optimization.crank_center_optimizer import (
    CrankCenterOptimizationConstraints,
    CrankCenterOptimizationTargets,
    CrankCenterOptimizer,
)
from campro.optimization.unified_framework import (
    OptimizationMethod,
    UnifiedOptimizationFramework,
)


class TestUnifiedFrameworkIntegration:
    """Test cases for unified framework integration with CrankCenterOptimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.framework = UnifiedOptimizationFramework()

        # Create test input data
        self.input_data = {
            "stroke": 20.0,
            "cycle_time": 1.0,
            "upstroke_duration_percent": 60.0,
            "zero_accel_duration_percent": 0.0,
            "motion_type": "minimum_jerk",
        }

    def test_framework_initialization_with_crank_center_optimizer(self):
        """Test that framework initializes with CrankCenterOptimizer."""
        assert isinstance(self.framework.tertiary_optimizer, CrankCenterOptimizer)
        assert self.framework.tertiary_optimizer.name == "TertiaryCrankCenterOptimizer"

    def test_crank_center_constraints_in_framework(self):
        """Test that crank center constraints are properly defined."""
        constraints = self.framework.constraints

        # Check crank center position bounds
        assert hasattr(constraints, "crank_center_x_min")
        assert hasattr(constraints, "crank_center_x_max")
        assert hasattr(constraints, "crank_center_y_min")
        assert hasattr(constraints, "crank_center_y_max")

        # Check crank geometry bounds
        assert hasattr(constraints, "crank_radius_min")
        assert hasattr(constraints, "crank_radius_max")
        assert hasattr(constraints, "rod_length_min")
        assert hasattr(constraints, "rod_length_max")

        # Check performance constraints
        assert hasattr(constraints, "min_torque_output")
        assert hasattr(constraints, "max_side_load_penalty")

        # Check default values
        assert constraints.crank_center_x_min == -50.0
        assert constraints.crank_center_x_max == 50.0
        assert constraints.crank_center_y_min == -50.0
        assert constraints.crank_center_y_max == 50.0
        assert constraints.crank_radius_min == 20.0
        assert constraints.crank_radius_max == 100.0
        assert constraints.rod_length_min == 100.0
        assert constraints.rod_length_max == 300.0
        assert constraints.min_torque_output == 100.0
        assert constraints.max_side_load_penalty == 500.0

    def test_crank_center_targets_in_framework(self):
        """Test that crank center optimization targets are properly defined."""
        targets = self.framework.targets

        # Check primary objectives
        assert hasattr(targets, "maximize_torque")
        assert hasattr(targets, "minimize_side_loading")
        assert hasattr(targets, "minimize_side_loading_during_compression")
        assert hasattr(targets, "minimize_side_loading_during_combustion")
        assert hasattr(targets, "minimize_torque_ripple")
        assert hasattr(targets, "maximize_power_output")

        # Check weighting factors
        assert hasattr(targets, "torque_weight")
        assert hasattr(targets, "side_load_weight")
        assert hasattr(targets, "compression_side_load_weight")
        assert hasattr(targets, "combustion_side_load_weight")
        assert hasattr(targets, "torque_ripple_weight")
        assert hasattr(targets, "power_output_weight")

        # Check default values
        assert targets.maximize_torque is True
        assert targets.minimize_side_loading is True
        assert targets.minimize_side_loading_during_compression is True
        assert targets.minimize_side_loading_during_combustion is True
        assert targets.minimize_torque_ripple is True
        assert targets.maximize_power_output is True

        assert targets.torque_weight == 1.0
        assert targets.side_load_weight == 0.8
        assert targets.compression_side_load_weight == 1.2
        assert targets.combustion_side_load_weight == 1.5
        assert targets.torque_ripple_weight == 0.3
        assert targets.power_output_weight == 0.5

    def test_crank_center_data_structure(self):
        """Test that UnifiedOptimizationData includes crank center optimization results."""
        data = self.framework.data

        # Check crank center parameters
        assert hasattr(data, "tertiary_crank_center_x")
        assert hasattr(data, "tertiary_crank_center_y")
        assert hasattr(data, "tertiary_crank_radius")
        assert hasattr(data, "tertiary_rod_length")

        # Check performance metrics
        assert hasattr(data, "tertiary_torque_output")
        assert hasattr(data, "tertiary_side_load_penalty")
        assert hasattr(data, "tertiary_max_torque")
        assert hasattr(data, "tertiary_torque_ripple")
        assert hasattr(data, "tertiary_power_output")
        assert hasattr(data, "tertiary_max_side_load")

        # Check initial values are None
        assert data.tertiary_crank_center_x is None
        assert data.tertiary_crank_center_y is None
        assert data.tertiary_crank_radius is None
        assert data.tertiary_rod_length is None
        assert data.tertiary_torque_output is None
        assert data.tertiary_side_load_penalty is None

    def test_tertiary_optimizer_configuration(self):
        """Test that tertiary optimizer is properly configured with crank center constraints and targets."""
        # Mock the configure method to capture the arguments
        with patch.object(
            self.framework.tertiary_optimizer, "configure",
        ) as mock_configure:
            self.framework._configure_optimizers()

            # Check that configure was called
            assert mock_configure.called

            # Get the call arguments
            call_args = mock_configure.call_args
            constraints = call_args[1]["constraints"]
            targets = call_args[1]["targets"]

            # Check constraints type and values
            assert isinstance(constraints, CrankCenterOptimizationConstraints)
            assert constraints.crank_center_x_min == -50.0
            assert constraints.crank_center_x_max == 50.0
            assert constraints.crank_center_y_min == -50.0
            assert constraints.crank_center_y_max == 50.0
            assert constraints.crank_radius_min == 20.0
            assert constraints.crank_radius_max == 100.0
            assert constraints.rod_length_min == 100.0
            assert constraints.rod_length_max == 300.0
            assert constraints.min_torque_output == 100.0
            assert constraints.max_side_load_penalty == 500.0

            # Check targets type and values
            assert isinstance(targets, CrankCenterOptimizationTargets)
            assert targets.maximize_torque is True
            assert targets.minimize_side_loading is True
            assert targets.minimize_side_loading_during_compression is True
            assert targets.minimize_side_loading_during_combustion is True
            assert targets.minimize_torque_ripple is True
            assert targets.maximize_power_output is True
            assert targets.torque_weight == 1.0
            assert targets.side_load_weight == 0.8
            assert targets.compression_side_load_weight == 1.2
            assert targets.combustion_side_load_weight == 1.5
            assert targets.torque_ripple_weight == 0.3
            assert targets.power_output_weight == 0.5

    def test_tertiary_optimization_data_preparation(self):
        """Test that tertiary optimization data is properly prepared."""
        # Set up mock data
        self.framework.data.primary_theta = np.linspace(0, 2 * np.pi, 100)
        self.framework.data.primary_position = 10.0 * np.sin(
            self.framework.data.primary_theta,
        )
        self.framework.data.primary_velocity = 10.0 * np.cos(
            self.framework.data.primary_theta,
        )
        self.framework.data.primary_acceleration = -10.0 * np.sin(
            self.framework.data.primary_theta,
        )
        self.framework.data.primary_load_profile = 1000.0 * np.ones_like(
            self.framework.data.primary_theta,
        )

        self.framework.data.secondary_base_radius = 25.0
        self.framework.data.secondary_cam_curves = {"x": np.array([1, 2, 3])}
        self.framework.data.secondary_psi = np.linspace(0, 2 * np.pi, 100)
        self.framework.data.secondary_R_psi = 45.0 * np.ones_like(
            self.framework.data.secondary_psi,
        )

        # Mock the tertiary optimizer
        with patch.object(
            self.framework.tertiary_optimizer, "optimize",
        ) as mock_optimize:
            mock_result = Mock()
            mock_result.status = OptimizationStatus.CONVERGED
            mock_result.solution = {
                "optimized_parameters": {
                    "crank_center_x": 2.0,
                    "crank_center_y": -1.0,
                    "crank_radius": 55.0,
                    "rod_length": 165.0,
                },
                "performance_metrics": {
                    "cycle_average_torque": 150.0,
                    "total_side_load_penalty": 100.0,
                    "max_torque": 200.0,
                    "torque_ripple": 0.1,
                    "power_output": 15000.0,
                    "max_side_load": 75.0,
                },
            }
            mock_result.objective_value = 0.5
            mock_result.iterations = 25
            mock_result.solve_time = 1.2
            mock_optimize.return_value = mock_result

            # Run tertiary optimization
            result = self.framework._optimize_tertiary()

            # Check that optimize was called
            assert mock_optimize.called

            # Get the call arguments
            call_args = mock_optimize.call_args
            primary_data = call_args[1]["primary_data"]
            secondary_data = call_args[1]["secondary_data"]
            initial_guess = call_args[1]["initial_guess"]

            # Check primary data
            assert "theta" in primary_data
            assert "displacement" in primary_data
            assert "velocity" in primary_data
            assert "acceleration" in primary_data
            assert "load_profile" in primary_data
            assert len(primary_data["theta"]) == 100

            # Check secondary data
            assert "optimized_parameters" in secondary_data
            assert "cam_curves" in secondary_data
            assert "psi" in secondary_data
            assert "R_psi" in secondary_data
            assert secondary_data["optimized_parameters"]["base_radius"] == 25.0

            # Check initial guess
            assert initial_guess["crank_center_x"] == 0.0
            assert initial_guess["crank_center_y"] == 0.0
            assert initial_guess["crank_radius"] == 50.0  # base_radius * 2.0
            assert initial_guess["rod_length"] == 150.0  # base_radius * 6.0

    def test_tertiary_data_update(self):
        """Test that tertiary optimization results are properly stored in data structure."""
        # Create mock optimization result
        mock_result = Mock()
        mock_result.status = OptimizationStatus.CONVERGED
        mock_result.solution = {
            "optimized_parameters": {
                "crank_center_x": 3.0,
                "crank_center_y": -1.5,
                "crank_radius": 55.0,
                "rod_length": 165.0,
            },
            "performance_metrics": {
                "cycle_average_torque": 150.0,
                "total_side_load_penalty": 100.0,
                "max_torque": 200.0,
                "torque_ripple": 0.1,
                "power_output": 15000.0,
                "max_side_load": 75.0,
            },
        }
        mock_result.objective_value = 0.5
        mock_result.iterations = 25
        mock_result.solve_time = 1.2

        # Update data from tertiary optimization
        self.framework._update_data_from_tertiary(mock_result)

        # Check that data was updated
        assert self.framework.data.tertiary_crank_center_x == 3.0
        assert self.framework.data.tertiary_crank_center_y == -1.5
        assert self.framework.data.tertiary_crank_radius == 55.0
        assert self.framework.data.tertiary_rod_length == 165.0
        assert self.framework.data.tertiary_torque_output == 150.0
        assert self.framework.data.tertiary_side_load_penalty == 100.0
        assert self.framework.data.tertiary_max_torque == 200.0
        assert self.framework.data.tertiary_torque_ripple == 0.1
        assert self.framework.data.tertiary_power_output == 15000.0
        assert self.framework.data.tertiary_max_side_load == 75.0

        # Check convergence info
        assert "tertiary" in self.framework.data.convergence_info
        tertiary_info = self.framework.data.convergence_info["tertiary"]
        assert tertiary_info["status"] == "converged"
        assert tertiary_info["objective_value"] == 0.5
        assert tertiary_info["iterations"] == 25
        assert tertiary_info["solve_time"] == 1.2

    def test_optimization_summary_includes_crank_center_results(self):
        """Test that optimization summary includes crank center optimization results."""
        # Set up mock data
        self.framework.data.tertiary_crank_center_x = 2.0
        self.framework.data.tertiary_crank_center_y = -1.0
        self.framework.data.tertiary_crank_radius = 55.0
        self.framework.data.tertiary_rod_length = 165.0
        self.framework.data.tertiary_torque_output = 150.0
        self.framework.data.tertiary_side_load_penalty = 100.0
        self.framework.data.tertiary_max_torque = 200.0
        self.framework.data.tertiary_torque_ripple = 0.1
        self.framework.data.tertiary_power_output = 15000.0
        self.framework.data.tertiary_max_side_load = 75.0

        # Get optimization summary
        summary = self.framework.get_optimization_summary()

        # Check that tertiary results are included
        assert "tertiary_results" in summary
        tertiary_results = summary["tertiary_results"]

        assert tertiary_results["crank_center_x"] == 2.0
        assert tertiary_results["crank_center_y"] == -1.0
        assert tertiary_results["crank_radius"] == 55.0
        assert tertiary_results["rod_length"] == 165.0
        assert tertiary_results["torque_output"] == 150.0
        assert tertiary_results["side_load_penalty"] == 100.0
        assert tertiary_results["max_torque"] == 200.0
        assert tertiary_results["torque_ripple"] == 0.1
        assert tertiary_results["power_output"] == 15000.0
        assert tertiary_results["max_side_load"] == 75.0

    @patch("campro.optimization.unified_framework.MotionOptimizer")
    @patch("campro.optimization.unified_framework.CamRingOptimizer")
    @patch("campro.optimization.unified_framework.CrankCenterOptimizer")
    def test_full_cascaded_optimization_integration(
        self, mock_crank_optimizer, mock_cam_optimizer, mock_motion_optimizer,
    ):
        """Test full cascaded optimization with CrankCenterOptimizer integration."""
        # Set up mock optimizers
        mock_motion_result = Mock()
        mock_motion_result.status = OptimizationStatus.CONVERGED
        mock_motion_result.solution = {
            "cam_angle": np.linspace(0, 2 * np.pi, 100),
            "position": 10.0 * np.sin(np.linspace(0, 2 * np.pi, 100)),
            "velocity": 10.0 * np.cos(np.linspace(0, 2 * np.pi, 100)),
            "acceleration": -10.0 * np.sin(np.linspace(0, 2 * np.pi, 100)),
        }
        mock_motion_result.objective_value = 0.1
        mock_motion_result.iterations = 15
        mock_motion_result.solve_time = 0.5

        mock_cam_result = Mock()
        mock_cam_result.status = OptimizationStatus.CONVERGED
        mock_cam_result.solution = {
            "optimized_parameters": {"base_radius": 25.0},
            "cam_curves": {"x": np.array([1, 2, 3])},
            "psi": np.linspace(0, 2 * np.pi, 100),
            "R_psi": 45.0 * np.ones(100),
        }
        mock_cam_result.objective_value = 0.2
        mock_cam_result.iterations = 20
        mock_cam_result.solve_time = 0.8

        mock_crank_result = Mock()
        mock_crank_result.status = OptimizationStatus.CONVERGED
        mock_crank_result.solution = {
            "optimized_parameters": {
                "crank_center_x": 2.0,
                "crank_center_y": -1.0,
                "crank_radius": 55.0,
                "rod_length": 165.0,
            },
            "performance_metrics": {
                "cycle_average_torque": 150.0,
                "total_side_load_penalty": 100.0,
                "max_torque": 200.0,
                "torque_ripple": 0.1,
                "power_output": 15000.0,
                "max_side_load": 75.0,
            },
        }
        mock_crank_result.objective_value = 0.3
        mock_crank_result.iterations = 25
        mock_crank_result.solve_time = 1.2

        # Configure mock optimizers
        mock_motion_optimizer.return_value.optimize.return_value = mock_motion_result
        mock_cam_optimizer.return_value.optimize.return_value = mock_cam_result
        mock_crank_optimizer.return_value.optimize.return_value = mock_crank_result

        # Create framework with mocked optimizers
        framework = UnifiedOptimizationFramework()

        # Replace the optimizers with mocked versions
        framework.primary_optimizer = mock_motion_optimizer.return_value
        framework.secondary_optimizer = mock_cam_optimizer.return_value
        framework.tertiary_optimizer = mock_crank_optimizer.return_value

        # Mock the data update methods to simulate proper data flow
        mock_update_primary = Mock(
            side_effect=lambda result: setattr(
                framework.data, "primary_theta", np.linspace(0, 2 * np.pi, 100),
            ),
        )
        mock_update_secondary = Mock(
            side_effect=lambda result: setattr(
                framework.data, "secondary_base_radius", 25.0,
            ),
        )
        mock_update_tertiary = Mock(
            side_effect=lambda result: setattr(
                framework.data, "tertiary_crank_center_x", 2.0,
            ),
        )

        with (
            patch.object(framework, "_update_data_from_primary", mock_update_primary),
            patch.object(
                framework, "_update_data_from_secondary", mock_update_secondary,
            ),
            patch.object(framework, "_update_data_from_tertiary", mock_update_tertiary),
        ):
            # Run cascaded optimization
            result = framework.optimize_cascaded(self.input_data)

        # Check that all data update methods were called
        assert mock_update_primary.called
        assert mock_update_secondary.called
        assert mock_update_tertiary.called

        # Check that result contains crank center optimization data
        assert result.tertiary_crank_center_x == 2.0
        # Note: Other fields are None because the mock only sets one field
        # In a real scenario, the tertiary optimization would set all fields

        # Check that the framework completed successfully
        assert result.optimization_method == OptimizationMethod.LEGENDRE_COLLOCATION
        assert result.total_solve_time > 0


class TestBackwardCompatibility:
    """Test backward compatibility with existing functionality."""

    def test_framework_still_has_primary_and_secondary_optimizers(self):
        """Test that framework still has primary and secondary optimizers."""
        framework = UnifiedOptimizationFramework()

        assert hasattr(framework, "primary_optimizer")
        assert hasattr(framework, "secondary_optimizer")
        assert hasattr(framework, "tertiary_optimizer")

        # Check that tertiary optimizer is now CrankCenterOptimizer
        assert isinstance(framework.tertiary_optimizer, CrankCenterOptimizer)

    def test_existing_data_fields_are_preserved(self):
        """Test that existing data fields are preserved."""
        framework = UnifiedOptimizationFramework()
        data = framework.data

        # Check that existing fields are still present
        assert hasattr(data, "stroke")
        assert hasattr(data, "cycle_time")
        assert hasattr(data, "primary_theta")
        assert hasattr(data, "primary_position")
        assert hasattr(data, "secondary_base_radius")
        assert hasattr(data, "secondary_cam_curves")
        assert hasattr(data, "secondary_psi")
        assert hasattr(data, "secondary_R_psi")

        # Check that new crank center fields are added
        assert hasattr(data, "tertiary_crank_center_x")
        assert hasattr(data, "tertiary_crank_center_y")
        assert hasattr(data, "tertiary_crank_radius")
        assert hasattr(data, "tertiary_rod_length")
        assert hasattr(data, "tertiary_torque_output")
        assert hasattr(data, "tertiary_side_load_penalty")

    def test_existing_constraints_are_preserved(self):
        """Test that existing constraints are preserved."""
        framework = UnifiedOptimizationFramework()
        constraints = framework.constraints

        # Check that existing constraints are still present
        assert hasattr(constraints, "stroke_min")
        assert hasattr(constraints, "stroke_max")
        assert hasattr(constraints, "base_radius_min")
        assert hasattr(constraints, "base_radius_max")
        assert hasattr(constraints, "min_curvature_radius")
        assert hasattr(constraints, "max_curvature")

        # Check that new crank center constraints are added
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

    def test_existing_targets_are_preserved(self):
        """Test that existing targets are preserved."""
        framework = UnifiedOptimizationFramework()
        targets = framework.targets

        # Check that existing targets are still present
        assert hasattr(targets, "minimize_jerk")
        assert hasattr(targets, "minimize_time")
        assert hasattr(targets, "minimize_energy")
        assert hasattr(targets, "minimize_ring_size")
        assert hasattr(targets, "minimize_cam_size")
        assert hasattr(targets, "minimize_curvature_variation")

        # Check that new crank center targets are added
        assert hasattr(targets, "maximize_torque")
        assert hasattr(targets, "minimize_side_loading")
        assert hasattr(targets, "minimize_side_loading_during_compression")
        assert hasattr(targets, "minimize_side_loading_during_combustion")
        assert hasattr(targets, "minimize_torque_ripple")
        assert hasattr(targets, "maximize_power_output")
