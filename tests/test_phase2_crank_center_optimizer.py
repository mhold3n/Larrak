"""
Unit tests for Phase 2 crank center optimizer.

Tests the CrankCenterOptimizer class that replaces the SunGearOptimizer
with physics-aware optimization for torque maximization and side-loading minimization.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from campro.optimization.base import OptimizationStatus
from campro.optimization.crank_center_optimizer import (
    CrankCenterOptimizationConstraints,
    CrankCenterOptimizationTargets,
    CrankCenterOptimizer,
    CrankCenterParameters,
)
from campro.physics.geometry.litvin import LitvinGearGeometry


class TestCrankCenterParameters:
    """Test cases for CrankCenterParameters."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = CrankCenterParameters()

        assert params.crank_center_x == 0.0
        assert params.crank_center_y == 0.0
        assert params.crank_radius == 50.0
        assert params.rod_length == 150.0
        assert params.bore_diameter == 100.0
        assert params.piston_clearance == 0.1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = CrankCenterParameters(
            crank_center_x=5.0,
            crank_center_y=-2.0,
            crank_radius=60.0,
            rod_length=180.0,
        )

        param_dict = params.to_dict()

        assert param_dict["crank_center_x"] == 5.0
        assert param_dict["crank_center_y"] == -2.0
        assert param_dict["crank_radius"] == 60.0
        assert param_dict["rod_length"] == 180.0
        assert param_dict["bore_diameter"] == 100.0
        assert param_dict["piston_clearance"] == 0.1


class TestCrankCenterOptimizationConstraints:
    """Test cases for CrankCenterOptimizationConstraints."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = CrankCenterOptimizationConstraints()

        # Crank center bounds
        assert constraints.crank_center_x_min == -50.0
        assert constraints.crank_center_x_max == 50.0
        assert constraints.crank_center_y_min == -50.0
        assert constraints.crank_center_y_max == 50.0

        # Crank geometry bounds
        assert constraints.crank_radius_min == 20.0
        assert constraints.crank_radius_max == 100.0
        assert constraints.rod_length_min == 100.0
        assert constraints.rod_length_max == 300.0

        # Performance constraints
        assert constraints.min_torque_output == 100.0
        assert constraints.max_side_load == 1000.0
        assert constraints.max_side_load_penalty == 500.0


class TestCrankCenterOptimizationTargets:
    """Test cases for CrankCenterOptimizationTargets."""

    def test_default_targets(self):
        """Test default target values."""
        targets = CrankCenterOptimizationTargets()

        # Primary objectives
        assert targets.maximize_torque is True
        assert targets.minimize_side_loading is True
        assert targets.minimize_side_loading_during_compression is True
        assert targets.minimize_side_loading_during_combustion is True

        # Weighting factors
        assert targets.torque_weight == 1.0
        assert targets.side_load_weight == 0.8
        assert targets.compression_side_load_weight == 1.2
        assert targets.combustion_side_load_weight == 1.5


class TestCrankCenterOptimizer:
    """Test cases for CrankCenterOptimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = CrankCenterOptimizer()

        # Create test motion law data
        self.theta = np.linspace(0, 2 * np.pi, 100)
        self.primary_data = {
            "theta": self.theta,
            "displacement": 10.0 * np.sin(self.theta),
            "velocity": 10.0 * np.cos(self.theta),
            "acceleration": -10.0 * np.sin(self.theta),
            "load_profile": 1000.0 * np.ones_like(self.theta),
        }

        # Create test secondary data (Litvin gear geometry)
        self.secondary_data = {
            "psi": self.theta,
            "R_psi": 45.0 * np.ones_like(self.theta),
            "optimized_parameters": {
                "base_radius": 25.0,
            },
        }

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = CrankCenterOptimizer()

        assert optimizer.name == "CrankCenterOptimizer"
        assert optimizer._is_configured is True
        assert isinstance(optimizer.constraints, CrankCenterOptimizationConstraints)
        assert isinstance(optimizer.targets, CrankCenterOptimizationTargets)
        assert optimizer._torque_calculator is not None
        assert optimizer._side_load_analyzer is not None
        assert optimizer._kinematics is not None

    def test_configure(self):
        """Test optimizer configuration."""
        constraints = CrankCenterOptimizationConstraints(
            crank_center_x_min=-30.0,
            crank_center_x_max=30.0,
        )
        targets = CrankCenterOptimizationTargets(
            torque_weight=1.5,
            side_load_weight=1.0,
        )

        self.optimizer.configure(constraints=constraints, targets=targets)

        assert self.optimizer.constraints.crank_center_x_min == -30.0
        assert self.optimizer.constraints.crank_center_x_max == 30.0
        assert self.optimizer.targets.torque_weight == 1.5
        assert self.optimizer.targets.side_load_weight == 1.0

    def test_extract_optimization_data(self):
        """Test extraction of optimization data from previous stages."""
        motion_law_data, load_profile, gear_geometry = (
            self.optimizer._extract_optimization_data(
                self.primary_data,
                self.secondary_data,
            )
        )

        # Check motion law data
        assert "theta" in motion_law_data
        assert "displacement" in motion_law_data
        assert "velocity" in motion_law_data
        assert "acceleration" in motion_law_data
        assert len(motion_law_data["theta"]) == 100

        # Check load profile
        assert len(load_profile) == 100
        assert np.allclose(load_profile, 1000.0)

        # Check gear geometry
        assert isinstance(gear_geometry, LitvinGearGeometry)
        assert hasattr(gear_geometry, "pressure_angle_rad")

    def test_extract_optimization_data_missing_load_profile(self):
        """Test extraction with missing load profile."""
        primary_data_no_load = {
            "theta": self.theta,
            "displacement": 10.0 * np.sin(self.theta),
            "velocity": 10.0 * np.cos(self.theta),
            "acceleration": -10.0 * np.sin(self.theta),
        }

        motion_law_data, load_profile, gear_geometry = (
            self.optimizer._extract_optimization_data(
                primary_data_no_load,
                self.secondary_data,
            )
        )

        # Should create default load profile
        assert len(load_profile) == 100
        assert np.allclose(load_profile, 1000.0)

    def test_extract_optimization_data_invalid_input(self):
        """Test extraction with invalid input data."""
        invalid_primary_data = {
            "displacement": 10.0 * np.sin(self.theta),
            "velocity": 10.0 * np.cos(self.theta),
            "acceleration": -10.0 * np.sin(self.theta),
            # Missing 'theta'
        }

        with pytest.raises(
            ValueError, match="Primary data must contain motion law data",
        ):
            self.optimizer._extract_optimization_data(
                invalid_primary_data, self.secondary_data,
            )

    def test_get_default_initial_guess(self):
        """Test default initial guess generation."""
        initial_guess = self.optimizer._get_default_initial_guess(
            self.primary_data,
            self.secondary_data,
        )

        assert "crank_center_x" in initial_guess
        assert "crank_center_y" in initial_guess
        assert "crank_radius" in initial_guess
        assert "rod_length" in initial_guess

        # Check default values
        assert initial_guess["crank_center_x"] == 0.0
        assert initial_guess["crank_center_y"] == 0.0
        assert initial_guess["crank_radius"] == 50.0  # base_radius * 2.0
        assert initial_guess["rod_length"] == 150.0

    def test_configure_physics_models(self):
        """Test physics model configuration."""
        params = {
            "crank_center_x": 5.0,
            "crank_center_y": -2.0,
            "crank_radius": 60.0,
            "rod_length": 180.0,
            "bore_diameter": 120.0,
            "piston_clearance": 0.15,
        }

        gear_geometry = LitvinGearGeometry(
            base_circle_cam=20.0,
            base_circle_ring=40.0,
            pressure_angle_rad=np.array([np.radians(25.0)]),
            contact_ratio=1.5,
            path_of_contact_arc_length=10.0,
            z_cam=30,
            z_ring=60,
            interference_flag=False,
        )

        self.optimizer._configure_physics_models(params, gear_geometry)

        # Check that physics models are configured
        assert self.optimizer._torque_calculator.is_configured()
        assert self.optimizer._side_load_analyzer.is_configured()
        assert self.optimizer._kinematics.is_configured()

    @patch("campro.optimization.crank_center_optimizer.minimize")
    def test_optimize_success(self, mock_minimize):
        """Test successful optimization."""
        # Mock successful optimization result
        mock_result = Mock()
        mock_result.success = True
        mock_result.fun = 0.5
        mock_result.nit = 25
        mock_result.nfev = 50
        mock_result.message = "Optimization terminated successfully"
        mock_result.x = np.array([2.0, -1.0, 55.0, 160.0])
        mock_minimize.return_value = mock_result

        # Mock physics model simulations
        with (
            patch.object(
                self.optimizer._torque_calculator, "simulate",
            ) as mock_torque_sim,
            patch.object(
                self.optimizer._side_load_analyzer, "simulate",
            ) as mock_side_sim,
            patch.object(self.optimizer._kinematics, "simulate") as mock_kin_sim,
        ):
            # Mock successful simulation results
            mock_torque_result = Mock()
            mock_torque_result.is_successful.return_value = True
            mock_torque_result.data = {"instantaneous_torque": np.array([100.0, 200.0])}
            mock_torque_result.metadata = {
                "torque_result": Mock(
                    cycle_average_torque=150.0,
                    max_torque=200.0,
                    torque_ripple=0.1,
                    power_output=15000.0,
                ),
            }
            mock_torque_sim.return_value = mock_torque_result

            mock_side_result = Mock()
            mock_side_result.is_successful.return_value = True
            mock_side_result.data = {"side_load_profile": np.array([50.0, 75.0])}
            mock_side_result.metadata = {
                "side_load_result": Mock(
                    total_penalty=100.0,
                    max_side_load=75.0,
                ),
            }
            mock_side_sim.return_value = mock_side_result

            mock_kin_result = Mock()
            mock_kin_result.is_successful.return_value = True
            mock_kin_result.data = {"rod_angles": np.array([0.1, 0.2])}
            mock_kin_sim.return_value = mock_kin_result

            # Run optimization
            result = self.optimizer.optimize(self.primary_data, self.secondary_data)

            # Check result
            assert result.status == OptimizationStatus.CONVERGED
            assert result.objective_value == 0.5
            assert result.iterations == 25
            assert "crank_center_parameters" in result.solution
            assert "performance_metrics" in result.solution

    @patch("campro.optimization.crank_center_optimizer.minimize")
    def test_optimize_failure(self, mock_minimize):
        """Test optimization failure."""
        # Mock failed optimization result
        mock_result = Mock()
        mock_result.success = False
        mock_result.message = "Optimization failed to converge"
        mock_result.nit = 100
        mock_minimize.return_value = mock_result

        # Run optimization
        result = self.optimizer.optimize(self.primary_data, self.secondary_data)

        # Check result
        assert result.status == OptimizationStatus.FAILED
        assert "error_message" in result.metadata
        assert result.metadata["error_message"] == "Optimization failed to converge"

    def test_optimize_not_configured(self):
        """Test optimization when not configured."""
        optimizer = CrankCenterOptimizer()
        optimizer._is_configured = False

        with pytest.raises(
            RuntimeError, match="Optimizer must be configured before optimization",
        ):
            optimizer.optimize(self.primary_data, self.secondary_data)

    def test_objective_function(self):
        """Test objective function calculation."""
        # Mock physics model simulations
        with (
            patch.object(
                self.optimizer._torque_calculator, "simulate",
            ) as mock_torque_sim,
            patch.object(
                self.optimizer._side_load_analyzer, "simulate",
            ) as mock_side_sim,
        ):
            # Mock successful simulation results
            mock_torque_result = Mock()
            mock_torque_result.is_successful.return_value = True
            mock_torque_result.metadata = {
                "torque_result": Mock(
                    cycle_average_torque=150.0,
                    torque_ripple=0.1,
                    power_output=15000.0,
                ),
            }
            mock_torque_sim.return_value = mock_torque_result

            mock_side_result = Mock()
            mock_side_result.is_successful.return_value = True
            mock_side_result.metadata = {
                "side_load_result": Mock(
                    total_penalty=100.0,
                ),
            }
            mock_side_sim.return_value = mock_side_result

            # Test parameters
            params = np.array([2.0, -1.0, 55.0, 160.0])
            param_names = [
                "crank_center_x",
                "crank_center_y",
                "crank_radius",
                "rod_length",
            ]

            gear_geometry = LitvinGearGeometry(
                base_circle_cam=20.0,
                base_circle_ring=40.0,
                pressure_angle_rad=np.array([np.radians(20.0)]),
                contact_ratio=1.5,
                path_of_contact_arc_length=10.0,
                z_cam=30,
                z_ring=60,
                interference_flag=False,
            )

            # Calculate objective
            objective = self.optimizer._objective_function(
                params,
                param_names,
                self.primary_data,
                self.primary_data["load_profile"],
                gear_geometry,
            )

            # Check that objective is calculated (should be negative for torque maximization)
            assert isinstance(objective, float)
            assert objective < 0  # Negative because we maximize torque

    def test_objective_function_failed_analysis(self):
        """Test objective function with failed physics analysis."""
        # Mock failed physics model simulations
        with patch.object(
            self.optimizer._torque_calculator, "simulate",
        ) as mock_torque_sim:
            mock_torque_result = Mock()
            mock_torque_result.is_successful.return_value = False
            mock_torque_sim.return_value = mock_torque_result

            # Test parameters
            params = np.array([2.0, -1.0, 55.0, 160.0])
            param_names = [
                "crank_center_x",
                "crank_center_y",
                "crank_radius",
                "rod_length",
            ]

            gear_geometry = LitvinGearGeometry(
                base_circle_cam=20.0,
                base_circle_ring=40.0,
                pressure_angle_rad=np.array([np.radians(20.0)]),
                contact_ratio=1.5,
                path_of_contact_arc_length=10.0,
                z_cam=30,
                z_ring=60,
                interference_flag=False,
            )

            # Calculate objective
            objective = self.optimizer._objective_function(
                params,
                param_names,
                self.primary_data,
                self.primary_data["load_profile"],
                gear_geometry,
            )

            # Should return large penalty for failed analysis
            assert objective == 1e6

    def test_define_constraints(self):
        """Test constraint definition."""
        gear_geometry = LitvinGearGeometry(
            base_circle_cam=20.0,
            base_circle_ring=40.0,
            pressure_angle_rad=np.array([np.radians(20.0)]),
            contact_ratio=1.5,
            path_of_contact_arc_length=10.0,
            z_cam=30,
            z_ring=60,
            interference_flag=False,
        )

        constraints = self.optimizer._define_constraints(
            self.primary_data,
            self.primary_data["load_profile"],
            gear_geometry,
        )

        # Should have performance constraints
        assert (
            len(constraints) >= 2
        )  # At least min torque and max side load constraints

        # Check constraint types
        for constraint in constraints:
            assert "type" in constraint
            assert "fun" in constraint
            assert constraint["type"] == "ineq"

    def test_generate_final_design(self):
        """Test final design generation."""
        optimized_params = {
            "crank_center_x": 3.0,
            "crank_center_y": -1.5,
            "crank_radius": 55.0,
            "rod_length": 165.0,
        }

        gear_geometry = LitvinGearGeometry(
            base_circle_cam=20.0,
            base_circle_ring=40.0,
            pressure_angle_rad=np.array([np.radians(20.0)]),
            contact_ratio=1.5,
            path_of_contact_arc_length=10.0,
            z_cam=30,
            z_ring=60,
            interference_flag=False,
        )
        gear_geometry.pressure_angle = np.radians(20.0)

        # Mock physics model simulations
        with (
            patch.object(
                self.optimizer._torque_calculator, "simulate",
            ) as mock_torque_sim,
            patch.object(
                self.optimizer._side_load_analyzer, "simulate",
            ) as mock_side_sim,
            patch.object(self.optimizer._kinematics, "simulate") as mock_kin_sim,
        ):
            # Mock successful simulation results
            mock_torque_result = Mock()
            mock_torque_result.is_successful.return_value = True
            mock_torque_result.data = {"instantaneous_torque": np.array([100.0, 200.0])}
            mock_torque_result.metadata = {
                "torque_result": Mock(
                    cycle_average_torque=150.0,
                    max_torque=200.0,
                    torque_ripple=0.1,
                    power_output=15000.0,
                ),
            }
            mock_torque_sim.return_value = mock_torque_result

            mock_side_result = Mock()
            mock_side_result.is_successful.return_value = True
            mock_side_result.data = {"side_load_profile": np.array([50.0, 75.0])}
            mock_side_result.metadata = {
                "side_load_result": Mock(
                    total_penalty=100.0,
                    max_side_load=75.0,
                ),
            }
            mock_side_sim.return_value = mock_side_result

            mock_kin_result = Mock()
            mock_kin_result.is_successful.return_value = True
            mock_kin_result.data = {"rod_angles": np.array([0.1, 0.2])}
            mock_kin_sim.return_value = mock_kin_result

            # Generate final design
            final_design = self.optimizer._generate_final_design(
                optimized_params,
                self.primary_data,
                self.primary_data["load_profile"],
                gear_geometry,
            )

            # Check final design structure
            assert "crank_center_parameters" in final_design
            assert "optimized_parameters" in final_design
            assert "torque_analysis" in final_design
            assert "side_load_analysis" in final_design
            assert "kinematics_analysis" in final_design
            assert "performance_metrics" in final_design

            # Check optimized parameters
            assert final_design["optimized_parameters"]["crank_center_x"] == 3.0
            assert final_design["optimized_parameters"]["crank_center_y"] == -1.5
            assert final_design["optimized_parameters"]["crank_radius"] == 55.0
            assert final_design["optimized_parameters"]["rod_length"] == 165.0


class TestIntegration:
    """Integration tests for CrankCenterOptimizer."""

    def test_physics_model_integration(self):
        """Test integration with Phase 1 physics models."""
        optimizer = CrankCenterOptimizer()

        # Create test data
        theta = np.linspace(0, 2 * np.pi, 50)
        primary_data = {
            "theta": theta,
            "displacement": 10.0 * np.sin(theta),
            "velocity": 10.0 * np.cos(theta),
            "acceleration": -10.0 * np.sin(theta),
            "load_profile": 1000.0 * np.ones_like(theta),
        }

        secondary_data = {
            "psi": theta,
            "R_psi": 45.0 * np.ones_like(theta),
            "optimized_parameters": {"base_radius": 25.0},
        }

        # Test data extraction
        motion_law_data, load_profile, gear_geometry = (
            optimizer._extract_optimization_data(
                primary_data,
                secondary_data,
            )
        )

        # Test physics model configuration
        params = {
            "crank_center_x": 0.0,
            "crank_center_y": 0.0,
            "crank_radius": 50.0,
            "rod_length": 150.0,
        }

        optimizer._configure_physics_models(params, gear_geometry)

        # Verify physics models are configured
        assert optimizer._torque_calculator.is_configured()
        assert optimizer._side_load_analyzer.is_configured()
        assert optimizer._kinematics.is_configured()

        # Test that physics models can run simulations
        crank_center_offset = (0.0, 0.0)

        torque_inputs = {
            "motion_law_data": motion_law_data,
            "load_profile": load_profile,
            "crank_center_offset": crank_center_offset,
        }

        torque_result = optimizer._torque_calculator.simulate(torque_inputs)
        assert torque_result.is_successful

        side_load_inputs = {
            "motion_law_data": motion_law_data,
            "load_profile": load_profile,
            "crank_center_offset": crank_center_offset,
        }

        side_load_result = optimizer._side_load_analyzer.simulate(side_load_inputs)
        assert side_load_result.is_successful
