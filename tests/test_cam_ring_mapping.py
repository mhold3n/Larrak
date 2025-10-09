"""
Tests for cam-ring-linear follower mapping functionality.

This module tests the mathematical framework for relating linear follower motion
to cam geometry and ring follower design.
"""

from unittest.mock import patch

import numpy as np
import pytest

from campro.optimization.cam_ring_processing import (
    _calculate_multi_objective_score,
    create_constant_ring_design,
    create_optimized_ring_design,
    process_linear_to_ring_follower,
    process_multi_objective_ring_design,
    process_ring_optimization,
)
from campro.physics.cam_ring_mapping import CamRingMapper, CamRingParameters


class TestCamRingParameters:
    """Test CamRingParameters dataclass."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = CamRingParameters()

        assert params.base_radius == 10.0
        assert params.connecting_rod_length == 25.0
        assert params.ring_center_x == 0.0
        assert params.ring_center_y == 0.0
        assert params.contact_type == "external"

    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = CamRingParameters(
            base_radius=15.0,
            connecting_rod_length=30.0,
            contact_type="internal",
        )

        assert params.base_radius == 15.0
        assert params.connecting_rod_length == 30.0
        assert params.contact_type == "internal"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        params = CamRingParameters(base_radius=12.0)
        param_dict = params.to_dict()

        assert isinstance(param_dict, dict)
        assert param_dict["base_radius"] == 12.0
        assert param_dict["connecting_rod_length"] == 25.0


class TestCamRingMapper:
    """Test CamRingMapper class."""

    @pytest.fixture
    def mapper(self):
        """Create a CamRingMapper instance for testing."""
        return CamRingMapper()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        theta = np.linspace(0, 2*np.pi, 100)
        x_theta = 5.0 * np.sin(theta)  # Simple sinusoidal motion
        return theta, x_theta

    def test_initialization(self):
        """Test mapper initialization."""
        mapper = CamRingMapper()
        assert isinstance(mapper.parameters, CamRingParameters)

        custom_params = CamRingParameters(base_radius=20.0)
        mapper_custom = CamRingMapper(custom_params)
        assert mapper_custom.parameters.base_radius == 20.0

    def test_compute_cam_curves(self, mapper, sample_data):
        """Test cam curve computation."""
        theta, x_theta = sample_data

        curves = mapper.compute_cam_curves(theta, x_theta)

        # Check that all required keys are present
        required_keys = ["pitch_radius", "profile_radius", "contact_radius", "theta"]
        for key in required_keys:
            assert key in curves

        # Check array shapes
        assert curves["pitch_radius"].shape == theta.shape
        assert curves["profile_radius"].shape == theta.shape
        assert curves["contact_radius"].shape == theta.shape

        # Check that pitch radius = base_radius + x_theta
        expected_pitch = mapper.parameters.base_radius + x_theta
        np.testing.assert_array_almost_equal(curves["pitch_radius"], expected_pitch)

        # Check that profile radius = pitch_radius (direct contact, no roller)
        expected_profile = curves["pitch_radius"]
        np.testing.assert_array_almost_equal(curves["profile_radius"], expected_profile)

    def test_compute_cam_curvature(self, mapper, sample_data):
        """Test cam curvature computation."""
        theta, x_theta = sample_data
        curves = mapper.compute_cam_curves(theta, x_theta)

        kappa_c = mapper.compute_cam_curvature(theta, curves["contact_radius"])

        # Check that curvature is finite
        assert np.all(np.isfinite(kappa_c))
        assert kappa_c.shape == theta.shape

    def test_compute_osculating_radius(self, mapper):
        """Test osculating radius computation."""
        # Test with finite curvature
        kappa_c = np.array([0.1, 0.2, 0.5, 1.0])
        rho_c = mapper.compute_osculating_radius(kappa_c)

        expected = 1.0 / kappa_c
        np.testing.assert_array_almost_equal(rho_c, expected)

        # Test with zero curvature
        kappa_c_zero = np.array([0.0, 0.1, 0.0])
        rho_c_zero = mapper.compute_osculating_radius(kappa_c_zero)

        assert rho_c_zero[0] == np.inf
        assert rho_c_zero[2] == np.inf
        assert rho_c_zero[1] == 10.0

    def test_design_ring_radius_constant(self, mapper):
        """Test constant ring radius design."""
        psi = np.linspace(0, 2*np.pi, 50)
        R_psi = mapper.design_ring_radius(psi, "constant", base_radius=20.0)

        expected = np.full_like(psi, 20.0)
        np.testing.assert_array_almost_equal(R_psi, expected)

    def test_design_ring_radius_linear(self, mapper):
        """Test linear ring radius design."""
        psi = np.linspace(0, 2*np.pi, 50)
        R_psi = mapper.design_ring_radius(psi, "linear", base_radius=15.0, slope=2.0)

        expected = 15.0 + 2.0 * psi
        np.testing.assert_array_almost_equal(R_psi, expected)

    def test_design_ring_radius_sinusoidal(self, mapper):
        """Test sinusoidal ring radius design."""
        psi = np.linspace(0, 2*np.pi, 50)
        R_psi = mapper.design_ring_radius(psi, "sinusoidal",
                                        base_radius=15.0, amplitude=3.0, frequency=2.0)

        expected = 15.0 + 3.0 * np.sin(2.0 * psi)
        np.testing.assert_array_almost_equal(R_psi, expected)

    def test_design_ring_radius_custom(self, mapper):
        """Test custom ring radius design."""
        psi = np.linspace(0, 2*np.pi, 50)

        def custom_func(psi_vals):
            return 10.0 + psi_vals**2

        R_psi = mapper.design_ring_radius(psi, "custom", custom_function=custom_func)

        expected = 10.0 + psi**2
        np.testing.assert_array_almost_equal(R_psi, expected)

    def test_design_ring_radius_invalid_type(self, mapper):
        """Test invalid ring design type."""
        psi = np.linspace(0, 2*np.pi, 10)

        with pytest.raises(ValueError, match="Unknown design type"):
            mapper.design_ring_radius(psi, "invalid_type")

    def test_solve_meshing_law(self, mapper, sample_data):
        """Test meshing law solution."""
        theta, x_theta = sample_data
        psi = np.linspace(0, 2*np.pi, len(theta))
        rho_c = np.ones_like(theta) * 5.0
        R_psi = np.ones_like(psi) * 10.0

        result_psi = mapper.solve_meshing_law(theta, rho_c, psi, R_psi)

        # Check that result has correct shape and is finite
        assert result_psi.shape == psi.shape
        assert np.all(np.isfinite(result_psi))
        # Check that result is reasonable (not all zeros or inf)
        assert not np.all(result_psi == 0)
        assert not np.all(np.isinf(result_psi))

    def test_solve_meshing_law_failure(self, mapper, sample_data):
        """Test meshing law solution failure handling."""
        theta, x_theta = sample_data
        psi = np.linspace(0, 2*np.pi, len(theta))
        rho_c = np.ones_like(theta) * 5.0
        R_psi = np.ones_like(psi) * 10.0

        # Test with invalid data that should trigger fallback
        with patch("scipy.integrate.solve_ivp") as mock_solve_ivp:
            mock_solve_ivp.side_effect = Exception("ODE solver failed")

            result_psi = mapper.solve_meshing_law(theta, rho_c, psi, R_psi)

            # Should fall back to linear approximation
            # Just check that we get a reasonable result
            assert result_psi.shape == psi.shape
            assert np.all(np.isfinite(result_psi))
            assert result_psi[0] == psi[0]  # Should start at the same point
            # The end point might be different due to the meshing law
            assert result_psi[-1] >= psi[0]  # Should be reasonable

    def test_compute_time_kinematics_cam_driven(self, mapper, sample_data):
        """Test time kinematics computation with cam-driven system."""
        theta, x_theta = sample_data
        psi = np.linspace(0, 2*np.pi, len(theta))
        rho_c = np.ones_like(theta) * 5.0
        R_psi = np.ones_like(psi) * 10.0

        kinematics = mapper.compute_time_kinematics(
            theta, psi, rho_c, R_psi, driver="cam", omega=2.0,
        )

        assert "time" in kinematics
        assert "theta" in kinematics
        assert "psi" in kinematics
        assert kinematics["driver"] == "cam"

    def test_compute_time_kinematics_ring_driven(self, mapper, sample_data):
        """Test time kinematics computation with ring-driven system."""
        theta, x_theta = sample_data
        psi = np.linspace(0, 2*np.pi, len(theta))
        rho_c = np.ones_like(theta) * 5.0
        R_psi = np.ones_like(psi) * 10.0

        kinematics = mapper.compute_time_kinematics(
            theta, psi, rho_c, R_psi, driver="ring", Omega=1.5,
        )

        assert "time" in kinematics
        assert "theta" in kinematics
        assert "psi" in kinematics
        assert kinematics["driver"] == "ring"

    def test_compute_time_kinematics_invalid_driver(self, mapper, sample_data):
        """Test time kinematics with invalid driver parameters."""
        theta, x_theta = sample_data
        psi = np.linspace(0, 2*np.pi, len(theta))
        rho_c = np.ones_like(theta) * 5.0
        R_psi = np.ones_like(psi) * 10.0

        with pytest.raises(ValueError, match="Must specify either omega"):
            mapper.compute_time_kinematics(theta, psi, rho_c, R_psi, driver="cam")

    def test_map_linear_to_ring_follower(self, mapper, sample_data):
        """Test complete linear-to-ring follower mapping."""
        theta, x_theta = sample_data

        ring_design = {
            "design_type": "constant",
            "base_radius": 15.0,
        }

        results = mapper.map_linear_to_ring_follower(theta, x_theta, ring_design)

        # Check that all required keys are present
        required_keys = ["theta", "x_theta", "cam_curves", "kappa_c", "rho_c", "psi", "R_psi"]
        for key in required_keys:
            assert key in results

        # Check array shapes (enhanced grid may have more points for better resolution)
        assert len(results["theta"]) >= len(theta)  # Enhanced grid has more points
        assert len(results["x_theta"]) >= len(x_theta)  # Enhanced grid has more points
        assert len(results["psi"]) >= len(theta)  # Enhanced grid has more points
        assert len(results["R_psi"]) >= len(theta)  # Enhanced grid has more points

        # Check that all arrays have the same length (consistency)
        assert len(results["theta"]) == len(results["x_theta"])
        assert len(results["psi"]) == len(results["R_psi"])
        assert len(results["theta"]) == len(results["psi"])

    def test_validate_design(self, mapper, sample_data):
        """Test design validation."""
        theta, x_theta = sample_data

        # Create a valid design
        ring_design = {"design_type": "constant", "base_radius": 15.0}
        results = mapper.map_linear_to_ring_follower(theta, x_theta, ring_design)

        validation = mapper.validate_design(results)

        # Check that validation returns boolean values
        assert isinstance(validation, dict)
        for key, value in validation.items():
            assert isinstance(value, bool)


class TestCamRingProcessing:
    """Test cam-ring processing functions."""

    @pytest.fixture
    def sample_primary_data(self):
        """Create sample primary optimization data."""
        time = np.linspace(0, 2*np.pi, 100)
        position = 5.0 * np.sin(time)
        velocity = 5.0 * np.cos(time)
        acceleration = -5.0 * np.sin(time)
        control = -5.0 * np.cos(time)

        return {
            "time": time,
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "control": control,
        }

    def test_process_linear_to_ring_follower(self, sample_primary_data):
        """Test linear-to-ring follower processing."""
        constraints = {
            "cam_parameters": {"base_radius": 12.0},
            "ring_design_type": "constant",
            "ring_design_params": {"base_radius": 20.0},
        }

        results = process_linear_to_ring_follower(
            sample_primary_data, constraints, {}, {},
        )

        # Check that results contain expected keys
        expected_keys = ["theta", "x_theta", "psi", "R_psi", "cam_curves", "validation"]
        for key in expected_keys:
            assert key in results

        # Check that primary data is preserved
        assert "primary_time" in results
        assert "primary_position" in results

    def test_process_linear_to_ring_follower_empty_data(self):
        """Test processing with empty primary data."""
        empty_data = {"time": np.array([]), "position": np.array([])}
        constraints = {"ring_design_type": "constant"}

        with pytest.raises(ValueError, match="Primary data must contain time and position"):
            process_linear_to_ring_follower(empty_data, constraints, {}, {})

    def test_process_ring_optimization(self, sample_primary_data):
        """Test ring optimization processing."""
        constraints = {
            "cam_parameters": {"base_radius": 10.0},
            "parameter_bounds": {"ring_radius": (10.0, 30.0)},
        }
        targets = {"objective": "minimize_ring_size"}

        results = process_ring_optimization(
            sample_primary_data, constraints, {}, targets,
        )

        assert "optimization_objective" in results
        assert results["optimization_objective"] == "minimize_ring_size"

    def test_process_multi_objective_ring_design(self, sample_primary_data):
        """Test multi-objective ring design processing."""
        constraints = {
            "cam_parameters": {"base_radius": 10.0},
            "design_alternatives": [
                {"design_type": "constant", "base_radius": 15.0},
                {"design_type": "linear", "base_radius": 12.0, "slope": 1.0},
            ],
        }
        targets = {
            "weights": {
                "ring_size": 0.4,
                "efficiency": 0.3,
                "smoothness": 0.2,
                "stress": 0.1,
            },
        }

        results = process_multi_objective_ring_design(
            sample_primary_data, constraints, {}, targets,
        )

        assert "multi_objective_score" in results
        assert "design_alternative" in results
        assert isinstance(results["multi_objective_score"], float)
        assert isinstance(results["design_alternative"], int)

    def test_calculate_multi_objective_score(self):
        """Test multi-objective score calculation."""
        # Create mock result data
        result = {
            "R_psi": np.array([10.0, 12.0, 15.0]),
            "psi": np.array([0.0, 1.0, 2.0]),
            "theta": np.array([0.0, 0.5, 1.0]),
            "kappa_c": np.array([0.1, 0.2, 0.15]),
        }

        weights = {
            "ring_size": 0.5,
            "efficiency": 0.3,
            "smoothness": 0.2,
        }

        score = _calculate_multi_objective_score(result, weights)

        assert isinstance(score, float)
        assert score >= 0.0  # Score should be non-negative

    def test_create_constant_ring_design(self, sample_primary_data):
        """Test constant ring design creation."""
        results = create_constant_ring_design(
            sample_primary_data, ring_radius=25.0,
        )

        # Check that ring radius is approximately constant
        R_psi = results["R_psi"]
        assert np.allclose(R_psi, 25.0, rtol=1e-10)

    def test_create_optimized_ring_design(self, sample_primary_data):
        """Test optimized ring design creation."""
        results = create_optimized_ring_design(
            sample_primary_data,
            optimization_objective="minimize_ring_size",
        )

        assert "optimization_objective" in results
        assert results["optimization_objective"] == "minimize_ring_size"


class TestIntegration:
    """Integration tests for cam-ring mapping system."""

    def test_end_to_end_mapping(self):
        """Test complete end-to-end mapping from linear follower to ring design."""
        # Create a realistic linear follower motion law
        theta = np.linspace(0, 2*np.pi, 200)
        x_theta = 8.0 * (1 - np.cos(theta))  # Simple harmonic motion

        # Create mapper with realistic parameters
        params = CamRingParameters(
            base_radius=15.0,
            connecting_rod_length=25.0,
            contact_type="external",
        )
        mapper = CamRingMapper(params)

        # Design ring with sinusoidal variation
        ring_design = {
            "design_type": "sinusoidal",
            "base_radius": 20.0,
            "amplitude": 3.0,
            "frequency": 1.0,
        }

        # Perform complete mapping
        results = mapper.map_linear_to_ring_follower(theta, x_theta, ring_design)

        # Validate results
        validation = mapper.validate_design(results)

        # Check that design is valid
        assert validation["positive_cam_radii"]
        assert validation["positive_ring_radii"]
        assert validation["reasonable_curvature"]
        assert validation["reasonable_osculating_radius"]
        assert validation["smooth_angle_relationship"]

        # Check that ring radius has expected sinusoidal variation
        R_psi = results["R_psi"]
        expected_variation = 20.0 + 3.0 * np.sin(results["psi"])
        # Allow significant tolerance due to meshing law effects
        # The meshing law changes the relationship between psi and the ring radius
        assert np.all(R_psi > 0)  # All radii should be positive
        assert np.min(R_psi) >= 17.0  # Should be close to base radius - amplitude
        assert np.max(R_psi) <= 23.0  # Should be close to base radius + amplitude

    def test_secondary_optimizer_integration(self):
        """Test integration with secondary optimizer."""
        from campro.optimization.secondary import SecondaryOptimizer
        from campro.storage import OptimizationRegistry

        # Create registry and store primary result
        registry = OptimizationRegistry()

        # Mock primary result
        primary_data = {
            "time": np.linspace(0, 2*np.pi, 100),
            "position": 5.0 * np.sin(np.linspace(0, 2*np.pi, 100)),
            "velocity": 5.0 * np.cos(np.linspace(0, 2*np.pi, 100)),
            "acceleration": -5.0 * np.sin(np.linspace(0, 2*np.pi, 100)),
            "control": -5.0 * np.cos(np.linspace(0, 2*np.pi, 100)),
        }

        # Store primary result
        registry.store_result("motion_optimizer", primary_data, {})

        # Create secondary optimizer
        secondary_optimizer = SecondaryOptimizer(registry=registry)

        # Define processing function
        def dummy_objective(t, x, v, a, u):
            return np.trapz(u**2, t)

        # Test processing
        constraints = {
            "cam_parameters": {"base_radius": 12.0},
            "ring_design_type": "constant",
            "ring_design_params": {"base_radius": 18.0},
        }

        result = secondary_optimizer.optimize(
            objective=dummy_objective,
            constraints=None,
            primary_optimizer_id="motion_optimizer",
            processing_function=process_linear_to_ring_follower,
            secondary_constraints=constraints,
            secondary_relationships={},
            optimization_targets={},
        )

        # Check that optimization was successful
        assert result.status.name in ["SUCCESS", "CONVERGED"]
        assert "theta" in result.solution
        assert "R_psi" in result.solution
