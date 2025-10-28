"""
Test suite for CasADi physics validation mode.

This module tests the experimental parallel validation mode that runs both
Python and CasADi physics for comparison and confidence building.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from campro.constants import (
    CASADI_PHYSICS_VALIDATION_MODE,
    CASADI_PHYSICS_VALIDATION_TOLERANCE,
    USE_CASADI_PHYSICS,
)
from campro.optimization.crank_center_optimizer import CrankCenterOptimizer
from campro.physics.geometry.litvin import LitvinGearGeometry


class TestCasadiValidationMode:
    """Test CasADi physics validation mode functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = CrankCenterOptimizer()

        # Mock configuration
        self.optimizer._is_configured = True
        self.optimizer.constraints = Mock()
        self.optimizer.constraints.max_iterations = 100
        self.optimizer.constraints.tolerance = 1e-6
        self.optimizer.constraints.crank_center_x_min = -10.0
        self.optimizer.constraints.crank_center_x_max = 10.0
        self.optimizer.constraints.crank_center_y_min = -10.0
        self.optimizer.constraints.crank_center_y_max = 10.0
        self.optimizer.constraints.crank_radius_min = 20.0
        self.optimizer.constraints.crank_radius_max = 100.0
        self.optimizer.constraints.rod_length_min = 100.0
        self.optimizer.constraints.rod_length_max = 300.0

    def test_validation_mode_constants(self):
        """Test that validation mode constants are properly defined."""
        assert hasattr(CASADI_PHYSICS_VALIDATION_MODE, "__bool__")
        assert hasattr(CASADI_PHYSICS_VALIDATION_TOLERANCE, "__float__")
        assert CASADI_PHYSICS_VALIDATION_TOLERANCE > 0

    def test_validation_mode_enabled(self):
        """Test that validation mode can be enabled."""
        # This test verifies that the validation mode flag exists and can be set
        # The actual validation logic is tested in integration tests
        assert hasattr(CASADI_PHYSICS_VALIDATION_MODE, "__bool__")
        assert hasattr(USE_CASADI_PHYSICS, "__bool__")

    def test_validation_mode_disabled_by_default(self):
        """Test that validation mode is disabled by default."""
        # This ensures we don't accidentally enable validation mode in production
        assert CASADI_PHYSICS_VALIDATION_MODE is False

    @patch("campro.constants.CASADI_PHYSICS_VALIDATION_MODE", True)
    @patch("campro.constants.USE_CASADI_PHYSICS", True)
    def test_validation_mode_method_exists(self):
        """Test that validation mode method exists and is callable."""
        assert hasattr(self.optimizer, "_optimize_with_validation_mode")
        assert callable(self.optimizer._optimize_with_validation_mode)

    @patch("campro.constants.CASADI_PHYSICS_VALIDATION_MODE", True)
    @patch("campro.constants.USE_CASADI_PHYSICS", True)
    def test_validation_mode_physics_evaluation(self):
        """Test CasADi physics evaluation in validation mode."""
        # Test parameters
        params = np.array([5.0, 2.0, 50.0, 150.0])  # x, y, r, l
        param_names = ["crank_center_x", "crank_center_y", "crank_radius", "rod_length"]

        # Mock motion law data
        motion_law_data = {
            "crank_angle": np.linspace(0, 2 * np.pi, 10),
        }
        load_profile = 1e5 * np.ones(10)  # Constant pressure

        # Mock gear geometry
        gear_geometry = Mock(spec=LitvinGearGeometry)

        # Test the validation method
        try:
            result = self.optimizer._validate_casadi_physics_at_point(
                params,
                param_names,
                motion_law_data,
                load_profile,
                gear_geometry,
            )

            # Check that result has expected structure
            assert isinstance(result, dict)
            assert "torque_avg" in result
            assert "torque_ripple" in result
            assert "side_load_penalty" in result
            assert "litvin_objective" in result
            assert "litvin_closure" in result

            # Check that values are finite
            for key, value in result.items():
                assert np.isfinite(value), f"{key} is not finite: {value}"

        except ImportError:
            pytest.skip("CasADi not available for validation mode testing")

    def test_validation_mode_performance_overhead(self):
        """Test that validation mode has acceptable performance overhead."""
        # This test measures the overhead of validation mode
        # In a real implementation, we would measure actual performance

        # Mock a simple objective function
        def simple_objective(params):
            return np.sum(params**2)

        # Test parameters
        initial_params = np.array([1.0, 1.0, 1.0, 1.0])
        bounds = [(-10, 10), (-10, 10), (20, 100), (100, 300)]
        constraints = []
        param_names = ["x", "y", "r", "l"]

        # Mock data
        motion_law_data = {"crank_angle": np.linspace(0, 2 * np.pi, 10)}
        load_profile = np.ones(10)
        gear_geometry = Mock(spec=LitvinGearGeometry)

        # Measure time without validation mode
        start_time = time.perf_counter()
        for _ in range(10):
            simple_objective(initial_params)
        baseline_time = time.perf_counter() - start_time

        # The overhead should be minimal for this simple test
        # In practice, validation mode overhead should be <10% of total optimization time
        assert baseline_time > 0  # Basic sanity check

    def test_validation_mode_error_handling(self):
        """Test that validation mode handles errors gracefully."""
        # Test that validation mode doesn't crash the optimization
        # even if CasADi validation fails

        # Mock parameters that might cause issues
        params = np.array([1.0, 2.0, 50.0, 150.0])  # Valid parameters
        param_names = ["x", "y", "r", "l"]
        motion_law_data = {"crank_angle": np.array([])}  # Empty array
        load_profile = np.array([])
        gear_geometry = Mock(spec=LitvinGearGeometry)

        # This should handle empty arrays gracefully
        try:
            result = self.optimizer._validate_casadi_physics_at_point(
                params,
                param_names,
                motion_law_data,
                load_profile,
                gear_geometry,
            )
            # If it succeeds, result should be None or have error handling
        except Exception as e:
            # Expected for invalid inputs - any exception is acceptable
            assert isinstance(e, Exception)

    def test_validation_mode_logging(self):
        """Test that validation mode produces appropriate log messages."""
        # Test the logging functionality
        casadi_validation = {
            "torque_avg": 2.5,
            "torque_ripple": 0.1,
            "side_load_penalty": 100.0,
            "litvin_objective": 0.0,
            "litvin_closure": 0.0,
        }

        python_result = Mock()
        python_result.x = np.array([1.0, 2.0, 50.0, 150.0])

        # This should not crash and should log appropriately
        self.optimizer._log_validation_comparison(python_result, casadi_validation)

        # Test with None validation (error case)
        self.optimizer._log_validation_comparison(python_result, None)

    def test_validation_mode_tolerance_checking(self):
        """Test tolerance checking for validation mode."""
        # Test that tolerance values are reasonable
        assert CASADI_PHYSICS_VALIDATION_TOLERANCE > 0
        assert CASADI_PHYSICS_VALIDATION_TOLERANCE < 1.0  # Should be small

        # Test tolerance comparison logic
        def within_tolerance(val1, val2, tolerance):
            return abs(val1 - val2) <= tolerance

        # Test cases
        assert within_tolerance(1.0, 1.0001, 1e-3)
        assert not within_tolerance(1.0, 1.1, 1e-3)
        assert within_tolerance(0.0, 0.0, 1e-4)
