"""
Integration tests for CasADi physics in CrankCenterOptimizer.

This test suite verifies that the CasADi physics integration works correctly
in the optimizer without producing NaN values or optimization failures.
"""

import casadi as ca
import numpy as np
import pytest

from campro.constants import USE_CASADI_PHYSICS
from campro.optimization.crank_center_optimizer import CrankCenterOptimizer
from campro.physics.geometry.litvin import LitvinGearGeometry


class TestCasadiOptimizerIntegration:
    """Test CasADi physics integration in the optimizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = CrankCenterOptimizer()

        # Create realistic test data
        self.motion_law_data = {
            "theta": np.linspace(0, 2 * np.pi, 100),
        }
        self.load_profile = 1000.0 * np.ones(100)

        # Create Litvin geometry for testing
        self.litvin_geometry = LitvinGearGeometry(
            base_circle_cam=25.0,
            base_circle_ring=10.0,
            pressure_angle_rad=np.array([np.radians(20.0)]),
            contact_ratio=1.5,
            path_of_contact_arc_length=5.0,
            z_cam=50,
            z_ring=20,
            interference_flag=False,
        )

    def test_casadi_physics_enabled(self):
        """Test that CasADi physics is enabled."""
        assert USE_CASADI_PHYSICS is True, "CasADi physics should be enabled"

    def test_optimizer_creation(self):
        """Test that optimizer can be created with CasADi physics."""
        assert self.optimizer is not None
        assert hasattr(self.optimizer, "_optimize_with_ipopt")

    def test_unified_physics_function_creation(self):
        """Test that unified physics function can be created."""
        from campro.physics.casadi import create_unified_physics

        unified_fn = create_unified_physics()
        assert unified_fn is not None

        # Test function signature
        inputs = unified_fn.sx_in()
        outputs = unified_fn.sx_out()

        assert len(inputs) == 9, f"Expected 9 inputs, got {len(inputs)}"
        assert len(outputs) == 5, f"Expected 5 outputs, got {len(outputs)}"

    def test_unified_physics_evaluation(self):
        """Test that unified physics function can be evaluated without NaN."""
        from campro.physics.casadi import create_unified_physics

        unified_fn = create_unified_physics()

        # Test with realistic parameters
        theta_vec = ca.DM(np.linspace(0, 2 * np.pi, 10))
        pressure_vec = ca.DM(1000.0 * np.ones(10))
        r, l = 50.0, 150.0
        x_off, y_off = 0.0, 0.0
        bore = 100.0
        max_side_threshold = 200.0
        litvin_config = ca.DM([50.0, 20.0, 2.0, 20.0, 25.0, 0.0])  # disabled

        # Evaluate function
        result = unified_fn(
            theta_vec,
            pressure_vec,
            r,
            l,
            x_off,
            y_off,
            bore,
            max_side_threshold,
            litvin_config,
        )

        # Check that all outputs are finite
        for i, output in enumerate(result):
            assert np.isfinite(float(output)), f"Output {i} is not finite: {output}"
            assert output.shape == (1, 1), f"Output {i} is not scalar: {output.shape}"

    def test_optimizer_without_litvin_geometry(self):
        """Test optimizer with CasADi physics but no Litvin geometry."""
        # Create primary and secondary data
        primary_data = {
            "theta": self.motion_law_data["theta"],
            "load_profile": self.load_profile,
        }
        secondary_data = {
            "gear_geometry": None,  # No Litvin geometry
        }

        # Test that optimizer can handle missing Litvin geometry
        result = self.optimizer.optimize(
            primary_data=primary_data,
            secondary_data=secondary_data,
        )

        # Should not crash and should return a result
        assert result is not None
        assert hasattr(result, "success")

    def test_optimizer_with_litvin_geometry(self):
        """Test optimizer with CasADi physics and Litvin geometry."""
        # Create primary and secondary data
        primary_data = {
            "theta": self.motion_law_data["theta"],
            "load_profile": self.load_profile,
        }
        secondary_data = {
            "gear_geometry": self.litvin_geometry,
        }

        # Test that optimizer can handle Litvin geometry
        result = self.optimizer.optimize(
            primary_data=primary_data,
            secondary_data=secondary_data,
        )

        # Should not crash and should return a result
        assert result is not None
        assert hasattr(result, "success")

    def test_validation_mode(self):
        """Test that validation mode works with CasADi physics."""
        # Test validation mode if available
        if hasattr(self.optimizer, "_validate_casadi_physics_at_point"):
            # Test validation at a specific point
            params = np.array([0.0, 0.0, 50.0, 150.0])  # [x_off, y_off, r, l]
            param_names = [
                "crank_center_x",
                "crank_center_y",
                "crank_radius",
                "rod_length",
            ]

            validation_result = self.optimizer._validate_casadi_physics_at_point(
                params,
                param_names,
                self.motion_law_data,
                self.load_profile,
                self.litvin_geometry,
            )

            # Should return validation metrics
            assert validation_result is not None

    def test_no_nan_values_in_outputs(self):
        """Test that no NaN values are produced in optimization outputs."""
        # Create primary and secondary data
        primary_data = {
            "theta": self.motion_law_data["theta"],
            "load_profile": self.load_profile,
        }
        secondary_data = {
            "gear_geometry": self.litvin_geometry,
        }

        result = self.optimizer.optimize(
            primary_data=primary_data,
            secondary_data=secondary_data,
        )

        # Check that result doesn't contain NaN values
        if hasattr(result, "x"):
            assert not np.any(np.isnan(result.x)), (
                "Optimization result contains NaN values"
            )

        if hasattr(result, "fun"):
            assert not np.isnan(result.fun), "Objective function value is NaN"

    def test_optimization_convergence(self):
        """Test that optimization converges successfully."""
        # Create primary and secondary data
        primary_data = {
            "theta": self.motion_law_data["theta"],
            "load_profile": self.load_profile,
        }
        secondary_data = {
            "gear_geometry": self.litvin_geometry,
        }

        result = self.optimizer.optimize(
            primary_data=primary_data,
            secondary_data=secondary_data,
        )

        # Check that optimization was successful
        assert result.success, f"Optimization failed: {result.message}"

    def test_performance_requirements(self):
        """Test that optimization meets performance requirements."""
        import time

        # Create primary and secondary data
        primary_data = {
            "theta": self.motion_law_data["theta"],
            "load_profile": self.load_profile,
        }
        secondary_data = {
            "gear_geometry": self.litvin_geometry,
        }

        start_time = time.time()
        result = self.optimizer.optimize(
            primary_data=primary_data,
            secondary_data=secondary_data,
        )
        end_time = time.time()

        # Check that optimization completes in reasonable time
        optimization_time = end_time - start_time
        assert optimization_time < 30.0, (
            f"Optimization took too long: {optimization_time:.2f}s"
        )

        # Check that optimization was successful
        assert result.success, f"Optimization failed: {result.message}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
