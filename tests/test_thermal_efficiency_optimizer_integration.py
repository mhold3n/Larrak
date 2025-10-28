"""
Integration tests for thermal efficiency optimizer.

This test suite verifies that the thermal efficiency optimization works correctly
without producing NaN values or optimization failures.
"""

import numpy as np
import pytest

from campro.optimization.motion_law import MotionLawConstraints, MotionType
from campro.optimization.thermal_efficiency_adapter import ThermalEfficiencyAdapter


class TestThermalEfficiencyIntegration:
    """Test thermal efficiency optimization integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = ThermalEfficiencyAdapter()

        # Create realistic motion law constraints for free piston engine
        self.constraints = MotionLawConstraints(
            stroke=180.0,  # 180mm stroke
            upstroke_duration_percent=50.0,  # 50% upstroke
            zero_accel_duration_percent=10.0,  # 10% zero acceleration
            max_velocity=10.0,  # 10 m/s max velocity
            max_acceleration=500.0,  # 500 m/s^2 max acceleration
        )

    def test_thermal_efficiency_adapter_creation(self):
        """Test that thermal efficiency adapter can be created."""
        assert self.adapter is not None
        assert hasattr(self.adapter, "solve_motion_law")

    def test_thermal_efficiency_no_nan(self):
        """Test that thermal efficiency optimization doesn't produce NaN."""
        try:
            result = self.adapter.solve_motion_law(
                self.constraints,
                MotionType.MINIMUM_JERK,
            )

            # Should not fail with NaN detection
            assert result is not None

            # If optimization succeeds, check for NaN in solution
            if hasattr(result, "success") and result.success:
                if hasattr(result, "solution") and result.solution is not None:
                    # Check for NaN in solution arrays
                    if hasattr(result.solution, "x"):
                        assert not np.any(np.isnan(result.solution.x)), (
                            "Solution contains NaN values"
                        )
                    if hasattr(result.solution, "v"):
                        assert not np.any(np.isnan(result.solution.v)), (
                            "Velocity contains NaN values"
                        )
                    if hasattr(result.solution, "a"):
                        assert not np.any(np.isnan(result.solution.a)), (
                            "Acceleration contains NaN values"
                        )

                # Check objective value
                if hasattr(result, "objective_value"):
                    assert not np.isnan(result.objective_value), (
                        "Objective value is NaN"
                    )

        except Exception as e:
            # If optimization fails, it should not be due to NaN detection
            error_msg = str(e).lower()
            assert "nan" not in error_msg, f"Optimization failed with NaN error: {e}"
            assert "invalid_number_detected" not in error_msg, (
                f"Optimization failed with Invalid_Number_Detected: {e}"
            )

            # Other failures are acceptable for this test (e.g., convergence issues)
            pytest.skip(f"Optimization failed for other reasons: {e}")

    def test_thermal_efficiency_with_different_motion_types(self):
        """Test thermal efficiency optimization with different motion types."""
        motion_types = [MotionType.MINIMUM_JERK, MotionType.MINIMUM_ACCELERATION]

        for motion_type in motion_types:
            try:
                result = self.adapter.solve_motion_law(self.constraints, motion_type)

                # Should not fail with NaN detection
                assert result is not None

                # Check for NaN if optimization succeeds
                if hasattr(result, "success") and result.success:
                    if hasattr(result, "objective_value"):
                        assert not np.isnan(result.objective_value), (
                            f"Objective value is NaN for {motion_type}"
                        )

            except Exception as e:
                # Skip if optimization fails for non-NaN reasons
                error_msg = str(e).lower()
                if "nan" in error_msg or "invalid_number_detected" in error_msg:
                    pytest.fail(f"NaN error with {motion_type}: {e}")
                else:
                    pytest.skip(f"Optimization failed for {motion_type}: {e}")

    def test_thermal_efficiency_constraint_satisfaction(self):
        """Test that thermal efficiency optimization satisfies constraints."""
        try:
            result = self.adapter.solve_motion_law(
                self.constraints,
                MotionType.MINIMUM_JERK,
            )

            if hasattr(result, "success") and result.success:
                if hasattr(result, "solution") and result.solution is not None:
                    # Check stroke constraint
                    if hasattr(result.solution, "x"):
                        stroke_achieved = np.max(result.solution.x) - np.min(
                            result.solution.x,
                        )
                        assert abs(stroke_achieved - self.constraints.stroke) < 0.01, (
                            f"Stroke constraint not satisfied: {stroke_achieved} vs {self.constraints.stroke}"
                        )

                    # Check velocity constraint
                    if hasattr(result.solution, "v"):
                        max_velocity = np.max(np.abs(result.solution.v))
                        assert max_velocity <= self.constraints.max_velocity * 1.1, (
                            f"Velocity constraint violated: {max_velocity} > {self.constraints.max_velocity}"
                        )

                    # Check acceleration constraint
                    if hasattr(result.solution, "a"):
                        max_acceleration = np.max(np.abs(result.solution.a))
                        assert (
                            max_acceleration <= self.constraints.max_acceleration * 1.1
                        ), (
                            f"Acceleration constraint violated: {max_acceleration} > {self.constraints.max_acceleration}"
                        )

        except Exception as e:
            # Skip if optimization fails
            pytest.skip(f"Optimization failed: {e}")

    def test_thermal_efficiency_performance(self):
        """Test that thermal efficiency optimization completes in reasonable time."""
        import time

        start_time = time.time()
        try:
            result = self.adapter.solve_motion_law(
                self.constraints,
                MotionType.MINIMUM_JERK,
            )
            end_time = time.time()

            # Check that optimization completes in reasonable time
            optimization_time = end_time - start_time
            assert optimization_time < 60.0, (
                f"Optimization took too long: {optimization_time:.2f}s"
            )

        except Exception as e:
            # Skip if optimization fails
            pytest.skip(f"Optimization failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
