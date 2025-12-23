"""
Unit tests for constraints module.

Tests ConstraintType, ConstraintViolation, BaseConstraints, and MotionConstraints.
"""

import numpy as np
import pytest

from campro.constraints.base import ConstraintType, ConstraintViolation
from campro.constraints.motion import MotionConstraints, MotionConstraintType


class TestConstraintType:
    """Tests for ConstraintType enum."""

    def test_all_types_defined(self):
        """All expected constraint types exist."""
        assert ConstraintType.POSITION.value == "position"
        assert ConstraintType.VELOCITY.value == "velocity"
        assert ConstraintType.ACCELERATION.value == "acceleration"
        assert ConstraintType.JERK.value == "jerk"
        assert ConstraintType.CONTROL.value == "control"

    def test_boundary_types(self):
        """Boundary constraint types exist."""
        assert ConstraintType.INITIAL_STATE.value == "initial_state"
        assert ConstraintType.FINAL_STATE.value == "final_state"


class TestConstraintViolation:
    """Tests for ConstraintViolation dataclass."""

    def test_violation_creation(self):
        """Create a constraint violation with required fields."""
        violation = ConstraintViolation(
            constraint_type=ConstraintType.POSITION,
            constraint_name="test_pos",
            violation_value=10.5,
            limit_value=10.0,
        )
        assert violation.constraint_type == ConstraintType.POSITION
        assert violation.constraint_name == "test_pos"
        assert violation.violation_value == 10.5
        assert violation.limit_value == 10.0

    def test_violation_auto_message(self):
        """Violation auto-generates message if not provided."""
        violation = ConstraintViolation(
            constraint_type=ConstraintType.VELOCITY,
            constraint_name="max_vel",
            violation_value=15.0,
            limit_value=10.0,
        )
        assert "max_vel" in violation.message
        assert "15.0" in violation.message or "15.000" in violation.message

    def test_violation_custom_message(self):
        """Custom message overrides auto-generated one."""
        violation = ConstraintViolation(
            constraint_type=ConstraintType.ACCELERATION,
            constraint_name="accel_limit",
            violation_value=100.0,
            limit_value=50.0,
            message="Custom error message",
        )
        assert violation.message == "Custom error message"

    def test_violation_with_time_index(self):
        """Violation can include time index."""
        violation = ConstraintViolation(
            constraint_type=ConstraintType.JERK,
            constraint_name="jerk_spike",
            violation_value=1000.0,
            limit_value=500.0,
            time_index=42,
        )
        assert violation.time_index == 42


class TestMotionConstraints:
    """Tests for MotionConstraints class."""

    def test_default_bounds(self):
        """Default motion constraints have None bounds (no restrictions)."""
        mc = MotionConstraints()
        # Defaults are None (meaning no bounds enforced)
        assert mc.position_bounds is None
        assert mc.velocity_bounds is None
        assert mc.acceleration_bounds is None
        assert mc.jerk_bounds is None

    def test_custom_bounds(self):
        """Custom bounds are stored correctly."""
        mc = MotionConstraints(
            position_bounds=(0.0, 1.0),
            velocity_bounds=(-5.0, 5.0),
            acceleration_bounds=(-10.0, 10.0),
            jerk_bounds=(-100.0, 100.0),
        )
        assert mc.position_bounds == (0.0, 1.0)
        assert mc.velocity_bounds == (-5.0, 5.0)

    def test_validate_with_valid_constraints(self):
        """Valid constraints pass validation."""
        mc = MotionConstraints(
            position_bounds=(0.0, 1.0),
            velocity_bounds=(-10.0, 10.0),
        )
        assert mc.validate() is True

    def test_validate_with_inverted_bounds(self):
        """Inverted bounds (min > max) fail validation."""
        mc = MotionConstraints(
            position_bounds=(1.0, 0.0),  # Invalid: min > max
        )
        # Should return False as bounds are inverted
        assert mc.validate() is False

    def test_check_violations_no_violation(self):
        """Solution within bounds has no violations."""
        mc = MotionConstraints(
            position_bounds=(0.0, 1.0),
            velocity_bounds=(-10.0, 10.0),
            acceleration_bounds=(-100.0, 100.0),
            jerk_bounds=(-1000.0, 1000.0),
        )
        solution = {
            "time": np.linspace(0, 1, 100),
            "position": np.linspace(0, 1, 100),  # 0 to 1
            "velocity": np.ones(100) * 1.0,  # Constant 1
            "acceleration": np.zeros(100),  # Zero
            "control": np.zeros(100),  # Zero jerk
        }
        violations = mc.check_violations(solution)
        assert len(violations) == 0

    def test_check_violations_detects_exceeds(self):
        """Solution exceeding bounds has violations."""
        mc = MotionConstraints(
            position_bounds=(0.0, 1.0),
            velocity_bounds=(-10.0, 10.0),
        )
        solution = {
            "time": np.linspace(0, 1, 100),
            "position": np.linspace(-0.5, 1.5, 100),  # Exceeds both bounds
            "velocity": np.ones(100) * 5.0,
            "acceleration": np.zeros(100),
            "control": np.zeros(100),
        }
        violations = mc.check_violations(solution)
        # Should have position violations
        assert len(violations) > 0
        assert any(v.constraint_name.startswith("position") for v in violations)

    def test_to_dict_round_trip(self):
        """Constraints can be serialized and deserialized."""
        mc = MotionConstraints(
            position_bounds=(0.0, 1.0),
            velocity_bounds=(-5.0, 5.0),
            initial_position=0.0,
            final_position=1.0,
        )
        data = mc.to_dict()
        mc2 = MotionConstraints.from_dict(data)

        assert mc2.position_bounds == mc.position_bounds
        assert mc2.velocity_bounds == mc.velocity_bounds
        assert mc2.initial_position == mc.initial_position
        assert mc2.final_position == mc.final_position


class TestMotionConstraintType:
    """Tests for MotionConstraintType enum."""

    def test_bounds_types(self):
        """Bounds constraint types are defined."""
        assert MotionConstraintType.POSITION_BOUNDS.value == "position_bounds"
        assert MotionConstraintType.VELOCITY_BOUNDS.value == "velocity_bounds"
        assert MotionConstraintType.ACCELERATION_BOUNDS.value == "acceleration_bounds"
        assert MotionConstraintType.JERK_BOUNDS.value == "jerk_bounds"

    def test_boundary_condition_types(self):
        """Boundary condition types are defined."""
        assert MotionConstraintType.INITIAL_POSITION.value == "initial_position"
        assert MotionConstraintType.FINAL_POSITION.value == "final_position"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
