from __future__ import annotations
from CamPro_OptimalMotion import (
    CamMotionConstraints,
    CollocationSettings,
    MotionConstraints,
    OptimalMotionSolver,
    solve_cam_motion_law,
    solve_minimum_energy_motion,
    solve_minimum_jerk_motion,
    solve_minimum_time_motion,
)
import pytest
import numpy as np

import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _basic_constraints() -> MotionConstraints:
    """Create basic motion constraints."""
    return MotionConstraints(
        initial_position=0.0,
        final_position=10.0,
        velocity_bounds=(-20.0, 20.0),
        acceleration_bounds=(-100.0, 100.0),
    )


def _basic_cam_constraints() -> CamMotionConstraints:
    """Create basic cam motion constraints."""
    return CamMotionConstraints(
        stroke=10.0,
        upstroke_duration_percent=60.0,
        zero_accel_duration_percent=0.0,
    )


def _solver() -> OptimalMotionSolver:
    """Create default solver instance."""
    return OptimalMotionSolver()


def test_motion_constraints() -> None:
    """Test MotionConstraints initialization."""
    constraints = MotionConstraints()
    assert constraints.position_bounds is None
    assert constraints.velocity_bounds is None

    custom = MotionConstraints(
        velocity_bounds=(-10.0, 10.0),
        initial_position=0.0,
        final_position=100.0,
    )
    assert custom.velocity_bounds == (-10.0, 10.0)
    assert custom.initial_position == 0.0


def test_collocation_settings() -> None:
    """Test CollocationSettings initialization."""
    settings = CollocationSettings()
    assert settings.degree == 3
    assert settings.method == "legendre"

    custom = CollocationSettings(degree=5, method="radau", max_iterations=500)
    assert custom.degree == 5
    assert custom.method == "radau"
    assert custom.max_iterations == 500


def test_solver_initialization() -> None:
    """Test OptimalMotionSolver initialization."""
    solver = OptimalMotionSolver()
    assert solver.settings.degree == 3

    settings = CollocationSettings(degree=5, method="radau")
    solver_custom = OptimalMotionSolver(settings)
    assert solver_custom.settings.degree == 5

    # Test invalid settings
    with pytest.raises(ValueError):
        CollocationSettings(degree=0)
    with pytest.raises(ValueError):
        CollocationSettings(method="invalid")


def test_minimum_time_motion() -> None:
    """Test minimum time motion solving."""
    solver = _solver()
    constraints = _basic_constraints()

    solution = solver.solve_minimum_time_motion(constraints, time_horizon=1.0)

    assert "time" in solution
    assert "position" in solution
    assert "velocity" in solution
    assert solution["position"][0] == pytest.approx(0.0, abs=1e-3)
    assert solution["position"][-1] == pytest.approx(10.0, abs=1e-3)


def test_minimum_energy_motion() -> None:
    """Test minimum energy motion solving."""
    solver = _solver()
    constraints = _basic_constraints()

    solution = solver.solve_minimum_energy_motion(
        constraints, time_horizon=1.0)

    assert "time" in solution
    assert "position" in solution
    assert "velocity" in solution
    assert solution["position"][0] == pytest.approx(0.0, abs=1e-3)
    assert solution["position"][-1] == pytest.approx(10.0, abs=1e-3)


def test_minimum_jerk_motion() -> None:
    """Test minimum jerk motion solving."""
    solver = _solver()
    constraints = _basic_constraints()

    solution = solver.solve_minimum_jerk_motion(constraints, time_horizon=1.0)

    assert "time" in solution
    assert "position" in solution
    assert "velocity" in solution
    assert solution["position"][0] == pytest.approx(0.0, abs=1e-3)
    assert solution["position"][-1] == pytest.approx(10.0, abs=1e-3)


def test_convenience_functions() -> None:
    """Test convenience functions for motion solving."""
    constraints = _basic_constraints()

    time_sol = solve_minimum_time_motion(constraints, time_horizon=1.0)
    assert "time" in time_sol

    energy_sol = solve_minimum_energy_motion(constraints, time_horizon=1.0)
    assert "time" in energy_sol

    jerk_sol = solve_minimum_jerk_motion(constraints, time_horizon=1.0)
    assert "time" in jerk_sol


def test_cam_motion_constraints() -> None:
    """Test CamMotionConstraints."""
    constraints = CamMotionConstraints()
    assert constraints.stroke > 0

    custom = CamMotionConstraints(
        stroke=20.0,
        upstroke_duration_percent=60.0,
        zero_accel_duration_percent=10.0,
    )
    assert custom.stroke == 20.0
    assert custom.upstroke_duration_percent == 60.0

    # Test invalid constraints
    with pytest.raises(ValueError):
        CamMotionConstraints(stroke=-10.0)
    with pytest.raises(ValueError):
        CamMotionConstraints(upstroke_duration_percent=150.0)


def test_cam_motion_law_solving() -> None:
    """Test cam motion law solving."""
    solver = _solver()
    constraints = _basic_cam_constraints()

    solution = solver.solve_cam_motion_law(
        constraints, motion_type="minimum_jerk")

    assert "cam_angle" in solution
    assert "position" in solution
    assert "velocity" in solution
    assert len(solution["cam_angle"]) > 0


@pytest.mark.parametrize("motion_type",
                         ["minimum_jerk", "minimum_time", "minimum_energy"])
def test_cam_motion_law_different_types(motion_type: str) -> None:
    """Test cam motion law with different motion types."""
    solver = _solver()
    constraints = _basic_cam_constraints()

    solution = solver.solve_cam_motion_law(
        constraints, motion_type=motion_type)

    assert "cam_angle" in solution
    assert "position" in solution
    assert solution["position"].max() <= constraints.stroke + 1e-3


def test_cam_motion_law_with_constraints() -> None:
    """Test cam motion law with velocity/acceleration constraints."""
    solver = _solver()
    constraints = CamMotionConstraints(
        stroke=10.0,
        upstroke_duration_percent=60.0,
        max_velocity=30.0,
        max_acceleration=150.0,
    )

    solution = solver.solve_cam_motion_law(
        constraints, motion_type="minimum_time")

    max_velocity = np.max(np.abs(solution["velocity"]))
    max_acceleration = np.max(np.abs(solution["acceleration"]))

    assert max_velocity <= 30.0 + 1e-6
    assert max_acceleration <= 150.0 + 1e-6


def test_solve_cam_motion_law_function() -> None:
    """Test solve_cam_motion_law convenience function."""
    constraints = _basic_cam_constraints()

    solution = solve_cam_motion_law(constraints, motion_type="minimum_jerk")

    assert "cam_angle" in solution
    assert "position" in solution


def test_invalid_cam_motion_type() -> None:
    """Test that invalid cam motion type raises ValueError."""
    solver = _solver()
    constraints = _basic_cam_constraints()

    with pytest.raises(ValueError, match="Unknown motion type"):
        solver.solve_cam_motion_law(constraints, motion_type="invalid_type")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
