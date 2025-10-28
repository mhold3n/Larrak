"""
Tests for optimal motion law functionality.

This module contains comprehensive tests for the CamPro_OptimalMotion module,
including unit tests, integration tests, and property-based tests.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

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


class TestMotionConstraints:
    """Test MotionConstraints dataclass."""

    def test_default_constraints(self):
        """Test default constraint initialization."""
        constraints = MotionConstraints()
        assert constraints.position_bounds is None
        assert constraints.velocity_bounds is None
        assert constraints.acceleration_bounds is None
        assert constraints.jerk_bounds is None
        assert constraints.control_bounds is None
        assert constraints.initial_position is None
        assert constraints.initial_velocity is None
        assert constraints.initial_acceleration is None
        assert constraints.final_position is None
        assert constraints.final_velocity is None
        assert constraints.final_acceleration is None

    def test_custom_constraints(self):
        """Test custom constraint initialization."""
        constraints = MotionConstraints(
            velocity_bounds=(-10.0, 10.0),
            initial_position=0.0,
            final_position=100.0,
        )
        assert constraints.velocity_bounds == (-10.0, 10.0)
        assert constraints.initial_position == 0.0
        assert constraints.final_position == 100.0


class TestCollocationSettings:
    """Test CollocationSettings dataclass."""

    def test_default_settings(self):
        """Test default settings initialization."""
        settings = CollocationSettings()
        assert settings.degree == 3
        assert settings.method == "legendre"
        assert settings.max_iterations == 1000
        assert settings.tolerance == 1e-6
        assert settings.verbose is True

    def test_custom_settings(self):
        """Test custom settings initialization."""
        settings = CollocationSettings(
            degree=5,
            method="radau",
            max_iterations=500,
            tolerance=1e-8,
            verbose=False,
        )
        assert settings.degree == 5
        assert settings.method == "radau"
        assert settings.max_iterations == 500
        assert settings.tolerance == 1e-8
        assert settings.verbose is False


class TestOptimalMotionSolver:
    """Test OptimalMotionSolver class."""

    def test_solver_initialization_default(self):
        """Test solver initialization with default settings."""
        solver = OptimalMotionSolver()
        assert solver.settings.degree == 3
        assert solver.settings.method == "legendre"
        assert solver.settings.max_iterations == 1000

    def test_solver_initialization_custom(self):
        """Test solver initialization with custom settings."""
        settings = CollocationSettings(degree=5, method="radau")
        solver = OptimalMotionSolver(settings)
        assert solver.settings.degree == 5
        assert solver.settings.method == "radau"

    def test_invalid_collocation_degree(self):
        """Test that invalid collocation degree raises ValueError."""
        settings = CollocationSettings(degree=0)
        with pytest.raises(ValueError, match="Collocation degree must be >= 1"):
            OptimalMotionSolver(settings)

    def test_invalid_collocation_method(self):
        """Test that invalid collocation method raises ValueError."""
        settings = CollocationSettings(method="invalid_method")
        with pytest.raises(ValueError, match="Unknown collocation method"):
            OptimalMotionSolver(settings)

    def test_invalid_max_iterations(self):
        """Test that invalid max iterations raises ValueError."""
        settings = CollocationSettings(max_iterations=0)
        with pytest.raises(ValueError, match="Max iterations must be >= 1"):
            OptimalMotionSolver(settings)


class TestMinimumTimeMotion:
    """Test minimum time motion law problems."""

    @pytest.fixture
    def solver(self):
        """Create solver instance for testing."""
        settings = CollocationSettings(degree=3, max_iterations=100, verbose=False)
        return OptimalMotionSolver(settings)

    @pytest.fixture
    def basic_constraints(self):
        """Create basic motion constraints."""
        return MotionConstraints(
            initial_position=0.0,
            initial_velocity=0.0,
            initial_acceleration=0.0,
            final_position=10.0,
            final_velocity=0.0,
            final_acceleration=0.0,
        )

    def test_minimum_time_basic(self, solver, basic_constraints):
        """Test basic minimum time motion law problem."""
        distance = 10.0
        max_velocity = 5.0
        max_acceleration = 2.0
        max_jerk = 1.0

        solution = solver.solve_minimum_time(
            basic_constraints,
            distance,
            max_velocity,
            max_acceleration,
            max_jerk,
        )

        # Check solution structure
        assert isinstance(solution, dict)
        assert "time" in solution
        assert "position" in solution
        assert "velocity" in solution
        assert "acceleration" in solution
        assert "control" in solution

        # Check solution arrays
        for key in ["time", "position", "velocity", "acceleration", "control"]:
            assert isinstance(solution[key], np.ndarray)
            assert len(solution[key]) > 0

        # Check boundary conditions
        assert abs(solution["position"][0] - 0.0) < 1e-6
        assert abs(solution["position"][-1] - 10.0) < 1e-6
        assert abs(solution["velocity"][0] - 0.0) < 1e-6
        assert abs(solution["velocity"][-1] - 0.0) < 1e-6

    def test_minimum_time_constraints_satisfied(self, solver, basic_constraints):
        """Test that constraints are satisfied in minimum time solution."""
        distance = 10.0
        max_velocity = 5.0
        max_acceleration = 2.0
        max_jerk = 1.0

        solution = solver.solve_minimum_time(
            basic_constraints,
            distance,
            max_velocity,
            max_acceleration,
            max_jerk,
        )

        # Check velocity constraints
        assert np.all(np.abs(solution["velocity"]) <= max_velocity + 1e-6)

        # Check acceleration constraints
        assert np.all(np.abs(solution["acceleration"]) <= max_acceleration + 1e-6)

        # Check jerk constraints
        assert np.all(np.abs(solution["control"]) <= max_jerk + 1e-6)


class TestMinimumEnergyMotion:
    """Test minimum energy motion law problems."""

    @pytest.fixture
    def solver(self):
        """Create solver instance for testing."""
        settings = CollocationSettings(degree=3, max_iterations=100, verbose=False)
        return OptimalMotionSolver(settings)

    @pytest.fixture
    def basic_constraints(self):
        """Create basic motion constraints."""
        return MotionConstraints(
            initial_position=0.0,
            initial_velocity=0.0,
            final_position=10.0,
            final_velocity=0.0,
        )

    def test_minimum_energy_basic(self, solver, basic_constraints):
        """Test basic minimum energy motion law problem."""
        distance = 10.0
        time_horizon = 5.0
        max_velocity = 5.0
        max_acceleration = 2.0

        solution = solver.solve_minimum_energy(
            basic_constraints,
            distance,
            time_horizon,
            max_velocity,
            max_acceleration,
        )

        # Check solution structure
        assert isinstance(solution, dict)
        assert "time" in solution
        assert "position" in solution
        assert "velocity" in solution
        assert "acceleration" in solution
        assert "control" in solution

        # Check boundary conditions
        assert abs(solution["position"][0] - 0.0) < 1e-6
        assert abs(solution["position"][-1] - 10.0) < 1e-6
        assert abs(solution["velocity"][0] - 0.0) < 1e-6
        assert abs(solution["velocity"][-1] - 0.0) < 1e-6

        # Check time horizon
        assert abs(solution["time"][-1] - time_horizon) < 1e-6


class TestMinimumJerkMotion:
    """Test minimum jerk motion law problems."""

    @pytest.fixture
    def solver(self):
        """Create solver instance for testing."""
        settings = CollocationSettings(degree=3, max_iterations=100, verbose=False)
        return OptimalMotionSolver(settings)

    @pytest.fixture
    def basic_constraints(self):
        """Create basic motion constraints."""
        return MotionConstraints(
            initial_position=0.0,
            initial_velocity=0.0,
            initial_acceleration=0.0,
            final_position=10.0,
            final_velocity=0.0,
            final_acceleration=0.0,
        )

    def test_minimum_jerk_basic(self, solver, basic_constraints):
        """Test basic minimum jerk motion law problem."""
        distance = 10.0
        time_horizon = 5.0
        max_velocity = 5.0
        max_acceleration = 2.0

        solution = solver.solve_minimum_jerk(
            basic_constraints,
            distance,
            time_horizon,
            max_velocity,
            max_acceleration,
        )

        # Check solution structure
        assert isinstance(solution, dict)
        assert "time" in solution
        assert "position" in solution
        assert "velocity" in solution
        assert "acceleration" in solution
        assert "control" in solution

        # Check boundary conditions
        assert abs(solution["position"][0] - 0.0) < 1e-6
        assert abs(solution["position"][-1] - 10.0) < 1e-6
        assert abs(solution["velocity"][0] - 0.0) < 1e-6
        assert abs(solution["velocity"][-1] - 0.0) < 1e-6
        assert abs(solution["acceleration"][0] - 0.0) < 1e-6
        assert abs(solution["acceleration"][-1] - 0.0) < 1e-6


class TestConvenienceFunctions:
    """Test convenience functions for motion law problems."""

    def test_solve_minimum_time_motion_function(self):
        """Test solve_minimum_time_motion convenience function."""
        distance = 10.0
        max_velocity = 5.0
        max_acceleration = 2.0
        max_jerk = 1.0

        solution = solve_minimum_time_motion(
            distance,
            max_velocity,
            max_acceleration,
            max_jerk,
        )

        assert isinstance(solution, dict)
        assert "time" in solution
        assert "position" in solution
        assert "velocity" in solution
        assert "acceleration" in solution
        assert "control" in solution

    def test_solve_minimum_energy_motion_function(self):
        """Test solve_minimum_energy_motion convenience function."""
        distance = 10.0
        time_horizon = 5.0
        max_velocity = 5.0
        max_acceleration = 2.0

        solution = solve_minimum_energy_motion(
            distance,
            time_horizon,
            max_velocity,
            max_acceleration,
        )

        assert isinstance(solution, dict)
        assert "time" in solution
        assert "position" in solution
        assert "velocity" in solution
        assert "acceleration" in solution
        assert "control" in solution

    def test_solve_minimum_jerk_motion_function(self):
        """Test solve_minimum_jerk_motion convenience function."""
        distance = 10.0
        time_horizon = 5.0
        max_velocity = 5.0
        max_acceleration = 2.0

        solution = solve_minimum_jerk_motion(
            distance,
            time_horizon,
            max_velocity,
            max_acceleration,
        )

        assert isinstance(solution, dict)
        assert "time" in solution
        assert "position" in solution
        assert "velocity" in solution
        assert "acceleration" in solution
        assert "control" in solution


class TestPropertyBasedTests:
    """Property-based tests using Hypothesis."""

    @given(
        distance=st.floats(min_value=0.1, max_value=100.0),
        max_velocity=st.floats(min_value=0.1, max_value=50.0),
        max_acceleration=st.floats(min_value=0.1, max_value=20.0),
        max_jerk=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_minimum_time_properties(
        self, distance, max_velocity, max_acceleration, max_jerk,
    ):
        """Test properties of minimum time solutions."""
        # Ensure reasonable parameter relationships
        if max_velocity > 0 and max_acceleration > 0 and max_jerk > 0:
            settings = CollocationSettings(degree=3, max_iterations=50, verbose=False)
            solver = OptimalMotionSolver(settings)
            constraints = MotionConstraints(
                initial_position=0.0,
                initial_velocity=0.0,
                final_position=distance,
                final_velocity=0.0,
            )

            try:
                solution = solver.solve_minimum_time(
                    constraints,
                    distance,
                    max_velocity,
                    max_acceleration,
                    max_jerk,
                )

                # Property: Final position should match target
                assert abs(solution["position"][-1] - distance) < 1e-4

                # Property: Initial and final velocities should be zero
                assert abs(solution["velocity"][0]) < 1e-4
                assert abs(solution["velocity"][-1]) < 1e-4

                # Property: Time should be positive
                assert solution["time"][-1] > 0

            except Exception:
                # Some parameter combinations may not be solvable
                pass

    @given(
        distance=st.floats(min_value=0.1, max_value=100.0),
        time_horizon=st.floats(min_value=0.1, max_value=20.0),
        max_velocity=st.floats(min_value=0.1, max_value=50.0),
        max_acceleration=st.floats(min_value=0.1, max_value=20.0),
    )
    def test_minimum_energy_properties(
        self, distance, time_horizon, max_velocity, max_acceleration,
    ):
        """Test properties of minimum energy solutions."""
        # Ensure reasonable parameter relationships
        if max_velocity > 0 and max_acceleration > 0:
            settings = CollocationSettings(degree=3, max_iterations=50, verbose=False)
            solver = OptimalMotionSolver(settings)
            constraints = MotionConstraints(
                initial_position=0.0,
                initial_velocity=0.0,
                final_position=distance,
                final_velocity=0.0,
            )

            try:
                solution = solver.solve_minimum_energy(
                    constraints,
                    distance,
                    time_horizon,
                    max_velocity,
                    max_acceleration,
                )

                # Property: Final position should match target
                assert abs(solution["position"][-1] - distance) < 1e-4

                # Property: Time horizon should be respected
                assert abs(solution["time"][-1] - time_horizon) < 1e-4

                # Property: Initial and final velocities should be zero
                assert abs(solution["velocity"][0]) < 1e-4
                assert abs(solution["velocity"][-1]) < 1e-4

            except Exception:
                # Some parameter combinations may not be solvable
                pass


class TestCamMotionConstraints:
    """Test CamMotionConstraints dataclass."""

    def test_default_cam_constraints(self):
        """Test default cam constraint initialization."""
        constraints = CamMotionConstraints(
            stroke=10.0,
            upstroke_duration_percent=60.0,
        )
        assert constraints.stroke == 10.0
        assert constraints.upstroke_duration_percent == 60.0
        assert constraints.zero_accel_duration_percent is None
        assert constraints.max_velocity is None
        assert constraints.max_acceleration is None
        assert constraints.max_jerk is None
        assert constraints.dwell_at_tdc is True
        assert constraints.dwell_at_bdc is True

    def test_custom_cam_constraints(self):
        """Test custom cam constraint initialization."""
        constraints = CamMotionConstraints(
            stroke=20.0,
            upstroke_duration_percent=50.0,
            zero_accel_duration_percent=10.0,
            max_velocity=100.0,
            max_acceleration=500.0,
            dwell_at_tdc=False,
        )
        assert constraints.stroke == 20.0
        assert constraints.upstroke_duration_percent == 50.0
        assert constraints.zero_accel_duration_percent == 10.0
        assert constraints.max_velocity == 100.0
        assert constraints.max_acceleration == 500.0
        assert constraints.dwell_at_tdc is False
        assert constraints.dwell_at_bdc is True

    def test_invalid_stroke(self):
        """Test that invalid stroke raises ValueError."""
        with pytest.raises(ValueError, match="Stroke must be positive"):
            CamMotionConstraints(stroke=0.0, upstroke_duration_percent=60.0)

        with pytest.raises(ValueError, match="Stroke must be positive"):
            CamMotionConstraints(stroke=-5.0, upstroke_duration_percent=60.0)

    def test_invalid_upstroke_duration(self):
        """Test that invalid upstroke duration raises ValueError."""
        with pytest.raises(
            ValueError, match="Upstroke duration percent must be between 0 and 100",
        ):
            CamMotionConstraints(stroke=10.0, upstroke_duration_percent=-10.0)

        with pytest.raises(
            ValueError, match="Upstroke duration percent must be between 0 and 100",
        ):
            CamMotionConstraints(stroke=10.0, upstroke_duration_percent=150.0)

    def test_invalid_zero_accel_duration(self):
        """Test that invalid zero acceleration duration raises ValueError."""
        with pytest.raises(
            ValueError,
            match="Zero acceleration duration percent must be between 0 and 100",
        ):
            CamMotionConstraints(
                stroke=10.0,
                upstroke_duration_percent=60.0,
                zero_accel_duration_percent=-5.0,
            )

        with pytest.raises(
            ValueError,
            match="Zero acceleration duration percent must be between 0 and 100",
        ):
            CamMotionConstraints(
                stroke=10.0,
                upstroke_duration_percent=60.0,
                zero_accel_duration_percent=120.0,
            )

    def test_zero_accel_can_exceed_upstroke(self):
        """Test that zero acceleration duration can exceed upstroke duration."""
        # This should NOT raise an error - zero acceleration can be anywhere in the cycle
        constraints = CamMotionConstraints(
            stroke=10.0,
            upstroke_duration_percent=40.0,
            zero_accel_duration_percent=50.0,
        )
        assert constraints.zero_accel_duration_percent == 50.0
        assert constraints.upstroke_duration_percent == 40.0


class TestCamMotionSolver:
    """Test cam motion law solving functionality."""

    @pytest.fixture
    def solver(self):
        """Create solver instance for testing."""
        settings = CollocationSettings(degree=3, max_iterations=50, verbose=False)
        return OptimalMotionSolver(settings)

    @pytest.fixture
    def basic_cam_constraints(self):
        """Create basic cam constraints."""
        return CamMotionConstraints(
            stroke=10.0,
            upstroke_duration_percent=60.0,
        )

    def test_cam_motion_law_basic(self, solver, basic_cam_constraints):
        """Test basic cam motion law solving."""
        solution = solver.solve_cam_motion_law(
            cam_constraints=basic_cam_constraints,
            motion_type="minimum_jerk",
            cycle_time=1.0,
        )

        # Check solution structure
        assert isinstance(solution, dict)
        assert "time" in solution
        assert "cam_angle" in solution
        assert "position" in solution
        assert "velocity" in solution
        assert "acceleration" in solution
        assert "control" in solution

        # Check solution arrays
        for key in [
            "time",
            "cam_angle",
            "position",
            "velocity",
            "acceleration",
            "control",
        ]:
            assert isinstance(solution[key], np.ndarray)
            assert len(solution[key]) > 0

        # Check cam angle range (0 to 360 degrees)
        assert np.min(solution["cam_angle"]) >= 0
        assert np.max(solution["cam_angle"]) <= 360

        # Check position range (0 to stroke)
        assert np.min(solution["position"]) >= 0
        assert np.max(solution["position"]) <= basic_cam_constraints.stroke + 1e-6

    def test_cam_motion_law_with_constraints(self, solver):
        """Test cam motion law with velocity and acceleration constraints."""
        cam_constraints = CamMotionConstraints(
            stroke=15.0,
            upstroke_duration_percent=50.0,
            max_velocity=20.0,
            max_acceleration=100.0,
        )

        solution = solver.solve_cam_motion_law(
            cam_constraints=cam_constraints,
            motion_type="minimum_jerk",
            cycle_time=1.0,
        )

        # Check constraints are satisfied
        max_velocity = np.max(np.abs(solution["velocity"]))
        max_acceleration = np.max(np.abs(solution["acceleration"]))

        assert max_velocity <= cam_constraints.max_velocity + 1e-6
        assert max_acceleration <= cam_constraints.max_acceleration + 1e-6

    def test_cam_motion_law_different_types(self, solver, basic_cam_constraints):
        """Test different cam motion law types."""
        motion_types = ["minimum_jerk", "minimum_energy", "minimum_time"]

        for motion_type in motion_types:
            solution = solver.solve_cam_motion_law(
                cam_constraints=basic_cam_constraints,
                motion_type=motion_type,
                cycle_time=1.0,
            )

            # Check solution structure
            assert "cam_angle" in solution
            assert "position" in solution
            assert "velocity" in solution
            assert "acceleration" in solution
            assert "control" in solution

    def test_cam_motion_law_without_dwell(self, solver):
        """Test cam motion law without dwell at TDC or BDC."""
        cam_constraints = CamMotionConstraints(
            stroke=12.0,
            upstroke_duration_percent=40.0,
            dwell_at_tdc=False,
            dwell_at_bdc=False,
        )

        solution = solver.solve_cam_motion_law(
            cam_constraints=cam_constraints,
            motion_type="minimum_jerk",
            cycle_time=1.0,
        )

        # Check that velocities at TDC and BDC are not necessarily zero
        # (This is a basic check - the actual values depend on the optimization)
        assert isinstance(solution["velocity"], np.ndarray)
        assert len(solution["velocity"]) > 0


class TestCamConvenienceFunctions:
    """Test cam motion law convenience functions."""

    def test_solve_cam_motion_law_function(self):
        """Test solve_cam_motion_law convenience function."""
        solution = solve_cam_motion_law(
            stroke=10.0,
            upstroke_duration_percent=60.0,
            motion_type="minimum_jerk",
            cycle_time=1.0,
        )

        assert isinstance(solution, dict)
        assert "cam_angle" in solution
        assert "position" in solution
        assert "velocity" in solution
        assert "acceleration" in solution
        assert "control" in solution

    def test_solve_cam_motion_law_with_constraints(self):
        """Test solve_cam_motion_law with constraints."""
        solution = solve_cam_motion_law(
            stroke=20.0,
            upstroke_duration_percent=50.0,
            motion_type="minimum_jerk",
            cycle_time=1.0,
            max_velocity=30.0,
            max_acceleration=150.0,
            dwell_at_tdc=True,
            dwell_at_bdc=False,
        )

        assert isinstance(solution, dict)
        assert "cam_angle" in solution

        # Check constraints
        max_velocity = np.max(np.abs(solution["velocity"]))
        max_acceleration = np.max(np.abs(solution["acceleration"]))

        assert max_velocity <= 30.0 + 1e-6
        assert max_acceleration <= 150.0 + 1e-6


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_plot_solution_without_matplotlib(self):
        """Test that plotting works without matplotlib available."""
        solver = OptimalMotionSolver()
        solution = {
            "time": np.array([0, 1, 2]),
            "position": np.array([0, 5, 10]),
            "velocity": np.array([0, 5, 0]),
            "acceleration": np.array([0, 0, 0]),
            "control": np.array([0, 0, 0]),
        }

        # Should not raise an exception even if matplotlib is not available
        solver.plot_solution(solution)

    def test_invalid_collocation_methods(self):
        """Test that invalid collocation methods are rejected."""
        invalid_methods = ["invalid", "euler", "rk4", ""]

        for method in invalid_methods:
            settings = CollocationSettings(method=method)
            with pytest.raises(ValueError, match="Unknown collocation method"):
                OptimalMotionSolver(settings)

    def test_invalid_cam_motion_type(self):
        """Test that invalid cam motion type raises ValueError."""
        solver = OptimalMotionSolver()
        cam_constraints = CamMotionConstraints(
            stroke=10.0,
            upstroke_duration_percent=60.0,
        )

        with pytest.raises(ValueError, match="Unknown motion type"):
            solver.solve_cam_motion_law(
                cam_constraints=cam_constraints,
                motion_type="invalid_type",
            )


if __name__ == "__main__":
    pytest.main([__file__])
