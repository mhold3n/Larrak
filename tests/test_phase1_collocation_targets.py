from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest
from numpy.typing import NDArray

from campro.optimization.motion_law import (
    MotionLawConstraints,
    MotionLawResult,
    MotionType,
)
from campro.optimization.motion_law_optimizer import MotionLawOptimizer

# cspell:ignore ndarray trapz allclose rtol pcurve imep dpdt


def _build_optimizer(**kwargs: Any) -> MotionLawOptimizer:
    """Return a lightweight optimizer tuned for deterministic pytest runs."""
    optimizer = MotionLawOptimizer()
    optimizer.configure(
        n_points=48,
        max_iterations=200,
        tolerance=1e-6,
        **kwargs,
    )
    return optimizer


def _s_curve(theta: NDArray[np.float64], upstroke_angle: float, stroke: float) -> NDArray[np.float64]:
    """Analytical S-curve used as the minimum-jerk golden profile."""
    tau = theta / max(upstroke_angle, 1e-9)
    poly = 6 * tau**5 - 15 * tau**4 + 10 * tau**3
    return (stroke * poly).astype(np.float64)


def _jerk_energy(result: MotionLawResult) -> float:
    """Integral of jerk^2 over the provided cam angle grid."""
    return float(np.trapz(result.jerk**2, result.cam_angle))


def _base_constraints(**overrides: float) -> MotionLawConstraints:
    params: dict[str, float] = {
        "stroke": 10.0,
        "upstroke_duration_percent": 50.0,
        "zero_accel_duration_percent": 0.0,
    }
    params.update(overrides)
    return MotionLawConstraints(**params)


def test_minimum_jerk_matches_polynomial_profile() -> None:
    optimizer = _build_optimizer()
    constraints = _base_constraints()

    result = optimizer.solve_motion_law(constraints, MotionType.MINIMUM_JERK)

    mask = result.cam_angle <= constraints.upstroke_angle + 1e-6
    expected = _s_curve(result.cam_angle[mask], constraints.upstroke_angle, constraints.stroke)
    np.testing.assert_allclose(result.position[mask], expected, rtol=5e-3, atol=5e-3)
    assert result.position.max() == pytest.approx(constraints.stroke, abs=1e-3)


def test_minimum_time_increases_peak_velocity_relative_to_minimum_jerk() -> None:
    optimizer = _build_optimizer()
    constraints = _base_constraints()

    min_jerk = optimizer.solve_motion_law(constraints, MotionType.MINIMUM_JERK)
    min_time = optimizer.solve_motion_law(constraints, MotionType.MINIMUM_TIME)

    assert min_time.convergence_status == "converged"
    assert min_time.velocity.max() > min_jerk.velocity.max()
    assert _jerk_energy(min_time) >= _jerk_energy(min_jerk)


def test_peak_acceleration_constraint_enforced() -> None:
    optimizer = _build_optimizer()
    limit = 35.0
    constraints = _base_constraints(max_acceleration=limit)

    limited = optimizer.solve_motion_law(constraints, MotionType.MINIMUM_TIME)
    assert np.max(np.abs(limited.acceleration)) <= limit + 1e-2


def test_pcurve_te_objective_drops_when_efficiency_weight_increases() -> None:
    constraints = _base_constraints()
    optimizer = _build_optimizer(weight_imep=0.0, weight_dpdt=0.0, weight_jerk=1.0)
    baseline = optimizer.solve_motion_law(constraints, MotionType.P_CURVE_TE)

    optimizer.configure(weight_imep=1.5, weight_dpdt=0.1, weight_jerk=0.2)
    efficiency_focused = optimizer.solve_motion_law(constraints, MotionType.P_CURVE_TE)

    assert efficiency_focused.objective_value < baseline.objective_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
