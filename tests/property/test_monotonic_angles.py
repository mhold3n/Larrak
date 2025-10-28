import numpy as np

from campro.optimization.motion_law import MotionLawConstraints, MotionType
from campro.optimization.motion_law_optimizer import MotionLawOptimizer


def test_cam_angle_monotonic():
    constraints = MotionLawConstraints(
        stroke=20.0,
        upstroke_duration_percent=60.0,
        zero_accel_duration_percent=0.0,
    )
    opt = MotionLawOptimizer()
    res = opt.solve_motion_law(constraints, MotionType.MINIMUM_JERK)
    theta = res.cam_angle
    # Strictly increasing sequence
    assert np.all(np.diff(theta) > 0)
