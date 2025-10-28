from __future__ import annotations

from campro.freepiston.opt.obj import smoothness_penalty


def test_smoothness_penalty_basic():
    # jerk-like squared accel integral over simple sequence
    acc = [0.0, 1.0, -1.0, 0.5]
    w = [0.25, 0.25, 0.25, 0.25]
    J = smoothness_penalty(accel=acc, weights=w)
    # non-negative and zero only if all accels are zero
    assert J >= 0.0
    assert smoothness_penalty(accel=[0.0, 0.0], weights=[0.5, 0.5]) == 0.0
