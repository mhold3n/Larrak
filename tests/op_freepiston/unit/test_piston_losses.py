from __future__ import annotations

from campro.freepiston.core.losses import coulomb_friction
from campro.freepiston.core.piston import clearance_penalty


def test_clearance_penalty_zero_above_gap():
    assert clearance_penalty(gap=0.002, gap_min=0.001, k=1e7) == 0.0


def test_clearance_penalty_positive_below_gap():
    # If gap < gap_min, penalty force is positive (repulsive)
    f = clearance_penalty(gap=0.0005, gap_min=0.001, k=1e7)
    assert f > 0.0


def test_coulomb_friction_direction():
    muN = 100.0
    assert coulomb_friction(v=1.0, muN=muN) == -muN
    assert coulomb_friction(v=-2.0, muN=muN) == muN
    assert coulomb_friction(v=0.0, muN=muN) == 0.0
