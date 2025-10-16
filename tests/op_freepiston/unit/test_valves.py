from __future__ import annotations

from campro.freepiston.core.valves import clamp_area, effective_area_linear


def test_effective_area_linear_and_clamp():
    Ain_max = 6.0e-4
    # Below zero lift
    assert effective_area_linear(lift=-0.001, A_max=Ain_max) == 0.0
    # Mid lift
    A_mid = effective_area_linear(lift=0.5, A_max=Ain_max)
    assert abs(A_mid - 0.5 * Ain_max) < 1e-12
    # Above normalized 1.0
    A_hi = effective_area_linear(lift=2.0, A_max=Ain_max)
    assert A_hi == Ain_max
    # Direct clamp
    assert clamp_area(2.0 * Ain_max, Ain_max) == Ain_max


