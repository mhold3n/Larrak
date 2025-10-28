from __future__ import annotations

from campro.freepiston.core.xfer import heat_loss_rate, woschni_h


def test_woschni_positive_and_scaling():
    h1 = woschni_h(p=1.0e5, T=800.0, B=0.082, w=10.0)
    h2 = woschni_h(p=2.0e5, T=800.0, B=0.082, w=10.0)
    assert h1 > 0.0 and h2 > h1


def test_heat_loss_rate_linear_in_area_and_deltaT():
    h = 100.0
    A = 0.02
    Tw = 450.0
    T = 650.0
    q = heat_loss_rate(h=h, area=A, T=T, Tw=Tw)
    # doubling area doubles q
    q2 = heat_loss_rate(h=h, area=2.0 * A, T=T, Tw=Tw)
    assert abs(q2 - 2.0 * q) < 1e-12
