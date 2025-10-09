from __future__ import annotations

from campro.freepiston.net1d.flux import hllc_flux


def test_hllc_flux_basic():
    U_L = (1.0, 0.0, 2.5e5)
    U_R = (0.5, 0.0, 1.25e5)
    F = hllc_flux(U_L, U_R)
    assert len(F) == 3
    assert all(f >= 0.0 for f in F)
