from __future__ import annotations

from campro.freepiston.zerod.cv import orifice_mdot


def test_orifice_mdot_simple_proportional():
    A = 1.0e-4
    Cd = 0.7
    rho = 1.2
    dp = 500.0  # Pa small delta
    md = orifice_mdot(A=A, Cd=Cd, rho=rho, dp=dp)
    assert md > 0.0
    # Halving area halves mass flow
    md2 = orifice_mdot(A=0.5 * A, Cd=Cd, rho=rho, dp=dp)
    assert abs(md2 - 0.5 * md) / md < 1e-12


