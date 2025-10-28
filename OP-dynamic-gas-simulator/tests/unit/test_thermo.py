from __future__ import annotations

from campro.freepiston.core.thermo import IdealMix


def test_ideal_mix_gas_constants_and_enthalpy():
    mix = IdealMix(gamma_ref=1.34, W_mix=0.02897)
    R, cp = mix.gas_constants()
    assert R > 0.0 and cp > R
    T1, T2 = 300.0, 800.0
    h1 = mix.h_T(T1)
    h2 = mix.h_T(T2)
    # For constant cp, delta h ~ cp*(T2-T1)
    assert abs((h2 - h1) - cp * (T2 - T1)) < 1e-9
