from __future__ import annotations

import math

from campro.freepiston.core.states import MechState
from campro.freepiston.zerod.cv import cv_residual, volume_from_pistons


def test_volume_from_pistons_basic():
    B = 0.082
    Vc = 3.2e-5
    x_L, x_R = 0.02, 0.16
    V = volume_from_pistons(B=B, Vc=Vc, x_L=x_L, x_R=x_R)
    A_p = math.pi * (B * 0.5) ** 2
    assert abs(V - (Vc + A_p * (x_R - x_L))) < 1e-9


def test_cv_residual_energy_only_no_flows():
    # Simple sanity: with no heat/flows and stationary pistons at fixed V,
    # residual should be zero for a steady state.
    mech = MechState(x_L=0.05, v_L=0.0, x_R=0.15, v_R=0.0)
    gas = {"rho": 1.2, "E": 2.5e5, "p": 1.0e5}
    params = {
        "geom": {"B": 0.082, "Vc": 3.2e-5},
        "xfer": {"h": 0.0},
        "flows": {"mdot_in": 0.0, "mdot_ex": 0.0},
    }
    r = cv_residual(mech, gas, params)
    assert isinstance(r, dict)
    # With no sources/sinks, residuals near zero
    assert abs(r["dm_dt"]) < 1e-12
    assert abs(r["dU_dt"]) < 1e-8
