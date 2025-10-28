from __future__ import annotations

from campro.freepiston.core.states import MechState
from campro.freepiston.zerod.cv import cv_residual


def test_energy_audit_no_flows_stationary():
    mech = MechState(x_L=0.05, v_L=0.0, x_R=0.15, v_R=0.0)
    gas = {"rho": 1.2, "E": 2.5e5, "p": 1.0e5, "T": 800.0}
    params = {
        "geom": {"B": 0.082, "Vc": 3.2e-5},
        "valves": {"Ain_max": 6.0e-4, "Aex_max": 6.0e-4},
        "flows": {"lift_in": 0.0, "lift_ex": 0.0},
        "xfer": {"Tw": 800.0},  # zero delta T
    }
    r = cv_residual(mech, gas, params)
    # With zero dV/dt, zero delta-T, and closed valves → dm_dt=0 and dU_dt≈0
    assert abs(r["dm_dt"]) < 1e-12
    assert abs(r["dU_dt"]) < 1e-8
