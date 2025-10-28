from __future__ import annotations

import pytest

from campro.freepiston.gas import build_gas_model


def test_build_gas_model_modes():
    gm0 = build_gas_model({"flow": {"mode": "0d"}})
    assert gm0.mode == "0d"

    gm1 = build_gas_model({"flow": {"mode": "1d"}})
    assert gm1.mode == "1d"


def test_mdot_symbolic_smoke():
    try:
        import casadi as ca  # type: ignore
    except Exception:
        pytest.skip("CasADi not available")
        return

    gm = build_gas_model({"flow": {"mode": "0d"}})

    # Symbolic temperature; other inputs as numeric for simplicity
    T_c = ca.SX.sym("T_c")
    mdot = gm.mdot_in(
        ca=ca,
        p_up=1.2e5,
        T_up=300.0,
        rho_up=1.2,
        p_down=1.0e5,
        T_down=T_c,
        A_eff=0.01,
        gamma=1.4,
        R=287.0,
    )

    # Should be a CasADi expression
    assert hasattr(mdot, "is_symbolic") or "casadi" in str(type(mdot)).lower()
