from __future__ import annotations

import pytest

from campro.freepiston.opt.nlp import build_collocation_nlp


def test_build_nlp_meta_and_objective_terms():
    P = {
        "num": {"K": 3, "C": 1},
        "flow": {"mode": "0d", "gamma": 1.4},
        "geometry": {"bore": 0.08, "clearance_volume": 3.2e-5},
        "bounds": {"x_gap_min": 8e-4},
        "obj": {"w": {"eta_th": 1.0, "short_circuit": 2.0, "smooth": 1e-3}},
        "walls": {"dynamic": True, "capacitance": 10.0},
        "constraints": {"short_circuit_max": 0.5, "scavenging_min": 0.1},
        "timing": {"Ain_t_cm": 0.25, "Aex_t_cm": 0.75, "tol": 1e-2},
    }
    try:
        nlp, meta = build_collocation_nlp(P)
    except RuntimeError:
        pytest.skip("CasADi not available")
        return

    assert isinstance(nlp, dict) and "x" in nlp and "f" in nlp and "g" in nlp
    assert meta.get("K") == 3 and meta.get("C") == 1
    assert meta.get("flow_mode") == "0d"
    assert meta.get("dynamic_wall") is True
    assert meta.get("scavenging_states") is True
    assert meta.get("timing_states") is True
