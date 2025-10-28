from __future__ import annotations

from campro.freepiston.opt.nlp import build_collocation_nlp


def test_build_nlp_1d_smoke_tiny_grid():
    P = {
        "num": {"K": 1, "C": 1},
        "flow": {"mode": "1d", "mesh_cells": 4},
        "geom": {"B": 0.082, "Vc": 3.2e-5},
        "bounds": {
            "Ain_max": 1e-3,
            "Aex_max": 1e-3,
            "T_min": 250.0,
            "T_max": 2000.0,
            "rho_min": 0.1,
            "rho_max": 10.0,
            "xL_min": -0.05,
            "xL_max": 0.05,
            "xR_min": 0.05,
            "xR_max": 0.20,
            "vL_min": -10.0,
            "vL_max": 10.0,
            "vR_min": -10.0,
            "vR_max": 10.0,
            "x_gap_min": 8.0e-4,
        },
    }
    nlp, meta = build_collocation_nlp(P)
    assert "x" in nlp and "f" in nlp and "g" in nlp
    assert meta["K"] == 1 and meta["C"] == 1
