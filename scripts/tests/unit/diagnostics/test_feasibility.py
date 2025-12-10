import importlib

import pytest


def casadi_available() -> bool:
    try:
        importlib.import_module("casadi")
        return True
    except Exception:
        return False


@pytest.mark.skip(reason="Segfaults on macOS due to CasADi instability")
def test_feasibility_nlp_relaxed_is_small():
    from campro.diagnostics.feasibility import check_feasibility_nlp

    rep = check_feasibility_nlp(
        {
            "stroke": 20.0,
            "cycle_time": 1.0,
            "upstroke_percent": 60.0,
        },
        {
            "max_velocity": 1e6,
            "max_acceleration": 1e8,
            "max_jerk": 1e12,
        },
    )
    # We expect a small residual even if not strictly feasible due to discretization
    assert rep.max_violation < 1e-2
