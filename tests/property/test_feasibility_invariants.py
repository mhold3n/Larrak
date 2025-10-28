import importlib

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def casadi_available() -> bool:
    try:
        importlib.import_module("casadi")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not casadi_available(), reason="CasADi not available")
@settings(max_examples=10)
@given(
    stroke=st.floats(min_value=5.0, max_value=50.0),
    up=st.floats(min_value=10.0, max_value=90.0),
)
def test_feasibility_residual_small_for_lenient_bounds(stroke, up):
    # Lenient bounds should produce small residuals for a wide range of inputs
    from campro.diagnostics.feasibility import check_feasibility_nlp

    rep = check_feasibility_nlp(
        {
            "stroke": float(stroke),
            "cycle_time": 1.0,
            "upstroke_percent": float(up),
        },
        {
            "max_velocity": 1e6,
            "max_acceleration": 1e8,
            "max_jerk": 1e12,
        },
    )
    assert rep.max_violation < 1e-2
