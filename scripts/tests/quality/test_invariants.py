import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

# Strategy for valid kinematic parameters
kinematic_strategy = st.fixed_dictionaries(
    {
        "stroke": st.floats(min_value=0.01, max_value=0.1),
        "outer_diameter": st.floats(min_value=0.03, max_value=0.2),
        "theta_bdc": st.just(np.pi),
    }
)


from hypothesis import settings
from hypothesis import strategies as st


# Use the full solver marker because it uses CasADi (even if not solving)
@pytest.mark.solver
@settings(deadline=None)
@given(params=kinematic_strategy)
def test_kinematic_nlp_construction(params):
    """
    Invariant: The build step should succeed and produce a valid NLP structure
    for any reasonable set of kinematic parameters.
    """

    # Create configuration
    P = {
        "problem_type": "kinematic",
        "planet_ring": {
            "stroke": params["stroke"],
            "outer_diameter": params["outer_diameter"],
            "theta_bdc": params["theta_bdc"],
            "gen_mode": "spline",
            "ratios": (2.0, 2.0),
            "mean_ratio": 2.0,
            "r_drive": params["stroke"] / 2.0,
            "attachment_radius": 0.0,
        },
        "geometry": {
            "bore": 0.016,
            "stroke": params["stroke"],
            "mass": 0.04,
        },
        "bounds": {
            "r_planet_min": 0.001,
            "r_planet_max": params["outer_diameter"] / 2,
            "R_ring_min": 0.001,
            "R_ring_max": params["outer_diameter"] * 2,
            "xL_min": -0.1,
            "xL_max": 0.1,
        },
        "constraints": {},
        "num": {
            "K": 20,
            "C": 3,
            "cycle_time": 0.02,
        },
        "solver": {
            "ipopt": {
                "print_level": 0,
            },
        },
        "obj": {"type": "kinematic_tracking", "w": {"ratio_tracking": 1.0}},
        "auto_plot": False,
    }

    # Verify construction only to avoid segfaults in loop
    from campro.optimization.nlp import build_collocation_nlp

    # build_collocation_nlp returns (nlp, meta)
    # It might raise if P is invalid, which Hypothesis captures as a failure (good)
    nlp, meta = build_collocation_nlp(P)

    # Check structure
    assert isinstance(nlp, dict)
    assert "x" in nlp
    assert "f" in nlp
    assert "g" in nlp

    # Check dimensions
    assert nlp["x"].shape[0] > 0
    assert nlp["g"].shape[0] > 0

    # Return None for Hypothesis compatibility
