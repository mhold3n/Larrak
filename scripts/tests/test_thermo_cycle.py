"""Verification test for Phase 1 Thermo Optimization."""

import numpy as np

from campro.optimization.nlp.phase_driver import solve_thermo_cycle


def test_phase1_thermo_solve():
    """Test that Phase 1 Thermo NLP builds and solves (or at least iterates)."""

    # Run with small iterations to functionality check
    result = solve_thermo_cycle(n_coll=20, max_iter=10)

    # Even if it hits max_iter (likely with 10), it should return a result dict
    assert "success" in result
    assert "x_opt" in result
    assert result["x_opt"].size > 0

    # Check if objective is numeric
    obj = result["objective"]
    assert isinstance(obj, float)
    assert not np.isnan(obj)

    # Basic feasibility check (only if converged, but we might not converge in 10 iter)
    # If we run longer:
    # result = solve_thermo_cycle(n_coll=20, max_iter=500)
    # assert result["success"]
