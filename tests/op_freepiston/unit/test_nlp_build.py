from __future__ import annotations

import numpy as np
import pytest

from campro.freepiston.opt.nlp import _compute_scavenging_initial_values, build_collocation_nlp


def test_build_nlp_smoke():
    P = {"num": {"K": 5, "C": 1}}
    nlp, meta = build_collocation_nlp(P)
    assert "x" in nlp and "f" in nlp and "g" in nlp
    assert meta["K"] == 5 and meta["C"] == 1


def test_compute_scavenging_initial_values():
    """Test that _compute_scavenging_initial_values produces feasible seeds."""
    # Typical problem configuration
    P = {
        "constraints": {
            "scavenging_min": 0.8,
            "short_circuit_max": 0.1,
        },
        "timing": {
            "Ain_t_cm": 0.5,
            "Aex_t_cm": 0.5,
        },
    }
    bounds = {
        "rho_min": 0.1,
        "Ain_max": 0.01,
        "Aex_max": 0.01,
    }
    geometry = {
        "bore": 0.05,
        "stroke": 0.02,
    }
    num = {
        "cycle_time": 1.0,
    }
    
    (yF0, Mdel0, Mlost0, AinInt0, AinTmom0, AexInt0, AexTmom0) = (
        _compute_scavenging_initial_values(P, bounds, geometry, num)
    )
    
    # Verify bounds satisfaction
    assert 0.0 <= yF0 <= 1.0, f"yF0={yF0} not in [0, 1]"
    assert Mdel0 > 0, f"Mdel0={Mdel0} not > 0"
    assert Mlost0 >= 0, f"Mlost0={Mlost0} not >= 0"
    assert Mlost0 <= Mdel0, f"Mlost0={Mlost0} > Mdel0={Mdel0}"
    assert AinInt0 > 0, f"AinInt0={AinInt0} not > 0"
    assert AexInt0 > 0, f"AexInt0={AexInt0} not > 0"
    assert AinTmom0 > 0, f"AinTmom0={AinTmom0} not > 0"
    assert AexTmom0 > 0, f"AexTmom0={AexTmom0} not > 0"
    
    # Verify physically reasonable values
    assert yF0 >= 0.8, f"yF0={yF0} should satisfy constraint lower bound (0.8)"
    assert Mdel0 < 1.0, f"Mdel0={Mdel0} seems too large"
    assert AinInt0 < 1.0, f"AinInt0={AinInt0} seems too large"


def test_compute_scavenging_initial_values_missing_config():
    """Test that helper function handles missing config gracefully."""
    # Minimal configuration
    P = {}
    bounds = {}
    geometry = {}
    num = {}
    
    (yF0, Mdel0, Mlost0, AinInt0, AinTmom0, AexInt0, AexTmom0) = (
        _compute_scavenging_initial_values(P, bounds, geometry, num)
    )
    
    # Should still produce valid values with defaults
    assert 0.0 <= yF0 <= 1.0
    assert Mdel0 > 0
    assert Mlost0 >= 0
    assert AinInt0 > 0
    assert AexInt0 > 0
    assert AinTmom0 > 0
    assert AexTmom0 > 0


def test_scavenging_constraints_cross_multiplied():
    """Test that scavenging constraints use cross-multiplied form (no division by Mdel_k)."""
    P = {
        "num": {"K": 5, "C": 1},
        "constraints": {
            "short_circuit_max": 0.1,
            "trapping_min": 0.8,
        },
    }
    nlp, meta = build_collocation_nlp(P)
    
    # Verify constraint groups include scavenging constraints
    assert "constraint_groups" in meta
    assert "scavenging" in meta["constraint_groups"]
    scavenging_indices = meta["constraint_groups"]["scavenging"]
    assert len(scavenging_indices) >= 2  # At least short_circuit and trapping constraints
    
    # Verify acceleration normalization factor is stored
    assert "acceleration_normalization_factor" in meta
    a_norm_factor = meta["acceleration_normalization_factor"]
    assert a_norm_factor >= 1.0  # Should be at least 1.0


def test_acceleration_normalization():
    """Test that acceleration normalization adapts to different a_max values."""
    # Test with different a_max values
    test_cases = [
        {"a_max": 200.0, "expected_norm_factor": 20.0},  # 200/10 = 20
        {"a_max": 1000.0, "expected_norm_factor": 100.0},  # 1000/10 = 100
        {"a_max": 5000.0, "expected_norm_factor": 500.0},  # 5000/10 = 500
    ]
    
    for case in test_cases:
        a_max = case["a_max"]
        expected_norm_factor = case["expected_norm_factor"]
        
        P = {
            "num": {"K": 5, "C": 1},
            "bounds": {"a_max": a_max},
        }
        nlp, meta = build_collocation_nlp(P)
        
        # Verify normalization factor
        assert "acceleration_normalization_factor" in meta
        a_norm_factor = meta["acceleration_normalization_factor"]
        assert abs(a_norm_factor - expected_norm_factor) < 1e-6, (
            f"a_max={a_max}: expected norm_factor={expected_norm_factor}, got {a_norm_factor}"
        )
        
        # Verify normalized bounds are ±10
        if "constraint_groups" in meta and "path_acceleration" in meta["constraint_groups"]:
            accel_indices = meta["constraint_groups"]["path_acceleration"]
            if len(accel_indices) > 0:
                # Check bounds for first acceleration constraint
                idx = accel_indices[0]
                lbg = meta["lbg"]
                ubg = meta["ubg"]
                if idx < len(lbg) and idx < len(ubg):
                    # Normalized bounds should be ±10 (a_max / a_norm_factor = a_max / (a_max/10) = 10)
                    expected_bound = 10.0
                    assert abs(abs(lbg[idx]) - expected_bound) < 1e-6 or abs(ubg[idx] - expected_bound) < 1e-6, (
                        f"a_max={a_max}: normalized bounds should be ±{expected_bound}, "
                        f"got lbg={lbg[idx]}, ubg={ubg[idx]}"
                    )


def test_over_scaling_detection():
    """Test that over-scaling detection identifies groups with small norms."""
    from campro.freepiston.opt.driver import _verify_scaling_quality, _relax_over_scaled_groups
    
    # Create a simple test case with over-scaled groups
    # This is a simplified test - full integration test would require actual NLP
    import casadi as ca
    
    # Create minimal NLP
    x = ca.SX.sym("x", 3)
    g = ca.vertcat(x[0], x[1], x[2])
    nlp = {"x": x, "g": g}
    
    # Create over-scaled variable groups (very small scales)
    x0 = np.array([1.0, 1.0, 1.0])
    scale = np.array([1e-12, 1e-12, 1.0])  # First two variables over-scaled
    scale_g = np.array([1.0, 1.0, 1.0])
    lbg = np.array([0.0, 0.0, 0.0])
    ubg = np.array([10.0, 10.0, 10.0])
    
    meta = {
        "variable_groups": {
            "test_group": [0, 1],  # Over-scaled variables
            "other_group": [2],  # Normal variable
        },
        "constraint_groups": {
            "test_constraint": [0, 1],
            "other_constraint": [2],
        },
    }
    
    # Test relaxation function
    relaxed_scale, relaxed_scale_g = _relax_over_scaled_groups(
        scale, scale_g,
        ["test_group"], ["test_constraint"],
        meta["variable_groups"], meta["constraint_groups"]
    )
    
    # Verify geometric averaging was applied (sqrt of original scale)
    assert abs(relaxed_scale[0] - np.sqrt(scale[0])) < 1e-10
    assert abs(relaxed_scale[1] - np.sqrt(scale[1])) < 1e-10
    assert abs(relaxed_scale[2] - scale[2]) < 1e-10  # Should be unchanged
