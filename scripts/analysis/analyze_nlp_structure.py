#!/usr/bin/env python3
"""
Analyze NLP structure: count variables and constraints by type.

This script builds the NLP and analyzes its structure to understand
the relationship between variables and constraints.
"""

import sys

import numpy as np

from campro.optimization.nlp import build_collocation_nlp


def analyze_nlp_structure(P: dict) -> dict:
    """Build NLP and analyze its structure."""
    print("Building NLP...")
    nlp, meta = build_collocation_nlp(P)
    
    # Get problem dimensions
    n_vars = meta.get("n_vars", 0)
    n_constraints = meta.get("n_constraints", 0)
    
    # Get bounds arrays
    lbw = meta.get("lbw", np.array([]))
    ubw = meta.get("ubw", np.array([]))
    lbg = meta.get("lbg", np.array([]))
    ubg = meta.get("ubg", np.array([]))
    
    # Count equality vs inequality constraints
    equality_mask = np.abs(lbg - ubg) < 1e-12
    n_equality = np.sum(equality_mask)
    n_inequality = n_constraints - n_equality
    
    # Get variable and constraint groups
    var_groups = meta.get("variable_groups", {})
    constraint_groups = meta.get("constraint_groups", {})
    
    # Count variables by group
    var_counts = {}
    for group_name, indices in var_groups.items():
        var_counts[group_name] = len(indices)
    
    # Count constraints by group
    constraint_counts = {}
    for group_name, indices in constraint_groups.items():
        constraint_counts[group_name] = len(indices)
    
    # Get problem parameters
    K = meta.get("K", 0)
    C = meta.get("C", 1)
    
    return {
        "n_vars": n_vars,
        "n_constraints": n_constraints,
        "n_equality": n_equality,
        "n_inequality": n_inequality,
        "K": K,
        "C": C,
        "var_counts": var_counts,
        "constraint_counts": constraint_counts,
        "var_groups": var_groups,
        "constraint_groups": constraint_groups,
        "lbw": lbw,
        "ubw": ubw,
        "lbg": lbg,
        "ubg": ubg,
    }


def print_analysis(results: dict):
    """Print analysis results."""
    print("\n" + "=" * 80)
    print("NLP STRUCTURE ANALYSIS")
    print("=" * 80)
    
    print("\nProblem Dimensions:")
    print(f"  K (stages): {results['K']}")
    print(f"  C (collocation points per stage): {results['C']}")
    print(f"  Total variables: {results['n_vars']}")
    print(f"  Total constraints: {results['n_constraints']}")
    print(f"  Equality constraints: {results['n_equality']}")
    print(f"  Inequality constraints: {results['n_inequality']}")
    print(f"  Degrees of freedom: {results['n_vars'] - results['n_equality']}")
    
    print("\nVariable Counts by Group:")
    total_vars_accounted = 0
    for group_name, count in sorted(results['var_counts'].items()):
        print(f"  {group_name:20s}: {count:4d} variables")
        total_vars_accounted += count
    print(f"  {'Total accounted':20s}: {total_vars_accounted:4d} variables")
    if total_vars_accounted != results['n_vars']:
        print(f"  {'UNACCOUNTED':20s}: {results['n_vars'] - total_vars_accounted:4d} variables")
    
    print("\nConstraint Counts by Group:")
    total_cons_accounted = 0
    for group_name, count in sorted(results['constraint_counts'].items()):
        print(f"  {group_name:20s}: {count:4d} constraints")
        total_cons_accounted += count
    print(f"  {'Total accounted':20s}: {total_cons_accounted:4d} constraints")
    if total_cons_accounted != results['n_constraints']:
        print(f"  {'UNACCOUNTED':20s}: {results['n_constraints'] - total_cons_accounted:4d} constraints")
    
    # Analyze equality constraints by group
    print("\nEquality Constraints by Group:")
    lbg = results['lbg']
    ubg = results['ubg']
    for group_name, indices in sorted(results['constraint_groups'].items()):
        if len(indices) > 0:
            group_lbg = lbg[indices]
            group_ubg = ubg[indices]
            equality_mask = np.abs(group_lbg - group_ubg) < 1e-12
            n_eq = np.sum(equality_mask)
            n_ineq = len(indices) - n_eq
            print(f"  {group_name:20s}: {n_eq:4d} equality, {n_ineq:4d} inequality")
    
    # Detailed breakdown
    print("\nDetailed Constraint Breakdown:")
    print(f"  Collocation residuals: {results['constraint_counts'].get('collocation_residuals', 0)}")
    print(f"    Expected: K × C × n_states = {results['K']} × {results['C']} × ?")
    print(f"  Continuity: {results['constraint_counts'].get('continuity', 0)}")
    print(f"    Expected: K × n_states = {results['K']} × ?")
    print(f"  Periodicity: {results['constraint_counts'].get('periodicity', 0)}")
    print("    Expected: 4 (xL, xR, vL, vR)")


if __name__ == "__main__":
    # Default problem configuration (matches phase1 test)
    P = {
        "num": {"K": 90, "C": 1},
        "geometry": {
            "bore": 0.1,
            "stroke": 0.1,
            "clearance_volume": 1e-4,
        },
        "bounds": {
            "rho_min": 0.1,
            "rho_max": 10.0,
        },
        "combustion": {
            "use_integrated_model": True,
            "fuel_type": "gasoline",
            "afr": 18.0,
            "fuel_mass_kg": 1e-5,
            "cycle_time_s": 1.0,
            "initial_temperature_K": 300.0,
            "initial_pressure_Pa": 1e5,
            "ignition_timing": 0.005,
        },
        "flow": {"mode": "0d"},
        "obj": {},
        "walls": {},
    }
    
    try:
        results = analyze_nlp_structure(P)
        print_analysis(results)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)







