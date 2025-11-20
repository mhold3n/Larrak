#!/usr/bin/env python3
"""
Detect redundant constraints in the NLP formulation.

This script analyzes the constraint structure to identify redundant constraints,
particularly focusing on continuity constraints that may be redundant with
collocation residuals.
"""

import sys
import numpy as np
from campro.freepiston.opt.nlp import build_collocation_nlp


def analyze_constraint_redundancy(P: dict) -> dict:
    """Analyze constraints for redundancy."""
    print("Building NLP...")
    nlp, meta = build_collocation_nlp(P)
    
    # Get problem dimensions
    n_vars = meta.get("n_vars", 0)
    n_constraints = meta.get("n_constraints", 0)
    
    # Get bounds arrays
    lbg = meta.get("lbg", np.array([]))
    ubg = meta.get("ubg", np.array([]))
    
    # Count equality vs inequality constraints
    equality_mask = np.abs(lbg - ubg) < 1e-12
    n_equality = np.sum(equality_mask)
    n_inequality = n_constraints - n_equality
    
    # Get constraint groups
    constraint_groups = meta.get("constraint_groups", {})
    
    # Analyze equality constraints by group
    equality_by_group = {}
    for group_name, indices in constraint_groups.items():
        if len(indices) > 0:
            group_lbg = lbg[indices]
            group_ubg = ubg[indices]
            group_equality_mask = np.abs(group_lbg - group_ubg) < 1e-12
            n_eq = np.sum(group_equality_mask)
            equality_by_group[group_name] = n_eq
    
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
        "equality_by_group": equality_by_group,
        "constraint_groups": constraint_groups,
        "equality_mask": equality_mask,
    }


def detect_redundancies(results: dict) -> list:
    """Detect redundant constraints."""
    redundancies = []
    
    K = results["K"]
    C = results["C"]
    equality_by_group = results["equality_by_group"]
    
    # Check for continuity vs collocation redundancy
    n_colloc = equality_by_group.get("collocation_residuals", 0)
    n_continuity = equality_by_group.get("continuity", 0)
    
    if n_colloc > 0 and n_continuity > 0:
        # Collocation residuals: K × C × n_states
        # Continuity: K × n_states
        # If they match, they're likely redundant
        expected_colloc = K * C * 13  # 13 states per collocation point
        expected_continuity = K * 13  # 13 states per stage
        
        if n_colloc == expected_colloc and n_continuity == expected_continuity:
            redundancies.append({
                "type": "continuity_redundant_with_collocation",
                "description": "Continuity constraints are redundant with collocation residuals",
                "evidence": {
                    "collocation_residuals": n_colloc,
                    "continuity": n_continuity,
                    "both_enforce": "State continuity between stages",
                    "collocation_enforces": "state[k+1] = state[k] + integral(derivative)",
                    "continuity_enforces": "state[k+1] = state[k]",
                },
                "impact": {
                    "constraints_to_remove": n_continuity,
                    "dof_before": results["n_vars"] - results["n_equality"],
                    "dof_after": results["n_vars"] - (results["n_equality"] - n_continuity),
                },
                "recommendation": "Remove continuity constraints (lines 2270-2307 in nlp.py)",
            })
    
    return redundancies


def print_redundancy_report(results: dict, redundancies: list):
    """Print redundancy analysis report."""
    print("\n" + "=" * 80)
    print("CONSTRAINT REDUNDANCY ANALYSIS")
    print("=" * 80)
    
    print(f"\nProblem Dimensions:")
    print(f"  Variables: {results['n_vars']}")
    print(f"  Total constraints: {results['n_constraints']}")
    print(f"  Equality constraints: {results['n_equality']}")
    print(f"  Inequality constraints: {results['n_inequality']}")
    print(f"  Current DOF: {results['n_vars'] - results['n_equality']}")
    
    print(f"\nEquality Constraints by Group:")
    for group_name, count in sorted(results['equality_by_group'].items()):
        if count > 0:
            print(f"  {group_name:25s}: {count:4d} equality constraints")
    
    if redundancies:
        print(f"\n{'=' * 80}")
        print("REDUNDANCIES DETECTED")
        print("=" * 80)
        
        for i, redundancy in enumerate(redundancies, 1):
            print(f"\n{i}. {redundancy['type']}")
            print(f"   Description: {redundancy['description']}")
            print(f"   Evidence:")
            for key, value in redundancy['evidence'].items():
                print(f"     - {key}: {value}")
            print(f"   Impact:")
            print(f"     - Constraints to remove: {redundancy['impact']['constraints_to_remove']}")
            print(f"     - DOF before: {redundancy['impact']['dof_before']}")
            print(f"     - DOF after: {redundancy['impact']['dof_after']}")
            print(f"   Recommendation: {redundancy['recommendation']}")
    else:
        print(f"\n{'=' * 80}")
        print("NO REDUNDANCIES DETECTED")
        print("=" * 80)
        print("All constraints appear to be necessary.")


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
        results = analyze_constraint_redundancy(P)
        redundancies = detect_redundancies(results)
        print_redundancy_report(results, redundancies)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)







