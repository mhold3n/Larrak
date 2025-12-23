"""
Diagnostic script to debug NLP feasibility using slack variables.
"""

import sys
import numpy as np
import casadi as ca
import datetime
from pathlib import Path

from thermo.nlp import build_thermo_nlp
from campro.logging import get_logger

log = get_logger(__name__)


def debug_feasibility_run():
    print("=== Feasibility Debug Run ===")

    # 1. Build NLP with debug_feasibility=True (adds slacks)
    print("Building NLP with slack variables...")
    nlp_dict, meta = build_thermo_nlp(n_coll=50, debug_feasibility=True)

    prob = {"x": nlp_dict["x"], "f": nlp_dict["f"], "g": nlp_dict["g"]}

    # 2. Solve
    print("Solving relaxed problem (min slack violation)...")
    opts = {"ipopt.max_iter": 500, "ipopt.print_level": 5, "ipopt.tol": 1e-4, "print_time": 0}
    solver = ca.nlpsol("solver", "ipopt", prob, opts)

    res = solver(x0=meta["w0"], lbx=meta["lbw"], ubx=meta["ubw"], lbg=meta["lbg"], ubg=meta["ubg"])

    w_opt = res["x"].full().flatten()
    status = solver.stats()["return_status"]
    print(f"Solver Status: {status}")
    print(f"Objective (Total Penalty): {res['f']}")

    # 3. Analyze Slacks
    print("\nAnalyzing constraint violations...")

    # Build reverse map for constraint groups
    # group_name -> [indices]
    idx_to_group = {}
    for group, indices in meta["constraint_groups"].items():
        for idx in indices:
            idx_to_group[idx] = group

    # Find slack variables in w
    # Names are not directly available in standard CasADi export unless simple syms
    # But usually creating SX symbols preserves names.
    # However, 'w' in export is a single Vertcat.

    # We can rely on the fact that slacks were appended to the END of w.
    # The original w had size N_orig.
    # New w has size N_total.
    # The new variables are slacks.
    # CollocationBuilder appends them: s_l_{i}, s_u_{i}.
    # Wait, we don't know the exact order or count without parsing names (which are lost in SX vertcat usually).
    # BUT, we can inspect constraint violations directly!
    # Even simpler: compute g(w_opt).

    # If we solved the relaxed problem, the constraints "lbg <= g(w) + sl - su <= ubg" are satisfied.
    # But we want to know the *magnitude* of sl/su which represents the violation of the ORIGINAL constraint.
    # Actually, simply checking the violation of the ORIGINAL constraints using the found w_opt
    # gives us the "unrelaxed" violation.
    # Wait, w_opt includes slacks.
    # The constraints in `nlp_dict` HAVE the slack terms added.
    # So `g(w_opt)` will satisfy the relaxed bounds.

    # We want to evaluate the ORIGINAL constraints.
    # OR, we can just recover the slack values from `w_opt`.
    # How to identify slack variables?
    # CollocationBuilder.w is a list of SX. The names might be preserved if we iterate the list before vertcat?
    # But here we only have the exported dict.

    # Strategy: Build a second NLP *without* relaxation to evaluate constraints?
    # Or rely on the fact that we know slacks are appended at the end.
    # But `CollocationBuilder` interweaves variables? No, `relax_constraints` appends to `self.w` AFTER everything else.
    # So all variables after `meta['n_vars_original']` are slacks.
    # Wait, `meta['n_vars']` is the NEW total.
    # We don't have n_vars_original.

    # Alternative strategy:
    # `w` symbolic variable has names? `nlp_dict['x']` is a symbolic expression.
    # `nlp_dict['x'].name()` usually gives "vertcat".
    # `ca.SX.sym` variables have names.

    # Let's inspect variable names if possible.
    # Or just use the fact that `relax_constraints` was called last.
    # The original build process adds states/controls.
    # `relax_constraints` adds slacks.
    # So we can look at the trailing variables.

    # Let's parse the variable names from the `nlp["x"]` symbols if possible.
    sx_vec = nlp_dict["x"]
    # Unrolling vertcat symbolic is hard.

    # Better approach:
    # Re-calculate the original constraints.
    # We have `build_thermo_nlp(..., debug_feasibility=False)`.
    # Let's build a "clean" NLP.
    nlp_clean, meta_clean = build_thermo_nlp(n_coll=50, debug_feasibility=False)

    # The `w_opt` from the relaxed problem has extra variables (slacks) at the end.
    # We can truncate `w_opt` to the size of `meta_clean['n_vars']`.
    n_clean = meta_clean["n_vars"]
    w_clean_opt = w_opt[:n_clean]

    # Evaluate clean constraints
    g_fn = ca.Function("g_clean", [nlp_clean["x"]], [nlp_clean["g"]])
    g_val = g_fn(w_clean_opt).full().flatten()

    # Check violations against clean bounds
    lbg = meta_clean["lbg"]
    ubg = meta_clean["ubg"]

    violations = []
    total_violation = 0.0

    for i in range(len(g_val)):
        val = g_val[i]
        lb = lbg[i]
        ub = ubg[i]

        viol = 0.0
        if val < lb - 1e-6:
            viol = lb - val
        elif val > ub + 1e-6:
            viol = val - ub

        if viol > 0:
            total_violation += viol
            # Identify group
            group = idx_to_group.get(i, "unknown")
            violations.append(
                {"index": i, "group": group, "violation": viol, "value": val, "bounds": (lb, ub)}
            )

    # Sort by violation
    violations.sort(key=lambda x: x["violation"], reverse=True)

    # Report Top Violations
    print(f"\nTotal Constraint Violation (L1): {total_violation:.6e}")
    print("Top 20 Violators:")
    print(f"{'Index':<6} | {'Group':<30} | {'Violation':<12} | {'Value':<12} | {'Bounds'}")
    print("-" * 90)
    for v in violations[:20]:
        print(
            f"{v['index']:<6} | {v['group']:<30} | {v['violation']:<12.4e} | {v['value']:<12.4e} | [{v['bounds'][0]:.2e}, {v['bounds'][1]:.2e}]"
        )

    # Group Summary
    print("\nViolation vs Group Summary:")
    group_stats = {}
    for v in violations:
        g = v["group"]
        if g not in group_stats:
            group_stats[g] = {"count": 0, "sum": 0.0, "max": 0.0}
        group_stats[g]["count"] += 1
        group_stats[g]["sum"] += v["violation"]
        group_stats[g]["max"] = max(group_stats[g]["max"], v["violation"])

    print(f"{'Group':<30} | {'Count':<6} | {'Sum':<12} | {'Max':<12}")
    print("-" * 65)
    for g, stats in sorted(group_stats.items(), key=lambda x: x[1]["sum"], reverse=True):
        print(f"{g:<30} | {stats['count']:<6} | {stats['sum']:<12.4e} | {stats['max']:<12.4e}")


if __name__ == "__main__":
    debug_feasibility_run()
