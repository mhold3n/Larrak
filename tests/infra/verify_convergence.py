import casadi as ca
import numpy as np
from thermo.nlp import build_thermo_nlp
import time


def verify_convergence():
    print("=== Convergence Verification ===")
    print("Building NLP...")
    # Use default settings (Phase 1 standard)
    nlp, meta = build_thermo_nlp(n_coll=50)

    print(f"Variables: {meta['n_vars']}")
    print(f"Constraints: {meta['n_constraints']}")

    # Solver configuration
    opts = {
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-6,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.print_level": 5,
        "ipopt.linear_solver": "mumps",  # Use mumps for stability/compatibility
        "print_time": 0,
    }

    print("Initializing Solver...")
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    print("Solving...")
    t0 = time.time()
    try:
        res = solver(
            x0=meta["w0"], lbx=meta["lbw"], ubx=meta["ubw"], lbg=meta["lbg"], ubg=meta["ubg"]
        )
        status = solver.stats()["return_status"]
    except Exception as e:
        print(f"Solver crashed: {e}")
        return

    dt = time.time() - t0

    print("\n=== Results ===")
    print(f"Status: {status}")
    print(f"Time: {dt:.2f} s")
    print(f"Objective: {res['f']}")
    print(f"Iterations: {solver.stats()['iter_count']}")

    # Check max constraint violation (unscaled)
    g_val = res["g"].full().flatten()
    lbg = meta["lbg"]
    ubg = meta["ubg"]

    viol = 0.0
    for i in range(len(g_val)):
        v = g_val[i]
        lb = lbg[i]
        ub = ubg[i]
        viol = max(viol, lb - v, v - ub)

    print(f"Max Constraint Violation: {viol:.2e}")

    if status == "Solve_Succeeded":
        print("✅ SUCCESS: Optimization converged!")
    else:
        print("❌ FAILURE: Optimization did not converge.")


if __name__ == "__main__":
    verify_convergence()
