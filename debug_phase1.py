import casadi as ca
import numpy as np


def main():
    # Simple NLP: min x^2 s.t. x >= 1
    x = ca.SX.sym("x")
    nlp = {"x": x, "f": x**2, "g": []}

    # Test 1: Nested Options (What we have now)
    opts_nested = {
        "expand": True,
        "print_time": False,
        "ipopt": {"print_level": 0, "max_iter": 100, "tol": 1e-4},
    }

    # Test 2: Flat Options (Standard)
    opts_flat = {
        "expand": True,
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.max_iter": 100,
        "ipopt.tol": 1e-4,
    }

    print("\n--- Testing Nested Options ---")
    try:
        solver = ca.nlpsol("solver", "ipopt", nlp, opts_nested)
        res = solver(x0=0, lbx=1)
        print("Status:", solver.stats()["return_status"])
    except Exception as e:
        print("Crash:", e)

    print("\n--- Testing Flat Options ---")
    try:
        solver = ca.nlpsol("solver", "ipopt", nlp, opts_flat)
        res = solver(x0=0, lbx=1)
        print("Status:", solver.stats()["return_status"])
    except Exception as e:
        print("Crash:", e)

    # Test 3: Flat with potentially invalid option 'acceptable_tol'
    # acceptable_tol is valid, but maybe not in all versions?
    opts_suspect = {
        "ipopt.print_level": 0,
        "ipopt.acceptable_tol": 1e-2,  # This was in the failing config
        "ipopt.linear_solver": "ma57",  # This requires HSL
    }
    print("\n--- Testing Suspect Options ---")
    try:
        solver = ca.nlpsol("solver", "ipopt", nlp, opts_suspect)
        res = solver(x0=0, lbx=1)
        print("Status:", solver.stats()["return_status"])
    except Exception as e:
        print("Crash:", e)


if __name__ == "__main__":
    main()
