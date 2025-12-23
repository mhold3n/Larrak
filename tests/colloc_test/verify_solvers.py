import casadi as ca
import sys
import os


def check_solver(solver_name):
    print(f"\n--- Testing Solver: {solver_name} ---")

    # Simple NLP from CasADi examples
    # min x^2 + 100*z^2 s.t. z + (1-x)^2 - y = 0
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    z = ca.SX.sym("z")
    nlp = {"x": ca.vertcat(x, y, z), "f": x**2 + 100 * z**2, "g": z + (1 - x) ** 2 - y}

    opts = {"ipopt.linear_solver": solver_name, "ipopt.print_level": 0, "print_time": False}

    try:
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # Try a solve
        res = solver(x0=[2.5, 3.0, 0.75], lbg=0, ubg=0)

        status = solver.stats()["return_status"]
        success = solver.stats()["success"]
        print(f"Result: {'PASS' if success else 'FAIL'} ({status})")
        return success
    except Exception as e:
        print(f"Result: FAIL (Exception: {str(e)})")
        return False


def main():
    print(f"CasADi Version: {ca.__version__}")
    print(f"Python Version: {sys.version}")
    print(f"Library Paths in Environment:")
    for path in os.environ.get("DYLD_LIBRARY_PATH", "").split(":"):
        print(f"  DYLD: {path}")
    for path in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        print(f"  LD:   {path}")

    solvers = ["mumps", "ma57", "ma86"]
    results = {}

    for s in solvers:
        results[s] = check_solver(s)

    print("\n--- Summary ---")
    for s, success in results.items():
        print(f"{s}: {'AVAILABLE' if success else 'MISSING/BROKEN'}")


if __name__ == "__main__":
    main()
