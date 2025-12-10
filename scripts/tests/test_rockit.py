
try:
    from rockit import FreeTime, MultipleShooting, Ocp

    print("Rockit import successful.")
except ImportError:
    print("Rockit NOT installed.")
    exit(1)


def test_rockit_simple():
    print("Running Rockit Smoke Test (Simple Integrator)...")

    # 1. Define OCP
    ocp = Ocp(t0=0, T=FreeTime(1.0))

    # 2. Define States and Controls
    x = ocp.state()
    u = ocp.control()

    # 3. Define Dynamics
    ocp.set_der(x, u)

    # 4. Constraints
    ocp.subject_to(x >= 0)
    ocp.subject_to(u >= -1)
    ocp.subject_to(u <= 1)

    # Boundary conditions
    ocp.subject_to(ocp.at_t0(x) == 0)
    ocp.subject_to(ocp.at_tf(x) == 1.0)

    # Objective: Minimize time
    ocp.add_objective(ocp.T)

    # 5. Solver
    ocp.solver("ipopt", {"ipopt": {"print_level": 0, "linear_solver": "ma57"}})

    # 6. Method
    ocp.method(MultipleShooting(N=20))

    # 7. Solve
    try:
        sol = ocp.solve()
        print("Solve successful!")
        print(f"Optimal Time: {sol.value(ocp.T):.4f}")
        print(f"Final State: {sol.value(ocp.at_tf(x)):.4f}")
    except Exception as e:
        print(f"Solve failed: {e}")


if __name__ == "__main__":
    test_rockit_simple()
