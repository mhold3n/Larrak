import importlib

import pytest


def casadi_available() -> bool:
    try:
        importlib.import_module("casadi")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not casadi_available(), reason="CasADi not available")
def test_ipopt_trivial_nlp_progress():
    import casadi as ca

    from campro.optimization.ipopt_factory import create_ipopt_solver

    x = ca.SX.sym("x")
    nlp = {"x": x, "f": (x - 1) ** 2}
    solver = create_ipopt_solver(
        "trivial",
        nlp,
        {"ipopt.max_iter": 30, "ipopt.tol": 1e-8, "ipopt.print_level": 0},
        linear_solver="ma27",
    )
    r = solver(x0=0.0)
    x_opt = float(r["x"])
    assert abs(x_opt - 1.0) < 1e-6
