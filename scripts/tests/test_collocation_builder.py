"""
Tests for CollocationBuilder.
"""

import casadi as ca
import numpy as np
from campro.optimization.framework.builder import CollocationBuilder

from examples.canonical_collocation import solve_double_integrator


def test_collocation_builder_simple_ode():
    """Test builder on x_dot = -x, x(0) = 1."""
    T = 1.0
    N = 10
    builder = CollocationBuilder(time_horizon=T, n_points=N)

    x = builder.add_state("x", bounds=(-2, 2), initial=1.0)

    def dynamics(states, controls):
        return {"x": -states["x"]}

    builder.set_dynamics(dynamics)
    builder.add_boundary_condition(lambda s, c: s["x"], 1.0, loc="initial")

    builder.build()

    from pathlib import Path

    log_path = Path("out/logs/debug_output.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        f.write(f"Number of constraints: {len(builder.g)}\n")
        f.write(f"C matrix:\n{builder.C}\n")
        f.write(f"D vector:\n{builder.D}\n")

        # Check if constraints are trivial
        for i, g_expr in enumerate(builder.g):
            f.write(f"g[{i}]: {g_expr}\n")

    # Evaluate g at w0
    g_func = ca.Function("g", [ca.vertcat(*builder.w)], [ca.vertcat(*builder.g)])
    g_val = g_func(builder.w0).full().flatten()
    print(f"Max constraint violation at w0: {np.max(np.abs(g_val))}")
    with open(log_path, "a") as f:
        f.write(f"Max constraint violation at w0: {np.max(np.abs(g_val))}\n")
        f.write(f"g_val: {g_val}\n")

    opts = {"expand": True, "print_time": True, "ipopt": {"print_level": 5}}
    results = builder.solve(opts)

    x_opt = results["x"]
    t_grid = builder.get_time_grid()

    # Analytical solution: x(t) = exp(-t)
    x_exact = np.exp(-t_grid)

    error = np.max(np.abs(x_opt - x_exact))
    assert error < 1e-4, f"Max error {error} too high"


def test_canonical_example():
    """Run the canonical example and verify results."""
    t, p, v, u = solve_double_integrator()

    assert p is not None

    # Check boundary conditions
    assert np.isclose(p[0], 0.0, atol=1e-5)
    assert np.isclose(v[0], 0.0, atol=1e-5)
    assert np.isclose(p[-1], 1.0, atol=1e-5)
    assert np.isclose(v[-1], 0.0, atol=1e-5)

    assert len(p) == 21  # N=20 -> 21 points
