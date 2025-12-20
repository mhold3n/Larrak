"""
Canonical example of using the CollocationBuilder DSL.

Problem: Double Integrator (Minimum Energy)
    Minimize integral(u^2 dt)
    Subject to:
        p_dot = v
        v_dot = u
        p(0) = 0, v(0) = 0
        p(T) = 1, v(T) = 0
        -1 <= u <= 1
"""

from campro.optimization.framework.builder import CollocationBuilder

# from campro.diagnostics.collocation_checks import check_collocation_residuals

def solve_double_integrator():
    # 1. Initialize Builder (no Opti needed)
    T = 2.0
    N = 20
    builder = CollocationBuilder(time_horizon=T, n_points=N, method="radau", degree=3)

    # 2. Define Variables
    p = builder.add_state("p", bounds=(-2, 2), initial=0.0)
    v = builder.add_state("v", bounds=(-5, 5), initial=0.0)
    u = builder.add_control("u", bounds=(-10, 10), initial=0.0)

    # 3. Define Dynamics
    # f(x, u) -> x_dot
    def dynamics(x, u):
        return {
            "p": x["v"],
            "v": u["u"]
        }
    builder.set_dynamics(dynamics)

    # 4. Boundary Conditions
    builder.add_boundary_condition(lambda x, u: x["p"], 0.0, loc="initial")
    builder.add_boundary_condition(lambda x, u: x["v"], 0.0, loc="initial")
    builder.add_boundary_condition(lambda x, u: x["p"], 1.0, loc="final")
    builder.add_boundary_condition(lambda x, u: x["v"], 0.0, loc="final")

    # 5. Objective
    # Minimize integral of u^2
    # J = sum(u_k^2 * h)
    # We need to access the symbolic variables. builder._U["u"] gives list of symbols.
    # But we should do this before build() if we want to set it?
    # Actually, builder.build() creates the variables.
    # So we must call build() first?
    # Wait, builder.add_state returns the symbol used for dynamics signature, but not the grid variables.
    # The grid variables are created in build().
    # But we can set the objective using the symbols? No, objective must be function of w.
    
    # In the new design, build() populates self.w.
    # So we call build(), then set objective?
    # But build() might need objective if we were using Opti.
    # Here we manage lists. So we can set objective after build().
    
    builder.build()
    
    # Now we can access builder._U["u"] which contains symbols for each interval
    U_vals = builder._U["u"]
    J = 0
    for k in range(N):
        J += U_vals[k]**2 * builder.h
    
    builder.set_objective(J)

    # 6. Solve
    # Options for CasADi nlpsol
    opts = {
        "expand": True,
        "print_time": False,
        "ipopt": {"print_level": 5}
    }
    
    try:
        results = builder.solve(opts)
        print("Optimization successful!")
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None, None, None, None

    # 7. Extract Results
    t_grid = builder.get_time_grid()
    p_opt = results["p"]
    v_opt = results["v"]
    u_opt = results["u"]
    
    # Check residuals (needs update to work without Opti)
    # max_resid = check_collocation_residuals(builder)
    # print(f"Max Collocation Residual: {max_resid}")

    return t_grid, p_opt, v_opt, u_opt

if __name__ == "__main__":
    t, p, v, u = solve_double_integrator()
    
    if p is not None:
        # Simple validation
        print(f"Final Position: {p[-1]}")
        print(f"Final Velocity: {v[-1]}")
    
    # Plot if running interactively
    # plt.figure()
    # plt.subplot(311); plt.plot(t, p); plt.ylabel("Position")
    # plt.subplot(312); plt.plot(t, v); plt.ylabel("Velocity")
    # plt.subplot(313); plt.step(t[:-1], u, where='post'); plt.ylabel("Control")
    # plt.show()
