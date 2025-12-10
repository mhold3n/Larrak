import logging

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

try:
    from campro_unaligned.freepiston.physics.casadi_1d import get_1d_derivatives
except ImportError:
    get_1d_derivatives = None  # type: ignore


log = logging.getLogger(__name__)


class ThermoSolver:
    """Base class for thermodynamic solvers."""

    def solve(self, geometry_profile: dict, params: dict) -> dict:
        raise NotImplementedError


class RungeKuttaShooter(ThermoSolver):
    """
    Solves for the thermodynamic Limit Cycle using RK45/Radau integration (Shooting Method).
    Decoupled from specific geometry classes - takes generic x(t), v(t) profile.
    """

    def __init__(self, method="Radau", rtol=1e-6, atol=1e-8):
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def solve(self, geometry_profile: dict, params: dict) -> dict:
        # Extract inputs
        t_grid = geometry_profile["t"]
        x_grid = geometry_profile["x"]
        v_grid = geometry_profile["v"]

        # Build interpolants for dynamics
        # (Linear or cubic spline interpolation of the input profile)
        from scipy.interpolate import interp1d

        x_func = interp1d(t_grid, x_grid, kind="cubic", fill_value="extrapolate")
        v_func = interp1d(t_grid, v_grid, kind="cubic", fill_value="extrapolate")

        cycle_time = t_grid[-1]

        # Thermo Params
        thermo = params.get("thermo", {})
        gamma = thermo.get("gamma", 1.4)
        R = thermo.get("R", 287.0)
        cv = R / (gamma - 1)

        # Geometry Params
        geom_params = params.get("geometry", {})
        bore = geom_params.get("bore", 0.1)
        area = np.pi * (bore / 2) ** 2

        # Initial guesses for finding limit cycle
        p0_guess = params.get("p0_guess", 1e5)
        T0_guess = params.get("T0_guess", 300.0)
        rho0_guess = p0_guess / (R * T0_guess)

        # Dynamics function
        def dynamics(t, y):
            # y = [rho, T]
            rho, T = y

            # Get kinematics at t
            # Handle t > cycle_time (periodicity)
            t_mod = t % cycle_time
            x_val = float(x_func(t_mod))
            v_val = float(v_func(t_mod))

            x_pos = abs(x_val)  # Ensure positive separation half-distance
            vol = area * 2.0 * x_pos
            vol = max(vol, 1e-9)

            dvol_dt = area * 2.0 * v_val

            # d(rho)/dt = -rho/V * dV/dt (Mass conservation, constant mass implies this?)
            # No, if open system (scavenging), it's complex.
            # For initialization, let's assume CLOSED system (Limit Cycle of compression/expansion)
            # d(rho)/dt = -rho * dV/dt / V
            drho_dt = -rho * dvol_dt / vol

            # d(T)/dt (Adiabatic)
            # T * V^(gamma-1) = const => T ~ V^(1-gamma)
            # dT/dt = - (gamma - 1) * T / V * dV/dt
            dT_dt = -(gamma - 1) * T * dvol_dt / vol

            return [drho_dt, dT_dt]

        # Root finding for Limit Cycle
        def residual(z):
            rho0, T0 = z
            sol = solve_ivp(
                dynamics,
                (0, cycle_time),
                [rho0, T0],
                method=self.method,
                rtol=self.rtol,
                atol=self.atol,
            )
            rho_end, T_end = sol.y[:, -1]
            return [rho_end - rho0, T_end - T0]

        z0 = [rho0_guess, T0_guess]
        res = root(residual, z0, method="hybr", tol=1e-3)

        if not res.success:
            log.warning(f"RungeKuttaShooter failed to find limit cycle: {res.message}")

        z_opt = res.x

        # Final Integration
        sol = solve_ivp(
            dynamics,
            (0, cycle_time),
            z_opt,
            method=self.method,
            dense_output=True,
            rtol=self.rtol,
            atol=self.atol,
        )

        t_sol = sol.t
        rho_sol = sol.y[0]
        T_sol = sol.y[1]

        # Calculate Pressure
        p_sol = rho_sol * R * T_sol

        return {
            "success": res.success,
            "t": t_sol,
            "rho": rho_sol,
            "T": T_sol,
            "p": p_sol,
            "sol": sol,  # Dense output for interpolation
            "method": "rk_shooting",
        }


class CasAdiCFDShooter(ThermoSolver):
    """
    Solves for the thermodynamic Limit Cycle using CasADi Integration of the FULL 1D CFD model.
    Uses 'shared physics' from campro_unaligned.freepiston.physics.casadi_1d.
    Force-driven geometry (input profile) -> Thermodynamics.
    """

    def __init__(self, n_cells=10):
        self.n_cells = n_cells

    def solve(self, geometry_profile: dict, params: dict) -> dict:
        t_grid = np.array(geometry_profile["t"])
        x_grid = np.array(geometry_profile["x"])
        v_grid = np.array(geometry_profile["v"])
        cycle_time = t_grid[-1]

        # Create Spline Interpolant for Geometry (Periodic)
        # CasADi BSpline needs strictly increasing knots.
        # We assume t_grid is sorted.
        t_knots = t_grid
        x_coeffs = x_grid
        v_coeffs = v_grid

        # Create lookup functions (interpolants)
        # 1D linear interpolation is simpler and robust for dense grid
        # or we use 'bspline'
        x_func = ca.interpolant("x_func", "linear", [t_knots], x_coeffs)
        v_func = ca.interpolant("v_func", "linear", [t_knots], v_coeffs)

        # Build CasADi Integrator with Shared Physics
        # States: [rho_0..N, u_0..N, E_0..N] (3*N states)
        # Note: xL, vL are inputs here, not states.

        rho_s = [ca.SX.sym(f"rho_{i}") for i in range(self.n_cells)]
        u_g_s = [ca.SX.sym(f"u_{i}") for i in range(self.n_cells)]
        E_s = [ca.SX.sym(f"E_{i}") for i in range(self.n_cells)]

        x_state = ca.vertcat(*rho_s, *u_g_s, *E_s)

        # Time is implicit in integrator, but we need it for lookup.
        # IDAS integrator supports 't' argument in equations?
        # Yes, 't' is time.

        t = ca.SX.sym("t")

        # Get kinematics from spline
        xL_val = x_func(t)
        vL_val = v_func(t)

        # Controls (Constant for now in initialization, or simple profiles)
        # We can reuse the simple sine profiles from physics_trajectory if we want
        # For now, let's assume simple constants or zero for robustness first,
        # OR implement the simple sine logic here too.
        # Better: assume 0.0 or simple fixed small opening to allow flow?
        # Actually, if valves are closed, we can't find a limit cycle with flow...
        # We NEED valve timing.
        # Let's implement the same simple timing logic as physics_trajectory using symbolic math.

        t_norm = t / cycle_time
        # Simple Sine Window
        # Intake: 0.4 - 0.6
        # Exhaust: 0.38 - 0.62
        pi = np.pi

        # Smooth window helper
        def smooth_window(t_n, t_s, t_e, w=0.02):
            return 0.5 * (ca.tanh((t_n - t_s) / w) - ca.tanh((t_n - t_e) / w))

        Ain_max = params.get("bounds", {}).get("Ain_max", 0.001)
        Aex_max = params.get("bounds", {}).get("Aex_max", 0.001)

        sin_in = ca.sin(pi * (t_norm - 0.4) / 0.2)
        Ain_val = ca.fmax(0, Ain_max * smooth_window(t_norm, 0.4, 0.6) * sin_in)

        sin_ex = ca.sin(pi * (t_norm - 0.38) / 0.24)
        Aex_val = ca.fmax(0, Aex_max * smooth_window(t_norm, 0.38, 0.62) * sin_ex)

        Q_comb_val = 0.0
        F_cam_dummy = 0.0  # Force calculation output is ignored for dynamics here

        # Call Physics Kernel
        geometry = params.get("geometry", {})
        flow_cfg = params.get("flow", {})
        viscosity_cfg = {"C_visc": 0.5, "C_lin": 0.05}  # Default

        # Use the kernel!
        if get_1d_derivatives is None:
            raise ImportError(
                "campro_unaligned module is required for CasAdiCFDShooter but was not found (likely archived)."
            )

        _, _, d_rho, d_u, d_E = get_1d_derivatives(
            xL_val,
            vL_val,
            rho_s,
            u_g_s,
            E_s,
            Ain_val,
            Aex_val,
            Q_comb_val,
            F_cam_dummy,
            geometry,
            flow_cfg,
            viscosity_cfg,
        )

        d_state = ca.vertcat(*d_rho, *d_u, *d_E)

        # Create Differential Equation dict
        ode = {"x": x_state, "t": t, "ode": d_state}

        # Create Integrator
        # 'idas' is good for stiff systems (CFD is stiff)
        opts = {
            "tf": cycle_time,
            "abstol": 1e-6,
            "reltol": 1e-6,
            "max_num_steps": 10000,
        }
        integrator = ca.integrator("cfd_integrator", "cvodes", ode, opts)

        # Simulation / Shooting Loop
        # Initial Guess (Uniform)
        p0 = 1e5
        T0 = 300.0
        R_gas = 287.0
        rho0 = p0 / (R_gas * T0)
        Cv = 718.0
        E0 = Cv * T0

        # Initial Vector: All cells uniform
        rho_vec = [rho0] * self.n_cells
        u_vec = [0.0] * self.n_cells
        E_vec = [E0] * self.n_cells
        x0_val = np.array(rho_vec + u_vec + E_vec)

        # Run for a few cycles to settle (Limit Cycle)
        # We can just chain integrations.
        n_cycles = 5
        x_curr = x0_val

        for i in range(n_cycles):
            res = integrator(x0=x_curr)
            x_curr = res["xf"].full().flatten()

        # Generate Dense Output for one final cycle
        # Getting dense output from CasADi integrator often requires setting grid
        # or we just rely on the 'solution' structure if we use a different interface.
        # 'integrator' usually returns just xf.
        # To get trajectory, we can make 'tf' a parameter or run in steps.
        # Running in steps is safer for obtaining the trace.

        N_steps = 100
        dt = cycle_time / N_steps
        t_span = np.linspace(0, cycle_time, N_steps + 1)

        # Create a step integrator or just loop
        opts_step = opts.copy()
        opts_step["tf"] = dt
        step_integrator = ca.integrator("step_int", "cvodes", ode, opts_step)

        trace_rho = []
        trace_u = []
        trace_E = []
        trace_t = []

        # Assuming x_curr is now the periodic start (limit cycle)
        x_trace = x_curr

        # Unpack initial
        rho_trace_row = x_trace[0 : self.n_cells]
        u_trace_row = x_trace[self.n_cells : 2 * self.n_cells]
        E_trace_row = x_trace[2 * self.n_cells :]

        trace_rho.append(rho_trace_row)
        trace_u.append(u_trace_row)
        trace_E.append(E_trace_row)
        trace_t.append(0.0)

        for i in range(N_steps):
            t_curr = t_span[i]
            # Need to shift time? IDAS 't' usually reset to 0?
            # CVODES resets time unless we manage state.
            # Actually, our 'ode' depends on absolute 't' for the spline lookups!
            # If step_integrator goes from 0 to dt, passing 't' to ode as 0..dt is wrong if we are at T=0.5.

            # Correction: We must pass 't' as a parameter or state?
            # Or construct a new integrator for the explicit interval? No.
            # Method: Add 'time' as a state dt/dt = 1 !
            # Then the integrator handles T automatically.

            # Let's rebuild the integrator with Time as state.
            pass  # See next block for refined implementation

        # --- Rebuild Integrator with Time State ---
        t_state = ca.SX.sym("t_state")
        x_state_aug = ca.vertcat(x_state, t_state)

        # ODE needs to replace generic 't' with 't_state'
        # Re-evaluate spline and controls using t_state
        xL_aug = x_func(t_state)
        vL_aug = v_func(t_state)
        t_norm_aug = t_state / cycle_time

        sin_in_aug = ca.sin(pi * (t_norm_aug - 0.4) / 0.2)
        Ain_aug = ca.fmax(0, Ain_max * smooth_window(t_norm_aug, 0.4, 0.6) * sin_in_aug)
        sin_ex_aug = ca.sin(pi * (t_norm_aug - 0.38) / 0.24)
        Aex_aug = ca.fmax(0, Aex_max * smooth_window(t_norm_aug, 0.38, 0.62) * sin_ex_aug)

        _, _, d_rho_aug, d_u_aug, d_E_aug = get_1d_derivatives(
            xL_aug,
            vL_aug,
            rho_s,
            u_g_s,
            E_s,
            Ain_aug,
            Aex_aug,
            Q_comb_val,
            F_cam_dummy,
            geometry,
            flow_cfg,
            viscosity_cfg,
        )

        rhs_aug = ca.vertcat(*d_rho_aug, *d_u_aug, *d_E_aug, 1.0)  # dt/dt = 1

        ode_aug = {"x": x_state_aug, "ode": rhs_aug}

        # Step Integrator (fixed step dt)
        step_int_aug = ca.integrator(
            "step_aug", "cvodes", ode_aug, {"tf": dt, "abstol": 1e-6, "reltol": 1e-6}
        )

        # Simulation Loop
        x_aug = np.concatenate([x_curr, [0.0]])  # Start at t=0

        trace = [x_aug]
        for i in range(N_steps):
            res = step_int_aug(x0=x_aug)
            x_aug = res["xf"].full().flatten()
            trace.append(x_aug)

        trace = np.array(trace)  # Shape [N+1, 3N+1]

        # Extract Results
        # rho: cols 0..9
        # u: cols 10..19
        # E: cols 20..29
        # t: col 30

        rho_res = trace[:, 0 : self.n_cells]
        u_res = trace[:, self.n_cells : 2 * self.n_cells]
        E_res = trace[:, 2 * self.n_cells : 3 * self.n_cells]
        t_res = trace[:, -1]

        # Create a callable Solution object or dictionary similar to what NLP expects
        # We need a wrapper that acts like 'dense_output'
        # We can implement a simple 'Solution' class or function

        from scipy.interpolate import interp1d

        class CasAdiDenseOutput:
            def __init__(self, t, y):
                self.t = t
                self.y = y  # Shape [n_vars, n_points]
                self.interpolator = interp1d(t, y, kind="cubic", fill_value="extrapolate", axis=1)

            def __call__(self, t):
                return self.interpolator(t)

        # y_sol needs to be shape [n_vars, n_points]
        # x_res is [n_points, n_vars]
        # We need to stack rho, u, E
        # x_res cols: [rho_0..N, u_0..N, E_0..N]
        y_sol = np.hstack([rho_res, u_res, E_res]).T  # Shape [3N, M]

        # Result object mimicking solve_ivp result
        class MockResult:
            def __init__(self, t, y):
                self.sol = CasAdiDenseOutput(t, y)
                self.t = t
                self.y = y
                self.success = True

        mock_sol = MockResult(t_res, y_sol)

        return {
            "success": True,
            "t": t_res,
            "rho": list(rho_res.T),  # List of arrays (one per cell)
            "u": list(u_res.T),
            "E": list(E_res.T),
            "xv_source": geometry_profile,  # Pass through for xL/vL lookup
            "method": "cfd_shooting",
            "sol": mock_sol,  # This object has .sol(t) method
        }


class SpectralThermoSolver(ThermoSolver):
    """
    Placeholder for Spectral/Collocation based thermo solver.
    """

    def solve(self, geometry_profile: dict, params: dict) -> dict:
        log.warning("SpectralThermoSolver not implemented yet. Returning None.")
        return {"success": False}
