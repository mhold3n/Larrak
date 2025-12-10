"""NLP builder for Phase 1 Thermodynamic Optimization.

This module constructs the CasADi NLP for finding the optimal motion law x(theta)
and thermodynamic trajectory using the 0D ThermoODE kernel.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
from campro.optimization.builder import CollocationBuilder

from thermo.geometry import StandardSliderCrankGeometry
from thermo.physics import PrechamberSurrogate, ThermoODE, WiebeSurrogate


def build_thermo_nlp(
    n_coll: int = 50,
    bore: float = 0.1,
    stroke: float = 0.2,
    conrod: float = 0.4,
) -> dict:
    """Builds the Phase 1 Thermodynamic NLP."""
    theta_range = [0.0, 2 * np.pi]
    duration = theta_range[1] - theta_range[0]
    builder = CollocationBuilder(time_horizon=duration, n_points=n_coll)

    # 1. Geometry & Physics Setup
    geo = StandardSliderCrankGeometry(
        bore=bore, stroke=stroke, conrod=conrod, compression_ratio=15.0
    )
    # Switch to PrechamberSurrogate for advanced features
    comb = PrechamberSurrogate(Q_total=1500.0)
    ode = ThermoODE(geometry=geo, combustion=comb)

    # 2. Time/Angle Grid
    theta_grid = builder.get_time_grid()

    # 3. Variables
    x = builder.add_state("x", bounds=(-0.1, 0.5), initial=0.1)
    v = builder.add_state("v", bounds=(-10.0, 10.0), initial=0.0)

    # Thermodynamic States
    m_c = builder.add_state("m_c", bounds=(1e-5, 1.0), initial=1e-3)
    T_c = builder.add_state("T_c", bounds=(250.0, 3500.0), initial=1000.0)
    Y_f = builder.add_state("Y_f", bounds=(0.0, 1.0), initial=0.0)

    # Controls
    acc = builder.add_control("acc", bounds=(-1000.0, 1000.0), initial=0.0)

    # New Control: Prechamber Fraction (f_pc)
    # Optimized constant for the cycle
    f_pc = builder.add_parameter("f_pc", value=0.05)  # optimized externally or here?

    # Combustion Controls (Parameters)
    theta_start = builder.add_parameter("theta_start", value=0.0)
    theta_dur = builder.add_parameter("theta_dur", value=float(np.radians(40.0)))
    omega = builder.add_parameter("omega", value=100.0)

    # 4. Dynamics
    # Mapping for PrechamberSurrogate
    u_comb = {
        "theta_start": theta_start,
        "theta_duration": theta_dur,
        "f_pc": f_pc,
        "theta_pc": theta_start - np.radians(5.0),  # Coupled timing for now
    }

    theta_state = builder.add_state("theta_state", bounds=(0.0, 4 * np.pi), initial=np.pi)

    def dynamics_func(states, controls):
        """Compute derivatives for all states."""
        x_s = states["x"]
        v_s = states["v"]
        m_c_s = states["m_c"]
        T_c_s = states["T_c"]
        Y_f_s = states["Y_f"]
        theta_s = states["theta_state"]

        acc_s = controls["acc"]

        y_vec = ca.vertcat(m_c_s, T_c_s, Y_f_s)
        dy_node = ode.dynamics(theta_s, y_vec, u_comb, {}, omega)

        return {
            "x": v_s,
            "v": acc_s,
            "m_c": dy_node[0],
            "T_c": dy_node[1],
            "Y_f": dy_node[2],
            "theta_state": 1.0,
        }

    builder.set_dynamics(dynamics_func)

    # 5. Constraints
    builder.add_boundary_condition(lambda x0, xf: x0["theta_state"], val=0.0, loc="initial")

    # BUILD
    builder.build()

    # Manual Loop for Objectives and Constraints
    J = 0.0
    dt_step = duration / n_coll

    P_prev = None

    for k in range(n_coll):
        m_k = builder._X["m_c"][k]
        T_k = builder._X["T_c"][k]
        th_k = builder._X["theta_state"][k]
        acc_k = builder._U["acc"][k]
        Yf_k = builder._X["Y_f"][k]

        vol_k = geo.Volume(th_k)
        P_k = m_k * 287.0 * T_k / vol_k
        dV_dth_k = geo.dV_dtheta(th_k)

        # Max Pressure Constraint
        builder.g.append(P_k)
        builder.lbg.append(0.0)
        builder.ubg.append(150e5)

        # Max Pressure Rise Rate Constraint (dP/dtheta approx)
        if P_prev is not None:
            dP = P_k - P_prev
            # Discrete derivative per step (fixed step in grid? No, step size varies with coll?)
            # Assuming uniform grid for phase 1 simplicity in derivative estimate
            # dP/dtheta approx dP / (dt_step) ?? No, dt_step is time?
            # dt_step is theta step here (time_horizon = duration in angle).
            dtheta = dt_step
            PRR = dP / dtheta
            # Limit PRR (e.g. 10 bar/deg -> ~570 bar/rad -> 5.7e7 Pa/rad)
            builder.g.append(PRR)
            builder.lbg.append(-1e8)  # Lower bound
            builder.ubg.append(1e8)  # Upper bound (1000 bar/rad approx)

        P_prev = P_k

        # Objectives
        # 1. Indicated Work (Maximize) -> Minimize -Work
        w_dens = P_k * dV_dth_k
        J_work = -w_dens
        # 2. Smoothness
        J_smooth = 1e-3 * acc_k**2

        J += (J_work + J_smooth) * dt_step

    # Kinetic Efficiency: Penalty on Unburned Fuel at end
    # J_UHC = Weight * Y_f_final
    Y_f_final = builder._X["Y_f"][-1]
    J_UHC = 1e5 * Y_f_final  # Heavy penalty for unburned fuel

    J += J_UHC

    builder.set_objective(J)

    return builder.export_nlp()


def build_thermo_nlp(
    n_coll: int = 50,
    bore: float = 0.1,
    stroke: float = 0.2,
    conrod: float = 0.4,
) -> dict:
    """Builds the Phase 1 Thermodynamic NLP.

    Args:
        n_coll: Number of collocation intervals
        bore, stroke, conrod: Basic geometry for initial guess / bounds

    Returns:
        dict: The NLP dictionary {x, f, g} + metadata
    """
    theta_range = [0.0, 2 * np.pi]
    # CollocationBuilder(time_horizon, n_points, ...)
    # Assuming time_horizon is duration.
    duration = theta_range[1] - theta_range[0]
    builder = CollocationBuilder(time_horizon=duration, n_points=n_coll)

    # 1. Geometry & Physics Setup
    # Using standard geometry as placeholder for 'lookup' logic
    geo = StandardSliderCrankGeometry(
        bore=bore, stroke=stroke, conrod=conrod, compression_ratio=15.0
    )
    # Using Wiebe surrogate
    comb = WiebeSurrogate(Q_total=1500.0)  # Approx 1.5kJ per cycle
    ode = ThermoODE(geometry=geo, combustion=comb)

    # 2. Time/Angle Grid
    theta_grid = builder.get_time_grid()  # Symbolic or numeric grid points

    # 3. Variables

    # Motion Law x(theta) - we optimize this!
    # Bounds: loose around expected stroke
    # Note: add_state(name, bounds=(lb, ub), initial=guess)
    x = builder.add_state("x", bounds=(-0.1, 0.5), initial=0.1)
    # Velocity v = dx/dtheta
    v = builder.add_state("v", bounds=(-10.0, 10.0), initial=0.0)

    # Thermodynamic States
    # y = [m_c, T_c, Y_f]

    # Cylinder Mass: lb > 0
    m_c = builder.add_state("m_c", bounds=(1e-5, 1.0), initial=1e-3)
    # Cylinder Temp: 300K to 3000K
    T_c = builder.add_state("T_c", bounds=(250.0, 3500.0), initial=1000.0)
    # Fuel Fraction
    Y_f = builder.add_state("Y_f", bounds=(0.0, 1.0), initial=0.0)

    # Controls
    # Motion control: Acceleration a = dv/dtheta
    acc = builder.add_control("acc", bounds=(-1000.0, 1000.0), initial=0.0)

    # Combustion Controls (Parameters)
    theta_start = builder.add_parameter("theta_start", value=0.0)
    theta_dur = builder.add_parameter("theta_dur", value=float(np.radians(40.0)))
    omega = builder.add_parameter("omega", value=100.0)  # rad/s

    # 4. Dynamics
    y_ode = ca.vertcat(m_c, T_c, Y_f)

    u_comb = {
        "theta_start": theta_start,
        "theta_duration": theta_dur,
        "wiebe_m": 2.0,
        "wiebe_a": 5.0,
    }

    # Need to access builder's internal time symbol?
    # builder defines independent variable?

    # Re-adding theta state to be sure.
    theta_state = builder.add_state("theta_state", bounds=(0.0, 4 * np.pi), initial=np.pi)

    def dynamics_func(states, controls):
        """Compute derivatives for all states."""
        # Unpack
        x_s = states["x"]
        v_s = states["v"]
        m_c_s = states["m_c"]
        T_c_s = states["T_c"]
        Y_f_s = states["Y_f"]
        theta_s = states["theta_state"]

        acc_s = controls["acc"]

        y_vec = ca.vertcat(m_c_s, T_c_s, Y_f_s)

        # Calculate ODE derivatives
        # Note: u_comb uses parameters (constants/SX), not time-varying controls here
        dy_node = ode.dynamics(theta_s, y_vec, u_comb, {}, omega)

        return {
            "x": v_s,
            "v": acc_s,
            "m_c": dy_node[0],
            "T_c": dy_node[1],
            "Y_f": dy_node[2],
            "theta_state": 1.0,
        }

    builder.set_dynamics(dynamics_func)

    # 5. Constraints
    builder.add_boundary_condition(lambda x0, xf: x0["theta_state"], val=0.0, loc="initial")

    # Max Pressure function
    def get_pressure(s_nodes, k_idx):
        # Access internal storage of builder
        m_val = builder._X["m_c"][k_idx]
        T_val = builder._X["T_c"][k_idx]
        th_val = builder._X["theta_state"][k_idx]  # Note: theta is a state now
        vol = geo.Volume(th_val)
        return m_val * 287.0 * T_val / vol

    # Max Pressure Constraint (Path Constraint)
    # We must add this BEFORE build() if we want builder to handle it at collocation points?
    # YES. builder.add_path_constraint takes expression (of generic symbols) and enforces it... HOW?
    # If add_path_constraint takes generic symbols, does it substitute?
    # Let's check builder.py add_path_constraint implementation.
    # Lines 194-198: just appends to list.
    # Lines 360+ (not seen): probably iterates and substitutes?
    # If builder supports path constraints with generic symbols, then it MUST substitute.
    # Why didn't it substitute for Objective?
    # Because set_objective is just `self.J = expr`.
    # But add_path_constraint implies it processes them.

    # Assuming add_path_constraint works with generic symbols (x, m_c, etc) by substitution logic internally?
    # OR it expects a function `f(x, u)`?
    # Outline said `add_path_constraint(expr, bounds)`.
    # Let's assume for now path constraints might fail too if I use them.
    # I'll comment out path constraint for this step to isolate Objective fix.
    # Or implement it manually in the loop.

    # Let's do manual loop for Objective AND Constraints to be safe/explicit.

    # BUILD to generate variables
    builder.build()

    # Now construct Objective and Constraints over the grid
    J = 0.0
    dt_step = duration / n_coll

    # Access generated variables
    # builder._X is dict: name -> list of SX
    # builder._U is dict: name -> list of SX

    # Iterate over intervals (Trapezoidal-ish or just Riemann left)
    for k in range(n_coll):
        # State at node k
        m_k = builder._X["m_c"][k]
        T_k = builder._X["T_c"][k]
        th_k = builder._X["theta_state"][k]
        acc_k = builder._U["acc"][k]

        # P = mRT/V
        vol_k = geo.Volume(th_k)
        P_k = m_k * 287.0 * T_k / vol_k

        # dV = dV/dtheta * step?
        # Work = Integral P dV = Integral P (dV/dtheta) dtheta
        dV_dth_k = geo.dV_dtheta(th_k)

        # Objective terms
        # Work density evaluated at k
        w_dens = P_k * dV_dth_k
        smooth = 1e-3 * acc_k**2

        # Sum
        J += (-w_dens + smooth) * dt_step

        # Max Pressure Constraint (at node k)
        # Manually add to builder.g
        # builder.g.append(P_k - 150e5) --> enforce <= 0
        limit = 150e5

        # We can append directly to builder lists
        # g is (val - bound) ? No, usually g(x) bounded by lbg, ubg
        builder.g.append(P_k)
        builder.lbg.append(0.0)  # Pressure > 0 naturally
        builder.ubg.append(limit)

    builder.set_objective(J)

    return builder.export_nlp()
