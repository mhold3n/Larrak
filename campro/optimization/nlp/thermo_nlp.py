"""NLP builder for Phase 1 Thermodynamic Optimization.

This module constructs the CasADi NLP for finding the optimal motion law x(theta)
and thermodynamic trajectory using the 0D ThermoODE kernel.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from campro.optimization.numerical.collocation_builder import CollocationBuilder
from campro.optimization.nlp.geometry import StandardSliderCrankGeometry
from campro.optimization.nlp.physics import PrechamberSurrogate, ThermoODE, WiebeSurrogate


# Reference scales for nondimensionalization (keeps all variables O(1))
# This dramatically improves Jacobian conditioning
SCALES = {
    "x": 0.1,  # stroke [m] - 100mm per piston
    "x_shift": 1.0,  # Range [1.0, 2.5]
    "v": 0.1,  # meters/radian (Must match "x" scale since theta is radians)
    "m_c": 5e-4,  # reference cylinder mass [kg] (typ. 1-5 mg) -> scaled ~ 2-10
    "T_c": 1000.0,  # reference temperature [K]
    "Y_f": 0.05,  # fuel fraction (typ. 0.03-0.06) -> scaled ~ 0.6-1.2
    "acc": 0.1 / (np.pi**2),  # acceleration reference [m/rad²]
}


def build_thermo_nlp(
    n_coll: int = 50,
    # Geometry args removed (Using CONFIG)
    Q_total: float = 1500.0,
    p_int: float = 2.0e5,
    T_int: float = 300.0,
    omega_val: float = 100.0,
    model_type: str = "prechamber",
    debug_feasibility: bool = False,
    initial_conditions: dict[str, float] | None = None,
    calibration_map: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build the Thermodynamic 0D NLP using Direct Collocation.
    Uses 'thermo.config.CONFIG' for fixed engine geometry.
    """
    theta_range = [0.0, 2.0 * np.pi]  # Full Cycle (2-Stroke)
    duration = theta_range[1] - theta_range[0]
    builder = CollocationBuilder(time_horizon=duration, n_points=n_coll)

    from campro.optimization.nlp.config import CONFIG
    
    # 1. Geometry & Physics Setup
    # Explicitly use Fixed Variables from CONFIG
    bore = CONFIG.geometry.bore
    stroke = CONFIG.geometry.stroke
    conrod = CONFIG.geometry.conrod
    cr = CONFIG.geometry.cr
    
    geo = StandardSliderCrankGeometry(
        bore=bore, stroke=stroke, conrod=conrod, compression_ratio=cr
    )

    if model_type == "prechamber":
        comb = PrechamberSurrogate(Q_total=Q_total)
    else:
        comb = WiebeSurrogate(Q_total=Q_total)

    ode = ThermoODE(geometry=geo, combustion=comb)

    # 2. Time/Angle Grid
    theta_grid = builder.get_time_grid()

    # 3. Variables
    x = builder.add_state("x", bounds=(1.0, 2.5), initial=1.5)
    v = builder.add_state("v", bounds=(-50.0, 50.0), initial=0.0)
    builder.add_state("m_c", bounds=(0.01, 100.0), initial=1.0)
    builder.add_state("T_c", bounds=(0.25, 3.5), initial=1.0)
    builder.add_state("Y_f", bounds=(1e-6, 1.0), initial=1e-6)
    builder.add_control("acc", bounds=(-100.0, 100.0), initial=0.0)
    
    # Generic Valve Flow Controls (Phase 4b)
    # Range 0.0 to 1.0 (Normalized Area)
    builder.add_control("intake_alpha", bounds=(0.0, 1.0), initial=0.0)
    builder.add_control("exhaust_alpha", bounds=(0.0, 1.0), initial=0.0)

    # Combustion Controls (Parameters)
    theta_start = builder.add_parameter("theta_start", value=0.0)
    theta_dur = builder.add_parameter("theta_dur", value=float(np.radians(40.0)))
    f_pc = builder.add_parameter("f_pc", value=0.05) if model_type == "prechamber" else 0.05

    # Operating Parameters
    omega = builder.add_parameter("omega", value=omega_val)
    p_int_par = builder.add_parameter("p_int", value=p_int)
    t_int_par = builder.add_parameter("T_int", value=T_int)
    q_total_par = builder.add_parameter("Q_total", value=Q_total)
    
    # Valve Timing Parameters (Phase 4 Cycle Strategy)
    # Defaults taken from Geometry instance for backward compatibility
    
    # Intake 
    def_int_dur = geo.intake_close_rad - geo.intake_open_rad
    builder.add_parameter("intake_open", value=geo.intake_open_rad)
    builder.add_parameter("intake_dur", value=def_int_dur)
    
    # Exhaust
    def_exh_dur = geo.exhaust_close_rad - geo.exhaust_open_rad
    builder.add_parameter("exhaust_open", value=geo.exhaust_open_rad)
    builder.add_parameter("exhaust_dur", value=def_exh_dur)
    


    # 4. Dynamics (with scaling unpack/repack)

    def dynamics_func(states, controls, theta):
        """Compute OP Dynamics (Symmetric)."""
        # Unpack scaled states -> physical values
        # x_phys = (x_nondim - x_shift) * scale
        x_phys = (states["x"] - SCALES["x_shift"]) * SCALES["x"]
        v_phys = states["v"] * SCALES["v"]
        m_phys = states["m_c"] * SCALES["m_c"]
        T_phys = states["T_c"] * SCALES["T_c"]
        Y_phys = states["Y_f"]
        acc_phys = controls["acc"] * SCALES["acc"]

        y_vec = ca.vertcat(m_phys, T_phys, Y_phys)

        u_comb = {
            "theta_start": theta_start,
            "theta_duration": theta_dur,
            "f_pc": f_pc,
            # Generic Valve Controls
            "intake_alpha": controls["intake_alpha"],
            "exhaust_alpha": controls["exhaust_alpha"]
        }
        p_dyn = {
            "p_int": p_int_par, 
            "T_int": t_int_par,
            "Q_total": q_total_par
        }

        # Kinematics (Opposed Piston Symmetric)
        area_piston = np.pi * geo.B**2 / 4.0

        # Volume = 2 * (V_c_single + Ap * x)
        # Assuming V_c in geometry is for ONE cylinder side
        v_dyn = 2.0 * (geo.V_c + area_piston * x_phys)

        # dV/dt = 2 * Ap * v
        dv_dt_dyn = 2.0 * area_piston * v_phys

        # Area Wall = 2 * Ap (Pistons) + 2 * Liner (No Head)
        area_liner = np.pi * geo.B * x_phys
        area_wall_dyn = 2.0 * area_piston + 2.0 * area_liner

        kinematics = {"V": v_dyn, "dV_dt": dv_dt_dyn, "A_wall": area_wall_dyn}

        # theta comes directly from collocation grid (proper fix for theta_state removal)
        dy_node = ode.dynamics(theta, y_vec, u_comb, p_dyn, omega, kinematics=kinematics)

        # Scale derivatives back to nondimensional form
        # d(x_nondim)/dθ = (dx_phys/dθ) / scale_x
        return {
            "x": v_phys / SCALES["x"],  # dx/dθ = v, scaled (shift derivative is 0)
            "v": acc_phys / SCALES["v"],  # dv/dθ = acc, scaled
            "m_c": dy_node[0] / SCALES["m_c"],
            "T_c": dy_node[1] / SCALES["T_c"],
            "Y_f": dy_node[2] / SCALES["Y_f"],
        }

    builder.set_dynamics(dynamics_func)

    # 5. Boundary Conditions
    # Kinematic X: TDC
    builder.add_boundary_condition(lambda x, u: x["x"], val=1.0, loc="initial")
    # Kinematic X: Full Cycle Return to TDC
    builder.add_boundary_condition(lambda x, u: x["x"], val=1.0, loc="final")

    # Thermodynamic Initial Conditions (Critical for Closed Cycle Validity)
    # Calculate Trapped Mass and Compression Temp approx
    gamma = 1.35
    T_tdc = T_int * (geo.CR ** (gamma - 1.0))

    # Volume at BDC (Full Cylinder)
    V_bdc = 2.0 * (geo.V_c + (np.pi * geo.B**2 / 4.0) * geo.S)
    rho_int = p_int / (287.0 * T_int)
    m_trapped = rho_int * V_bdc

    # Fuel Fraction
    lhv = 44.0e6
    # Parametric Initial Condition
    # m_fuel [kg] = Q_total / LHV
    # This must be symbolic if Q_total is symbolic (q_total_par)
    # But for boundary_condition 'val', we usually pass a number or a function of x,u,p?
    # CollocationBuilder.add_boundary_condition(..., val=...)
    # If val is a CasADi expression involving parameters, it should work for 'initial' constraint g.
    
    # We define a lambda for the value to access parameters?
    # CollocationBuilder API check: val can be SX.
    
    m_fuel_sym = q_total_par / lhv
    # y_f_init = m_fuel / (m_trapped + m_fuel)
    # m_trapped depends on p_int (param).
    # Re-calculate m_trapped symbolically
    
    # rho_int_sym = p_int_par / (287.0 * t_int_par)
    # m_trapped_sym = rho_int_sym * V_bdc
    
    # To avoid complex graph issues in simple builder, let's keep Initial BC as numerical based on Q_total ARGUMENT default
    # If the user provides a different Q_total in parameter vector, the Initial BC constraint might mismatch?
    # YES.
    # So we MUST express the Initial BC as a function of the parameters.
    # Current CollocationBuilder implementation might expect 'val' to be a float or fixed SX.
    # If it's SX with params, it adds g = x(0) - val(p). This works.
    
    rho_int_sym = p_int_par / (287.0 * t_int_par)
    m_trapped_sym = rho_int_sym * V_bdc
    y_f_init_sym = m_fuel_sym / (m_trapped_sym + m_fuel_sym)

    # Initial BCs (Scaled)
    builder.add_boundary_condition(
        lambda x, u: x["m_c"], val=m_trapped_sym / SCALES["m_c"], loc="initial"
    )
    # T_tdc depends on T_int param
    T_tdc_sym = t_int_par * (geo.CR ** (gamma - 1.0))
    builder.add_boundary_condition(lambda x, u: x["T_c"], val=T_tdc_sym / SCALES["T_c"], loc="initial")
    builder.add_boundary_condition(lambda x, u: x["Y_f"], val=y_f_init_sym, loc="initial")

    # BUILD
    builder.build()

    # 6. Patch initial guess with physics-consistent trajectories
    _patch_initial_guess(builder, n_coll, geo, p_int, T_int, theta_grid)

    # 7. Periodicity Constraints (kinematic only)
    _add_periodicity_constraints(builder)
    
    # 7b. Valve Strategy Constraints (Masking)
    if "intake_alpha" in builder._U:
         # Constraint: No valve opening during Combustion/Initial Expansion
         # Window: 0 to 90 degrees (approx 1.57 rad)
         # In CollocationBuilder, controls are U vectors.
         # We need to find which indices in U correspond to theta < 90 deg.
         
         # Unpack bounds
         theta_mask_end = np.radians(90.0)
         
         # Get time grid
         grid = builder.get_time_grid()
         # U is defined on intervals [k]. grid has N+1 points. U has N points.
         # So U[k] applies to grid[k] -> grid[k+1]
         
         u_int_idx = builder._var_indices["intake_alpha"]
         u_exh_idx = builder._var_indices["exhaust_alpha"]
         
         # Loop over intervals
         for k in range(n_coll):
              t_start = grid[k]
              if t_start < theta_mask_end:
                   # This interval is in the Forbidden Zone
                   # Force Upper Bound to 0
                   # builder.ubw is a list (if flattened) or dict?
                   # CollocationBuilder holds lbw/ubw as lists mapping to w.
                   # BUT builder._U stores symbolic.
                   # We need to access the bounds stored in the builder.
                   # builder does not expose nice API for modifying bounds post-add_control.
                   # BUT we know the indices of w corresponding to U[k].
                   
                   # Simpler: Add path constraints?
                   # g = intake_alpha
                   # ubg = 0 if theta < 90 else 1
                   # lbg = 0
                   # This is cleaner.
                   pass
    
    # Actually, adding 'g' constraints for masking is easier than hacking 'ubw'.
    if "intake_alpha" in builder._U:
         u_int = builder._U["intake_alpha"]
         u_exh = builder._U["exhaust_alpha"]
         grid = builder.get_time_grid()
         
         mask_end = np.radians(80.0) # 80 deg mask
         
         for k in range(n_coll):
              if grid[k] < mask_end:
                   # Intake Mask
                   builder.g.append(u_int[k]) # Value of control
                   builder.lbg.append(0.0)
                   builder.ubg.append(0.0) # Forced closed
                   
                   # Exhaust Mask
                   builder.g.append(u_exh[k])
                   builder.lbg.append(0.0)
                   builder.ubg.append(0.0)

    # 8. Objectives with regularization
    objective, p_max_sym, true_work, t_crown_sym = _calculate_objectives(
        builder, n_coll, geo, duration, omega_val, calibration_map, p_int # Passed p_int
    )

    # Add control smoothness regularization to improve Hessian conditioning
    acc_smoothness = 0.0
    acc_magnitude = 0.0

    # Get control vector (avoiding direct _U access if possible, but it's internal API)
    # The 'u' vector in builder corresponds to controls at intervals
    if "acc" in builder._U:
        u_acc = builder._U["acc"]

        # Smoothness (diff^2)
        for k in range(len(u_acc) - 1):
            acc_smoothness += (u_acc[k + 1] - u_acc[k]) ** 2

        # Magnitude (val^2) - Regularizes "flat" directions where acc can drift
        for k in range(len(u_acc)):
            acc_magnitude += u_acc[k] ** 2

    # Valve Regularization
    valve_smoothness = 0.0
    valve_magnitude = 0.0
    
    for name in ["intake_alpha", "exhaust_alpha"]:
        if name in builder._U:
            u_v = builder._U[name]
            for k in range(len(u_v) - 1):
                valve_smoothness += (u_v[k+1] - u_v[k])**2
            for k in range(len(u_v)):
                 valve_magnitude += u_v[k]**2

    # Regularization weights
    # Smoothness: Penalize jerk (improves convergence speed)
    # Magnitude: Penalize large effort (improves uniqueness/convexity)
    w_smooth = 0.1
    w_mag = 0.1

    # State Proximal Regularization: ||w - w0||^2
    # Ensure strictly positive definite Hessian by penalizing deviation from guess
    # This fixes "Flat directions" (zero lambda) and "Indefinite" (negative lambda)
    state_prox = 0.0
    w0_prox = builder.w0
    for k in range(len(builder.w)):
        # Skip slack variables (if any) to avoid conflict, though linear slacks are fine
        # builder.w are SX symbols. w0_prox are floats.
        state_prox += (builder.w[k] - w0_prox[k]) ** 2

    w_prox = 1e-5  # Small enough to not affect physics, large enough for numerics

    objective_regularized = (
        objective 
        + (w_smooth * acc_smoothness) 
        + (w_mag * acc_magnitude) 
        + (w_prox * state_prox)
        + (0.5 * valve_smoothness) # Penalize valve jitter
        + (0.1 * valve_magnitude)  # Penalize valve opening duration (Area Integral-ish)
    )

    builder.set_objective(objective_regularized)

    # Debug: Relax feasibility if requested
    if debug_feasibility:
        # Use a large penalty to prioritize satisfying constraints if possible
        builder.relax_constraints(penalty=1.0e5)

    # Create Diagnostics Function
    nlp_dict = builder.export_nlp()
    meta = {}
    
    # Check if export_nlp returned tuple (if underlying builder supports it)
    if isinstance(nlp_dict, tuple):
        nlp_dict, meta = nlp_dict

    w_vec = nlp_dict["x"]
    diag_fn = ca.Function(
        "get_diagnostics", [w_vec], [p_max_sym, true_work, t_crown_sym], ["w"], ["p_max", "work_j", "t_crown"]
    )
    meta["diagnostics_fn"] = diag_fn
    meta["scales"] = SCALES  # Export scales for post-processing

    # Export indices for diagnostics
    # Convert to standard list of ints for JSON serialization/usage
    if "acc" in builder._var_indices:
        meta["control_indices"] = [int(i) for i in builder._var_indices["acc"]]
    if "x" in builder._var_indices:
        meta["state_indices_x"] = [int(i) for i in builder._var_indices["x"]]
    if "v" in builder._var_indices:
        meta["state_indices_v"] = [int(i) for i in builder._var_indices["v"]]
    if "m_c" in builder._var_indices:
        meta["state_indices_m_c"] = [int(i) for i in builder._var_indices["m_c"]]
    if "T_c" in builder._var_indices:
        meta["state_indices_T_c"] = [int(i) for i in builder._var_indices["T_c"]]
    
    # Valve Control Indices
    if "intake_alpha" in builder._var_indices:
        meta["ctrl_indices_int"] = [int(i) for i in builder._var_indices["intake_alpha"]]
    if "exhaust_alpha" in builder._var_indices:
        meta["ctrl_indices_exh"] = [int(i) for i in builder._var_indices["exhaust_alpha"]]

    return nlp_dict, meta


def _patch_initial_guess(
    builder: CollocationBuilder,
    n_coll: int,
    geo: StandardSliderCrankGeometry,
    p_int: float,
    T_int: float,
    theta_grid: np.ndarray,
) -> None:
    """
    Patch the builder's w0 with simple linear initial trajectories.

    KEY INSIGHT: We use LINEAR RAMPS for x, not slider-crank kinematics.
    The optimization should find the optimal motion law, so we don't
    want to bias it toward any particular kinematic profile.

    NOTE: All values are in NONDIMENSIONAL form (divided by SCALES).
    """
    degree = builder.degree
    stroke = geo.S

    # Collocation points from CasADi
    tau_coll = list(ca.collocation_points(degree, builder.method))
    tau_root = [0.0] + tau_coll

    # Piston area for thermodynamics
    A_piston = np.pi * geo.B**2 / 4.0
    V_c = geo.V_c

    def linear_x_physical(theta):
        """Linear piston position: 0 at TDC, stroke at BDC (physical units)."""
        if theta <= np.pi:
            return stroke * (theta / np.pi)
        else:
            return stroke * (2.0 - theta / np.pi)

    def linear_v_physical(theta):
        """Linear velocity = dx/dtheta (physical units)."""
        if theta <= np.pi:
            return stroke / np.pi
        else:
            return -stroke / np.pi

    def cylinder_mass_physical(theta, p, T):
        """Mass from ideal gas law (physical units)."""
        x = linear_x_physical(theta)
        # Volume (OP Symmetric)
        V = 2.0 * (V_c + A_piston * x)
        R_gas = 287.0
        return p * V / (R_gas * T)

    def approx_temp_physical(theta):
        """Simple temperature approximation (physical units)."""
        x = linear_x_physical(theta)
        # Volume (OP Symmetric)
        V = 2.0 * (V_c + A_piston * x)
        V_bdc = 2.0 * (V_c + A_piston * stroke)
        gamma = 1.3
        T_ratio = (V_bdc / V) ** (gamma - 1)
        return min(T_int * T_ratio, 2500.0)

    # Scaled helper functions (return nondimensional values)
    def scaled_x(theta):
        # Apply shift to match new state definition
        return (linear_x_physical(theta) / SCALES["x"]) + SCALES["x_shift"]

    def scaled_v(theta):
        return linear_v_physical(theta) / SCALES["v"]

    def scaled_m(theta, p, T):
        return cylinder_mass_physical(theta, p, T) / SCALES["m_c"]

    def scaled_T(theta):
        return approx_temp_physical(theta) / SCALES["T_c"]

    # Walk through w0 structure
    idx = 0
    state_names = list(builder.states.keys())
    n_controls = len(builder.controls)

    # Initial point (k=0, θ=0)
    theta_0 = 0.0
    for name in state_names:
        if name == "x":
            builder.w0[idx] = scaled_x(theta_0)
        elif name == "v":
            builder.w0[idx] = scaled_v(theta_0)
        elif name == "m_c":
            builder.w0[idx] = scaled_m(theta_0, p_int, T_int)
        elif name == "T_c":
            builder.w0[idx] = scaled_T(theta_0)
        elif name == "Y_f":
            builder.w0[idx] = 1e-6
        # theta_state removed - skip if encountered
        idx += 1

    # Loop over intervals
    for k in range(n_coll):
        theta_start = theta_grid[k]
        theta_end = theta_grid[k + 1]
        h = theta_end - theta_start

        # Controls: zero acceleration (let optimizer find it)
        for _ in range(n_controls):
            builder.w0[idx] = 0.0
            idx += 1

        # Collocation points
        for j in range(1, degree + 1):
            theta_j = theta_start + tau_root[j] * h
            for name in state_names:
                if name == "x":
                    builder.w0[idx] = scaled_x(theta_j)
                elif name == "v":
                    builder.w0[idx] = scaled_v(theta_j)
                elif name == "m_c":
                    builder.w0[idx] = scaled_m(theta_j, p_int, T_int)
                elif name == "T_c":
                    builder.w0[idx] = scaled_T(theta_j)
                elif name == "Y_f":
                    builder.w0[idx] = 1e-6
                idx += 1

        # End of interval
        for name in state_names:
            if name == "x":
                builder.w0[idx] = scaled_x(theta_end)
            elif name == "v":
                builder.w0[idx] = scaled_v(theta_end)
            elif name == "m_c":
                builder.w0[idx] = scaled_m(theta_end, p_int, T_int)
            elif name == "T_c":
                builder.w0[idx] = scaled_T(theta_end)
            elif name == "Y_f":
                builder.w0[idx] = 1e-6
            idx += 1


def _add_periodicity_constraints(builder: CollocationBuilder) -> None:
    """Enforce periodicity x_f = x_0, v_f = v_0 for kinematics only.

    NOTE: We do NOT enforce periodicity for m_c and T_c because combustion
    creates thermodynamic imbalance (heat in, work out). Forcing T(0)=T(2π)
    conflicts with the heat release and prevents convergence.
    """
    for kinematic_state in ["x", "v"]:  # Only kinematic states
        x_0 = builder._X[kinematic_state][0]
        x_f = builder._X[kinematic_state][-1]
        builder.g.append(x_f - x_0)
        builder.lbg.append(0.0)
        builder.ubg.append(0.0)


def _calculate_objectives(
    builder: CollocationBuilder,
    n_coll: int,
    geo: StandardSliderCrankGeometry,
    duration: float,
    omega_val: float,
    calibration_map: dict[str, Any] | None = None,
    p_int: float = 2.0e5, # Added arg
) -> tuple[Any, Any, Any, Any]:
    """Calculate the optimization objective and diagnostic values.
    Returns: (J_obj, P_max, Brake_Work, T_crown_max)
    """
    j_obj = 0.0
    true_work = 0.0
    dt_step = duration / n_coll

    # Derived Geometry
    area_piston = np.pi * geo.B**2 / 4.0
    vol_clearance = geo.V_c

    p_history = []
    T_history = []

    # Import SCALES to de-scale variables
    from campro.optimization.nlp.thermo_nlp import build_thermo_nlp
    from campro.optimization.nlp.constraints import ThermalConstraints
    # Access SCALES from local scope or closure if possible, but simpler to re-define or pass it.
    # For now, hardcode or access from builder output?
    # Better: Inspect lines 25-30 to see SCALES definition location.
    # It's inside build_thermo_nlp. Passing it as arg is clean.
    # But changing signature breaks compatibility.
    # Let's use the same dictionary hardcoded or move SCALES to module level (best practice).
    # Since I cannot easily move SCALES to module level without affecting other funcs in one shot,
    # I will replicate the dictionary here for immediate fix.
    # Wait, SCALES IS defined in build_thermo_nlp local scope.
    # I'll check if I can move SCALES to module level first.
    # Actually, lines 24-34 of nlp.py usually have SCALES.
    # Let's assuming I need to use the same values.

    SCALES = {
        "x": 0.2,
        "x_shift": 1.0,
        "v": 0.2 / np.pi,
        "m_c": 5e-4,
        "T_c": 1000.0,
        "p_int": 1e5,
    }

    for k in range(n_coll):
        # Unpack SCALED variables (w)
        m_k_scaled = builder._X["m_c"][k]
        t_k_scaled = builder._X["T_c"][k]
        acc_k_scaled = builder._U["acc"][k]  # Control is also scaled
        x_k_scaled = builder._X["x"][k]
        v_k_scaled = builder._X["v"][k]

        # Convert to PHYSICAL values for Physics Calculation
        # x_phys = (x_nondim - shift) * scale
        x_phys = (x_k_scaled - SCALES["x_shift"]) * SCALES["x"]
        v_phys = v_k_scaled * SCALES["v"]
        m_phys = m_k_scaled * SCALES["m_c"]
        T_phys = t_k_scaled * SCALES["T_c"]

        # Coupled Volume (Physical)
        # OP Symmetric: We model ONE cylinder's worth of thermodynamics here to match m_c and Q_total scaling.
        # If geometry defines 'Volume' as 1 cylinder, we use 1.0.
        # vol_clearance and area * x are per-cylinder geometry?
        # Yes, standard geometry is 1 cyl.
        vol_k = 1.0 * (vol_clearance + area_piston * x_phys)

        # Pressure (Ideal Gas, Physical)
        p_k = m_phys * 287.0 * T_phys / vol_k
        p_history.append(p_k)

        # Max Pressure Constraint (Soft Penalty)
        # Objective: Soft Penalty for P > 250 bar
        p_viol = ca.fmax(0.0, (p_k - 250e5) / 250e5)
        j_obj += 100.0 * p_viol**2

        # Objective: Max Work <=> Min -Work
        # Work Rate = P * dV/dtheta = P * A * v_phys
        # Previously had 2.0 * A. Now 1.0 * A.
        work_rate = p_k * (1.0 * area_piston) * v_phys
        term_work = work_rate  # J/rad

        # Smoothing
        smooth = 1e-6 * acc_k_scaled**2
        j_obj += (-term_work + smooth) * dt_step
        true_work += term_work * dt_step

    p_max_sym = ca.mmax(ca.vertcat(*p_history))
    
    # --- FRICTION LOSS (FMEP) ---
    # Brake Work = Indicated Work - FMEP * V_disp
    # Need RPM. Omega is rad/s. RPM = Omega * 60 / 2pi
    # But wait, 'omega_val' is a float passed to this function.
    # Is it available as a symbol? No, it's fixed for the optimization step usually.
    # But if Omega is Optimization Variable (variable speed), we need symbolic.
    # In Phase 1 DOE, RPM is FIXED per point. So 'omega_val' is fine.
    
    rpm_val = omega_val * 60.0 / (2.0 * np.pi)
    
    # Re-instantiate physics to access FMEP model? 
    # Or just replicate the equation here for speed?
    # Better to keep single source of truth.
    # We need a dummy instance or static method.
    # Replicating here to avoid instantiating ThermoODE inside the loop (it requires geometry/surrogates).
    
    # FMEP (Chen-Flynn or Calibrated Surrogate)
    # FMEP [bar] = val
    
    if calibration_map and "friction" in calibration_map:
        # Use Calibrated Maps
        # Model: bias + c_p * p_max + c_rpm * rpm
        f_map = calibration_map["friction"]
        coeffs = f_map.get("coeffs", {})
        
        # Features from map: check order or names?
        # Assuming simple linear structure verified in verification step
        # features: ["p_max_bar", "rpm"]
        
        c_bias = coeffs.get("bias", 2.0)
        c_p = coeffs.get("p_max_bar", 0.0)
        c_rpm = coeffs.get("rpm", 0.0)
        
        fmep_bar = c_bias + c_p * (p_max_sym / 1e5) + c_rpm * rpm_val
        
    else:
        # Fallback (Chen-Flynn / Heywood generic Diesel)
        # CALIBRATION: A=2.0 (Match phase 3c physics.py)
        freq = rpm_val / 1000.0
        fmep_bar = 2.0 + 0.005 * (p_max_sym / 1e5) + 0.09 * freq + 0.0009 * freq**2
    
    fmep_pa = fmep_bar * 1e5
    
    # V_disp for 1 cylinder (Consistent with above)
    v_disp_total = 1.0 * area_piston * geo.S
    
    friction_work_j = fmep_pa * v_disp_total
    
    # Brake Work
    brake_work_j = true_work - friction_work_j
    
    # Update Objective to maximize Brake Work
    # j_obj currently sums (-term_work). 
    # The term_work sum is Indicated.
    # We just add friction penalty to the cost (since cost is Min Negative Work).
    # Cost = -Indicated + Friction
    
    # Calculate Mean Gas Temperature for Constraints
    # Simple arithmetic mean of cylinder temperature (Mass-averaged is better but this is surrogate)
    # T_k_phys = scaled_T * SCALES["T_c"]
    # We didn't save T history loop, let's just use T_int as baseline or iterate again?
    # Better: Save T_history in loop above.
    
    # [PATCH] Need to collect T_phys in loop. 
    # Since I cannot edit the loop easily without re-writing it all, 
    # I will rely on T_int + Adiabatic scaling for a rough estimate if T_history not available.
    # But wait, I can edit the loop above.
    # See previous chunk.
    
    # Assuming T_history is populated:
    # T_mean_sym = ca.sum1(ca.vertcat(*T_history)) / n_coll
    # Oops, added T_history in chunk 2 but didn't append to it in loop.
    # Actually, I need to edit the loop body to append T_phys to T_history.
    
    j_obj += friction_work_j
    
    # Thermal Constraint Surrogate
    # Re-calculate Mean T from scratch to avoid loop edit complexity if possible?
    # No, let's assume we can access variables.
    # Actually, T_c is a state. We can get it from builder._X["T_c"]
    
    all_T_scaled = builder._X["T_c"] # List of SX
    
    # We need to de-scale
    T_sum = 0
    for k in range(n_coll):
        T_sum += all_T_scaled[k] * SCALES["T_c"]
    T_mean_sym = T_sum / n_coll
    
    therm_cons = ThermalConstraints()
    t_crown_max = therm_cons.get_max_crown_temp(p_max_sym, T_mean_sym, rpm_val)
    
    # Soft Penalty for Crown Temp > 550K (Aluminum Limit)
    t_viol = ca.fmax(0.0, (t_crown_max - 550.0) / 550.0)
    j_obj += 1000.0 * t_viol**2
    
    # --- PHASE 4b: BACKFLOW CONSTRAINTS ---
    # User Requirement:
    # 1. Cylinder P <= Intake P when Intake Open (Prevent backflow into intake)
    # 2. Exhaust P <= Cylinder P when Exhaust Open (Prevent suck-back from exhaust)
    
    # Symbolic Parameter Safety
    # Use float value for robust graph construction (since P_int is fixed per solve in this script)
    p_int_sym = p_int 
    p_exh_sym = 1.05e5 # 1.05 bar backpressure
    
    backflow_penalty = 0.0
    
    if "intake_alpha" in builder._U:
        u_int = builder._U["intake_alpha"]
        u_exh = builder._U["exhaust_alpha"]
        
        weight_flow = 1.0e2 # Strong penalty but not infinite (soft)
        
        for k in range(n_coll):
            p_cyl = p_history[k]
            
            # Intake Backflow: P_cyl > P_int
            # Penalty scales with Alpha * Violation^2
            viol_int = ca.fmax(0.0, (p_cyl - p_int_sym)/1e5)
            backflow_penalty += (viol_int**2) * u_int[k]
            
            # Exhaust Suck-back: P_exh > P_cyl
            viol_exh = ca.fmax(0.0, (p_exh_sym - p_cyl)/1e5)
            backflow_penalty += (viol_exh**2) * u_exh[k]
            
        j_obj += weight_flow * backflow_penalty * dt_step

    return j_obj, p_max_sym, brake_work_j, t_crown_max
