"""NLP builder for Phase 1 Thermodynamic Optimization.

This module constructs the CasADi NLP for finding the optimal motion law x(theta)
and thermodynamic trajectory using the 0D ThermoODE kernel.
"""

from __future__ import annotations

from typing import Any

import casadi as ca
import numpy as np

from campro.optimization.numerical.collocation_builder import CollocationBuilder
from thermo.geometry import StandardSliderCrankGeometry
from thermo.physics import PrechamberSurrogate, ThermoODE, WiebeSurrogate


# Reference scales for nondimensionalization (keeps all variables O(1))
# This dramatically improves Jacobian conditioning
SCALES = {
    "x": 0.1,  # stroke [m] - 100mm per piston
    "x_shift": 1.0,  # Range [1.0, 2.5]
    "v": 0.1 / np.pi,  # 100mm/pi
    "m_c": 5e-4,  # reference cylinder mass [kg] (typ. 1-5 mg) -> scaled ~ 2-10
    "T_c": 1000.0,  # reference temperature [K]
    "Y_f": 0.05,  # fuel fraction (typ. 0.03-0.06) -> scaled ~ 0.6-1.2
    "acc": 0.1 / (np.pi**2),  # acceleration reference [m/rad²]
}


def build_thermo_nlp(
    n_coll: int = 50,
    bore: float = 0.1,
    stroke: float = 0.1,  # 100mm Per Piston (OP)
    conrod: float = 0.4,
    Q_total: float = 1500.0,
    p_int: float = 2.0e5,
    T_int: float = 300.0,
    omega_val: float = 100.0,
    model_type: str = "prechamber",
    debug_feasibility: bool = False,
) -> dict:
    """Builds the Phase 1 Thermodynamic NLP (Opposed Piston).

    OP Configuration:
    - Two symmetric pistons.
    - Total Stroke = 2 * stroke parameter.
    - Total Volume = 2 * (Vc + Ap*x).
    - Mean Piston Speed = 2 * stroke * RPM / 60 (Healthy 20m/s @ 6000).
    """
    theta_range = [0.0, np.pi]  # Expansion Stroke Only
    duration = theta_range[1] - theta_range[0]
    builder = CollocationBuilder(time_horizon=duration, n_points=n_coll)

    # 1. Geometry & Physics Setup
    geo = StandardSliderCrankGeometry(
        bore=bore, stroke=stroke, conrod=conrod, compression_ratio=15.0
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

    # Combustion Controls (Parameters)
    theta_start = builder.add_parameter("theta_start", value=0.0)
    theta_dur = builder.add_parameter("theta_dur", value=float(np.radians(40.0)))
    f_pc = builder.add_parameter("f_pc", value=0.05) if model_type == "prechamber" else 0.05

    # Operating Parameters
    omega = builder.add_parameter("omega", value=omega_val)
    p_int_par = builder.add_parameter("p_int", value=p_int)
    t_int_par = builder.add_parameter("T_int", value=T_int)

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
        }
        p_dyn = {"p_int": p_int_par, "T_int": t_int_par}

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
    # Kinematic X: BDC
    builder.add_boundary_condition(lambda x, u: x["x"], val=2.0, loc="final")

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
    m_fuel = Q_total / lhv
    y_f_init = m_fuel / (m_trapped + m_fuel)

    # Initial BCs (Scaled)
    builder.add_boundary_condition(
        lambda x, u: x["m_c"], val=m_trapped / SCALES["m_c"], loc="initial"
    )
    builder.add_boundary_condition(lambda x, u: x["T_c"], val=T_tdc / SCALES["T_c"], loc="initial")
    builder.add_boundary_condition(lambda x, u: x["Y_f"], val=y_f_init, loc="initial")

    # BUILD
    builder.build()

    # 6. Patch initial guess with physics-consistent trajectories
    _patch_initial_guess(builder, n_coll, geo, p_int, T_int, theta_grid)

    # 7. Periodicity Constraints (kinematic only)
    # _add_periodicity_constraints(builder)

    # 8. Objectives with regularization
    objective, p_max_sym, true_work = _calculate_objectives(
        builder, n_coll, geo, duration, omega_val
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

    # Regularization weights
    # Smoothness: Penalize jerk (improves convergence speed)
    # Magnitude: Penalize large effort (improves uniqueness/convexity)
    w_smooth = 1e-1
    w_mag = 1e-2

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
        objective + (w_smooth * acc_smoothness) + (w_mag * acc_magnitude) + (w_prox * state_prox)
    )

    builder.set_objective(objective_regularized)

    # Debug: Relax feasibility if requested
    if debug_feasibility:
        # Use a large penalty to prioritize satisfying constraints if possible
        builder.relax_constraints(penalty=1.0e5)

    # Create Diagnostics Function
    res = builder.export_nlp()

    if isinstance(res, tuple):
        w_vec = res[0]["x"]
        diag_fn = ca.Function(
            "get_diagnostics", [w_vec], [p_max_sym, true_work], ["w"], ["p_max", "work_j"]
        )
        res[1]["diagnostics_fn"] = diag_fn
        res[1]["scales"] = SCALES  # Export scales for post-processing

        # Export indices for diagnostics
        # Convert to standard list of ints for JSON serialization/usage
        if "acc" in builder._var_indices:
            res[1]["control_indices"] = [int(i) for i in builder._var_indices["acc"]]
        if "x" in builder._var_indices:
            res[1]["state_indices_x"] = [int(i) for i in builder._var_indices["x"]]

    return res


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
) -> tuple[Any, Any, Any]:
    """Calculate the optimization objective and diagnostic values."""
    j_obj = 0.0
    true_work = 0.0
    dt_step = duration / n_coll

    # Derived Geometry
    area_piston = np.pi * geo.B**2 / 4.0
    vol_clearance = geo.V_c

    p_history = []

    # Import SCALES to de-scale variables
    from thermo.nlp import build_thermo_nlp
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
        # OP Symmetric: 2x Cylinder Volume
        vol_k = 2.0 * (vol_clearance + area_piston * x_phys)

        # Pressure (Ideal Gas, Physical)
        p_k = m_phys * 287.0 * T_phys / vol_k
        p_history.append(p_k)

        # Max Pressure Constraint (Soft Penalty)
        # Objective: Soft Penalty for P > 250 bar
        p_viol = ca.fmax(0.0, (p_k - 250e5) / 250e5)
        j_obj += 100.0 * p_viol**2

        # Objective: Max Work <=> Min -Work
        # Work Rate = P * dV/dtheta = P * (2 * A) * v_phys (since v_phys is m/rad)
        work_rate = p_k * (2.0 * area_piston) * v_phys
        term_work = work_rate  # J/rad

        # Smoothing
        smooth = 1e-6 * acc_k_scaled**2

        j_obj += (-term_work + smooth) * dt_step
        true_work += term_work * dt_step

    p_max_sym = ca.mmax(ca.vertcat(*p_history))
    return j_obj, p_max_sym, true_work
