"""
Refactored NLP construction for Phase 3: "Breathing Gears" Optimization (Litvin-Compatible).
"""

from __future__ import annotations

import math
from typing import Any

import casadi as ca
import numpy as np

from campro.litvin.casadi_litvin import casadi_conjugacy_residual
from campro.logging import get_logger
from campro.optimization.numerical.collocation_builder import CollocationBuilder

log = get_logger(__name__)


def generate_bspline_basis(knots, grid, degree=3, derivative=0):
    """
    Generate B-spline Basis Matrix M such that Values = M @ Weights.
    Uses generic B-spline evaluation.
    This is calculated numerically (numpy) once during setup.
    """
    # Simple uniform B-spline basis for now
    # Scipy is useful here
    try:
        from scipy.interpolate import BSpline
    except ImportError:
        raise ImportError("Scipy required for B-spline basis generation")

    n_weights = len(knots) - degree - 1
    # Create identity weights to get basis functions
    M = []
    for i in range(n_weights):
        # Weight vector with 1 at i, 0 elsewhere
        c = np.zeros(n_weights)
        c[i] = 1.0
        spl = BSpline(knots, c, degree)
        if derivative > 0:
            spl = spl.derivative(derivative)
        M.append(spl(grid))

    return np.array(M).T  # Shape (N_points, N_weights)


def build_collocation_nlp(
    P: dict[str, Any],
    return_builder: bool = False,
    initial_trajectory: dict[str, Any] | None = None,
) -> (
    tuple[dict[str, Any], dict[str, Any]]
    | tuple[dict[str, Any], dict[str, Any], CollocationBuilder]
):
    """
    Build Phase 3 NLP: Breathing Gears Optimization.
    Optimizes global B-spline parameters for planet/ring radii to match target ratio
    while satisfying kinematic (Litvin) constraints.
    """

    # 1. Setup Grid
    num = P.get("num", {})
    K = int(num.get("K", 40))
    C = int(num.get("C", 3))
    phi_horizon = 2.0 * math.pi

    # 2. B-Spline Configuration
    n_knots = int(num.get("n_knots", 10))
    degree = 3
    # Uniform clamped knots? Or periodic?
    # Periodic Spline logic is ideal for gears.
    # For now, use standard clamped with periodic constraint on params later.
    t = np.linspace(0, phi_horizon, n_knots)
    # Augment knots for clamping/periodicity (simple approach first)
    # Actually, let's use global variables w_rp, w_rr directly in dynamics?
    # No, use the lifted approach: Controls r, R linked to w via constraints.

    builder = CollocationBuilder(time_horizon=phi_horizon, n_points=K, method="radau", degree=C)

    # 3. Define Variables (Controls/States)

    # "Breathing" Variables (Node-based, driven by Splines)
    # Bounds will be set, but also constrained by splines
    bounds_cfg = P.get("bounds", {})
    r_min = bounds_cfg.get("r_min", 0.005)
    r_max = bounds_cfg.get("r_max", 0.100)

    builder.add_control("r_planet", bounds=(r_min, r_max))
    builder.add_control("R_ring", bounds=(r_min, r_max))
    builder.add_control("r_journal", bounds=(0.0, 0.05))  # Rod offset

    # Contact State (Algebraic variable for Litvin check)
    # phi_contact: Roll angle on the ring gear at contact
    # Bounds: +/- 1.0 rad? Depends on tooth depth.
    # Ring active flank is roughly +/- sqrt((ra/rb)^2 - 1).
    # Let's say +/- 0.5 rad is safe range.
    builder.add_control("phi_contact", bounds=(-0.5, 0.5))

    # Kinematic State
    builder.add_state("psi", bounds=(-100.0, 100.0), initial=0.0)

    # 4. Dynamics
    def dynamics_mech(x, u, p=None):
        r = u["r_planet"]
        R = u["R_ring"]
        # Ratio i = R/r
        # dpsi/dphi = i - 1
        return {"psi": (R / r) - 1.0}

    builder.set_dynamics(dynamics_mech)
    builder.build()

    nl, meta = builder.export_nlp()

    # 5. Add Global B-Spline Parameters (Decision Variables)
    # Weights for r_planet, R_ring, r_journal
    # Let's assume N_W weights per curve.
    N_W = n_knots  # Simplified

    # Add to X
    w_rp = ca.SX.sym("w_rp", N_W)
    w_rr = ca.SX.sym("w_rr", N_W)
    w_rj = ca.SX.sym("w_rj", N_W)

    nl["x"] = ca.vertcat(nl["x"], w_rp, w_rr, w_rj)

    # Initial Guess for Weights (Flat)
    w0_base = np.array(meta["w0"])
    w_rp_0 = np.full(N_W, 0.025)
    w_rr_0 = np.full(N_W, 0.050)
    w_rj_0 = np.full(N_W, 0.010)
    meta["w0"] = np.concatenate([w0_base, w_rp_0, w_rr_0, w_rj_0])

    # Bounds for Weights
    lbw_base = np.array(meta["lbw"])
    ubw_base = np.array(meta["ubw"])

    meta["lbw"] = np.concatenate(
        [lbw_base, np.full(N_W, r_min), np.full(N_W, r_min), np.full(N_W, 0.0)]
    )
    meta["ubw"] = np.concatenate(
        [ubw_base, np.full(N_W, r_max), np.full(N_W, r_max), np.full(N_W, 0.05)]
    )

    # 6. Constraints to link Splines to Controls
    # We evaluate Splines at the Grid Points (Collocation points too?)
    # builder.get_time_grid() gives State nodes (K+1).
    # Controls are typically K segments.
    # Collocation is Radau, controls are usually piecewise polynomial or constant?
    # In 'driver.py' we saw controls size K.
    # CollocationBuilder.py: "Controls are piecewise constant" (degree=0?) or polynomial?
    # Usually CasADi collocation examples use piecewise constant U.
    # But if we want smooth derivatives d/dphi, piecewise constant U is bad for Litvin!
    # The Global Splines w_rp provide C2 smoothness.
    # We should enforce that the Control variables U_k MATCH the Spline value at the Time Node `t_k`.
    # AND, explicit derivatives dR/dphi, dr/dphi should come from the Spline analytic derivative.

    # Extract grid for controls
    # Controls are usually defined at the *start* of the interval k.
    # time_grid has K+1 points. U has K points.
    grid_full = np.array(builder.get_time_grid())
    grid_u = grid_full[:-1]  # Start points

    # Generate Basis M for grid_u
    # We need knots covering the domain.
    # Standard uniform knots with degree pad
    knots = np.linspace(0, phi_horizon, N_W - degree + 1)  # Internal
    pad = (np.max(knots) - np.min(knots)) / len(knots)
    knots = np.concatenate(
        [
            [knots[0] - pad * i for i in range(degree, 0, -1)],
            knots,
            [knots[-1] + pad * i for i in range(1, degree + 1)],
        ]
    )

    M_val = generate_bspline_basis(knots, grid_u, degree, derivative=0)
    M_der = generate_bspline_basis(knots, grid_u, degree, derivative=1)

    # Vectorized Spline Evaluation (SX Compatible via Matrix Mult)
    # Weights w_rp are SX vectors. M_val is numeric matrix.
    # r_vals_vec = M_val @ w_rp
    r_vals_vec = ca.mtimes(M_val, w_rp)
    R_vals_vec = ca.mtimes(M_val, w_rr)
    rj_vals_vec = ca.mtimes(M_val, w_rj)

    r_prime_vec = ca.mtimes(M_der, w_rp)
    R_prime_vec = ca.mtimes(M_der, w_rr)

    # Global Config
    track_weight = P.get("weights", {}).get("track", 10.0)
    eff_weight = P.get("weights", {}).get("eff", 1.0)
    smooth_weight = P.get("weights", {}).get("smooth", 0.1)

    # Ring Base Radius (Fixed Parameter)
    # If not provided, assume standard 20 deg pressure angle at INITIAL configuration
    # rb = R_pitch * cos(alpha)
    # Let's demand it in P or derive from init R_ring
    R_nominal = (r_min + r_max) / 2  # This should be R_ring nominal, not r_planet
    alpha_std = 20.0 * math.pi / 180.0
    rb_ring = P.get("rb_ring", R_nominal * math.cos(alpha_std))

    # Load Profile Interpolant
    load_profile = P.get("load_profile")
    F_gas_func = None
    if load_profile:
        angles = np.array(load_profile.get("angle", np.linspace(0, 2 * np.pi, 100)))
        forces = np.array(load_profile.get("F_gas", np.zeros_like(angles)))
        # Sort
        idx = np.argsort(angles)
        F_gas_func = ca.interpolant("F_gas", "linear", [angles[idx]], forces[idx])
    else:
        F_gas_func = ca.Function("F_gas_zero", [ca.MX.sym("phi")], [0.0])

    target_profile = P.get("target_ratio_profile")

    # Objectives
    J = 0.0
    g_spline = []
    lbg_spline = []
    ubg_spline = []

    # Additional constraints (e.g. Conjugacy)
    g_litvin = []
    lbg_litvin = []
    ubg_litvin = []

    # Get controls list from builder
    r_ctrls = builder._U["r_planet"]
    R_ctrls = builder._U["R_ring"]
    rj_ctrls = builder._U["r_journal"]
    phi_c_ctrls = builder._U["phi_contact"]
    psi_states = builder._X["psi"]

    # Grid for constraints (Start of each interval)
    grid_full = np.array(builder.get_time_grid())
    grid_u = grid_full[:-1]

    for k in range(K):
        phi_k = grid_u[k]

        # 1. Spline Evaluation (Value & Derivative)
        r_val = r_vals_vec[k]
        R_val = R_vals_vec[k]
        rj_val = rj_vals_vec[k]

        r_prime = r_prime_vec[k]
        R_prime = R_prime_vec[k]
        # rj_prime unused for now

        # 2. Link Controls to Splines (Equality)
        g_spline.append(r_ctrls[k] - r_val)
        g_spline.append(R_ctrls[k] - R_val)
        g_spline.append(rj_ctrls[k] - rj_val)

        lbg_spline.extend([0.0, 0.0, 0.0])
        ubg_spline.extend([0.0, 0.0, 0.0])

        # 3. Litvin Conjugacy Constraint
        # center distance d = R - r
        # d' = R' - r'
        d_k = R_ctrls[k] - r_ctrls[k]
        d_prime_k = R_prime - r_prime

        # Planet angle theta_p (approx psi)
        # Dynamics: dpsi/dphi = R/r - 1
        # theta_p' = R/r - 1
        i_ratio = R_ctrls[k] / r_ctrls[k]
        theta_p_k = psi_states[k]
        theta_p_prime_k = i_ratio - 1.0

        # Contact Angle
        phi_c = phi_c_ctrls[k]

        conjugacy_res = casadi_conjugacy_residual(
            rb_ring,
            phi_k,  # theta_r (Ring Angle)
            d_k,
            theta_p_k,
            phi_c,
            d_prime_k,
            theta_p_prime_k,
        )
        g_litvin.append(conjugacy_res)
        lbg_litvin.append(0.0)
        ubg_litvin.append(0.0)

        # 4. Objectives

        # Ratio Tracking
        if target_profile is not None:
            idx = int(k * len(target_profile) / K)
            i_target = target_profile[idx]
            J += track_weight * (i_ratio - i_target) ** 2

        # Efficiency (Friction Work)
        # F_gas
        F_gas = F_gas_func(phi_k)
        # dxL/dphi = -d * sin(psi) * (i-1)
        dx_dphi = -d_k * ca.sin(theta_p_k) * (theta_p_prime_k)

        # Normal Force (approx)
        # Normal Force (approx)
        # alpha_op is defined by cos(alpha_op) = rb_ring / R_ring
        # So cos_alpha = rb_ring / R_ctrls[k]
        # We clamp denominator to avoid singular forces if cos_alpha -> 0 (though R < rb is invalid anyway)
        cos_alpha = rb_ring / (R_ctrls[k] + 1e-6)
        # If R < rb, cos_alpha > 1, physically impossible for involute but numerically we just want smoothness.
        # But wait, if R < rb, contact happens below base circle?
        # Let's just use the ratio blindly for smoothness.
        N_c = ca.fabs(F_gas) / (cos_alpha + 1e-4)

        mu = 0.05
        T_loss = mu * N_c * r_ctrls[k]
        J += eff_weight * (T_loss**2)

        # Smoothness (Penalize Spline Weights variance or Derivatives)
        # Here we penalize derivatives magnitude
        J += smooth_weight * (r_prime**2 + R_prime**2)

    # 6. Global Piston Path Constraints
    # (Optional, reusing builder's X bounds for psi)

    # Aggregate Constraints
    nl["g"] = ca.vertcat(nl["g"], *g_spline, *g_litvin)
    lbg_final = np.concatenate([meta["lbg"], lbg_spline, lbg_litvin])
    ubg_final = np.concatenate([meta["ubg"], ubg_spline, ubg_litvin])

    # Add global variables to variable_groups so driver can extract them
    # Current x size before appending:
    n_vars_ocp = nl["x"].size1() - 3 * N_W

    # Indices for w_rp, w_rr, w_rj
    # Range [n_vars_ocp, n_vars_ocp + N_W)
    vb = builder._var_indices.copy()
    vb["w_rp"] = list(range(n_vars_ocp, n_vars_ocp + N_W))
    vb["w_rr"] = list(range(n_vars_ocp + N_W, n_vars_ocp + 2 * N_W))
    vb["w_rj"] = list(range(n_vars_ocp + 2 * N_W, n_vars_ocp + 3 * N_W))

    meta_data = {
        "n_vars": nl["x"].size1(),
        "n_constraints": nl["g"].size1(),
        "w0": meta["w0"],
        "lbw": meta["lbw"],
        "ubw": meta["ubw"],
        "lbg": lbg_final,
        "ubg": ubg_final,
        "variable_groups": vb,
        "K": K,
        "C": C,
        "time_grid": np.array(builder.get_time_grid()),
    }

    nl["f"] = J

    return nl, meta_data
