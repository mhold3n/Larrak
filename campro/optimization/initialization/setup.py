"""Initialization setup utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


# Scaling group configuration: defines max_ratio and log-scaling settings per variable group
# max_ratio: Maximum allowed ratio between min and max scales within a group
# use_log_scale: Whether variables in this group should be transformed to log space in NLP
SCALING_GROUP_CONFIG = {
    "positions": {"max_ratio": 10.0},
    "velocities": {"max_ratio": 10.0},
    "pressures": {"max_ratio": 1e3, "use_log_scale": True},  # 1000:1 ratio allowed
    "densities": {"max_ratio": 1e2, "use_log_scale": True},  # 100:1 ratio allowed
    "temperatures": {"max_ratio": 10.0},
    "valve_areas": {"max_ratio": 1e3, "use_log_scale": True},  # 1000:1 ratio allowed
    "ignition": {"max_ratio": 10.0},
    "scavenging_fractions": {"max_ratio": 10.0},  # yF (dimensionless)
    "scavenging_masses": {"max_ratio": 100.0},  # Mdel, Mlost (kg)
    "scavenging_area_integrals": {"max_ratio": 1000.0},  # AinInt, AexInt (m^2*s)
    "scavenging_time_moments": 5e-5,  # m^2*s^2 (AinTmom, AexTmom)
    "cycle_time": {"max_ratio": 10.0},  # T_cycle (s)
}


def compute_interior_point(
    lb: float,
    ub: float,
    margin: float = 0.05,
) -> float:
    """
    Compute interior point within bounds with given margin.

    Returns a point that is margin% inside the interval from the lower bound.
    For unbounded cases, returns a reasonable default.

    Args:
        lb: Lower bound (can be -inf)
        ub: Upper bound (can be inf)
        margin: Margin factor (0.05 = 5% inside from lower bound)

    Returns:
        Interior point value
    """
    # Handle fully unbounded case
    if np.isinf(lb) and np.isinf(ub):
        return 0.0

    # Handle semi-bounded cases
    if np.isinf(lb):
        # Upper bound only: return something below ub
        return ub - abs(ub) * margin if abs(ub) > 1e-6 else ub - 0.1
    if np.isinf(ub):
        # Lower bound only: return something above lb
        return lb + abs(lb) * margin if abs(lb) > 1e-6 else lb + 0.1

    # Bounded case: linear interpolation
    width = ub - lb
    if width <= 0:
        return lb  # Degenerate interval

    return lb + width * margin


def clamp_initial_guess(
    x0: np.ndarray[Any, Any],
    lbx: np.ndarray[Any, Any],
    ubx: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """
    Clamp initial guess to bounds and validate.

    Ensures all values in x0 are within [lbx, ubx] bounds.
    Logs warnings when values are clamped.

    Args:
        x0: Initial guess array
        lbx: Lower bounds
        ubx: Upper bounds

    Returns:
        Clamped initial guess array
    """
    # Check for NaN/Inf in x0
    if not np.all(np.isfinite(x0)):
        n_nan = np.sum(~np.isfinite(x0))
        log.warning(f"Initial guess contains {n_nan} NaN/Inf values, replacing with 0.0")
        x0 = np.nan_to_num(x0, nan=0.0, posinf=1.0, neginf=-1.0)

    # Check for bound violations
    violations_lower = x0 < lbx
    violations_upper = x0 > ubx
    n_lower = np.sum(violations_lower)
    n_upper = np.sum(violations_upper)

    if n_lower > 0 or n_upper > 0:
        log.warning(
            f"Initial guess outside bounds: {n_lower} lower violations, "
            f"{n_upper} upper violations. Clamping to bounds.",
        )
        # Clone to avoid modifying input if it's a view
        x0 = x0.copy()
        x0 = np.clip(x0, lbx, ubx)

    return x0


def generate_physics_based_initial_guess(
    n_vars: int, params: dict[str, Any]
) -> np.ndarray[Any, Any]:
    """
    Generate physics-based initial guess.
    Uses quasi-steady approximation with interior point seeding.
    """
    x0 = np.zeros(n_vars)

    # 1. Extract Context
    ctx = _extract_simulation_context(params)

    # 2. Pre-calculate Thermo Limits
    thermo_limits = _calculate_thermo_limits(ctx, params.get("bounds", {}))

    # 3. Trajectory Generation
    n_per_point = 6
    n_points = n_vars // n_per_point

    # Interior points for fallback
    interior = _get_interior_points(params.get("bounds", {}))

    for i in range(n_points):
        idx = i * n_per_point
        t = i * ctx["dt"]
        phase = (t / ctx["cycle_time"]) if ctx["cycle_time"] > 0 else 0.0

        # A. Kinematics
        pos_left, vel_left, pos_right, vel_right = _compute_piston_kinematics(phase, ctx)

        # B. Thermodynamics
        rho, temp = _compute_thermo_state(phase, ctx, thermo_limits)

        # C. Assign with Safety Checks
        _assign_state_to_guess(
            x0,
            idx,
            n_vars,
            (pos_left, vel_left, pos_right, vel_right, rho, temp),
            interior,
        )

    # 4. Final Cleanup (Gaps and NaNs)
    _finalize_initial_guess(x0, n_vars, n_per_point, ctx["gap_min"], interior)

    log.info(
        f"Generated robust physics-based initial guess for {n_vars} variables using quasi-steady approximation"
    )
    return x0


def _extract_simulation_context(params: dict[str, Any]) -> dict[str, Any]:
    """Extract and derive all necessary simulation constants."""
    geom = params.get("geometry", {})
    combustion = params.get("combustion", {})
    bounds = params.get("bounds", {})
    thermo = params.get("thermodynamics", {})

    # Basic
    cycle_time = combustion.get("cycle_time_s", 1.0)
    stroke = geom.get("stroke", 0.1)

    # Derived
    return {
        "cycle_time": cycle_time,
        "stroke": stroke,
        "bore": geom.get("bore", 0.1),
        "compression_ratio": geom.get("compression_ratio", 10.0),
        "clearance_volume": geom.get("clearance_volume", 1e-4),
        "gap_min": bounds.get("gap_min", 0.0008),
        "R": thermo.get("R", 287.0),
        "gamma": thermo.get("gamma", 1.4),
        "p_initial": combustion.get("initial_pressure_Pa", 1e5),
        "T_initial": combustion.get("initial_temperature_K", 300.0),
        "xL_min": bounds.get("xL_min", -0.1),
        "xL_max": bounds.get("xL_max", 0.1),
        "xR_min": bounds.get("xR_min", 0.0),
        "xR_max": bounds.get("xR_max", 0.2),
        "dt": cycle_time / max(1, 49),  # Default 50 point approx
    }


def _calculate_thermo_limits(ctx: dict[str, Any], bounds: dict[str, Any]) -> dict[str, float]:
    """Pre-calculate compressed and expanded states."""
    # Geometry
    area_piston = np.pi * (ctx["bore"] / 2.0) ** 2
    v_max = ctx["clearance_volume"] + area_piston * ctx["stroke"]
    v_compressed = v_max / ctx["compression_ratio"]

    # Compression (Polytropic)
    n_poly = ctx["gamma"] * 0.9
    p_comp = ctx["p_initial"] * ((v_max / v_compressed) ** n_poly)
    t_comp = ctx["T_initial"] * ((v_max / v_compressed) ** (n_poly - 1))

    # Clamping
    p_comp = min(p_comp, bounds.get("p_max", 10e6) * 0.8)
    t_comp = min(t_comp, bounds.get("T_max", 2000.0) * 0.8)
    rho_comp = p_comp / (ctx["R"] * t_comp)

    # Expansion (Scavenging loss)
    scavenging_factor = 0.7
    p_exp = p_comp * ((v_compressed / v_max) ** n_poly)
    t_exp = t_comp * ((v_compressed / v_max) ** (n_poly - 1))

    # Clamping
    p_exp = max(p_exp, bounds.get("p_min", 0.01e6) * 1.2)
    t_exp = max(t_exp, bounds.get("T_min", 200.0) * 1.2)
    rho_exp = (p_exp / (ctx["R"] * t_exp)) * scavenging_factor

    return {
        "rho_initial": ctx["p_initial"] / (ctx["R"] * ctx["T_initial"]),
        "rho_compressed": rho_comp,
        "T_compressed": t_comp,
        "rho_expanded": rho_exp,
        "T_expanded": t_exp,
        "p_compressed": p_comp,  # stored for potential debug, technically redundant with rho/T
    }


def _get_interior_points(bounds: dict[str, float]) -> dict[str, float]:
    """Compute safe fallback values."""
    return {
        "xL": compute_interior_point(bounds.get("xL_min", -0.1), bounds.get("xL_max", 0.1)),
        "xR": compute_interior_point(bounds.get("xR_min", 0.0), bounds.get("xR_max", 0.2)),
        "vL": compute_interior_point(bounds.get("vL_min", -50), bounds.get("vL_max", 50)),
        "vR": compute_interior_point(bounds.get("vR_min", -50), bounds.get("vR_max", 50)),
        "rho": compute_interior_point(bounds.get("rho_min", 0.01), bounds.get("rho_max", 100)),
        "T": compute_interior_point(bounds.get("T_min", 200), bounds.get("T_max", 3000)),
    }


def _compute_piston_kinematics(
    phase: float, ctx: dict[str, Any]
) -> tuple[float, float, float, float]:
    """Compute sinusoidal piston motion."""
    theta = 2.0 * np.pi * phase

    # Centers and Amplitudes
    pos_left_center = (ctx["xL_min"] + ctx["xL_max"]) / 2.0
    pos_right_center = (ctx["xR_min"] + ctx["xR_max"]) / 2.0

    # Amplitude limited by stroke or bounds
    amp_left = min(ctx["stroke"] * 0.4, (ctx["xL_max"] - ctx["xL_min"]) * 0.4)
    amp_right = min(ctx["stroke"] * 0.4, (ctx["xR_max"] - ctx["xR_min"]) * 0.4)

    # Position
    pos_left = pos_left_center + amp_left * np.cos(theta)
    pos_right = pos_right_center - amp_right * np.cos(theta)

    # Velocity
    omega = 2.0 * np.pi / ctx["cycle_time"] if ctx["cycle_time"] > 0 else 0.0
    vel_left = -amp_left * omega * np.sin(theta)
    vel_right = amp_right * omega * np.sin(theta)

    # Gap enforcement
    gap = pos_right - pos_left
    if gap < ctx["gap_min"]:
        diff = ctx["gap_min"] - gap
        pos_left -= diff / 2.0
        pos_right += diff / 2.0

    return pos_left, vel_left, pos_right, vel_right


def _compute_thermo_state(
    phase: float, ctx: dict[str, Any], limits: dict[str, float]
) -> tuple[float, float]:
    """Interpolate thermo state based on phase."""
    rho_i, rho_c, rho_e = limits["rho_initial"], limits["rho_compressed"], limits["rho_expanded"]
    t_i, t_c, t_e = ctx["T_initial"], limits["T_compressed"], limits["T_expanded"]

    if phase < 0.25:
        # Compression Start
        prog = phase / 0.25
        rho = rho_i + (rho_c - rho_i) * prog**1.5
        temp = t_i + (t_c - t_i) * prog**1.3
    elif phase < 0.5:
        # Compression End / TDC
        prog = (phase - 0.25) / 0.25
        rho = rho_c * (1.0 + 0.05 * np.sin(prog * np.pi))
        temp = t_c * (1.0 - 0.05 * prog)
    elif phase < 0.75:
        # Expansion
        prog = (phase - 0.5) / 0.25
        rho = rho_c * (1.0 - prog * 0.6)
        temp = t_c * (1.0 - prog * 0.4)
    else:
        # Scavenging
        prog = (phase - 0.75) / 0.25
        rho = rho_e + (rho_i - rho_e) * prog**0.7
        temp = t_e + (t_i - t_e) * prog**0.7

    return rho, temp


def _assign_state_to_guess(
    x0: np.ndarray,
    idx: int,
    n_vars: int,
    values: tuple[float, float, float, float, float, float],
    interior: dict[str, float],
) -> None:
    """Safe assignment with bounds/NaN checking."""
    pos_left, vel_left, pos_right, vel_right, rho, temp = values

    # Map to indices 0..5 locally, check limits
    to_assign = [
        (idx, pos_left, interior["xL"]),
        (idx + 1, vel_left, interior["vL"]),
        (idx + 2, pos_right, interior["xR"]),
        (idx + 3, vel_right, interior["vR"]),
        (idx + 4, rho, interior["rho"]),
        (idx + 5, temp, interior["T"]),
    ]

    for i, val, fallback in to_assign:
        if i < n_vars:
            x0[i] = val if np.isfinite(val) else fallback


def _finalize_initial_guess(
    x0: np.ndarray,
    n_vars: int,
    n_per: int,
    gap_min: float,
    interior: dict[str, float],
) -> None:
    """Final pass to fix gaps and NaNs."""
    n_points = n_vars // n_per

    # Fix gaps
    _fix_trajectory_gaps(x0, n_points, n_per, gap_min)

    # Fix NaNs globally
    _replace_nans_with_defaults(x0, n_vars, n_per, interior)


def _fix_trajectory_gaps(x0: np.ndarray, n_points: int, n_per: int, gap_min: float) -> None:
    for i in range(n_points):
        idx = i * n_per
        if idx + 2 < len(x0):
            gap = x0[idx + 2] - x0[idx]
            if gap < gap_min:
                x0[idx + 2] = x0[idx] + gap_min


def _replace_nans_with_defaults(
    x0: np.ndarray, n_vars: int, n_per: int, interior: dict[str, float]
) -> None:
    for i in range(n_vars):
        if not np.isfinite(x0[i]):
            rem = i % n_per
            if rem == 0:
                x0[i] = interior["xL"]
            elif rem == 1:
                x0[i] = interior["vL"]
            elif rem == 2:
                x0[i] = interior["xR"]
            elif rem == 3:
                x0[i] = interior["vR"]
            elif rem == 4:
                x0[i] = interior["rho"]
            elif rem == 5:
                x0[i] = interior["T"]
            else:
                x0[i] = 0.0


def apply_problem_bounds(
    lbx: np.ndarray[Any, Any],
    ubx: np.ndarray[Any, Any],
    lbg: np.ndarray[Any, Any],
    ubg: np.ndarray[Any, Any],
    params: dict[str, Any],
) -> None:
    """Apply problem-specific bounds.

    If CEM envelope is available in params['_cem_envelope'], uses those
    bounds instead of hardcoded defaults for boost/fuel ranges.
    """
    # Get problem parameters
    constraints = params.get("constraints", {})

    # === CEM Envelope Integration ===
    # If CEM was queried in driver.py, use its envelope bounds
    cem_envelope = params.get("_cem_envelope")
    if cem_envelope is not None:
        log.info("Applying CEM envelope bounds to optimization")
        # CEM provides boost_range and fuel_range
        # These map to pressure/density bounds in the NLP
        boost_min, boost_max = cem_envelope.boost_range
        fuel_min, fuel_max = cem_envelope.fuel_range

        # Convert boost (bar) to pressure bounds (Pa)
        # boost_range is intake pressure in bar
        p_boost_min = boost_min * 1e5  # bar to Pa
        p_boost_max = boost_max * 1e5

        # Update constraints with CEM bounds (override defaults)
        constraints = {
            **constraints,
            "_cem_boost_min": p_boost_min,
            "_cem_boost_max": p_boost_max,
            "_cem_fuel_min": fuel_min,
            "_cem_fuel_max": fuel_max,
        }
        log.debug(
            f"CEM bounds: boost={boost_min:.2f}-{boost_max:.2f} bar, "
            f"fuel={fuel_min:.4f}-{fuel_max:.4f}"
        )

    # Piston position bounds
    pos_left_min = constraints.get("x_L_min", 0.01)
    pos_left_max = constraints.get("x_L_max", 0.45)
    pos_right_min = constraints.get("x_R_min", 0.55)
    pos_right_max = constraints.get("x_R_max", 0.99)

    # State bounds (rho, T)
    lbx[4::6] = 0.1  # Min density (assuming rho is at index 4 in each block of 6)
    ubx[4::6] = 100.0  # Max density

    temp_min = constraints.get("T_min", 200.0)
    temp_max = constraints.get("T_max", 3000.0)
    lbx[5::6] = temp_min  # Min temperature (assuming T is at index 5)
    ubx[5::6] = temp_max

    # Constraint bounds (gap > min_gap)
    min_gap = constraints.get("min_gap", 0.01)
    lbg[:] = min_gap
    ubg[:] = np.inf

    # Apply bounds to variables (assuming specific ordering)
    # This is a simplified version - in practice, you'd need to know the exact variable ordering
    n_vars = len(lbx)
    n_per_point = 6  # x_L, v_L, x_R, v_R, rho, T

    # Piston velocity bounds
    v_max = constraints.get("v_max", 10.0)

    for i in range(0, n_vars, n_per_point):
        if i < n_vars:
            lbx[i] = pos_left_min  # x_L
            ubx[i] = pos_left_max
        if i + 1 < n_vars:
            lbx[i + 1] = -v_max  # v_L
            ubx[i + 1] = v_max
        if i + 2 < n_vars:
            lbx[i + 2] = pos_right_min  # x_R
            ubx[i + 2] = pos_right_max
        if i + 3 < n_vars:
            lbx[i + 3] = -v_max  # v_R
            ubx[i + 3] = v_max
        if i + 4 < n_vars:
            lbx[i + 4] = 0.1  # rho (density)
            ubx[i + 4] = 100.0
        if i + 5 < n_vars:
            lbx[i + 5] = temp_min  # T (temperature)
            ubx[i + 5] = temp_max


def setup_optimization_bounds(
    n_vars: int,
    n_constraints: int,
    params: dict[str, Any],
    builder: Any = None,
    warm_start: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> tuple[np.ndarray | None, ...]:
    """
    Setup initial guess (x0) and bounds (lbx, ubx, lbg, ubg).
    Tries NLP builder first, then warm start, then physics-based fallback.
    """
    try:
        # 1. Try NLP Builder export
        if builder is not None:
            res = _try_nlp_initialization(builder, n_vars, n_constraints, warm_start, meta)
            if res is not None:
                return res

        # 2. Fallback
        return _setup_fallback_bounds(n_vars, n_constraints, params, warm_start, meta)

    except Exception as e:
        log.error(f"Failed to set up optimization bounds: {e!s}")
        return None, None, None, None, None, np.array([])


def _try_nlp_initialization(
    builder: Any,
    n_vars: int,
    n_constraints: int,
    warm_start: dict[str, Any] | None,
    meta: dict[str, Any] | None,
) -> tuple[np.ndarray, ...] | None:
    """Attempt to extract bounds and init from NLP builder."""
    nlp_dat = builder.export_nlp()
    if not isinstance(nlp_dat, tuple):
        return None

    nlp_def, _ = nlp_dat
    x0 = np.array(nlp_def["x0"])
    lbx = np.array(nlp_def.get("lbx", -np.inf * np.ones(n_vars)))
    ubx = np.array(nlp_def.get("ubx", np.inf * np.ones(n_vars)))
    lbg = np.array(nlp_def.get("lbg", -np.inf * np.ones(n_constraints)))
    ubg = np.array(nlp_def.get("ubg", np.inf * np.ones(n_constraints)))

    # Apply Warm Start if valid
    if warm_start and "x0" in warm_start:
        ws_x0 = np.array(warm_start["x0"])
        if len(ws_x0) == len(x0):
            x0 = ws_x0
            log.info("Applied warm start to NLP initialization")

    # Scaling
    x0 = clamp_initial_guess(x0, lbx, ubx)
    _apply_log_scaling(x0, meta)

    return x0, lbx, ubx, lbg, ubg, np.array([])


def _setup_fallback_bounds(
    n_vars: int,
    n_constraints: int,
    params: dict[str, Any],
    warm_start: dict[str, Any] | None,
    meta: dict[str, Any] | None,
) -> tuple[np.ndarray | None, ...]:
    log.info("NLP-provided bounds/initial guess not available, generating from problem parameters")

    lbx = -np.inf * np.ones(n_vars)
    ubx = np.inf * np.ones(n_vars)
    lbg = -np.inf * np.ones(n_constraints)
    ubg = np.inf * np.ones(n_constraints)

    if warm_start and "x0" in warm_start:
        x0 = np.array(warm_start["x0"])
        if len(x0) != n_vars:
            log.warning(f"Warm start x0 length {len(x0)} != problem size {n_vars}")
            x0 = generate_physics_based_initial_guess(n_vars, params)
    else:
        x0 = generate_physics_based_initial_guess(n_vars, params)

    apply_problem_bounds(lbx, ubx, lbg, ubg, params)
    x0 = clamp_initial_guess(x0, lbx, ubx)
    _apply_log_scaling(x0, meta)

    return x0, lbx, ubx, lbg, ubg, np.array([])


def _apply_log_scaling(x0: np.ndarray, meta: dict[str, Any] | None) -> None:
    variable_groups = meta.get("variable_groups", {}) if meta else {}
    log_space_groups = ["densities", "valve_areas"]

    for group_name in log_space_groups:
        if group_name in variable_groups:
            group_indices = variable_groups[group_name]
            group_config = SCALING_GROUP_CONFIG.get(group_name, {})
            if isinstance(group_config, dict) and group_config.get("use_log_scale", False):
                for idx in group_indices:
                    if 0 <= idx < len(x0):
                        x0[idx] = np.log(max(x0[idx], 1e-3))
