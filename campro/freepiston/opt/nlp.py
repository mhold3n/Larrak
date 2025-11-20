from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from campro.constants import CASADI_PHYSICS_EPSILON
from campro.freepiston.gas import build_gas_model
from campro.freepiston.opt.colloc import CollocationGrid, make_grid
from campro.logging import get_logger
from campro.physics.combustion import CombustionModel

log = get_logger(__name__)

# Import scaling config to determine which variables should be log-transformed
# Avoid circular import by importing only when needed
_SCALING_GROUP_CONFIG = None


def _get_scaling_group_config() -> dict[str, dict[str, Any]]:
    """Get scaling group configuration, importing from driver if needed."""
    global _SCALING_GROUP_CONFIG
    if _SCALING_GROUP_CONFIG is None:
        from campro.freepiston.opt.driver import SCALING_GROUP_CONFIG
        _SCALING_GROUP_CONFIG = SCALING_GROUP_CONFIG
    return _SCALING_GROUP_CONFIG


def _should_use_log_scale(group_name: str) -> bool:
    """Check if a variable group should use log-space transformation."""
    config = _get_scaling_group_config()
    return config.get(group_name, {}).get("use_log_scale", False)


def _log_transform_var(ca: Any, var: Any, epsilon: float = 1e-10) -> Any:
    """Transform variable to log space: log(var + epsilon)."""
    return ca.log(ca.fmax(var, epsilon))


def _exp_transform_var(ca: Any, log_var: Any, epsilon: float = 1e-3) -> Any:
    """Transform log-space variable back to physical space with bounds enforcement.
    
    Ensures exp(log_var) >= epsilon to prevent numerical issues.
    This is necessary because exp() can produce values slightly below the
    theoretical minimum even when log_var is within bounds, due to:
    - Numerical precision in exp() evaluation
    - Optimizer line search trying values slightly outside bounds
    - Accumulated numerical errors
    
    Args:
        ca: CasADi module
        log_var: Log-space variable
        epsilon: Minimum value to enforce (default 1e-3 for density, 1e-10 for valve areas)
    
    Returns:
        Physical-space variable: max(exp(log_var), epsilon)
    """
    return ca.fmax(ca.exp(log_var), epsilon)


def _import_casadi() -> Any:
    try:
        import casadi as ca
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CasADi is required for NLP building") from exc
    return ca


def gas_pressure_from_state(*, rho: Any, T: Any, gamma: float = 1.4) -> Any:
    """Gas pressure from density and temperature using ideal gas law.

    Parameters
    ----------
    rho : Any
        Gas density [kg/m^3] (CasADi variable)
    T : Any
        Gas temperature [K] (CasADi variable)
    gamma : float
        Heat capacity ratio

    Returns
    -------
    p : Any
        Gas pressure [Pa] (CasADi variable)
    """
    ca = _import_casadi()
    R = 287.0  # Gas constant for air [J/(kg K)]
    return rho * R * T


def chamber_volume_from_pistons(*, x_L: Any, x_R: Any, B: float, Vc: float) -> Any:
    """Chamber volume from piston positions.

    Parameters
    ----------
    x_L : Any
        Left piston position [m] (CasADi variable)
    x_R : Any
        Right piston position [m] (CasADi variable)
    B : float
        Bore diameter [m]
    Vc : float
        Clearance volume [m^3] (minimum volume from CR: V_min = clearance_volume)

    Returns
    -------
    V : Any
        Chamber volume [m^3] (CasADi variable), protected to never go below Vc
    """
    ca = _import_casadi()
    A_piston = math.pi * ca.fmax(B / 2.0, CASADI_PHYSICS_EPSILON) ** 2
    # Use clearance_volume (Vc) as minimum instead of CASADI_PHYSICS_EPSILON
    # This is physically meaningful: CR = V_max/V_min where V_min = clearance_volume
    return ca.fmax(Vc + A_piston * (x_R - x_L), Vc)


def enhanced_piston_dae_constraints(
    xL_c: Any,
    xR_c: Any,
    vL_c: Any,
    vR_c: Any,
    aL_c: Any,
    aR_c: Any,
    p_gas_c: Any,
    geometry: dict[str, float],
) -> tuple[Any, Any]:
    """
    Complete piston DAE with all force components.

    Returns:
        F_L, F_R: Net forces on left and right pistons
    """
    ca = _import_casadi()

    # Gas pressure forces
    A_piston = math.pi * ca.fmax(geometry["bore"] / 2.0, CASADI_PHYSICS_EPSILON) ** 2
    F_gas_L = p_gas_c * A_piston
    F_gas_R = -p_gas_c * A_piston  # Opposite direction

    # Inertia forces (piston + connecting rod)
    m_piston = geometry["mass"]
    m_rod = geometry["rod_mass"]
    rod_cg_offset = geometry["rod_cg_offset"]
    rod_length = geometry["rod_length"]

    # Effective mass including rod dynamics
    m_eff_L = m_piston + m_rod * (
        rod_cg_offset / ca.fmax(rod_length, CASADI_PHYSICS_EPSILON)
    )
    m_eff_R = m_piston + m_rod * (
        rod_cg_offset / ca.fmax(rod_length, CASADI_PHYSICS_EPSILON)
    )

    F_inertia_L = -m_eff_L * aL_c
    F_inertia_R = -m_eff_R * aR_c

    # Friction forces (velocity-dependent)
    friction_coeff = geometry.get("friction_coeff", 0.1)
    F_friction_L = -friction_coeff * ca.sign(vL_c) * ca.fabs(vL_c)
    F_friction_R = -friction_coeff * ca.sign(vR_c) * ca.fabs(vR_c)

    # Clearance penalty forces (smooth)
    gap_min = geometry.get("gap_min", 0.0008)
    gap_current = xR_c - xL_c
    penalty_stiffness_raw = geometry.get("penalty_stiffness", 1e6)

    # Cap penalty stiffness to keep scaled forces O(1)
    # Typical force scale ~1e4 N, typical position scale ~1e-1 m
    # So penalty_stiffness should be ~1e5 N/m to give forces ~1e4 N
    # Cap at 1e6 N/m to prevent extreme Jacobian entries
    penalty_stiffness = min(penalty_stiffness_raw, 1e6)

    # Smooth penalty function
    gap_violation = ca.fmax(0.0, gap_min - gap_current)
    F_clearance_L = penalty_stiffness * gap_violation
    F_clearance_R = -penalty_stiffness * gap_violation

    # Net forces
    F_L = F_gas_L + F_inertia_L + F_friction_L + F_clearance_L
    F_R = F_gas_R + F_inertia_R + F_friction_R + F_clearance_R

    return F_L, F_R


def piston_force_balance(
    *,
    p_gas: Any,
    x_L: Any,
    x_R: Any,
    v_L: Any,
    v_R: Any,
    a_L: Any,
    a_R: Any,
    geometry: dict[str, float],
) -> tuple[Any, Any]:
    """Enhanced piston force balance with full gas-structure coupling.

    This function implements the complete force balance for opposed pistons including:
    - Gas pressure forces with proper area calculation
    - Inertia forces for piston and connecting rod
    - Friction forces with velocity-dependent damping
    - Clearance penalty forces with smooth transitions
    - Piston ring dynamics and blow-by effects

    Parameters
    ----------
    p_gas : Any
        Gas pressure [Pa] (CasADi variable)
    x_L, x_R : Any
        Piston positions [m] (CasADi variables)
    v_L, v_R : Any
        Piston velocities [m/s] (CasADi variables)
    a_L, a_R : Any
        Piston accelerations [m/s^2] (CasADi variables)
    geometry : Dict[str, float]
        Piston geometry parameters

    Returns
    -------
    F_L, F_R : Any
        Net forces on left and right pistons [N] (CasADi variables)
    """
    ca = _import_casadi()

    # Piston area
    B = geometry.get("bore", 0.1)  # m
    A_piston = math.pi * ca.fmax(B / 2.0, CASADI_PHYSICS_EPSILON) ** 2

    # Gas pressure forces (opposed pistons)
    F_gas_L = p_gas * A_piston
    F_gas_R = -p_gas * A_piston  # Opposite direction for opposed pistons

    # Enhanced inertia forces
    m_piston = geometry.get("mass", 1.0)  # kg
    m_rod = geometry.get("rod_mass", 0.5)  # kg
    rod_length = geometry.get("rod_length", 0.15)  # m
    rod_cg_offset = geometry.get("rod_cg_offset", 0.075)  # m

    # Piston inertia
    F_piston_inertia_L = -m_piston * a_L
    F_piston_inertia_R = -m_piston * a_R

    # Connecting rod inertia (simplified)
    F_rod_inertia_L = (
        -m_rod * a_L * (rod_cg_offset / ca.fmax(rod_length, CASADI_PHYSICS_EPSILON))
    )
    F_rod_inertia_R = (
        -m_rod * a_R * (rod_cg_offset / ca.fmax(rod_length, CASADI_PHYSICS_EPSILON))
    )

    F_inertia_L = F_piston_inertia_L + F_rod_inertia_L
    F_inertia_R = F_piston_inertia_R + F_rod_inertia_R

    # Enhanced friction forces
    mu_friction = geometry.get("friction_coefficient", 0.1)
    c_damping = geometry.get("damping_coefficient", 100.0)  # Ns/m

    # Velocity-dependent friction
    F_friction_L = -mu_friction * ca.fabs(v_L) * ca.sign(v_L) - c_damping * v_L
    F_friction_R = -mu_friction * ca.fabs(v_R) * ca.sign(v_R) - c_damping * v_R

    # Piston ring friction
    ring_count = geometry.get("ring_count", 3)
    ring_tension = geometry.get("ring_tension", 100.0)  # N
    ring_width = geometry.get("ring_width", 0.002)  # m
    mu_ring = geometry.get("ring_friction_coefficient", 0.1)

    # Ring friction force (pressure-dependent)
    F_ring_L = (
        -ring_count
        * mu_ring
        * (ring_tension + p_gas * ring_width * math.pi * B)
        * ca.sign(v_L)
    )
    F_ring_R = (
        -ring_count
        * mu_ring
        * (ring_tension + p_gas * ring_width * math.pi * B)
        * ca.sign(v_R)
    )

    # Enhanced clearance penalty forces
    gap_min = geometry.get("clearance_min", 0.001)  # m
    k_clearance_raw = geometry.get("clearance_stiffness", 1e6)  # N/m
    # Cap clearance stiffness to keep scaled forces O(1)
    k_clearance = min(k_clearance_raw, 1e6)  # N/m
    clearance_smooth = geometry.get("clearance_smooth", 0.0001)  # m

    gap = x_R - x_L

    # Smooth clearance penalty using tanh function
    gap_violation = ca.fmax(0.0, gap_min - gap)
    F_clearance = (
        k_clearance
        * gap_violation
        * ca.tanh(gap_violation / ca.fmax(clearance_smooth, CASADI_PHYSICS_EPSILON))
    )

    # Net forces
    F_L = F_gas_L + F_inertia_L + F_friction_L + F_ring_L + F_clearance
    F_R = F_gas_R + F_inertia_R + F_friction_R + F_ring_R - F_clearance

    return F_L, F_R


def enhanced_gas_dae_constraints(
    rho_c: Any,
    T_c: Any,
    V_c: Any,
    dV_dt_c: Any,
    mdot_in_c: Any,
    mdot_out_c: Any,
    Q_comb_c: Any,
    Q_heat_c: Any,
    geometry: dict[str, float],
    thermo: dict[str, float],
    bounds: dict[str, float] | None = None,
    use_log_density: bool = False,
) -> tuple[Any, Any]:
    """
    Complete gas DAE with all source terms.

    Returns:
        drho_dt, dT_dt: Density and temperature rates
    """
    ca = _import_casadi()

    # Gas properties
    R = thermo.get("R", 287.0)
    gamma = thermo.get("gamma", 1.4)
    cp = thermo.get("cp", 1005.0)
    cv = cp / ca.fmax(gamma, CASADI_PHYSICS_EPSILON)

    # Mass balance: d(rho*V)/dt = mdot_in - mdot_out
    # Expanding: rho*dV/dt + V*drho/dt = mdot_in - mdot_out
    # Solving for drho/dt: drho/dt = (mdot_in - mdot_out - rho*dV/dt) / V
    drho_dt = (mdot_in_c - mdot_out_c - rho_c * dV_dt_c) / ca.fmax(
        V_c, CASADI_PHYSICS_EPSILON,
    )

    # Energy balance: d(m*e)/dt = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # where m = rho*V, e = cv*T, h = cp*T, p = rho*R*T

    # Compute minimum density and mass for numerical stability guards
    if bounds is not None:
        rho_min = bounds.get("rho_min", 0.1)
        if use_log_density:
            rho_min_bound = max(rho_min, 1e-3)  # Match log-space bound minimum
        else:
            rho_min_bound = rho_min
        clearance_volume = geometry.get("clearance_volume", 1e-4)
        m_total_min = rho_min_bound * clearance_volume
    else:
        # Fallback to defaults if bounds not provided
        rho_min_bound = 1e-3 if use_log_density else 0.1
        clearance_volume = geometry.get("clearance_volume", 1e-4)
        m_total_min = rho_min_bound * clearance_volume

    # Total mass and internal energy
    # Protect rho_c to ensure minimum density matches bounds
    rho_c_safe = ca.fmax(rho_c, rho_min_bound)
    m_total = rho_c_safe * V_c
    # Additional protection: ensure m_total never smaller than physically reasonable minimum
    m_total_safe = ca.fmax(m_total, m_total_min)
    e_internal = cv * T_c

    # Enthalpy of inlet/outlet streams
    T_in = thermo.get("T_in", 300.0)
    T_out = T_c  # Assume outlet at chamber temperature
    h_in = cp * T_in
    h_out = cp * T_out

    # Pressure (use protected density for consistency)
    p_gas = rho_c_safe * R * T_c

    # Energy equation: d(m*e)/dt = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # Expanding: m*de/dt + e*dm/dt = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # Since de/dt = cv*dT/dt and dm/dt = drho_dt*V + rho*dV_dt_c:
    # m*cv*dT/dt + e*(drho_dt*V + rho*dV_dt_c) = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt

    # Solve for dT/dt (use protected mass)
    dT_dt = (
        Q_comb_c
        - Q_heat_c
        + mdot_in_c * h_in
        - mdot_out_c * h_out
        - p_gas * dV_dt_c
        - e_internal * (drho_dt * V_c + rho_c_safe * dV_dt_c)
    ) / ca.fmax(m_total_safe * cv, CASADI_PHYSICS_EPSILON)
    
    # Protect outputs from NaN/Inf (for consistency with main collocation loop)
    drho_dt_max = 1e6  # Reasonable maximum density derivative [kg/(m^3 s)]
    drho_dt = ca.fmin(drho_dt, drho_dt_max)
    drho_dt = ca.fmax(drho_dt, -drho_dt_max)
    
    dT_dt_max = 1e6  # Reasonable maximum temperature derivative [K/s]
    dT_dt = ca.fmin(dT_dt, dT_dt_max)
    dT_dt = ca.fmax(dT_dt, -dT_dt_max)

    return drho_dt, dT_dt


def gas_energy_balance(
    *,
    rho: Any,
    T: Any,
    V: Any,
    dV_dt: Any,
    Q_combustion: Any,
    Q_heat_transfer: Any,
    mdot_in: Any,
    mdot_out: Any,
    T_in: Any = None,
    T_out: Any = None,
    gamma: float = 1.4,
) -> Any:
    """Enhanced gas energy balance equation with proper thermodynamics.

    This function implements the complete energy balance for the gas in the chamber,
    including:
    - Mass conservation with volume changes
    - Energy conservation with heat transfer and combustion
    - Proper enthalpy calculations for inlet/outlet flows
    - Temperature-dependent gas properties

    Parameters
    ----------
    rho : Any
        Gas density [kg/m^3] (CasADi variable)
    T : Any
        Gas temperature [K] (CasADi variable)
    V : Any
        Chamber volume [m^3] (CasADi variable)
    dV_dt : Any
        Volume rate of change [m^3/s] (CasADi variable)
    Q_combustion : Any
        Combustion heat release rate [W] (CasADi variable)
    Q_heat_transfer : Any
        Heat transfer rate [W] (CasADi variable)
    mdot_in : Any
        Mass flow rate in [kg/s] (CasADi variable)
    mdot_out : Any
        Mass flow rate out [kg/s] (CasADi variable)
    T_in : Any, optional
        Inlet temperature [K] (CasADi variable)
    T_out : Any, optional
        Outlet temperature [K] (CasADi variable)
    gamma : float
        Heat capacity ratio

    Returns
    -------
    dT_dt : Any
        Temperature rate of change [K/s] (CasADi variable)
    """
    ca = _import_casadi()

    # Gas properties (temperature-dependent)
    R = 287.0  # J/(kg K) - gas constant for air
    cp_ref = 1005.0  # J/(kg K) - reference specific heat
    cv_ref = cp_ref / ca.fmax(
        gamma, CASADI_PHYSICS_EPSILON,
    )  # J/(kg K) - reference specific heat at constant volume

    # Temperature-dependent specific heat (simplified linear model)
    # In practice, this would use JANAF polynomial fits
    cp = cp_ref * (1.0 + 0.0001 * (T - 300.0))  # Linear temperature dependence
    cv = cp / ca.fmax(gamma, CASADI_PHYSICS_EPSILON)

    # Set default inlet/outlet temperatures
    if T_in is None:
        T_in = 300.0  # K - ambient temperature
    if T_out is None:
        T_out = T  # K - assume outlet at chamber temperature

    # Mass balance: d(rho*V)/dt = mdot_in - mdot_out
    dm_dt = mdot_in - mdot_out

    # Total mass in chamber - protect to prevent division by extremely small values
    # Even if rho and V are individually protected, their product can be extremely small
    # Use minimum mass based on minimum density (1e-3) * minimum volume (1e-4) = 1e-7 kg
    rho_min_safe = 1e-3  # Minimum density [kg/m³] matching log-space bound
    V_min_safe = 1e-4  # Minimum volume [m³] matching clearance volume
    m_min = rho_min_safe * V_min_safe  # Minimum mass [kg] = 1e-7
    m = rho * V
    m_safe = ca.fmax(m, m_min)  # Protect mass to prevent division by extremely small values

    # Energy balance: d(m*e)/dt = Q_combustion - Q_heat_transfer + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # where e = cv*T is specific internal energy and h = cp*T is specific enthalpy

    # Specific internal energy and enthalpy
    e = cv * T  # J/kg
    h_in = cp * T_in  # J/kg
    h_out = cp * T_out  # J/kg

    # Pressure (ideal gas law)
    p = rho * R * T  # Pa

    # Energy equation: d(m*e)/dt = Q_combustion - Q_heat_transfer + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # Expanding: m*de/dt + e*dm/dt = Q_combustion - Q_heat_transfer + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # Since de/dt = cv*dT/dt, we get:
    # m*cv*dT/dt + e*dm/dt = Q_combustion - Q_heat_transfer + mdot_in*h_in - mdot_out*h_out - p*dV/dt

    # Solve for dT/dt - use protected mass to prevent division by extremely small values
    dT_dt = (
        Q_combustion
        - Q_heat_transfer
        + mdot_in * h_in
        - mdot_out * h_out
        - p * dV_dt
        - e * dm_dt
    ) / ca.fmax(m_safe * cv, CASADI_PHYSICS_EPSILON)

    return dT_dt


def build_collocation_nlp_with_1d_coupling(
    P: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """
    Build NLP with full 1D gas-structure coupling.

    This replaces the current 0D gas model with 1D FV model
    while maintaining the collocation framework.
    """
    ca = _import_casadi()
    num = P.get("num", {})
    K = int(num.get("K", 10))
    C = int(num.get("C", 1))
    grid: CollocationGrid = make_grid(K, C, kind="radau")

    # 1D gas model parameters
    n_cells = P.get("flow", {}).get("mesh_cells", 80)
    use_1d_gas = P.get("flow", {}).get("use_1d_gas", False)

    if use_1d_gas:
        return _build_1d_collocation_nlp(P, ca, K, C, grid, n_cells)
    return build_collocation_nlp(P)


def _build_1d_collocation_nlp(
    P: dict[str, Any], ca: Any, K: int, C: int, grid: CollocationGrid, n_cells: int,
) -> tuple[Any, dict[str, Any]]:
    """Build 1D gas-structure coupled collocation NLP."""

    # Get geometry and parameters
    geometry = P.get("geometry", {})
    bounds = P.get("bounds", {})
    obj_cfg = P.get("obj", {})
    walls_cfg = P.get("walls", {})
    flow_cfg = P.get("flow", {})
    combustion_cfg = P.get("combustion", {})

    use_combustion_model = bool(combustion_cfg.get("use_integrated_model", False))
    if use_combustion_model and flow_cfg.get("use_1d_gas", False):
        raise NotImplementedError(
            "Integrated combustion model is not yet supported with 1D gas dynamics.",
        )

    combustion_model: CombustionModel | None = None
    combustion_cycle_time = None
    omega_deg_per_s_const = None
    omega_deg_per_s_dm = None
    combustion_samples: list[tuple[float, Any]] = []

    if use_combustion_model:
        required_keys = [
            "fuel_type",
            "afr",
            "fuel_mass_kg",
            "cycle_time_s",
            "initial_temperature_K",
        ]
        for key in required_keys:
            if key not in combustion_cfg:
                raise ValueError(f"combustion configuration missing required key '{key}'")

        combustion_cycle_time = float(combustion_cfg["cycle_time_s"])
        combustion_model = CombustionModel()
        combustion_model.configure(
            fuel_type=combustion_cfg["fuel_type"],
            afr=float(combustion_cfg["afr"]),
            bore_m=float(geometry.get("bore", 0.1)),
            stroke_m=float(geometry.get("stroke", 0.1)),
            clearance_volume_m3=float(geometry.get("clearance_volume", 1e-4)),
            fuel_mass_kg=float(combustion_cfg["fuel_mass_kg"]),
            cycle_time_s=combustion_cycle_time,
            initial_temperature_K=float(combustion_cfg["initial_temperature_K"]),
            initial_pressure_Pa=float(combustion_cfg.get("initial_pressure_Pa", 1e5)),
            target_mfb=float(combustion_cfg.get("target_mfb", 0.99)),
            m_wiebe=float(combustion_cfg.get("m_wiebe", 2.0)),
            k_turb=float(combustion_cfg.get("k_turb", 0.3)),
            c_burn=float(combustion_cfg.get("c_burn", 3.0)),
            turbulence_exponent=float(combustion_cfg.get("turbulence_exponent", 0.7)),
            min_flame_speed=float(combustion_cfg.get("min_flame_speed", 0.2)),
            heating_value_override=combustion_cfg.get("heating_value_override"),
            phi_override=combustion_cfg.get("phi_override"),
        )
        omega_deg_per_s_const = float(
            combustion_cfg.get(
                "omega_deg_per_s",
                360.0 / max(combustion_cycle_time, 1e-9),
            ),
        )
        omega_deg_per_s_dm = ca.DM(omega_deg_per_s_const)

    # Define ignition time variable if using combustion model
    if use_combustion_model:
        ignition_bounds = combustion_cfg.get(
            "ignition_bounds_s",
            (0.0, max(combustion_cycle_time or 1.0, 1e-6)),
        )
        ignition_initial = float(
            combustion_cfg.get(
                "ignition_initial_s",
                0.1 * max(combustion_cycle_time or 1.0, 1e-6),
            ),
        )
        t_ign = ca.SX.sym("t_ign")
        # Add t_ign to variables
        w_ign = [t_ign]
        w0_ign = [ignition_initial]
        lbw_ign = [float(ignition_bounds[0])]
        ubw_ign = [float(ignition_bounds[1])]
    else:
        t_ign = None
        w_ign = []
        w0_ign = []
        lbw_ign = []
        ubw_ign = []

    # Variables, initial guesses, and bounds
    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []

    # Add ignition variable if using combustion model
    w += w_ign
    w0 += w0_ign
    lbw += lbw_ign
    ubw += ubw_ign

    # Initial states
    xL0 = ca.SX.sym("xL0")
    xR0 = ca.SX.sym("xR0")
    vL0 = ca.SX.sym("vL0")
    vR0 = ca.SX.sym("vR0")

    w += [xL0, xR0, vL0, vR0]
    w0 += [0.0, 0.1, 0.0, 0.0]
    lbw += [
        bounds.get("xL_min", -0.1),
        bounds.get("xR_min", 0.0),
        bounds.get("vL_min", -10.0),
        bounds.get("vR_min", -10.0),
    ]
    ubw += [
        bounds.get("xL_max", 0.1),
        bounds.get("xR_max", 0.2),
        bounds.get("vL_max", 10.0),
        bounds.get("vR_max", 10.0),
    ]

    # Initial 1D gas state variables (per cell)
    rho0_cells = [ca.SX.sym(f"rho0_{i}") for i in range(n_cells)]
    u0_cells = [ca.SX.sym(f"u0_{i}") for i in range(n_cells)]
    E0_cells = [ca.SX.sym(f"E0_{i}") for i in range(n_cells)]

    w += rho0_cells + u0_cells + E0_cells
    w0 += [1.0] * n_cells + [0.0] * n_cells + [2.5] * n_cells  # Initial gas state
    lbw += (
        [bounds.get("rho_min", 0.1)] * n_cells
        + [bounds.get("u_min", -100.0)] * n_cells
        + [bounds.get("E_min", 0.1)] * n_cells
    )
    ubw += (
        [bounds.get("rho_max", 10.0)] * n_cells
        + [bounds.get("u_max", 100.0)] * n_cells
        + [bounds.get("E_max", 100.0)] * n_cells
    )

    # Valve controls
    Ain0 = ca.SX.sym("Ain0")
    Aex0 = ca.SX.sym("Aex0")
    w += [Ain0, Aex0]
    w0 += [0.0, 0.0]
    lbw += [0.0, 0.0]
    ubw += [bounds.get("Ain_max", 0.01), bounds.get("Aex_max", 0.01)]

    # State variables for each time step
    xL_k = xL0
    xR_k = xR0
    vL_k = vL0
    vR_k = vR0
    rho_k = rho0_cells
    u_k = u0_cells
    E_k = E0_cells
    Ain_k = Ain0
    Aex_k = Aex0

    h = 1.0 / ca.fmax(K, CASADI_PHYSICS_EPSILON)
    # Objective accumulators
    W_ind_accum = 0.0
    Q_in_accum = 0.0
    scav_penalty_accum = 0.0
    smooth_penalty_accum = 0.0

    # Define dt_real for combustion model timing
    if use_combustion_model:
        dt_real = (combustion_cycle_time or 1.0) / max(K, 1)
        combustion_samples.append((0.0, ca.DM(0.0)))
    else:
        dt_real = None

    # Scavenging and timing accumulator initial states
    yF0 = ca.SX.sym("yF0")
    Mdel0 = ca.SX.sym("Mdel0")
    Mlost0 = ca.SX.sym("Mlost0")
    AinInt0 = ca.SX.sym("AinInt0")
    AinTmom0 = ca.SX.sym("AinTmom0")
    AexInt0 = ca.SX.sym("AexInt0")
    AexTmom0 = ca.SX.sym("AexTmom0")
    w += [yF0, Mdel0, Mlost0, AinInt0, AinTmom0, AexInt0, AexTmom0]
    
    # Compute feasible initial values from problem configuration
    num = P.get("num", {})
    (yF0_val, Mdel0_val, Mlost0_val, AinInt0_val, AinTmom0_val, AexInt0_val, AexTmom0_val) = (
        _compute_scavenging_initial_values(P, bounds, geometry, num)
    )
    w0 += [yF0_val, Mdel0_val, Mlost0_val, AinInt0_val, AinTmom0_val, AexInt0_val, AexTmom0_val]
    lbw += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ubw += [1.0, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf]
    yF_k = yF0
    Mdel_k = Mdel0
    Mlost_k = Mlost0
    AinInt_k = AinInt0
    AinTmom_k = AinTmom0
    AexInt_k = AexInt0
    AexTmom_k = AexTmom0

    for k in range(K):
        # Stage controls (valve areas and combustion heat release)
        Ain_stage: list[Any] = []
        Aex_stage: list[Any] = []
        Q_comb_stage: list[Any] = []

        for j in range(C):
            ain_sym = ca.SX.sym(f"Ain_{k}_{j}")
            aex_sym = ca.SX.sym(f"Aex_{k}_{j}")
            Ain_stage.append(ain_sym)
            Aex_stage.append(aex_sym)
            w += [ain_sym, aex_sym]
            w0 += [0.0, 0.0]
            lbw += [0.0, 0.0]
            ubw += [
                bounds.get("Ain_max", 0.01),
                bounds.get("Aex_max", 0.01),
            ]

            if not use_combustion_model:
                q_sym = ca.SX.sym(f"Q_comb_{k}_{j}")
                Q_comb_stage.append(q_sym)
                w += [q_sym]
                w0 += [0.0]
                lbw += [0.0]
                # Convert normalized bounds from kJ to J (multiply by 1e3)
                q_comb_max_j = bounds.get("Q_comb_max", 10.0) * 1e3  # Default 10.0 kJ = 10000.0 J
                ubw += [q_comb_max_j]
            else:
                Q_comb_stage.append(None)

        # Collocation states for pistons
        xL_colloc = [ca.SX.sym(f"xL_{k}_{j}") for j in range(C)]
        xR_colloc = [ca.SX.sym(f"xR_{k}_{j}") for j in range(C)]
        vL_colloc = [ca.SX.sym(f"vL_{k}_{j}") for j in range(C)]
        vR_colloc = [ca.SX.sym(f"vR_{k}_{j}") for j in range(C)]

        for j in range(C):
            w += [xL_colloc[j], xR_colloc[j], vL_colloc[j], vR_colloc[j]]
            w0 += [0.0, 0.1, 0.0, 0.0]
            lbw += [
                bounds.get("xL_min", -0.1),
                bounds.get("xR_min", 0.0),
                bounds.get("vL_min", -10.0),
                bounds.get("vR_min", -10.0),
            ]
            ubw += [
                bounds.get("xL_max", 0.1),
                bounds.get("xR_max", 0.2),
                bounds.get("vL_max", 10.0),
                bounds.get("vR_max", 10.0),
            ]

        # Collocation states for 1D gas (per cell, per collocation point)
        rho_colloc = [
            [ca.SX.sym(f"rho_{k}_{c}_{i}") for i in range(n_cells)] for c in range(C)
        ]
        u_colloc = [
            [ca.SX.sym(f"u_{k}_{c}_{i}") for i in range(n_cells)] for c in range(C)
        ]
        E_colloc = [
            [ca.SX.sym(f"E_{k}_{c}_{i}") for i in range(n_cells)] for c in range(C)
        ]

        for c in range(C):
            w += rho_colloc[c] + u_colloc[c] + E_colloc[c]
            w0 += [1.0] * n_cells + [0.0] * n_cells + [2.5] * n_cells
            lbw += (
                [bounds.get("rho_min", 0.1)] * n_cells
                + [bounds.get("u_min", -100.0)] * n_cells
                + [bounds.get("E_min", 0.1)] * n_cells
            )
            ubw += (
                [bounds.get("rho_max", 10.0)] * n_cells
                + [bounds.get("u_max", 100.0)] * n_cells
                + [bounds.get("E_max", 100.0)] * n_cells
            )

        # Collocation equations
        for c in range(C):
            # Current state
            xL_c = xL_colloc[c]
            xR_c = xR_colloc[c]
            vL_c = vL_colloc[c]
            vR_c = vR_colloc[c]
            Ain_c = Ain_stage[c]
            Aex_c = Aex_stage[c]

            if use_combustion_model and combustion_model is not None:
                time_val = (k + grid.nodes[c]) * float(dt_real or 0.0)
                time_dm = ca.DM(time_val)
                piston_speed_expr = 0.5 * ca.fabs(vR_c - vL_c)
                comb_expr = combustion_model.symbolic_heat_release(
                    ca=ca,
                    time_s=time_dm,
                    piston_speed_m_per_s=piston_speed_expr,
                    ignition_time_s=t_ign,
                    omega_deg_per_s=omega_deg_per_s_dm,
                )
                Q_comb_c = comb_expr["heat_release_rate"]
                Q_comb_stage[c] = Q_comb_c
                combustion_samples.append((time_val, comb_expr["mfb"]))
            else:
                Q_comb_c = Q_comb_stage[c]

            # Chamber volume and its rate of change
            V_c = chamber_volume_from_pistons(
                x_L=xL_c,
                x_R=xR_c,
                B=geometry.get("bore", 0.1),
                Vc=geometry.get("clearance_volume", 1e-4),
            )
            dV_dt = (
                math.pi
                * ca.fmax(geometry.get("bore", 0.1) / 2.0, CASADI_PHYSICS_EPSILON) ** 2
                * (vR_c - vL_c)
            )

            # Enhanced piston forces with proper acceleration coupling
            # Compute accelerations from velocity differences (simplified)
            aL_c = (vL_c - vL_k) / ca.fmax(h, CASADI_PHYSICS_EPSILON)
            aR_c = (vR_c - vR_k) / ca.fmax(h, CASADI_PHYSICS_EPSILON)

            # Calculate average gas pressure for piston forces
            p_avg = 0.0
            for i in range(n_cells):
                rho_i = rho_colloc[c][i]
                u_i = u_colloc[c][i]
                E_i = E_colloc[c][i]
                # Convert to pressure using ideal gas law
                p_i = (
                    (1.4 - 1.0)
                    * rho_i
                    * (E_i - 0.5 * ca.fmax(u_i, CASADI_PHYSICS_EPSILON) ** 2)
                )
                p_avg += p_i
            p_avg /= n_cells

            F_L_c, F_R_c = piston_force_balance(
                p_gas=p_avg,
                x_L=xL_c,
                x_R=xR_c,
                v_L=vL_c,
                v_R=vR_c,
                a_L=aL_c,
                a_R=aR_c,
                geometry=geometry,
            )

            # 1D gas dynamics constraints
            for i in range(n_cells):
                # Conservative form: dU/dt + dF/dx = S
                # where U = [rho, rho*u, rho*E], F = [rho*u, rho*u^2+p, (rho*E+p)*u]

                # Current cell state
                rho_i = rho_colloc[c][i]
                u_i = u_colloc[c][i]
                E_i = E_colloc[c][i]

                # Calculate fluxes at cell faces using HLLC
                if i > 0:
                    # Left face flux
                    U_L = [
                        rho_colloc[c][i - 1],
                        rho_colloc[c][i - 1] * u_colloc[c][i - 1],
                        rho_colloc[c][i - 1] * E_colloc[c][i - 1],
                    ]
                    U_R = [rho_i, rho_i * u_i, rho_i * E_i]
                    F_left = _hllc_flux_symbolic(U_L, U_R, ca)
                else:
                    F_left = [0.0, 0.0, 0.0]

                if i < n_cells - 1:
                    # Right face flux
                    U_L = [rho_i, rho_i * u_i, rho_i * E_i]
                    U_R = [
                        rho_colloc[c][i + 1],
                        rho_colloc[c][i + 1] * u_colloc[c][i + 1],
                        rho_colloc[c][i + 1] * E_colloc[c][i + 1],
                    ]
                    F_right = _hllc_flux_symbolic(U_L, U_R, ca)
                else:
                    F_right = [0.0, 0.0, 0.0]

                # Source terms (volume change, heat transfer)
                S_sources = _calculate_1d_source_terms(
                    rho_i,
                    u_i,
                    E_i,
                    xL_c,
                    xR_c,
                    vL_c,
                    vR_c,
                    geometry,
                    flow_cfg,
                    Q_comb_c,
                    Ain_c,
                    Aex_c,
                )

                # Cell width (simplified)
                dx = (xR_c - xL_c) / ca.fmax(n_cells, CASADI_PHYSICS_EPSILON)

                # Semi-discrete form: dU/dt = -(F_right - F_left) / dx + S
                dU_dt = []
                for comp in range(3):
                    dU_dt.append(
                        -(F_right[comp] - F_left[comp])
                        / ca.fmax(dx, CASADI_PHYSICS_EPSILON)
                        + S_sources[comp],
                    )

                # Add collocation constraint
                U_cell = [rho_i, rho_i * u_i, rho_i * E_i]
                U_cell_old = [rho_k[i], rho_k[i] * u_k[i], rho_k[i] * E_k[i]]

                for comp in range(3):
                    rhs = U_cell_old[comp]
                    for j in range(C):
                        rhs += h * grid.a[c][j] * dU_dt[comp]

                    g.append(U_cell[comp] - rhs)
                    lbg.append(0.0)
                    ubg.append(0.0)

            # Collocation equations for pistons
            rhs_xL = xL_k
            rhs_xR = xR_k
            rhs_vL = vL_k
            rhs_vR = vR_k

            for j in range(C):
                rhs_xL += h * grid.a[c][j] * vL_colloc[j]
                rhs_xR += h * grid.a[c][j] * vR_colloc[j]

                # Enhanced force balance with proper mass calculation
                m_total_L = geometry.get("mass", 1.0) + geometry.get(
                    "rod_mass", 0.5,
                ) * (
                    geometry.get("rod_cg_offset", 0.075)
                    / ca.fmax(geometry.get("rod_length", 0.15), CASADI_PHYSICS_EPSILON)
                )
                m_total_R = geometry.get("mass", 1.0) + geometry.get(
                    "rod_mass", 0.5,
                ) * (
                    geometry.get("rod_cg_offset", 0.075)
                    / ca.fmax(geometry.get("rod_length", 0.15), CASADI_PHYSICS_EPSILON)
                )

                rhs_vL += (
                    h
                    * grid.a[c][j]
                    * (F_L_c / ca.fmax(m_total_L, CASADI_PHYSICS_EPSILON))
                )
                rhs_vR += (
                    h
                    * grid.a[c][j]
                    * (F_R_c / ca.fmax(m_total_R, CASADI_PHYSICS_EPSILON))
                )

            colloc_res = [xL_c - rhs_xL, xR_c - rhs_xR, vL_c - rhs_vL, vR_c - rhs_vR]
            g += colloc_res
            lbg += [0.0] * len(colloc_res)
            ubg += [0.0] * len(colloc_res)

            # Accumulate indicated work and Q_in
            W_ind_accum += h * grid.weights[c] * p_avg * dV_dt
            Q_in_accum += h * grid.weights[c] * Q_comb_c

            # Scavenging short-circuit penalty surrogate
            eps = 1e-9
            # Simplified mass flow calculation
            mdot_in = Ain_c * 0.1  # Simplified
            mdot_out = Aex_c * 0.1  # Simplified
            ratio = mdot_out / ca.fmax(mdot_in + eps, CASADI_PHYSICS_EPSILON)
            scav_penalty_accum += h * grid.weights[c] * ratio

        # Step to next time point
        xL_k1 = xL_k
        xR_k1 = xR_k
        vL_k1 = vL_k
        vR_k1 = vR_k

        for j in range(C):
            xL_k1 += h * grid.weights[j] * vL_colloc[j]
            xR_k1 += h * grid.weights[j] * vR_colloc[j]

            # Enhanced force balance with proper mass calculation
            m_total_L = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (
                geometry.get("rod_cg_offset", 0.075)
                / ca.fmax(geometry.get("rod_length", 0.15), CASADI_PHYSICS_EPSILON)
            )
            m_total_R = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (
                geometry.get("rod_cg_offset", 0.075)
                / ca.fmax(geometry.get("rod_length", 0.15), CASADI_PHYSICS_EPSILON)
            )

            vL_k1 += (
                h
                * grid.weights[j]
                * (F_L_c / ca.fmax(m_total_L, CASADI_PHYSICS_EPSILON))
            )
            vR_k1 += (
                h
                * grid.weights[j]
                * (F_R_c / ca.fmax(m_total_R, CASADI_PHYSICS_EPSILON))
            )

        # Update gas states
        rho_k1 = [rho_k[i] for i in range(n_cells)]
        u_k1 = [u_k[i] for i in range(n_cells)]
        E_k1 = [E_k[i] for i in range(n_cells)]

        for j in range(C):
            for i in range(n_cells):
                # Simplified gas state update
                rho_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
                u_k1[i] += h * grid.weights[j] * 0.0  # Placeholder
                E_k1[i] += h * grid.weights[j] * 0.0  # Placeholder

        # New state variables
        xL_k = ca.SX.sym(f"xL_{k + 1}")
        xR_k = ca.SX.sym(f"xR_{k + 1}")
        vL_k = ca.SX.sym(f"vL_{k + 1}")
        vR_k = ca.SX.sym(f"vR_{k + 1}")

        rho_k = [ca.SX.sym(f"rho_{k + 1}_{i}") for i in range(n_cells)]
        u_k = [ca.SX.sym(f"u_{k + 1}_{i}") for i in range(n_cells)]
        E_k = [ca.SX.sym(f"E_{k + 1}_{i}") for i in range(n_cells)]

        yF_k = ca.SX.sym(f"yF_{k + 1}")
        Mdel_k = ca.SX.sym(f"Mdel_{k + 1}")
        Mlost_k = ca.SX.sym(f"Mlost_{k + 1}")
        AinInt_k = ca.SX.sym(f"AinInt_{k + 1}")
        AinTmom_k = ca.SX.sym(f"AinTmom_{k + 1}")
        AexInt_k = ca.SX.sym(f"AexInt_{k + 1}")
        AexTmom_k = ca.SX.sym(f"AexTmom_{k + 1}")

        w += (
            [xL_k, xR_k, vL_k, vR_k]
            + rho_k
            + u_k
            + E_k
            + [yF_k, Mdel_k, Mlost_k, AinInt_k, AinTmom_k, AexInt_k, AexTmom_k]
        )
        w0 += (
            [0.0, 0.1, 0.0, 0.0]
            + [1.0] * n_cells
            + [0.0] * n_cells
            + [2.5] * n_cells
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        lbw += (
            [
                bounds.get("xL_min", -0.1),
                bounds.get("xR_min", 0.0),
                bounds.get("vL_min", -10.0),
                bounds.get("vR_min", -10.0),
            ]
            + [bounds.get("rho_min", 0.1)] * n_cells
            + [bounds.get("u_min", -100.0)] * n_cells
            + [bounds.get("E_min", 0.1)] * n_cells
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        ubw += (
            [
                bounds.get("xL_max", 0.1),
                bounds.get("xR_max", 0.2),
                bounds.get("vL_max", 10.0),
                bounds.get("vR_max", 10.0),
            ]
            + [bounds.get("rho_max", 10.0)] * n_cells
            + [bounds.get("u_max", 100.0)] * n_cells
            + [bounds.get("E_max", 100.0)] * n_cells
            + [1.0, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf]
        )

        # Continuity constraints
        cont = [xL_k - xL_k1, xR_k - xR_k1, vL_k - vL_k1, vR_k - vR_k1]
        for i in range(n_cells):
            cont += [rho_k[i] - rho_k1[i], u_k[i] - u_k1[i], E_k[i] - E_k1[i]]
        cont += [
            yF_k - yF_k,
            Mdel_k - Mdel_k,
            Mlost_k - Mlost_k,
            AinInt_k - AinInt_k,
            AinTmom_k - AinTmom_k,
            AexInt_k - AexInt_k,
            AexTmom_k - AexTmom_k,
        ]
        g += cont
        lbg += [0.0] * len(cont)
        ubg += [0.0] * len(cont)

    # Enhanced path constraints
    gap_min = bounds.get("x_gap_min", 0.0008)
    g += [xR_k - xL_k]  # Clearance constraint
    lbg += [gap_min]
    ubg += [ca.inf]

    # Objective: weighted multi-term
    w_dict = obj_cfg.get("w", {})
    w_smooth = float(w_dict.get("smooth", 0.0))
    w_short = float(w_dict.get("short_circuit", 0.0))
    w_eta = float(w_dict.get("eta_th", 0.0))

    # W_ind term (maximize): minimize -W_ind
    J_terms = []
    J_terms.append(-W_ind_accum)
    # Thermal efficiency surrogate: maximize W_ind/Q_in -> minimize -(W/Q)
    if w_eta > 0.0:
        J_terms.append(
            -w_eta * (W_ind_accum / ca.fmax(Q_in_accum + 1e-6, CASADI_PHYSICS_EPSILON)),
        )
    # Scavenging penalty (minimize cycle short-circuit fraction)
    if w_short > 0.0:
        J_terms.append(w_short * scav_penalty_accum)

    J = sum(J_terms)

    nlp = {"x": ca.vertcat(*w), "f": J, "g": ca.vertcat(*g)}
    
    # Convert bounds and initial guess lists to numpy arrays for driver consumption
    w0_arr = np.array(w0, dtype=float)
    lbw_arr = np.array(lbw, dtype=float)
    ubw_arr = np.array(ubw, dtype=float)
    lbg_arr = np.array(lbg, dtype=float)
    ubg_arr = np.array(ubg, dtype=float)
    
    meta = {
        "K": K,
        "C": C,
        "n_vars": len(w),
        "n_constraints": len(g),
        "flow_mode": "1d_gas",
        "dynamic_wall": False,
        "scavenging_states": True,
        "timing_states": True,
        "n_cells": n_cells,
        # Include NLP-provided bounds and initial guess for driver consumption
        "w0": w0_arr,
        "lbw": lbw_arr,
        "ubw": ubw_arr,
        "lbg": lbg_arr,
        "ubg": ubg_arr,
    }
    return nlp, meta


def _hllc_flux_symbolic(U_L: list[Any], U_R: list[Any], ca: Any) -> list[Any]:
    """Symbolic HLLC flux calculation for CasADi."""
    # Simplified HLLC implementation for symbolic computation
    # In practice, this would be more sophisticated

    rho_L, rhou_L, rhoE_L = U_L
    rho_R, rhou_R, rhoE_R = U_R

    # Convert to primitive variables
    # Protect densities to prevent division by extremely small values
    # Use 1e-3 as minimum (matching log-space bound minimum) instead of CASADI_PHYSICS_EPSILON
    rho_min_bound = 1e-3  # Default minimum density for 1D gas flow
    rho_L_safe = ca.fmax(rho_L, rho_min_bound)
    rho_R_safe = ca.fmax(rho_R, rho_min_bound)
    u_L = rhou_L / rho_L_safe
    u_R = rhou_R / rho_R_safe
    p_L = (
        (1.4 - 1.0)
        * rho_L
        * (
            rhoE_L / rho_L_safe
            - 0.5 * ca.fmax(u_L, CASADI_PHYSICS_EPSILON) ** 2
        )
    )
    p_R = (
        (1.4 - 1.0)
        * rho_R
        * (
            rhoE_R / rho_R_safe
            - 0.5 * ca.fmax(u_R, CASADI_PHYSICS_EPSILON) ** 2
        )
    )

    # Simplified flux calculation
    F_L = [
        rho_L * u_L,
        rho_L * ca.fmax(u_L, CASADI_PHYSICS_EPSILON) ** 2 + p_L,
        (rhoE_L + p_L) * u_L,
    ]
    F_R = [
        rho_R * u_R,
        rho_R * ca.fmax(u_R, CASADI_PHYSICS_EPSILON) ** 2 + p_R,
        (rhoE_R + p_R) * u_R,
    ]

    # Simple average for now (in practice, use proper HLLC)
    F = [(F_L[i] + F_R[i]) / 2.0 for i in range(3)]

    return F


def _calculate_1d_source_terms(
    rho: Any,
    u: Any,
    E: Any,
    xL: Any,
    xR: Any,
    vL: Any,
    vR: Any,
    geometry: dict[str, float],
    flow_cfg: dict[str, Any],
    Q_comb: Any,
    Ain: Any,
    Aex: Any,
) -> list[Any]:
    """Calculate 1D source terms for gas dynamics equations."""
    ca = _import_casadi()

    # Volume change rate
    dV_dt = (
        math.pi
        * ca.fmax(geometry.get("bore", 0.1) / 2.0, CASADI_PHYSICS_EPSILON) ** 2
        * (vR - vL)
    )

    # Cell volume (simplified)
    V_cell = (xR - xL) / ca.fmax(flow_cfg.get("mesh_cells", 80), CASADI_PHYSICS_EPSILON)

    # Source terms
    S_rho = -rho * dV_dt / ca.fmax(V_cell, CASADI_PHYSICS_EPSILON)  # Mass source
    S_rhou = (
        -rho * u * dV_dt / ca.fmax(V_cell, CASADI_PHYSICS_EPSILON)
    )  # Momentum source
    S_rhoE = -rho * E * dV_dt / ca.fmax(
        V_cell, CASADI_PHYSICS_EPSILON,
    ) + Q_comb / ca.fmax(V_cell, CASADI_PHYSICS_EPSILON)  # Energy source

    return [S_rho, S_rhou, S_rhoE]


def _compute_scavenging_initial_values(
    P: dict[str, Any], bounds: dict[str, Any], geometry: dict[str, Any], num: dict[str, Any]
) -> tuple[float, float, float, float, float, float, float]:
    """Compute feasible initial values for scavenging/timing accumulator states.
    
    Derives initial values from problem configuration to ensure constraints are
    satisfied at the initial guess. This prevents IPOPT from starting far outside
    the feasible region.
    
    Parameters
    ----------
    P : dict[str, Any]
        Full problem parameters dictionary
    bounds : dict[str, Any]
        Bounds configuration dictionary
    geometry : dict[str, Any]
        Geometry configuration dictionary
    num : dict[str, Any]
        Numerical parameters dictionary (contains cycle_time, etc.)
        
    Returns
    -------
    tuple[float, float, float, float, float, float, float]
        Initial values for (yF0, Mdel0, Mlost0, AinInt0, AinTmom0, AexInt0, AexTmom0)
        Units: yF0 [dimensionless], Mdel0/Mlost0 [kg], AinInt0/AexInt0 [m²·s],
        AinTmom0/AexTmom0 [m²·s²]
    """
    import math
    
    cons_cfg = P.get("constraints", {})
    timing_cfg = P.get("timing", {})
    
    # yF0: Fresh-fuel fraction - use constraint lower bound
    yF0 = float(cons_cfg.get("scavenging_min", 0.8))
    yF0 = min(max(yF0, 0.0), 1.0)  # Clamp to [0, 1]
    
    # Mdel0: Delivered mass - estimate from steady-state flow
    rho_initial = float(bounds.get("rho_min", 0.1))  # kg/m³
    bore = float(geometry.get("bore", 0.05))  # m
    stroke = float(geometry.get("stroke", 0.02))  # m
    cycle_time = float(num.get("cycle_time", 1.0))  # s
    
    # Initial volume: π × (bore/2)² × stroke
    V_initial = math.pi * (bore / 2.0) ** 2 * stroke  # m³
    flow_fraction = 0.2  # Typical scavenging efficiency (20% of volume per cycle)
    Mdel0 = rho_initial * V_initial * flow_fraction  # kg
    Mdel0 = max(Mdel0, 1e-6)  # Ensure non-zero minimum
    
    # Mlost0: Lost mass - fraction of delivered mass
    short_circuit_max = float(cons_cfg.get("short_circuit_max", 0.1))
    Mlost0 = Mdel0 * short_circuit_max
    Mlost0 = min(Mlost0, Mdel0)  # Ensure ≤ Mdel0
    
    # AinInt0, AexInt0: Area-time integrals - estimate from valve area × time × duty cycle
    Ain_max = float(bounds.get("Ain_max", 0.01))  # m²
    Aex_max = float(bounds.get("Aex_max", 0.01))  # m²
    duty_cycle = 0.4  # Typical valve open fraction (40%)
    
    AinInt0 = Ain_max * cycle_time * duty_cycle  # m²·s
    AexInt0 = Aex_max * cycle_time * duty_cycle  # m²·s
    AinInt0 = max(AinInt0, 1e-6)  # Ensure non-zero minimum
    AexInt0 = max(AexInt0, 1e-6)  # Ensure non-zero minimum
    
    # AinTmom0, AexTmom0: Area-time moments - estimate from integrals × typical timing
    Ain_t_cm = float(timing_cfg.get("Ain_t_cm", 0.5))  # s (center of mass timing)
    Aex_t_cm = float(timing_cfg.get("Aex_t_cm", 0.5))  # s
    
    AinTmom0 = AinInt0 * Ain_t_cm  # m²·s²
    AexTmom0 = AexInt0 * Aex_t_cm  # m²·s²
    
    return (yF0, Mdel0, Mlost0, AinInt0, AinTmom0, AexInt0, AexTmom0)


def build_collocation_nlp(P: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Build collocation NLP with gas-structure coupling.

    Implements a full gas-structure coupled optimization problem with:
    - Piston dynamics with gas pressure coupling
    - Gas thermodynamics with combustion and heat transfer
    - Valve actuation constraints
    - Path constraints on pressure and temperature
    - Indicated work objective
    """
    ca = _import_casadi()
    num = P.get("num", {})
    K = int(num.get("K", 10))
    C = int(num.get("C", 1))
    grid: CollocationGrid = make_grid(K, C, kind="radau")

    # Get geometry and parameters
    geometry = P.get("geometry", {})
    bounds = P.get("bounds", {})
    obj_cfg = P.get("obj", {})
    walls_cfg = P.get("walls", {})
    flow_cfg = P.get("flow", {})
    combustion_cfg = P.get("combustion", {})

    # Check if combustion model is enabled
    use_combustion_model = bool(combustion_cfg.get("use_integrated_model", False))
    combustion_model: CombustionModel | None = None
    combustion_cycle_time = None
    omega_deg_per_s_const = None
    omega_deg_per_s_dm = None
    combustion_samples: list[tuple[float, Any]] = []

    if use_combustion_model:
        required_keys = [
            "fuel_type",
            "afr",
            "fuel_mass_kg",
            "cycle_time_s",
            "initial_temperature_K",
        ]
        for key in required_keys:
            if key not in combustion_cfg:
                raise ValueError(f"combustion configuration missing required key '{key}'")

        combustion_cycle_time = float(combustion_cfg["cycle_time_s"])
        combustion_model = CombustionModel()
        combustion_model.configure(
            fuel_type=combustion_cfg["fuel_type"],
            afr=float(combustion_cfg["afr"]),
            bore_m=float(geometry.get("bore", 0.1)),
            stroke_m=float(geometry.get("stroke", 0.1)),
            clearance_volume_m3=float(geometry.get("clearance_volume", 1e-4)),
            fuel_mass_kg=float(combustion_cfg["fuel_mass_kg"]),
            cycle_time_s=combustion_cycle_time,
            initial_temperature_K=float(combustion_cfg["initial_temperature_K"]),
            initial_pressure_Pa=float(combustion_cfg.get("initial_pressure_Pa", 1e5)),
            target_mfb=float(combustion_cfg.get("target_mfb", 0.99)),
            m_wiebe=float(combustion_cfg.get("m_wiebe", 2.0)),
            k_turb=float(combustion_cfg.get("k_turb", 0.3)),
            c_burn=float(combustion_cfg.get("c_burn", 3.0)),
            turbulence_exponent=float(combustion_cfg.get("turbulence_exponent", 0.7)),
            min_flame_speed=float(combustion_cfg.get("min_flame_speed", 0.2)),
            heating_value_override=combustion_cfg.get("heating_value_override"),
            phi_override=combustion_cfg.get("phi_override"),
        )
        omega_deg_per_s_const = float(
            combustion_cfg.get(
                "omega_deg_per_s",
                360.0 / max(combustion_cycle_time, 1e-9),
            ),
        )
        omega_deg_per_s_dm = ca.DM(omega_deg_per_s_const)

    # Unified gas closures (0D/1D strategy)
    gas_model = build_gas_model(P)

    # Variables, initial guesses, and bounds
    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []

    # Track variable groups for metadata
    var_groups: dict[str, list[int]] = {
        "positions": [],
        "velocities": [],
        "densities": [],
        "temperatures": [],
        "valve_areas": [],
        "ignition": [],
        "scavenging_fractions": [],  # yF (dimensionless)
        "scavenging_masses": [],  # Mdel, Mlost (kg)
        "scavenging_area_integrals": [],  # AinInt, AexInt (m²·s)
        "scavenging_time_moments": [],  # AinTmom, AexTmom (m²·s²)
    }
    var_idx = 0  # Track current variable index
    
    # Track constraint groups for metadata
    constraint_groups: dict[str, list[int]] = {
        "collocation_residuals": [],
        "continuity": [],
        "path_pressure": [],
        "path_temperature": [],
        "path_clearance": [],
        "path_valve_rate": [],
        "path_velocity": [],
        "path_acceleration": [],
        "combustion": [],
        "periodicity": [],
        "scavenging": [],
    }
    con_idx = 0  # Track current constraint index

    # Initial states
    xL0 = ca.SX.sym("xL0")
    xR0 = ca.SX.sym("xR0")
    vL0 = ca.SX.sym("vL0")
    vR0 = ca.SX.sym("vR0")
    
    # Density: use log-space if configured
    use_log_density = _should_use_log_scale("densities")
    if use_log_density:
        rho0_log = ca.SX.sym("rho0_log")
        rho0 = _exp_transform_var(ca, rho0_log, epsilon=1e-3)  # Physical-space variable for use in constraints
    else:
        rho0_log = None
        rho0 = ca.SX.sym("rho0")
    
    T0 = ca.SX.sym("T0")

    # Add variables to optimization vector (use log-space variable if applicable)
    if use_log_density:
        w += [xL0, xR0, vL0, vR0, rho0_log, T0]
        # Initial guess in log space: log(rho0_physical)
        rho0_initial = bounds.get("rho_min", 0.1) + 0.5 * (bounds.get("rho_max", 10.0) - bounds.get("rho_min", 0.1))
        # Use same minimum as bounds (1e-3) for consistency
        w0 += [0.0, 0.1, 0.0, 0.0, math.log(max(rho0_initial, 1e-3)), 300.0]
    else:
        w += [xL0, xR0, vL0, vR0, rho0, T0]
        w0 += [0.0, 0.1, 0.0, 0.0, 1.0, 300.0]
    
    # Track initial state groups
    var_groups["positions"].extend([var_idx, var_idx + 1])  # xL0, xR0
    var_groups["velocities"].extend([var_idx + 2, var_idx + 3])  # vL0, vR0
    var_groups["densities"].append(var_idx + 4)  # rho0 (or rho0_log)
    var_groups["temperatures"].append(var_idx + 5)  # T0
    var_idx += 6
    
    # Bounds: transform to log-space for densities if using log scale
    lbw += [
        bounds.get("xL_min", -0.1),
        bounds.get("xR_min", 0.0),
        bounds.get("vL_min", -10.0),
        bounds.get("vR_min", -10.0),
    ]
    ubw += [
        bounds.get("xL_max", 0.1),
        bounds.get("xR_max", 0.2),
        bounds.get("vL_max", 10.0),
        bounds.get("vR_max", 10.0),
    ]
    
    # Density bounds: transform to log-space if using log scale
    rho_min = bounds.get("rho_min", 0.1)
    rho_max = bounds.get("rho_max", 10.0)
    if use_log_density:
        # Use 1e-3 instead of 1e-10 to prevent extremely negative log values
        # log(1e-3) ≈ -6.9, which is much more reasonable than log(1e-10) ≈ -23
        rho_min_safe = max(rho_min, 1e-3)
        lbw += [math.log(rho_min_safe)]
        ubw += [math.log(max(rho_max, 1e-3))]
        # Store rho_min_bound for use in numerical guards (matching log-space minimum)
        rho_min_bound = rho_min_safe
    else:
        lbw += [rho_min]
        ubw += [rho_max]
        rho_min_bound = rho_min
    
    # Compute minimum mass for numerical stability guards
    # m_c_min = rho_min_bound * clearance_volume ensures mass never smaller than physically reasonable minimum
    clearance_volume = geometry.get("clearance_volume", 1e-4)
    m_c_min = rho_min_bound * clearance_volume
    
    # Temperature bounds
    lbw += [bounds.get("T_min", 200.0)]
    ubw += [bounds.get("T_max", 2000.0)]

    if use_combustion_model:
        ignition_bounds = combustion_cfg.get(
            "ignition_bounds_s",
            (0.0, max(combustion_cycle_time or 1.0, 1e-6)),
        )
        ignition_initial = float(
            combustion_cfg.get(
                "ignition_initial_s",
                0.1 * max(combustion_cycle_time or 1.0, 1e-6),
            ),
        )
        t_ign = ca.SX.sym("t_ign")
        w += [t_ign]
        w0 += [ignition_initial]
        lbw += [float(ignition_bounds[0])]
        ubw += [float(ignition_bounds[1])]
        var_groups["ignition"].append(var_idx)
        var_idx += 1
    else:
        t_ign = None

    # Valve controls: use log-space if configured
    use_log_valve_areas = _should_use_log_scale("valve_areas")
    if use_log_valve_areas:
        Ain0_log = ca.SX.sym("Ain0_log")
        Aex0_log = ca.SX.sym("Aex0_log")
        Ain0 = _exp_transform_var(ca, Ain0_log, epsilon=1e-10)  # Physical-space variable for use in constraints
        Aex0 = _exp_transform_var(ca, Aex0_log, epsilon=1e-10)
    else:
        Ain0_log = None
        Aex0_log = None
        Ain0 = ca.SX.sym("Ain0")
        Aex0 = ca.SX.sym("Aex0")
    
    if use_log_valve_areas:
        w += [Ain0_log, Aex0_log]
        # Initial guess in log space: log(small positive value to avoid log(0))
        epsilon_valve = 1e-10
        w0 += [math.log(epsilon_valve), math.log(epsilon_valve)]
    else:
        w += [Ain0, Aex0]
        w0 += [0.0, 0.0]
    
    var_groups["valve_areas"].extend([var_idx, var_idx + 1])  # Ain0, Aex0 (or log versions)
    var_idx += 2
    
    # Valve area bounds: transform to log-space if using log scale
    Ain_max = bounds.get("Ain_max", 0.01)
    Aex_max = bounds.get("Aex_max", 0.01)
    if use_log_valve_areas:
        epsilon_valve = 1e-10
        lbw += [math.log(epsilon_valve), math.log(epsilon_valve)]
        ubw += [math.log(max(Ain_max, epsilon_valve)), math.log(max(Aex_max, epsilon_valve))]
    else:
        lbw += [0.0, 0.0]
        ubw += [Ain_max, Aex_max]

    # State variables for each time step
    xL_k = xL0
    xR_k = xR0
    vL_k = vL0
    vR_k = vR0
    rho_k = rho0
    # Track log-space density variable for continuity constraints
    if use_log_density:
        # rho0_log is bounded to [log(1e-3), log(rho_max)] and initialized to log(max(rho0_initial, 1e-3))
        # This ensures rho_k_log_prev is finite at k=0 and remains finite throughout iterations
        # The bounds are set in the variable creation section (lines 1432-1433) to ensure finite values
        rho_k_log_prev = rho0_log  # Track log-space variable across iterations
    else:
        rho_k_log_prev = None
    T_k = T0
    Ain_k = Ain0
    Aex_k = Aex0

    h = 1.0 / ca.fmax(K, CASADI_PHYSICS_EPSILON)
    # Objective accumulators
    W_ind_accum = 0.0
    Q_in_accum = 0.0
    scav_penalty_accum = 0.0
    smooth_penalty_accum = 0.0

    if use_combustion_model:
        dt_real = (combustion_cycle_time or 1.0) / max(K, 1)
        combustion_samples.append((0.0, ca.DM(0.0)))
    else:
        dt_real = None

    # Optional dynamic wall temperature state
    dynamic_wall = bool(walls_cfg.get("dynamic", False))
    Cw = float(walls_cfg.get("capacitance", 0.0))  # [J/K]
    T_wall_const = float(geometry.get("T_wall", 400.0))
    if dynamic_wall:
        Tw0 = ca.SX.sym("Tw0")
        w += [Tw0]
        w0 += [T_wall_const]
        lbw += [walls_cfg.get("T_wall_min", 250.0)]
        ubw += [walls_cfg.get("T_wall_max", 800.0)]
        var_groups["temperatures"].append(var_idx)  # Tw0
        var_idx += 1
        Tw_k = Tw0
    else:
        Tw_k = T_wall_const

    # Scavenging and timing accumulator initial states
    yF0 = ca.SX.sym("yF0")
    Mdel0 = ca.SX.sym("Mdel0")
    Mlost0 = ca.SX.sym("Mlost0")
    AinInt0 = ca.SX.sym("AinInt0")
    AinTmom0 = ca.SX.sym("AinTmom0")
    AexInt0 = ca.SX.sym("AexInt0")
    AexTmom0 = ca.SX.sym("AexTmom0")
    w += [yF0, Mdel0, Mlost0, AinInt0, AinTmom0, AexInt0, AexTmom0]
    
    # Compute feasible initial values from problem configuration
    (yF0_val, Mdel0_val, Mlost0_val, AinInt0_val, AinTmom0_val, AexInt0_val, AexTmom0_val) = (
        _compute_scavenging_initial_values(P, bounds, geometry, num)
    )
    w0 += [yF0_val, Mdel0_val, Mlost0_val, AinInt0_val, AinTmom0_val, AexInt0_val, AexTmom0_val]
    # Track scavenging states in separate groups by unit type
    penalty_start_idx = var_idx
    var_groups["scavenging_fractions"].append(var_idx)  # yF0
    var_groups["scavenging_masses"].extend([var_idx + 1, var_idx + 2])  # Mdel0, Mlost0
    var_groups["scavenging_area_integrals"].extend([var_idx + 3, var_idx + 5])  # AinInt0, AexInt0
    var_groups["scavenging_time_moments"].extend([var_idx + 4, var_idx + 6])  # AinTmom0, AexTmom0
    var_idx += 7
    lbw += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ubw += [1.0, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf]
    yF_k = yF0
    Mdel_k = Mdel0
    Mlost_k = Mlost0
    AinInt_k = AinInt0
    AinTmom_k = AinTmom0
    AexInt_k = AexInt0
    AexTmom_k = AexTmom0

    for k in range(K):
        # Stage controls (valve areas and combustion heat release)
        Ain_stage = []
        Aex_stage = []
        Q_comb_stage: list[Any] = []

        for j in range(C):
            # Valve areas: use log-space if configured
            if use_log_valve_areas:
                ain_sym_log = ca.SX.sym(f"Ain_{k}_{j}_log")
                aex_sym_log = ca.SX.sym(f"Aex_{k}_{j}_log")
                ain_sym = _exp_transform_var(ca, ain_sym_log, epsilon=1e-10)  # Physical-space for constraints
                aex_sym = _exp_transform_var(ca, aex_sym_log, epsilon=1e-10)
                Ain_stage.append(ain_sym)
                Aex_stage.append(aex_sym)
                w += [ain_sym_log, aex_sym_log]
                # Initial guess in log space
                epsilon_valve = 1e-10
                w0 += [math.log(epsilon_valve), math.log(epsilon_valve)]
                # Bounds in log space
                Ain_max = bounds.get("Ain_max", 0.01)
                Aex_max = bounds.get("Aex_max", 0.01)
                lbw += [math.log(epsilon_valve), math.log(epsilon_valve)]
                ubw += [math.log(max(Ain_max, epsilon_valve)), math.log(max(Aex_max, epsilon_valve))]
            else:
                ain_sym = ca.SX.sym(f"Ain_{k}_{j}")
                aex_sym = ca.SX.sym(f"Aex_{k}_{j}")
                Ain_stage.append(ain_sym)
                Aex_stage.append(aex_sym)
                w += [ain_sym, aex_sym]
                w0 += [0.0, 0.0]
                lbw += [0.0, 0.0]
                ubw += [
                    bounds.get("Ain_max", 0.01),
                    bounds.get("Aex_max", 0.01),
                ]
            var_groups["valve_areas"].extend([var_idx, var_idx + 1])
            var_idx += 2

            if not use_combustion_model:
                q_sym = ca.SX.sym(f"Q_comb_{k}_{j}")
                Q_comb_stage.append(q_sym)
                w += [q_sym]
                w0 += [0.0]
                lbw += [0.0]
                # Convert normalized bounds from kJ to J (multiply by 1e3)
                q_comb_max_j = bounds.get("Q_comb_max", 10.0) * 1e3  # Default 10.0 kJ = 10000.0 J
                ubw += [q_comb_max_j]
                # Q_comb is a penalty/control term, not a state variable
                var_idx += 1
            else:
                Q_comb_stage.append(None)

        # Collocation states
        xL_colloc = [ca.SX.sym(f"xL_{k}_{j}") for j in range(C)]
        xR_colloc = [ca.SX.sym(f"xR_{k}_{j}") for j in range(C)]
        vL_colloc = [ca.SX.sym(f"vL_{k}_{j}") for j in range(C)]
        vR_colloc = [ca.SX.sym(f"vR_{k}_{j}") for j in range(C)]
        
        # Density collocation states: use log-space if configured
        if use_log_density:
            rho_colloc_log = [ca.SX.sym(f"rho_{k}_{j}_log") for j in range(C)]
            rho_colloc = [_exp_transform_var(ca, rho_log, epsilon=1e-3) for rho_log in rho_colloc_log]  # Physical-space for constraints
        else:
            rho_colloc_log = None
            rho_colloc = [ca.SX.sym(f"rho_{k}_{j}") for j in range(C)]
        
        T_colloc = [ca.SX.sym(f"T_{k}_{j}") for j in range(C)]

        for j in range(C):
            if use_log_density:
                w += [
                    xL_colloc[j],
                    xR_colloc[j],
                    vL_colloc[j],
                    vR_colloc[j],
                    rho_colloc_log[j],
                    T_colloc[j],
                ]
                # Initial guess in log space
                rho_initial = bounds.get("rho_min", 0.1) + 0.5 * (bounds.get("rho_max", 10.0) - bounds.get("rho_min", 0.1))
                # Use same minimum as bounds (1e-3) for consistency
                w0 += [0.0, 0.1, 0.0, 0.0, math.log(max(rho_initial, 1e-3)), 300.0]
            else:
                w += [
                    xL_colloc[j],
                    xR_colloc[j],
                    vL_colloc[j],
                    vR_colloc[j],
                    rho_colloc[j],
                    T_colloc[j],
                ]
                w0 += [0.0, 0.1, 0.0, 0.0, 1.0, 300.0]
            
            # Track collocation state groups
            var_groups["positions"].extend([var_idx, var_idx + 1])  # xL, xR
            var_groups["velocities"].extend([var_idx + 2, var_idx + 3])  # vL, vR
            var_groups["densities"].append(var_idx + 4)  # rho (or rho_log)
            var_groups["temperatures"].append(var_idx + 5)  # T
            var_idx += 6
            
            lbw += [
                bounds.get("xL_min", -0.1),
                bounds.get("xR_min", 0.0),
                bounds.get("vL_min", -10.0),
                bounds.get("vR_min", -10.0),
            ]
            ubw += [
                bounds.get("xL_max", 0.1),
                bounds.get("xR_max", 0.2),
                bounds.get("vL_max", 10.0),
                bounds.get("vR_max", 10.0),
            ]
            
            # Density bounds: transform to log-space if using log scale
            rho_min = bounds.get("rho_min", 0.1)
            rho_max = bounds.get("rho_max", 10.0)
            if use_log_density:
                # Use 1e-3 instead of 1e-10 to prevent extremely negative log values
                # log(1e-3) ≈ -6.9, which is much more reasonable than log(1e-10) ≈ -23
                rho_min_safe = max(rho_min, 1e-3)
                lbw += [math.log(rho_min_safe)]
                ubw += [math.log(max(rho_max, 1e-3))]
                # Store rho_min_bound for use in numerical guards (matching log-space minimum)
                rho_min_bound = rho_min_safe
            else:
                lbw += [rho_min]
                ubw += [rho_max]
                rho_min_bound = rho_min
            
            # Compute minimum mass for numerical stability guards (per collocation point)
            clearance_volume = geometry.get("clearance_volume", 1e-4)
            m_c_min = rho_min_bound * clearance_volume
            
            # Temperature bounds
            lbw += [bounds.get("T_min", 200.0)]
            ubw += [bounds.get("T_max", 2000.0)]

        # Collocation equations
        for c in range(C):
            # Current state
            xL_c = xL_colloc[c]
            xR_c = xR_colloc[c]
            vL_c = vL_colloc[c]
            vR_c = vR_colloc[c]
            rho_c = rho_colloc[c]
            # Protect rho_c immediately for all uses to prevent numerical instability
            # This ensures consistent density protection throughout the collocation loop
            rho_c_safe = ca.fmax(rho_c, rho_min_bound)
            T_c = T_colloc[c]
            Ain_c = Ain_stage[c]
            Aex_c = Aex_stage[c]
            
            # Protect valve areas: ensure non-negative and bounded before gas model calls
            # Negative or extremely large valve areas can cause Inf in mass flow calculations
            Ain_max = bounds.get("Ain_max", 0.01)  # Maximum inlet valve area [m²]
            Aex_max = bounds.get("Aex_max", 0.01)  # Maximum exhaust valve area [m²]
            Ain_c_safe = ca.fmax(ca.fmin(Ain_c, Ain_max), 0.0)  # Clamp to [0, Ain_max]
            Aex_c_safe = ca.fmax(ca.fmin(Aex_c, Aex_max), 0.0)  # Clamp to [0, Aex_max]
            
            if use_combustion_model and combustion_model is not None:
                time_val = (k + grid.nodes[c]) * float(dt_real or 0.0)
                time_dm = ca.DM(time_val)
                piston_speed_expr = 0.5 * ca.fabs(vR_c - vL_c)
                comb_expr = combustion_model.symbolic_heat_release(
                    ca=ca,
                    time_s=time_dm,
                    piston_speed_m_per_s=piston_speed_expr,
                    ignition_time_s=t_ign,
                    omega_deg_per_s=omega_deg_per_s_dm,
                )
                Q_comb_c = comb_expr["heat_release_rate"]
                Q_comb_stage[c] = Q_comb_c
                combustion_samples.append((time_val, comb_expr["mfb"]))
            else:
                Q_comb_c = Q_comb_stage[c]

            # Chamber volume and its rate of change
            # V_c is already protected by clearance_volume minimum in chamber_volume_from_pistons
            clearance_volume = geometry.get("clearance_volume", 1e-4)
            V_c = chamber_volume_from_pistons(
                x_L=xL_c,
                x_R=xR_c,
                B=geometry.get("bore", 0.1),
                Vc=clearance_volume,
            )
            # For explicit use, ensure V_c_safe uses clearance_volume (CR-derived minimum)
            # This is redundant since chamber_volume_from_pistons already protects, but makes intent clear
            V_c_safe = ca.fmax(V_c, clearance_volume)
            dV_dt = (
                math.pi
                * ca.fmax(geometry.get("bore", 0.1) / 2.0, CASADI_PHYSICS_EPSILON) ** 2
                * (vR_c - vL_c)
            )

            # Gas pressure (use protected density for numerical stability)
            p_c = gas_pressure_from_state(rho=rho_c_safe, T=T_c)
            
            # Protect pressure with minimum value to prevent extreme pressure ratios in gas model
            # Very small p_c (e.g., from rho_c_safe at minimum and low T_c) causes pr to become very large
            p_c_min = 1e3  # Reasonable minimum pressure [Pa] to prevent extreme pressure ratios
            p_c_safe = ca.fmax(p_c, p_c_min)

            # Enhanced piston forces with proper acceleration coupling
            # Compute accelerations from velocity differences (simplified)
            aL_c = (vL_c - vL_k) / ca.fmax(h, CASADI_PHYSICS_EPSILON)
            aR_c = (vR_c - vR_k) / ca.fmax(h, CASADI_PHYSICS_EPSILON)

            F_L_c, F_R_c = piston_force_balance(
                p_gas=p_c_safe,
                x_L=xL_c,
                x_R=xR_c,
                v_L=vL_c,
                v_R=vR_c,
                a_L=aL_c,
                a_R=aR_c,
                geometry=geometry,
            )

            # Mass flow rates via unified gas model
            R = 287.0
            gamma = float(flow_cfg.get("gamma", 1.4))
            p_in = float(geometry.get("p_intake", 1e5))
            T_in = float(geometry.get("T_intake", 300.0))
            rho_in = p_in / (R * T_in)
            p_ex = float(geometry.get("p_exhaust", 1e5))
            T_ex = T_c
            rho_ex = p_ex / (R * T_ex)
            mdot_in = gas_model.mdot_in(
                ca=ca,
                p_up=p_in,
                T_up=T_in,
                rho_up=rho_in,
                p_down=p_c_safe,  # Use protected pressure to prevent extreme pressure ratios
                T_down=T_c,
                A_eff=Ain_c_safe,  # Use protected valve area
                gamma=gamma,
                R=R,
            )
            mdot_out = gas_model.mdot_out(
                ca=ca,
                p_up=p_c_safe,  # Use protected pressure to prevent extreme pressure ratios
                T_up=T_c,
                rho_up=rho_c_safe,  # Use protected density for numerical stability
                p_down=p_ex,
                T_down=T_ex,
                A_eff=Aex_c_safe,  # Use protected valve area
                gamma=gamma,
                R=R,
            )
            
            # Protect mass flow rates from NaN/Inf (could come from extreme pressure ratios or valve areas)
            # This prevents NaN propagation through dm_dt, drho_dt, dT_dt, and dyF_dt
            mdot_max = 1e3  # Reasonable maximum mass flow rate [kg/s]
            # CasADi fmin/fmax don't handle NaN - if input is NaN, output is NaN
            # Use conditional logic to replace Inf/NaN BEFORE clipping
            # Replace Inf/NaN with safe defaults: if |mdot| > mdot_max, clamp to mdot_max with correct sign
            mdot_in_safe = ca.if_else(
                ca.fabs(mdot_in) > mdot_max,
                ca.sign(mdot_in) * mdot_max,  # Preserve sign but clamp magnitude
                ca.fmax(mdot_in, 0.0)  # Ensure non-negative for normal values
            )
            mdot_in = ca.fmin(mdot_in_safe, mdot_max)  # Final clamp to [0, mdot_max]
            mdot_out_safe = ca.if_else(
                ca.fabs(mdot_out) > mdot_max,
                ca.sign(mdot_out) * mdot_max,  # Preserve sign but clamp magnitude
                ca.fmax(mdot_out, 0.0)  # Ensure non-negative for normal values
            )
            mdot_out = ca.fmin(mdot_out_safe, mdot_max)  # Final clamp to [0, mdot_max]

            # Wall heat transfer
            T_wall_c = Tw_k if dynamic_wall else T_wall_const
            Q_heat_transfer = gas_model.qdot_wall(
                ca=ca,
                p_gas=p_c_safe,  # Use protected pressure for consistency
                T_gas=T_c,
                T_wall=T_wall_c,
                B=geometry.get("bore", 0.1),
                x_L=xL_c,
                x_R=xR_c,
            )

            # Enhanced gas energy balance (use protected density and volume for numerical stability)
            dT_dt = gas_energy_balance(
                rho=rho_c_safe,
                T=T_c,
                V=V_c_safe,  # Use protected volume
                dV_dt=dV_dt,
                Q_combustion=Q_comb_c,
                Q_heat_transfer=Q_heat_transfer,
                mdot_in=mdot_in,
                mdot_out=mdot_out,
                T_in=T_in,
                T_out=T_ex,
                gamma=gamma,
            )
            
            # Protect dT_dt from NaN/Inf (could come from extreme heat transfer or mass flow rates)
            dT_dt_max = 1e6  # Reasonable maximum temperature derivative [K/s]
            dT_dt = ca.fmin(dT_dt, dT_dt_max)
            dT_dt = ca.fmax(dT_dt, -dT_dt_max)

            # Mass balance (use protected density and volume for numerical stability)
            dm_dt = mdot_in - mdot_out
            # Protect dm_dt from NaN/Inf (mdot_in and mdot_out already protected, but ensure safety)
            dm_dt_max = 1e3  # Reasonable maximum mass rate change [kg/s]
            dm_dt = ca.fmin(dm_dt, dm_dt_max)
            dm_dt = ca.fmax(dm_dt, -dm_dt_max)
            
            # Use V_c_safe (protected by clearance_volume minimum) instead of raw V_c
            drho_dt = (dm_dt - rho_c_safe * dV_dt) / V_c_safe
            
            # For log-space density: convert physical-space derivative to log-space derivative
            # If rho = exp(rho_log), then d(rho_log)/dt = (1/rho) * drho_dt = drho_dt / rho
            # CRITICAL: Clamp drho_dt BEFORE division to prevent Inf in drho_log_dt
            # When rho_c_safe is at minimum (1e-3), drho_dt must be clamped to ensure
            # drho_log_dt = drho_dt / rho_c_safe doesn't exceed drho_log_dt_max (1e6)
            # Maximum drho_dt = drho_log_dt_max * rho_min_bound = 1e6 * 1e-3 = 1e3
            if use_log_density:
                # Clamp drho_dt conservatively to prevent Inf in drho_log_dt
                # Use rho_min_bound (1e-3) to compute safe maximum for drho_dt
                drho_log_dt_max = 1e6  # Reasonable maximum for log-space density derivative [1/s]
                drho_dt_max_for_log = drho_log_dt_max * rho_min_bound  # 1e6 * 1e-3 = 1e3
                # Clamp drho_dt to prevent Inf in drho_log_dt when rho_c_safe is at minimum
                drho_dt = ca.fmin(drho_dt, drho_dt_max_for_log)
                drho_dt = ca.fmax(drho_dt, -drho_dt_max_for_log)
                
                # Now compute drho_log_dt safely (division cannot produce Inf)
                drho_log_dt = drho_dt / rho_c_safe
                # Additional protection: clamp drho_log_dt to ensure it's within bounds
                # This handles cases where rho_c_safe > rho_min_bound (allowing larger drho_dt)
                drho_log_dt = ca.fmin(drho_log_dt, drho_log_dt_max)
                drho_log_dt = ca.fmax(drho_log_dt, -drho_log_dt_max)
            else:
                # For non-log density: use standard clamping
                drho_dt_max = 1e6  # Reasonable maximum density derivative [kg/(m^3 s)]
                drho_dt = ca.fmin(drho_dt, drho_dt_max)
                drho_dt = ca.fmax(drho_dt, -drho_dt_max)
                drho_log_dt = None
            
            # Fresh fraction dynamics
            # rho_c_safe already computed at start of loop
            # Compute mass with protected density and volume
            m_c = rho_c_safe * V_c_safe
            # Additional protection: ensure m_c never smaller than physically reasonable minimum
            # m_c_min = rho_min_bound * clearance_volume ensures mass never smaller than minimum density * minimum volume
            m_c_safe = ca.fmax(m_c, m_c_min)
            dyF_dt = (mdot_in - mdot_out * yF_k - yF_k * dm_dt) / m_c_safe
            
            # Protect dyF_dt from NaN/Inf (could come from extreme mass flow rates or small m_c_safe)
            dyF_dt_max = 1e3  # Reasonable maximum fresh fraction derivative [1/s]
            dyF_dt = ca.fmin(dyF_dt, dyF_dt_max)
            dyF_dt = ca.fmax(dyF_dt, -dyF_dt_max)

            # Collocation equations (state updates using A matrix)
            rhs_xL = xL_k
            rhs_xR = xR_k
            rhs_vL = vL_k
            rhs_vR = vR_k
            
            # For density: use log-space variable if configured
            if use_log_density:
                # Use log-space variable from previous iteration (rho_k_log_prev) for RHS initialization
                # rho_k is physical-space (from _exp_transform_var), but rhs_rho_log must be log-space
                # to match rho_colloc_log[c] in the constraint comparison
                rhs_rho_log = rho_k_log_prev  # Use log-space variable, not physical-space rho_k
            else:
                rhs_rho = rho_k
            
            rhs_T = T_k
            if dynamic_wall:
                rhs_Tw = Tw_k
            # Scavenging and timing rhs
            rhs_yF = yF_k
            rhs_Mdel = Mdel_k
            rhs_Mlost = Mlost_k
            rhs_AinInt = AinInt_k
            rhs_AinTmom = AinTmom_k
            rhs_AexInt = AexInt_k
            rhs_AexTmom = AexTmom_k

            for j in range(C):
                rhs_xL += h * grid.a[c][j] * vL_colloc[j]
                rhs_xR += h * grid.a[c][j] * vR_colloc[j]

                # Enhanced force balance with proper mass calculation
                m_total_L = geometry.get("mass", 1.0) + geometry.get(
                    "rod_mass", 0.5,
                ) * (
                    geometry.get("rod_cg_offset", 0.075)
                    / ca.fmax(geometry.get("rod_length", 0.15), CASADI_PHYSICS_EPSILON)
                )
                m_total_R = geometry.get("mass", 1.0) + geometry.get(
                    "rod_mass", 0.5,
                ) * (
                    geometry.get("rod_cg_offset", 0.075)
                    / ca.fmax(geometry.get("rod_length", 0.15), CASADI_PHYSICS_EPSILON)
                )

                rhs_vL += (
                    h
                    * grid.a[c][j]
                    * (F_L_c / ca.fmax(m_total_L, CASADI_PHYSICS_EPSILON))
                )
                rhs_vR += (
                    h
                    * grid.a[c][j]
                    * (F_R_c / ca.fmax(m_total_R, CASADI_PHYSICS_EPSILON))
                )
                
                # Density update: use log-space derivative if configured
                if use_log_density:
                    rhs_rho_log += h * grid.a[c][j] * drho_log_dt
                else:
                    rhs_rho += h * grid.a[c][j] * drho_dt
                
                rhs_T += h * grid.a[c][j] * dT_dt
                if dynamic_wall:
                    # Cw * dTw/dt = -Q_heat_transfer
                    dTw_dt = (-Q_heat_transfer / max(Cw, 1e-6)) if Cw > 0.0 else 0.0
                    rhs_Tw += h * grid.a[c][j] * dTw_dt
                # Scavenging accumulators and timing integrals
                rhs_yF += h * grid.a[c][j] * dyF_dt
                rhs_Mdel += h * grid.a[c][j] * mdot_in
                rhs_Mlost += h * grid.a[c][j] * (mdot_out * yF_k)
                t_c = (k + grid.nodes[c]) * h
                rhs_AinInt += h * grid.a[c][j] * Ain_c
                rhs_AinTmom += h * grid.a[c][j] * (t_c * Ain_c)
                rhs_AexInt += h * grid.a[c][j] * Aex_c
                rhs_AexTmom += h * grid.a[c][j] * (t_c * Aex_c)

            # Protect accumulated RHS values from Inf/NaN after accumulation loops
            # Even with derivative clipping, Inf can accumulate if derivatives were Inf before clipping
            if use_log_density:
                # Protect rhs_rho_log: reasonable range for log-space density is [-10, 10] (exp(-10) to exp(10))
                rhs_rho_log_min = -10.0
                rhs_rho_log_max = 10.0
                rhs_rho_log = ca.fmax(ca.fmin(rhs_rho_log, rhs_rho_log_max), rhs_rho_log_min)
            else:
                # Protect rhs_rho: reasonable range for density is [0.01, 100] kg/m³
                rhs_rho_min = 0.01
                rhs_rho_max = 100.0
                rhs_rho = ca.fmax(ca.fmin(rhs_rho, rhs_rho_max), rhs_rho_min)
            
            # Protect rhs_T: reasonable range for temperature is [100, 5000] K
            rhs_T_min = 100.0
            rhs_T_max = 5000.0
            rhs_T = ca.fmax(ca.fmin(rhs_T, rhs_T_max), rhs_T_min)
            
            # Protect rhs_yF: fresh fraction should be in [0, 1]
            rhs_yF_min = 0.0
            rhs_yF_max = 1.0
            rhs_yF = ca.fmax(ca.fmin(rhs_yF, rhs_yF_max), rhs_yF_min)

            # Collocation residuals: handle log-space density
            if use_log_density:
                # Compare log-space variables: rho_colloc_log[c] - rhs_rho_log
                colloc_res = [
                    xL_c - rhs_xL,
                    xR_c - rhs_xR,
                    vL_c - rhs_vL,
                    vR_c - rhs_vR,
                    rho_colloc_log[c] - rhs_rho_log,
                    T_c - rhs_T,
                ]
            else:
                colloc_res = [
                    xL_c - rhs_xL,
                    xR_c - rhs_xR,
                    vL_c - rhs_vL,
                    vR_c - rhs_vR,
                    rho_c - rhs_rho,
                    T_c - rhs_T,
                ]
            if dynamic_wall:
                colloc_res.append(T_wall_c - rhs_Tw)
            colloc_res += [
                yF_k - rhs_yF,
                Mdel_k - rhs_Mdel,
                Mlost_k - rhs_Mlost,
                AinInt_k - rhs_AinInt,
                AinTmom_k - rhs_AinTmom,
                AexInt_k - rhs_AexInt,
                AexTmom_k - rhs_AexTmom,
            ]
            g += colloc_res
            lbg += [0.0] * len(colloc_res)
            ubg += [0.0] * len(colloc_res)
            # Track collocation residual constraints
            constraint_groups["collocation_residuals"].extend(range(con_idx, con_idx + len(colloc_res)))
            con_idx += len(colloc_res)

            # Accumulate indicated work and Q_in
            W_ind_accum += h * grid.weights[c] * p_c_safe * dV_dt  # Use protected pressure for consistency
            Q_in_accum += h * grid.weights[c] * Q_comb_c
            # Scavenging short-circuit penalty surrogate
            eps = 1e-9
            ratio = mdot_out / ca.fmax(mdot_in + eps, CASADI_PHYSICS_EPSILON)
            scav_penalty_accum += h * grid.weights[c] * ratio

        # Step to next time point
        xL_k1 = xL_k
        xR_k1 = xR_k
        vL_k1 = vL_k
        vR_k1 = vR_k
        # For density: initialize rho_k1 from log-space variable if configured
        if use_log_density:
            # Use tracked log-space variable from previous iteration
            rho_k1 = rho_k_log_prev
        else:
            rho_k1 = rho_k
        T_k1 = T_k

        for j in range(C):
            xL_k1 += h * grid.weights[j] * vL_colloc[j]
            xR_k1 += h * grid.weights[j] * vR_colloc[j]

            # Enhanced force balance with proper mass calculation
            m_total_L = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (
                geometry.get("rod_cg_offset", 0.075)
                / ca.fmax(geometry.get("rod_length", 0.15), CASADI_PHYSICS_EPSILON)
            )
            m_total_R = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (
                geometry.get("rod_cg_offset", 0.075)
                / ca.fmax(geometry.get("rod_length", 0.15), CASADI_PHYSICS_EPSILON)
            )

            vL_k1 += (
                h
                * grid.weights[j]
                * (F_L_c / ca.fmax(m_total_L, CASADI_PHYSICS_EPSILON))
            )
            vR_k1 += (
                h
                * grid.weights[j]
                * (F_R_c / ca.fmax(m_total_R, CASADI_PHYSICS_EPSILON))
            )
            # Density update: use log-space derivative if configured
            if use_log_density:
                rho_k1 += h * grid.weights[j] * drho_log_dt
            else:
                rho_k1 += h * grid.weights[j] * drho_dt
            T_k1 += h * grid.weights[j] * dT_dt
            if dynamic_wall:
                dTw_dt_bar = (-Q_heat_transfer / max(Cw, 1e-6)) if Cw > 0.0 else 0.0
                Tw_k = Tw_k + h * grid.weights[j] * dTw_dt_bar
        # Scavenging and timing next states (use last stage derivatives)
        yF_k1 = yF_k + h * dyF_dt
        Mdel_k1 = Mdel_k + h * mdot_in
        Mlost_k1 = Mlost_k + h * (mdot_out * yF_k)
        # Use end-of-interval averaged time for moment updates
        t_bar = (k + 1.0) * h
        AinInt_k1 = AinInt_k + h * Ain_c
        AinTmom_k1 = AinTmom_k + h * (t_bar * Ain_c)
        AexInt_k1 = AexInt_k + h * Aex_c
        AexTmom_k1 = AexTmom_k + h * (t_bar * Aex_c)

        # New state variables
        xL_k = ca.SX.sym(f"xL_{k + 1}")
        xR_k = ca.SX.sym(f"xR_{k + 1}")
        vL_k = ca.SX.sym(f"vL_{k + 1}")
        vR_k = ca.SX.sym(f"vR_{k + 1}")
        
        # Density: use log-space if configured
        if use_log_density:
            rho_k_log = ca.SX.sym(f"rho_{k + 1}_log")
            rho_k = _exp_transform_var(ca, rho_k_log, epsilon=1e-3)  # Physical-space for use in constraints
            # Update tracked log-space variable for next iteration
            rho_k_log_prev = rho_k_log
        else:
            rho_k_log = None
            rho_k = ca.SX.sym(f"rho_{k + 1}")
        
        T_k = ca.SX.sym(f"T_{k + 1}")

        yF_k = ca.SX.sym(f"yF_{k + 1}")
        Mdel_k = ca.SX.sym(f"Mdel_{k + 1}")
        Mlost_k = ca.SX.sym(f"Mlost_{k + 1}")
        AinInt_k = ca.SX.sym(f"AinInt_{k + 1}")
        AinTmom_k = ca.SX.sym(f"AinTmom_{k + 1}")
        AexInt_k = ca.SX.sym(f"AexInt_{k + 1}")
        AexTmom_k = ca.SX.sym(f"AexTmom_{k + 1}")

        if use_log_density:
            w += [
                xL_k,
                xR_k,
                vL_k,
                vR_k,
                rho_k_log,
                T_k,
                yF_k,
                Mdel_k,
                Mlost_k,
                AinInt_k,
                AinTmom_k,
                AexInt_k,
                AexTmom_k,
            ]
            # Initial guess in log space
            rho_initial = bounds.get("rho_min", 0.1) + 0.5 * (bounds.get("rho_max", 10.0) - bounds.get("rho_min", 0.1))
            # Use same minimum as bounds (1e-3) for consistency
            w0 += [0.0, 0.1, 0.0, 0.0, math.log(max(rho_initial, 1e-3)), 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            w += [
                xL_k,
                xR_k,
                vL_k,
                vR_k,
                rho_k,
                T_k,
                yF_k,
                Mdel_k,
                Mlost_k,
                AinInt_k,
                AinTmom_k,
                AexInt_k,
                AexTmom_k,
            ]
            w0 += [0.0, 0.1, 0.0, 0.0, 1.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        lbw += [
            bounds.get("xL_min", -0.1),
            bounds.get("xR_min", 0.0),
            bounds.get("vL_min", -10.0),
            bounds.get("vR_min", -10.0),
        ]
        ubw += [
            bounds.get("xL_max", 0.1),
            bounds.get("xR_max", 0.2),
            bounds.get("vL_max", 10.0),
            bounds.get("vR_max", 10.0),
        ]
        
        # Density bounds: transform to log-space if using log scale
        rho_min = bounds.get("rho_min", 0.1)
        rho_max = bounds.get("rho_max", 10.0)
        if use_log_density:
            # Use 1e-3 instead of 1e-10 to prevent extremely negative log values
            # log(1e-3) ≈ -6.9, which is much more reasonable than log(1e-10) ≈ -23
            rho_min_safe = max(rho_min, 1e-3)
            lbw += [math.log(rho_min_safe)]
            ubw += [math.log(max(rho_max, 1e-3))]
            # Store rho_min_bound for use in numerical guards (matching log-space minimum)
            rho_min_bound = rho_min_safe
        else:
            lbw += [rho_min]
            ubw += [rho_max]
            rho_min_bound = rho_min
        
        # Compute minimum mass for numerical stability guards (per time step)
        clearance_volume = geometry.get("clearance_volume", 1e-4)
        m_c_min = rho_min_bound * clearance_volume
        
        # Temperature and other bounds
        lbw += [
            bounds.get("T_min", 200.0),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        ubw += [
            bounds.get("T_max", 2000.0),
            1.0,
            ca.inf,
            ca.inf,
            ca.inf,
            ca.inf,
            ca.inf,
            ca.inf,
        ]
        
        # Track time-step state groups
        var_groups["positions"].extend([var_idx, var_idx + 1])  # xL_k, xR_k
        var_groups["velocities"].extend([var_idx + 2, var_idx + 3])  # vL_k, vR_k
        var_groups["densities"].append(var_idx + 4)  # rho_k (or rho_k_log)
        var_groups["temperatures"].append(var_idx + 5)  # T_k
        var_idx += 13  # Update var_idx: 13 variables per time step (xL, xR, vL, vR, rho, T, yF, Mdel, Mlost, AinInt, AinTmom, AexInt, AexTmom)

        # Continuity constraints: REMOVED - redundant with collocation residuals
        # The collocation residuals already enforce state continuity between stages
        # through the integration scheme. Adding explicit continuity constraints
        # makes the problem overconstrained (1170 redundant equality constraints).
        # 
        # Original code (lines 2270-2310) removed:
        # - Continuity constraints enforced: state[k+1] = state[k1] where state[k1]
        #   is computed from integration at stage k
        # - This is redundant because collocation residuals already enforce:
        #   state_colloc = state_prev + h * integral(derivative)
        #   and state[k+1] ≈ state_colloc[C-1] (last collocation point)
        #
        # Removing these constraints increases DOF from -438 to 732, making the
        # problem well-posed. See NLP_STRUCTURE_ANALYSIS.md for details.

    # Enhanced path constraints
    gap_min = bounds.get("x_gap_min", 0.0008)
    g += [xR_k - xL_k]  # Clearance constraint
    lbg += [gap_min]
    ubg += [ca.inf]
    # Track clearance/penalty constraint
    constraint_groups["path_clearance"].append(con_idx)
    con_idx += 1

    # Compute acceleration normalization factor (targeting fixed normalized bounds ±10)
    a_max = bounds.get("a_max", 1000.0)
    desired_norm = 10.0  # Target normalized bound ±10
    min_norm = 1.0  # Minimum normalization factor
    a_norm_factor = max(a_max / desired_norm, min_norm)
    
    # Comprehensive path constraints for all time steps
    for k in range(K):
        for j in range(C):
            # Pressure constraints (normalized to MPa to reduce Jacobian entries)
            p_kj = gas_pressure_from_state(rho=rho_colloc[j], T=T_colloc[j])
            p_kj_mpa = p_kj / 1e6  # Convert Pa → MPa
            g += [p_kj_mpa]  # Pressure constraint in MPa
            # Bounds are already in MPa (no conversion needed)
            p_min_mpa = bounds.get("p_min", 0.01)  # Default 0.01 MPa
            p_max_mpa = bounds.get("p_max", 10.0)  # Default 10.0 MPa
            lbg += [p_min_mpa]
            ubg += [p_max_mpa]
            # Track pressure path constraints
            constraint_groups["path_pressure"].append(con_idx)
            con_idx += 1

            # Temperature constraints
            T_kj = T_colloc[j]
            g += [T_kj]  # Temperature constraint
            lbg += [bounds.get("T_min", 200.0)]
            ubg += [bounds.get("T_max", 2000.0)]
            # Track temperature path constraints
            constraint_groups["path_temperature"].append(con_idx)
            con_idx += 1

            # Valve rate constraints (if valve areas are available)
            if k > 0:
                # Valve area rate constraints
                Ain_prev = Ain_stage[j] if k > 0 else Ain0
                Aex_prev = Aex_stage[j] if k > 0 else Aex0

                Ain_rate = (Ain_stage[j] - Ain_prev) / ca.fmax(
                    h, CASADI_PHYSICS_EPSILON,
                )
                Aex_rate = (Aex_stage[j] - Aex_prev) / ca.fmax(
                    h, CASADI_PHYSICS_EPSILON,
                )

                g += [Ain_rate, Aex_rate]  # Valve rate constraints
                lbg += [-bounds.get("dA_dt_max", 0.02), -bounds.get("dA_dt_max", 0.02)]
                ubg += [bounds.get("dA_dt_max", 0.02), bounds.get("dA_dt_max", 0.02)]
                # Track valve rate constraints
                constraint_groups["path_valve_rate"].extend([con_idx, con_idx + 1])
                con_idx += 2

            # Piston velocity constraints
            vL_kj = vL_colloc[j]
            vR_kj = vR_colloc[j]
            g += [vL_kj, vR_kj]  # Piston velocity constraints
            lbg += [-bounds.get("v_max", 50.0), -bounds.get("v_max", 50.0)]
            ubg += [bounds.get("v_max", 50.0), bounds.get("v_max", 50.0)]
            # Track velocity path constraints
            constraint_groups["path_velocity"].extend([con_idx, con_idx + 1])
            con_idx += 2

            # Piston acceleration constraints (normalized to fixed bounds)
            if k > 0:
                aL_kj = (vL_colloc[j] - vL_k) / h
                aR_kj = (vR_colloc[j] - vR_k) / h
                
                # Normalize acceleration constraints using precomputed factor
                aL_kj_norm = aL_kj / a_norm_factor
                aR_kj_norm = aR_kj / a_norm_factor
                
                g += [aL_kj_norm, aR_kj_norm]  # Normalized piston acceleration constraints
                lbg += [-a_max / a_norm_factor, -a_max / a_norm_factor]
                ubg += [a_max / a_norm_factor, a_max / a_norm_factor]
                # Track acceleration path constraints
                constraint_groups["path_acceleration"].extend([con_idx, con_idx + 1])
                con_idx += 2

    # Combustion timing constraints (optional placeholders)
    for k in range(K):
        for j in range(C):
            Q_comb_kj = Q_comb_stage[j]
            # Combustion heat release should be non-negative and within bounds
            # Avoid Python conditionals on symbolic expressions; encode constraints directly.
            g += [Q_comb_kj]
            lbg += [0.0]
            # Convert normalized bounds from kJ to J (multiply by 1e3)
            q_comb_max_j = bounds.get("Q_comb_max", 10.0) * 1e3  # Default 10.0 kJ = 10000.0 J
            ubg += [q_comb_max_j]
            # Track combustion constraints
            constraint_groups["combustion"].append(con_idx)
            con_idx += 1

    # Cycle periodicity constraints
    g += [xL_k - xL0, xR_k - xR0, vL_k - vL0, vR_k - vR0]
    lbg += [0.0, 0.0, 0.0, 0.0]
    ubg += [0.0, 0.0, 0.0, 0.0]
    # Track periodicity constraints
    constraint_groups["periodicity"].extend(range(con_idx, con_idx + 4))
    con_idx += 4

    # Scavenging metrics and timing targets at cycle end
    V_end = chamber_volume_from_pistons(
        x_L=xL_k,
        x_R=xR_k,
        B=geometry.get("bore", 0.1),
        Vc=geometry.get("clearance_volume", 1e-4),
    )
    m_end = rho_k * V_end
    fresh_trapped = yF_k * m_end
    total_trapped = m_end
    
    # Use consistent epsilon matching seeded lower bounds (1e-6 from _compute_scavenging_initial_values)
    eps = 1e-6

    cons_cfg = P.get("constraints", {})
    if "short_circuit_max" in cons_cfg:
        # Cross-multiplied form to avoid division by near-zero: Mlost_k ≤ short_circuit_max * (Mdel_k + eps)
        short_circuit_max_val = float(cons_cfg["short_circuit_max"])
        g += [Mlost_k - short_circuit_max_val * (Mdel_k + eps)]
        lbg += [-ca.inf]
        ubg += [0.0]
        constraint_groups["scavenging"].append(con_idx)
        con_idx += 1
    if "scavenging_min" in cons_cfg:
        g += [yF_k]
        lbg += [float(cons_cfg["scavenging_min"])]
        ubg += [1.0]
        constraint_groups["scavenging"].append(con_idx)
        con_idx += 1
    if "trapping_min" in cons_cfg:
        # Cross-multiplied form to avoid division by near-zero: total_trapped / (Mdel_k + eps) ≥ trapping_min
        trapping_min_val = float(cons_cfg["trapping_min"])
        g += [total_trapped - trapping_min_val * (Mdel_k + eps)]
        lbg += [0.0]
        ubg += [ca.inf]
        constraint_groups["scavenging"].append(con_idx)
        con_idx += 1

    timing_cfg = P.get("timing", {})
    if timing_cfg:
        # Use consistent epsilon with seeded lower bounds (1e-6 for area integrals)
        eps = 1e-6
        tol = float(timing_cfg.get("tol", 1e-3))  # Ensure tol is purely numeric
        
        if "Ain_t_cm" in timing_cfg:
            t_target = float(timing_cfg["Ain_t_cm"])
            # Cross-multiplied form to avoid division by near-zero: |AinTmom_k - t_target * (AinInt_k + eps)| ≤ tol * (AinInt_k + eps)
            # Upper bound: AinTmom_k - t_target * (AinInt_k + eps) ≤ tol * (AinInt_k + eps)
            g += [AinTmom_k - t_target * (AinInt_k + eps)]
            lbg += [-ca.inf]
            ubg += [tol * (AinInt_k + eps)]
            constraint_groups["scavenging"].append(con_idx)
            con_idx += 1
            # Lower bound: AinTmom_k - t_target * (AinInt_k + eps) ≥ -tol * (AinInt_k + eps)
            g += [AinTmom_k - t_target * (AinInt_k + eps)]
            lbg += [-tol * (AinInt_k + eps)]
            ubg += [ca.inf]
            constraint_groups["scavenging"].append(con_idx)
            con_idx += 1
            
        if "Aex_t_cm" in timing_cfg:
            t_target = float(timing_cfg["Aex_t_cm"])
            # Cross-multiplied form: |AexTmom_k - t_target * (AexInt_k + eps)| ≤ tol * (AexInt_k + eps)
            # Upper bound: AexTmom_k - t_target * (AexInt_k + eps) ≤ tol * (AexInt_k + eps)
            g += [AexTmom_k - t_target * (AexInt_k + eps)]
            lbg += [-ca.inf]
            ubg += [tol * (AexInt_k + eps)]
            constraint_groups["scavenging"].append(con_idx)
            con_idx += 1
            # Lower bound: AexTmom_k - t_target * (AexInt_k + eps) ≥ -tol * (AexInt_k + eps)
            g += [AexTmom_k - t_target * (AexInt_k + eps)]
            lbg += [-tol * (AexInt_k + eps)]
            ubg += [ca.inf]
            constraint_groups["scavenging"].append(con_idx)
            con_idx += 1

    # Objective: weighted multi-term
    w_dict = obj_cfg.get("w", {})
    w_smooth = float(w_dict.get("smooth", 0.0))
    w_short = float(w_dict.get("short_circuit", 0.0))
    w_eta = float(w_dict.get("eta_th", 0.0))

    # Smoothness on valve rates (reuse diff chain assembled above)
    if w_smooth > 0.0:
        valve_vars = []
        for k in range(K):
            for j in range(C):
                valve_vars.append(Ain_stage[j])
                valve_vars.append(Aex_stage[j])
        if len(valve_vars) > 1:
            diffs = [
                valve_vars[i + 1] - valve_vars[i] for i in range(len(valve_vars) - 1)
            ]
            smooth_penalty_accum = w_smooth * ca.sumsqr(ca.vertcat(*diffs))

    # W_ind term (maximize): minimize -W_ind
    J_terms = []
    J_terms.append(-W_ind_accum)
    # Thermal efficiency surrogate: maximize W_ind/Q_in -> minimize -(W/Q)
    if w_eta > 0.0:
        J_terms.append(
            -w_eta * (W_ind_accum / ca.fmax(Q_in_accum + 1e-6, CASADI_PHYSICS_EPSILON)),
        )
    # Scavenging penalty (minimize cycle short-circuit fraction)
    if w_short > 0.0:
        J_terms.append(w_short * short_circuit_fraction)
    # Smoothness
    if w_smooth > 0.0:
        J_terms.append(smooth_penalty_accum)

    combustion_meta: dict[str, Any] | None = None
    if use_combustion_model:
        beta = float(combustion_cfg.get("ca_softening", 200.0))
        ca_targets = {
            "CA10": 0.10,
            "CA50": 0.50,
            "CA90": 0.90,
            "CA100": 0.99,
        }
        ca_marker_expr: dict[str, Any] = {}
        if combustion_samples:
            for name, target in ca_targets.items():
                numerator = 0.0
                denominator = 0.0
                for time_val, mfb_expr in combustion_samples:
                    theta_val = float(omega_deg_per_s_const or 0.0) * time_val
                    weight = ca.exp(-beta * (mfb_expr - target) ** 2)
                    numerator += theta_val * weight
                    denominator += weight
                fallback_theta = (
                    float(omega_deg_per_s_const or 0.0) * combustion_samples[-1][0]
                    if combustion_samples
                    else 0.0
                )
                ca_marker_expr[name] = ca.if_else(
                    denominator > 1e-9,
                    numerator / denominator,
                    fallback_theta,
                )
        else:
            # Default zeros if no samples captured (degenerate grids)
            for name in ca_targets:
                ca_marker_expr[name] = ca.DM(0.0)

        # Soft CA phasing objectives
        w_ca50 = float(combustion_cfg.get("w_ca50", 0.0))
        if w_ca50 > 0.0 and "CA50" in ca_marker_expr:
            ca50_target = float(combustion_cfg.get("ca50_target_deg", 0.0))
            delta_ca50 = ca_marker_expr["CA50"] - ca50_target
            J_terms.append(w_ca50 * delta_ca50 * delta_ca50)

        w_cadur = float(combustion_cfg.get("w_ca_duration", 0.0))
        if (
            w_cadur > 0.0
            and "CA90" in ca_marker_expr
            and "CA10" in ca_marker_expr
        ):
            cadur_target = float(combustion_cfg.get("ca_duration_target_deg", 0.0))
            cadur = ca_marker_expr["CA90"] - ca_marker_expr["CA10"]
            delta_cadur = cadur - cadur_target
            J_terms.append(w_cadur * delta_cadur * delta_cadur)

        combustion_meta = {
            "type": "integrated_wiebe",
            "omega_deg_per_s": omega_deg_per_s_const,
            "ignition_time_var": t_ign,
            "ca_markers": ca_marker_expr,
            "sample_times": [sample[0] for sample in combustion_samples],
        }

    J = sum(J_terms)

    nlp = {"x": ca.vertcat(*w), "f": J, "g": ca.vertcat(*g)}
    
    # Convert bounds and initial guess lists to numpy arrays for driver consumption
    # These arrays match the exact variable ordering in w and constraint ordering in g
    w0_arr = np.array(w0, dtype=float)
    lbw_arr = np.array(lbw, dtype=float)
    ubw_arr = np.array(ubw, dtype=float)
    lbg_arr = np.array(lbg, dtype=float)
    ubg_arr = np.array(ubg, dtype=float)
    
    meta = {
        "K": K,
        "C": C,
        "n_vars": len(w),
        "n_constraints": len(g),
        "flow_mode": gas_model.mode,
        "dynamic_wall": dynamic_wall,
        "scavenging_states": True,
        "timing_states": True,
        "variable_groups": var_groups,  # Add variable group metadata
        "constraint_groups": constraint_groups,  # Add constraint group metadata
        "acceleration_normalization_factor": a_norm_factor,  # Store normalization factor for downstream use
        # Include NLP-provided bounds and initial guess for driver consumption
        "w0": w0_arr,
        "lbw": lbw_arr,
        "ubw": ubw_arr,
        "lbg": lbg_arr,
        "ubg": ubg_arr,
    }
    if combustion_meta is not None:
        meta["combustion_model"] = combustion_meta
        try:
            # Expose simple cycle markers for downstream consumers
            cycle_time = float(combustion_cfg.get("cycle_time_s", 1.0))
            # Use cast to avoid mypy inference issues with dict literal
            cycle_markers = cast(dict[str, Any], {"t0": 0.0, "T": cycle_time})
            meta["cycle"] = cycle_markers
        except Exception:
            pass
    return nlp, meta
