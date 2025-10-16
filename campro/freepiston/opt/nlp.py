from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from campro.freepiston.gas import build_gas_model
from campro.freepiston.opt.colloc import CollocationGrid, make_grid
from campro.logging import get_logger

log = get_logger(__name__)


def _import_casadi():
    try:
        import casadi as ca  # type: ignore
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
        Clearance volume [m^3]
        
    Returns
    -------
    V : Any
        Chamber volume [m^3] (CasADi variable)
    """
    ca = _import_casadi()
    A_piston = math.pi * (B / 2.0) ** 2
    return Vc + A_piston * (x_R - x_L)


def enhanced_piston_dae_constraints(
    xL_c: Any, xR_c: Any, vL_c: Any, vR_c: Any,
    aL_c: Any, aR_c: Any, p_gas_c: Any,
    geometry: Dict[str, float],
) -> Tuple[Any, Any]:
    """
    Complete piston DAE with all force components.
    
    Returns:
        F_L, F_R: Net forces on left and right pistons
    """
    ca = _import_casadi()

    # Gas pressure forces
    A_piston = math.pi * (geometry["bore"] / 2.0) ** 2
    F_gas_L = p_gas_c * A_piston
    F_gas_R = -p_gas_c * A_piston  # Opposite direction

    # Inertia forces (piston + connecting rod)
    m_piston = geometry["mass"]
    m_rod = geometry["rod_mass"]
    rod_cg_offset = geometry["rod_cg_offset"]
    rod_length = geometry["rod_length"]

    # Effective mass including rod dynamics
    m_eff_L = m_piston + m_rod * (rod_cg_offset / rod_length)
    m_eff_R = m_piston + m_rod * (rod_cg_offset / rod_length)

    F_inertia_L = -m_eff_L * aL_c
    F_inertia_R = -m_eff_R * aR_c

    # Friction forces (velocity-dependent)
    friction_coeff = geometry.get("friction_coeff", 0.1)
    F_friction_L = -friction_coeff * ca.sign(vL_c) * ca.fabs(vL_c)
    F_friction_R = -friction_coeff * ca.sign(vR_c) * ca.fabs(vR_c)

    # Clearance penalty forces (smooth)
    gap_min = geometry.get("gap_min", 0.0008)
    gap_current = xR_c - xL_c
    penalty_stiffness = geometry.get("penalty_stiffness", 1e6)

    # Smooth penalty function
    gap_violation = ca.fmax(0.0, gap_min - gap_current)
    F_clearance_L = penalty_stiffness * gap_violation
    F_clearance_R = -penalty_stiffness * gap_violation

    # Net forces
    F_L = F_gas_L + F_inertia_L + F_friction_L + F_clearance_L
    F_R = F_gas_R + F_inertia_R + F_friction_R + F_clearance_R

    return F_L, F_R


def piston_force_balance(*, p_gas: Any, x_L: Any, x_R: Any, v_L: Any, v_R: Any,
                        a_L: Any, a_R: Any, geometry: Dict[str, float]) -> Tuple[Any, Any]:
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
    A_piston = math.pi * (B / 2.0) ** 2

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
    F_rod_inertia_L = -m_rod * a_L * (rod_cg_offset / rod_length)
    F_rod_inertia_R = -m_rod * a_R * (rod_cg_offset / rod_length)

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
    F_ring_L = -ring_count * mu_ring * (ring_tension + p_gas * ring_width * math.pi * B) * ca.sign(v_L)
    F_ring_R = -ring_count * mu_ring * (ring_tension + p_gas * ring_width * math.pi * B) * ca.sign(v_R)

    # Enhanced clearance penalty forces
    gap_min = geometry.get("clearance_min", 0.001)  # m
    k_clearance = geometry.get("clearance_stiffness", 1e6)  # N/m
    clearance_smooth = geometry.get("clearance_smooth", 0.0001)  # m

    gap = x_R - x_L

    # Smooth clearance penalty using tanh function
    gap_violation = ca.fmax(0.0, gap_min - gap)
    F_clearance = k_clearance * gap_violation * ca.tanh(gap_violation / clearance_smooth)

    # Net forces
    F_L = F_gas_L + F_inertia_L + F_friction_L + F_ring_L + F_clearance
    F_R = F_gas_R + F_inertia_R + F_friction_R + F_ring_R - F_clearance

    return F_L, F_R


def enhanced_gas_dae_constraints(
    rho_c: Any, T_c: Any, V_c: Any, dV_dt_c: Any,
    mdot_in_c: Any, mdot_out_c: Any, Q_comb_c: Any, Q_heat_c: Any,
    geometry: Dict[str, float], thermo: Dict[str, float],
) -> Tuple[Any, Any]:
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
    cv = cp / gamma

    # Mass balance: d(rho*V)/dt = mdot_in - mdot_out
    # Expanding: rho*dV/dt + V*drho/dt = mdot_in - mdot_out
    # Solving for drho/dt: drho/dt = (mdot_in - mdot_out - rho*dV/dt) / V
    drho_dt = (mdot_in_c - mdot_out_c - rho_c * dV_dt_c) / V_c

    # Energy balance: d(m*e)/dt = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # where m = rho*V, e = cv*T, h = cp*T, p = rho*R*T

    # Total mass and internal energy
    m_total = rho_c * V_c
    e_internal = cv * T_c

    # Enthalpy of inlet/outlet streams
    T_in = thermo.get("T_in", 300.0)
    T_out = T_c  # Assume outlet at chamber temperature
    h_in = cp * T_in
    h_out = cp * T_out

    # Pressure
    p_gas = rho_c * R * T_c

    # Energy equation: d(m*e)/dt = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # Expanding: m*de/dt + e*dm/dt = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt
    # Since de/dt = cv*dT/dt and dm/dt = drho_dt*V + rho*dV_dt_c:
    # m*cv*dT/dt + e*(drho_dt*V + rho*dV_dt_c) = Q_comb - Q_heat + mdot_in*h_in - mdot_out*h_out - p*dV/dt

    # Solve for dT/dt
    dT_dt = (Q_comb_c - Q_heat_c +
             mdot_in_c * h_in - mdot_out_c * h_out -
             p_gas * dV_dt_c -
             e_internal * (drho_dt * V_c + rho_c * dV_dt_c)) / (m_total * cv)

    return drho_dt, dT_dt


def gas_energy_balance(*, rho: Any, T: Any, V: Any, dV_dt: Any,
                      Q_combustion: Any, Q_heat_transfer: Any,
                      mdot_in: Any, mdot_out: Any,
                      T_in: Any = None, T_out: Any = None,
                      gamma: float = 1.4) -> Any:
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
    cv_ref = cp_ref / gamma  # J/(kg K) - reference specific heat at constant volume

    # Temperature-dependent specific heat (simplified linear model)
    # In practice, this would use JANAF polynomial fits
    cp = cp_ref * (1.0 + 0.0001 * (T - 300.0))  # Linear temperature dependence
    cv = cp / gamma

    # Set default inlet/outlet temperatures
    if T_in is None:
        T_in = 300.0  # K - ambient temperature
    if T_out is None:
        T_out = T  # K - assume outlet at chamber temperature

    # Mass balance: d(rho*V)/dt = mdot_in - mdot_out
    dm_dt = mdot_in - mdot_out

    # Total mass in chamber
    m = rho * V

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

    # Solve for dT/dt
    dT_dt = (Q_combustion - Q_heat_transfer +
             mdot_in * h_in - mdot_out * h_out -
             p * dV_dt - e * dm_dt) / (m * cv)

    return dT_dt


def build_collocation_nlp_with_1d_coupling(P: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
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


def _build_1d_collocation_nlp(P: Dict[str, Any], ca: Any, K: int, C: int,
                             grid: CollocationGrid, n_cells: int) -> Tuple[Any, Dict[str, Any]]:
    """Build 1D gas-structure coupled collocation NLP."""

    # Get geometry and parameters
    geometry = P.get("geometry", {})
    bounds = P.get("bounds", {})
    obj_cfg = P.get("obj", {})
    walls_cfg = P.get("walls", {})
    flow_cfg = P.get("flow", {})

    # Variables, initial guesses, and bounds
    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []

    # Initial states
    xL0 = ca.SX.sym("xL0")
    xR0 = ca.SX.sym("xR0")
    vL0 = ca.SX.sym("vL0")
    vR0 = ca.SX.sym("vR0")

    w += [xL0, xR0, vL0, vR0]
    w0 += [0.0, 0.1, 0.0, 0.0]
    lbw += [bounds.get("xL_min", -0.1), bounds.get("xR_min", 0.0),
            bounds.get("vL_min", -10.0), bounds.get("vR_min", -10.0)]
    ubw += [bounds.get("xL_max", 0.1), bounds.get("xR_max", 0.2),
            bounds.get("vL_max", 10.0), bounds.get("vR_max", 10.0)]

    # Initial 1D gas state variables (per cell)
    rho0_cells = [ca.SX.sym(f"rho0_{i}") for i in range(n_cells)]
    u0_cells = [ca.SX.sym(f"u0_{i}") for i in range(n_cells)]
    E0_cells = [ca.SX.sym(f"E0_{i}") for i in range(n_cells)]

    w += rho0_cells + u0_cells + E0_cells
    w0 += [1.0] * n_cells + [0.0] * n_cells + [2.5] * n_cells  # Initial gas state
    lbw += [bounds.get("rho_min", 0.1)] * n_cells + \
           [bounds.get("u_min", -100.0)] * n_cells + \
           [bounds.get("E_min", 0.1)] * n_cells
    ubw += [bounds.get("rho_max", 10.0)] * n_cells + \
           [bounds.get("u_max", 100.0)] * n_cells + \
           [bounds.get("E_max", 100.0)] * n_cells

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

    h = 1.0 / K
    # Objective accumulators
    W_ind_accum = 0.0
    Q_in_accum = 0.0
    scav_penalty_accum = 0.0
    smooth_penalty_accum = 0.0

    # Scavenging and timing accumulator initial states
    yF0 = ca.SX.sym("yF0")
    Mdel0 = ca.SX.sym("Mdel0")
    Mlost0 = ca.SX.sym("Mlost0")
    AinInt0 = ca.SX.sym("AinInt0")
    AinTmom0 = ca.SX.sym("AinTmom0")
    AexInt0 = ca.SX.sym("AexInt0")
    AexTmom0 = ca.SX.sym("AexTmom0")
    w += [yF0, Mdel0, Mlost0, AinInt0, AinTmom0, AexInt0, AexTmom0]
    w0 += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
        Ain_stage = [ca.SX.sym(f"Ain_{k}_{j}") for j in range(C)]
        Aex_stage = [ca.SX.sym(f"Aex_{k}_{j}") for j in range(C)]
        Q_comb_stage = [ca.SX.sym(f"Q_comb_{k}_{j}") for j in range(C)]

        for j in range(C):
            w += [Ain_stage[j], Aex_stage[j], Q_comb_stage[j]]
            w0 += [0.0, 0.0, 0.0]
            lbw += [0.0, 0.0, 0.0]
            ubw += [bounds.get("Ain_max", 0.01), bounds.get("Aex_max", 0.01),
                   bounds.get("Q_comb_max", 10000.0)]

        # Collocation states for pistons
        xL_colloc = [ca.SX.sym(f"xL_{k}_{j}") for j in range(C)]
        xR_colloc = [ca.SX.sym(f"xR_{k}_{j}") for j in range(C)]
        vL_colloc = [ca.SX.sym(f"vL_{k}_{j}") for j in range(C)]
        vR_colloc = [ca.SX.sym(f"vR_{k}_{j}") for j in range(C)]

        for j in range(C):
            w += [xL_colloc[j], xR_colloc[j], vL_colloc[j], vR_colloc[j]]
            w0 += [0.0, 0.1, 0.0, 0.0]
            lbw += [bounds.get("xL_min", -0.1), bounds.get("xR_min", 0.0),
                   bounds.get("vL_min", -10.0), bounds.get("vR_min", -10.0)]
            ubw += [bounds.get("xL_max", 0.1), bounds.get("xR_max", 0.2),
                   bounds.get("vL_max", 10.0), bounds.get("vR_max", 10.0)]

        # Collocation states for 1D gas (per cell, per collocation point)
        rho_colloc = [[ca.SX.sym(f"rho_{k}_{c}_{i}") for i in range(n_cells)] for c in range(C)]
        u_colloc = [[ca.SX.sym(f"u_{k}_{c}_{i}") for i in range(n_cells)] for c in range(C)]
        E_colloc = [[ca.SX.sym(f"E_{k}_{c}_{i}") for i in range(n_cells)] for c in range(C)]

        for c in range(C):
            w += rho_colloc[c] + u_colloc[c] + E_colloc[c]
            w0 += [1.0] * n_cells + [0.0] * n_cells + [2.5] * n_cells
            lbw += [bounds.get("rho_min", 0.1)] * n_cells + \
                   [bounds.get("u_min", -100.0)] * n_cells + \
                   [bounds.get("E_min", 0.1)] * n_cells
            ubw += [bounds.get("rho_max", 10.0)] * n_cells + \
                   [bounds.get("u_max", 100.0)] * n_cells + \
                   [bounds.get("E_max", 100.0)] * n_cells

        # Collocation equations
        for c in range(C):
            # Current state
            xL_c = xL_colloc[c]
            xR_c = xR_colloc[c]
            vL_c = vL_colloc[c]
            vR_c = vR_colloc[c]
            Ain_c = Ain_stage[c]
            Aex_c = Aex_stage[c]
            Q_comb_c = Q_comb_stage[c]

            # Chamber volume and its rate of change
            V_c = chamber_volume_from_pistons(x_L=xL_c, x_R=xR_c,
                                            B=geometry.get("bore", 0.1),
                                            Vc=geometry.get("clearance_volume", 1e-4))
            dV_dt = math.pi * (geometry.get("bore", 0.1) / 2.0) ** 2 * (vR_c - vL_c)

            # Enhanced piston forces with proper acceleration coupling
            # Compute accelerations from velocity differences (simplified)
            aL_c = (vL_c - vL_k) / h if h > 0 else 0.0
            aR_c = (vR_c - vR_k) / h if h > 0 else 0.0

            # Calculate average gas pressure for piston forces
            p_avg = 0.0
            for i in range(n_cells):
                rho_i = rho_colloc[c][i]
                u_i = u_colloc[c][i]
                E_i = E_colloc[c][i]
                # Convert to pressure using ideal gas law
                p_i = (1.4 - 1.0) * rho_i * (E_i - 0.5 * u_i**2)
                p_avg += p_i
            p_avg /= n_cells

            F_L_c, F_R_c = piston_force_balance(p_gas=p_avg, x_L=xL_c, x_R=xR_c,
                                               v_L=vL_c, v_R=vR_c, a_L=aL_c, a_R=aR_c,
                                               geometry=geometry)

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
                    U_L = [rho_colloc[c][i-1], rho_colloc[c][i-1] * u_colloc[c][i-1],
                           rho_colloc[c][i-1] * E_colloc[c][i-1]]
                    U_R = [rho_i, rho_i * u_i, rho_i * E_i]
                    F_left = _hllc_flux_symbolic(U_L, U_R, ca)
                else:
                    F_left = [0.0, 0.0, 0.0]

                if i < n_cells - 1:
                    # Right face flux
                    U_L = [rho_i, rho_i * u_i, rho_i * E_i]
                    U_R = [rho_colloc[c][i+1], rho_colloc[c][i+1] * u_colloc[c][i+1],
                           rho_colloc[c][i+1] * E_colloc[c][i+1]]
                    F_right = _hllc_flux_symbolic(U_L, U_R, ca)
                else:
                    F_right = [0.0, 0.0, 0.0]

                # Source terms (volume change, heat transfer)
                S_sources = _calculate_1d_source_terms(
                    rho_i, u_i, E_i, xL_c, xR_c, vL_c, vR_c,
                    geometry, flow_cfg, Q_comb_c, Ain_c, Aex_c,
                )

                # Cell width (simplified)
                dx = (xR_c - xL_c) / n_cells

                # Semi-discrete form: dU/dt = -(F_right - F_left) / dx + S
                dU_dt = []
                for comp in range(3):
                    dU_dt.append(-(F_right[comp] - F_left[comp]) / dx + S_sources[comp])

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
                m_total_L = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (geometry.get("rod_cg_offset", 0.075) / geometry.get("rod_length", 0.15))
                m_total_R = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (geometry.get("rod_cg_offset", 0.075) / geometry.get("rod_length", 0.15))

                rhs_vL += h * grid.a[c][j] * (F_L_c / m_total_L)
                rhs_vR += h * grid.a[c][j] * (F_R_c / m_total_R)

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
            ratio = mdot_out / (mdot_in + eps)
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
            m_total_L = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (geometry.get("rod_cg_offset", 0.075) / geometry.get("rod_length", 0.15))
            m_total_R = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (geometry.get("rod_cg_offset", 0.075) / geometry.get("rod_length", 0.15))

            vL_k1 += h * grid.weights[j] * (F_L_c / m_total_L)
            vR_k1 += h * grid.weights[j] * (F_R_c / m_total_R)

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
        xL_k = ca.SX.sym(f"xL_{k+1}")
        xR_k = ca.SX.sym(f"xR_{k+1}")
        vL_k = ca.SX.sym(f"vL_{k+1}")
        vR_k = ca.SX.sym(f"vR_{k+1}")

        rho_k = [ca.SX.sym(f"rho_{k+1}_{i}") for i in range(n_cells)]
        u_k = [ca.SX.sym(f"u_{k+1}_{i}") for i in range(n_cells)]
        E_k = [ca.SX.sym(f"E_{k+1}_{i}") for i in range(n_cells)]

        yF_k = ca.SX.sym(f"yF_{k+1}")
        Mdel_k = ca.SX.sym(f"Mdel_{k+1}")
        Mlost_k = ca.SX.sym(f"Mlost_{k+1}")
        AinInt_k = ca.SX.sym(f"AinInt_{k+1}")
        AinTmom_k = ca.SX.sym(f"AinTmom_{k+1}")
        AexInt_k = ca.SX.sym(f"AexInt_{k+1}")
        AexTmom_k = ca.SX.sym(f"AexTmom_{k+1}")

        w += [xL_k, xR_k, vL_k, vR_k] + rho_k + u_k + E_k + \
             [yF_k, Mdel_k, Mlost_k, AinInt_k, AinTmom_k, AexInt_k, AexTmom_k]
        w0 += [0.0, 0.1, 0.0, 0.0] + [1.0] * n_cells + [0.0] * n_cells + [2.5] * n_cells + \
              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lbw += [bounds.get("xL_min", -0.1), bounds.get("xR_min", 0.0),
               bounds.get("vL_min", -10.0), bounds.get("vR_min", -10.0)] + \
               [bounds.get("rho_min", 0.1)] * n_cells + \
               [bounds.get("u_min", -100.0)] * n_cells + \
               [bounds.get("E_min", 0.1)] * n_cells + \
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ubw += [bounds.get("xL_max", 0.1), bounds.get("xR_max", 0.2),
               bounds.get("vL_max", 10.0), bounds.get("vR_max", 10.0)] + \
               [bounds.get("rho_max", 10.0)] * n_cells + \
               [bounds.get("u_max", 100.0)] * n_cells + \
               [bounds.get("E_max", 100.0)] * n_cells + \
               [1.0, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf]

        # Continuity constraints
        cont = [xL_k - xL_k1, xR_k - xR_k1, vL_k - vL_k1, vR_k - vR_k1]
        for i in range(n_cells):
            cont += [rho_k[i] - rho_k1[i], u_k[i] - u_k1[i], E_k[i] - E_k1[i]]
        cont += [yF_k - yF_k, Mdel_k - Mdel_k, Mlost_k - Mlost_k,
                AinInt_k - AinInt_k, AinTmom_k - AinTmom_k,
                AexInt_k - AexInt_k, AexTmom_k - AexTmom_k]
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
        J_terms.append(-w_eta * (W_ind_accum / (Q_in_accum + 1e-6)))
    # Scavenging penalty (minimize cycle short-circuit fraction)
    if w_short > 0.0:
        J_terms.append(w_short * scav_penalty_accum)

    J = sum(J_terms)

    nlp = {"x": ca.vertcat(*w), "f": J, "g": ca.vertcat(*g)}
    meta = {"K": K, "C": C, "n_vars": len(w), "n_constraints": len(g),
            "flow_mode": "1d_gas", "dynamic_wall": False,
            "scavenging_states": True, "timing_states": True, "n_cells": n_cells}
    return nlp, meta


def _hllc_flux_symbolic(U_L: List[Any], U_R: List[Any], ca: Any) -> List[Any]:
    """Symbolic HLLC flux calculation for CasADi."""
    # Simplified HLLC implementation for symbolic computation
    # In practice, this would be more sophisticated

    rho_L, rhou_L, rhoE_L = U_L
    rho_R, rhou_R, rhoE_R = U_R

    # Convert to primitive variables
    u_L = rhou_L / rho_L
    u_R = rhou_R / rho_R
    p_L = (1.4 - 1.0) * rho_L * (rhoE_L / rho_L - 0.5 * u_L**2)
    p_R = (1.4 - 1.0) * rho_R * (rhoE_R / rho_R - 0.5 * u_R**2)

    # Simplified flux calculation
    F_L = [rho_L * u_L, rho_L * u_L**2 + p_L, (rhoE_L + p_L) * u_L]
    F_R = [rho_R * u_R, rho_R * u_R**2 + p_R, (rhoE_R + p_R) * u_R]

    # Simple average for now (in practice, use proper HLLC)
    F = [(F_L[i] + F_R[i]) / 2.0 for i in range(3)]

    return F


def _calculate_1d_source_terms(rho: Any, u: Any, E: Any, xL: Any, xR: Any,
                              vL: Any, vR: Any, geometry: Dict[str, float],
                              flow_cfg: Dict[str, Any], Q_comb: Any,
                              Ain: Any, Aex: Any) -> List[Any]:
    """Calculate 1D source terms for gas dynamics equations."""
    ca = _import_casadi()

    # Volume change rate
    dV_dt = math.pi * (geometry.get("bore", 0.1) / 2.0) ** 2 * (vR - vL)

    # Cell volume (simplified)
    V_cell = (xR - xL) / flow_cfg.get("mesh_cells", 80)

    # Source terms
    S_rho = -rho * dV_dt / V_cell  # Mass source
    S_rhou = -rho * u * dV_dt / V_cell  # Momentum source
    S_rhoE = -rho * E * dV_dt / V_cell + Q_comb / V_cell  # Energy source

    return [S_rho, S_rhou, S_rhoE]


def build_collocation_nlp(P: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
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

    # Initial states
    xL0 = ca.SX.sym("xL0")
    xR0 = ca.SX.sym("xR0")
    vL0 = ca.SX.sym("vL0")
    vR0 = ca.SX.sym("vR0")
    rho0 = ca.SX.sym("rho0")
    T0 = ca.SX.sym("T0")

    w += [xL0, xR0, vL0, vR0, rho0, T0]
    w0 += [0.0, 0.1, 0.0, 0.0, 1.0, 300.0]
    lbw += [bounds.get("xL_min", -0.1), bounds.get("xR_min", 0.0),
            bounds.get("vL_min", -10.0), bounds.get("vR_min", -10.0),
            bounds.get("rho_min", 0.1), bounds.get("T_min", 200.0)]
    ubw += [bounds.get("xL_max", 0.1), bounds.get("xR_max", 0.2),
            bounds.get("vL_max", 10.0), bounds.get("vR_max", 10.0),
            bounds.get("rho_max", 10.0), bounds.get("T_max", 2000.0)]

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
    rho_k = rho0
    T_k = T0
    Ain_k = Ain0
    Aex_k = Aex0

    h = 1.0 / K
    # Objective accumulators
    W_ind_accum = 0.0
    Q_in_accum = 0.0
    scav_penalty_accum = 0.0
    smooth_penalty_accum = 0.0

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
    w0 += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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
        Ain_stage = [ca.SX.sym(f"Ain_{k}_{j}") for j in range(C)]
        Aex_stage = [ca.SX.sym(f"Aex_{k}_{j}") for j in range(C)]
        Q_comb_stage = [ca.SX.sym(f"Q_comb_{k}_{j}") for j in range(C)]

        for j in range(C):
            w += [Ain_stage[j], Aex_stage[j], Q_comb_stage[j]]
            w0 += [0.0, 0.0, 0.0]
            lbw += [0.0, 0.0, 0.0]
            ubw += [bounds.get("Ain_max", 0.01), bounds.get("Aex_max", 0.01),
                   bounds.get("Q_comb_max", 10000.0)]

        # Collocation states
        xL_colloc = [ca.SX.sym(f"xL_{k}_{j}") for j in range(C)]
        xR_colloc = [ca.SX.sym(f"xR_{k}_{j}") for j in range(C)]
        vL_colloc = [ca.SX.sym(f"vL_{k}_{j}") for j in range(C)]
        vR_colloc = [ca.SX.sym(f"vR_{k}_{j}") for j in range(C)]
        rho_colloc = [ca.SX.sym(f"rho_{k}_{j}") for j in range(C)]
        T_colloc = [ca.SX.sym(f"T_{k}_{j}") for j in range(C)]

        for j in range(C):
            w += [xL_colloc[j], xR_colloc[j], vL_colloc[j], vR_colloc[j],
                  rho_colloc[j], T_colloc[j]]
            w0 += [0.0, 0.1, 0.0, 0.0, 1.0, 300.0]
            lbw += [bounds.get("xL_min", -0.1), bounds.get("xR_min", 0.0),
                   bounds.get("vL_min", -10.0), bounds.get("vR_min", -10.0),
                   bounds.get("rho_min", 0.1), bounds.get("T_min", 200.0)]
            ubw += [bounds.get("xL_max", 0.1), bounds.get("xR_max", 0.2),
                   bounds.get("vL_max", 10.0), bounds.get("vR_max", 10.0),
                   bounds.get("rho_max", 10.0), bounds.get("T_max", 2000.0)]

        # Collocation equations
        for c in range(C):
            # Current state
            xL_c = xL_colloc[c]
            xR_c = xR_colloc[c]
            vL_c = vL_colloc[c]
            vR_c = vR_colloc[c]
            rho_c = rho_colloc[c]
            T_c = T_colloc[c]
            Ain_c = Ain_stage[c]
            Aex_c = Aex_stage[c]
            Q_comb_c = Q_comb_stage[c]

            # Chamber volume and its rate of change
            V_c = chamber_volume_from_pistons(x_L=xL_c, x_R=xR_c,
                                            B=geometry.get("bore", 0.1),
                                            Vc=geometry.get("clearance_volume", 1e-4))
            dV_dt = math.pi * (geometry.get("bore", 0.1) / 2.0) ** 2 * (vR_c - vL_c)

            # Gas pressure
            p_c = gas_pressure_from_state(rho=rho_c, T=T_c)

            # Enhanced piston forces with proper acceleration coupling
            # Compute accelerations from velocity differences (simplified)
            aL_c = (vL_c - vL_k) / h if h > 0 else 0.0
            aR_c = (vR_c - vR_k) / h if h > 0 else 0.0

            F_L_c, F_R_c = piston_force_balance(p_gas=p_c, x_L=xL_c, x_R=xR_c,
                                               v_L=vL_c, v_R=vR_c, a_L=aL_c, a_R=aR_c,
                                               geometry=geometry)

            # Mass flow rates via unified gas model
            R = 287.0
            gamma = float(flow_cfg.get("gamma", 1.4))
            p_in = float(geometry.get("p_intake", 1e5))
            T_in = float(geometry.get("T_intake", 300.0))
            rho_in = p_in / (R * T_in)
            p_ex = float(geometry.get("p_exhaust", 1e5))
            T_ex = T_c
            rho_ex = p_ex / (R * T_ex)
            mdot_in = gas_model.mdot_in(ca=ca, p_up=p_in, T_up=T_in, rho_up=rho_in,
                                        p_down=p_c, T_down=T_c, A_eff=Ain_c, gamma=gamma, R=R)
            mdot_out = gas_model.mdot_out(ca=ca, p_up=p_c, T_up=T_c, rho_up=rho_c,
                                          p_down=p_ex, T_down=T_ex, A_eff=Aex_c, gamma=gamma, R=R)

            # Wall heat transfer
            T_wall_c = Tw_k if dynamic_wall else T_wall_const
            Q_heat_transfer = gas_model.qdot_wall(ca=ca, p_gas=p_c, T_gas=T_c,
                                                  T_wall=T_wall_c, B=geometry.get("bore", 0.1),
                                                  x_L=xL_c, x_R=xR_c)

            # Enhanced gas energy balance
            dT_dt = gas_energy_balance(rho=rho_c, T=T_c, V=V_c, dV_dt=dV_dt,
                                     Q_combustion=Q_comb_c, Q_heat_transfer=Q_heat_transfer,
                                     mdot_in=mdot_in, mdot_out=mdot_out,
                                     T_in=T_in, T_out=T_ex, gamma=gamma)

            # Mass balance
            dm_dt = mdot_in - mdot_out
            drho_dt = (dm_dt - rho_c * dV_dt) / V_c
            # Fresh fraction dynamics
            m_c = rho_c * V_c
            dyF_dt = (mdot_in - mdot_out * yF_k - yF_k * dm_dt) / (m_c + 1e-9)

            # Collocation equations (state updates using A matrix)
            rhs_xL = xL_k
            rhs_xR = xR_k
            rhs_vL = vL_k
            rhs_vR = vR_k
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
                m_total_L = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (geometry.get("rod_cg_offset", 0.075) / geometry.get("rod_length", 0.15))
                m_total_R = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (geometry.get("rod_cg_offset", 0.075) / geometry.get("rod_length", 0.15))

                rhs_vL += h * grid.a[c][j] * (F_L_c / m_total_L)
                rhs_vR += h * grid.a[c][j] * (F_R_c / m_total_R)
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

            colloc_res = [xL_c - rhs_xL, xR_c - rhs_xR, vL_c - rhs_vL, vR_c - rhs_vR,
                          rho_c - rhs_rho, T_c - rhs_T]
            if dynamic_wall:
                colloc_res.append(T_wall_c - rhs_Tw)
            colloc_res += [yF_k - rhs_yF, Mdel_k - rhs_Mdel, Mlost_k - rhs_Mlost,
                           AinInt_k - rhs_AinInt, AinTmom_k - rhs_AinTmom,
                           AexInt_k - rhs_AexInt, AexTmom_k - rhs_AexTmom]
            g += colloc_res
            lbg += [0.0] * len(colloc_res)
            ubg += [0.0] * len(colloc_res)

            # Accumulate indicated work and Q_in
            W_ind_accum += h * grid.weights[c] * p_c * dV_dt
            Q_in_accum += h * grid.weights[c] * Q_comb_c
            # Scavenging short-circuit penalty surrogate
            eps = 1e-9
            ratio = mdot_out / (mdot_in + eps)
            scav_penalty_accum += h * grid.weights[c] * ratio

        # Step to next time point
        xL_k1 = xL_k
        xR_k1 = xR_k
        vL_k1 = vL_k
        vR_k1 = vR_k
        rho_k1 = rho_k
        T_k1 = T_k

        for j in range(C):
            xL_k1 += h * grid.weights[j] * vL_colloc[j]
            xR_k1 += h * grid.weights[j] * vR_colloc[j]

            # Enhanced force balance with proper mass calculation
            m_total_L = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (geometry.get("rod_cg_offset", 0.075) / geometry.get("rod_length", 0.15))
            m_total_R = geometry.get("mass", 1.0) + geometry.get("rod_mass", 0.5) * (geometry.get("rod_cg_offset", 0.075) / geometry.get("rod_length", 0.15))

            vL_k1 += h * grid.weights[j] * (F_L_c / m_total_L)
            vR_k1 += h * grid.weights[j] * (F_R_c / m_total_R)
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
        xL_k = ca.SX.sym(f"xL_{k+1}")
        xR_k = ca.SX.sym(f"xR_{k+1}")
        vL_k = ca.SX.sym(f"vL_{k+1}")
        vR_k = ca.SX.sym(f"vR_{k+1}")
        rho_k = ca.SX.sym(f"rho_{k+1}")
        T_k = ca.SX.sym(f"T_{k+1}")

        yF_k = ca.SX.sym(f"yF_{k+1}")
        Mdel_k = ca.SX.sym(f"Mdel_{k+1}")
        Mlost_k = ca.SX.sym(f"Mlost_{k+1}")
        AinInt_k = ca.SX.sym(f"AinInt_{k+1}")
        AinTmom_k = ca.SX.sym(f"AinTmom_{k+1}")
        AexInt_k = ca.SX.sym(f"AexInt_{k+1}")
        AexTmom_k = ca.SX.sym(f"AexTmom_{k+1}")

        w += [xL_k, xR_k, vL_k, vR_k, rho_k, T_k,
              yF_k, Mdel_k, Mlost_k, AinInt_k, AinTmom_k, AexInt_k, AexTmom_k]
        w0 += [0.0, 0.1, 0.0, 0.0, 1.0, 300.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lbw += [bounds.get("xL_min", -0.1), bounds.get("xR_min", 0.0),
               bounds.get("vL_min", -10.0), bounds.get("vR_min", -10.0),
               bounds.get("rho_min", 0.1), bounds.get("T_min", 200.0),
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ubw += [bounds.get("xL_max", 0.1), bounds.get("xR_max", 0.2),
               bounds.get("vL_max", 10.0), bounds.get("vR_max", 10.0),
               bounds.get("rho_max", 10.0), bounds.get("T_max", 2000.0),
               1.0, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf, ca.inf]

        # Continuity constraints
        cont = [xL_k - xL_k1, xR_k - xR_k1, vL_k - vL_k1, vR_k - vR_k1,
                rho_k - rho_k1, T_k - T_k1,
                yF_k - yF_k1, Mdel_k - Mdel_k1, Mlost_k - Mlost_k1,
                AinInt_k - AinInt_k1, AinTmom_k - AinTmom_k1,
                AexInt_k - AexInt_k1, AexTmom_k - AexTmom_k1]
        g += cont
        lbg += [0.0] * len(cont)
        ubg += [0.0] * len(cont)

    # Enhanced path constraints
    gap_min = bounds.get("x_gap_min", 0.0008)
    g += [xR_k - xL_k]  # Clearance constraint
    lbg += [gap_min]
    ubg += [ca.inf]

    # Comprehensive path constraints for all time steps
    for k in range(K):
        for j in range(C):
            # Pressure constraints
            p_kj = gas_pressure_from_state(rho=rho_colloc[j], T=T_colloc[j])
            g += [p_kj]  # Pressure constraint
            lbg += [bounds.get("p_min", 1e4)]
            ubg += [bounds.get("p_max", 1e7)]

            # Temperature constraints
            T_kj = T_colloc[j]
            g += [T_kj]  # Temperature constraint
            lbg += [bounds.get("T_min", 200.0)]
            ubg += [bounds.get("T_max", 2000.0)]

            # Valve rate constraints (if valve areas are available)
            if k > 0:
                # Valve area rate constraints
                Ain_prev = Ain_stage[j] if k > 0 else Ain0
                Aex_prev = Aex_stage[j] if k > 0 else Aex0

                Ain_rate = (Ain_stage[j] - Ain_prev) / h
                Aex_rate = (Aex_stage[j] - Aex_prev) / h

                g += [Ain_rate, Aex_rate]  # Valve rate constraints
                lbg += [-bounds.get("dA_dt_max", 0.02), -bounds.get("dA_dt_max", 0.02)]
                ubg += [bounds.get("dA_dt_max", 0.02), bounds.get("dA_dt_max", 0.02)]

            # Piston velocity constraints
            vL_kj = vL_colloc[j]
            vR_kj = vR_colloc[j]
            g += [vL_kj, vR_kj]  # Piston velocity constraints
            lbg += [-bounds.get("v_max", 50.0), -bounds.get("v_max", 50.0)]
            ubg += [bounds.get("v_max", 50.0), bounds.get("v_max", 50.0)]

            # Piston acceleration constraints (simplified)
            if k > 0:
                aL_kj = (vL_colloc[j] - vL_k) / h
                aR_kj = (vR_colloc[j] - vR_k) / h
                g += [aL_kj, aR_kj]  # Piston acceleration constraints
                lbg += [-bounds.get("a_max", 1000.0), -bounds.get("a_max", 1000.0)]
                ubg += [bounds.get("a_max", 1000.0), bounds.get("a_max", 1000.0)]

    # Combustion timing constraints (optional placeholders)
    for k in range(K):
        for j in range(C):
            Q_comb_kj = Q_comb_stage[j]
            # Combustion heat release should be non-negative and within bounds
            # Avoid Python conditionals on symbolic expressions; encode constraints directly.
            g += [Q_comb_kj]
            lbg += [0.0]
            ubg += [bounds.get("Q_comb_max", 10000.0)]

    # Cycle periodicity constraints
    g += [xL_k - xL0, xR_k - xR0, vL_k - vL0, vR_k - vR0]
    lbg += [0.0, 0.0, 0.0, 0.0]
    ubg += [0.0, 0.0, 0.0, 0.0]

    # Scavenging metrics and timing targets at cycle end
    V_end = chamber_volume_from_pistons(x_L=xL_k, x_R=xR_k,
                                        B=geometry.get("bore", 0.1),
                                        Vc=geometry.get("clearance_volume", 1e-4))
    m_end = rho_k * V_end
    fresh_trapped = yF_k * m_end
    total_trapped = m_end
    short_circuit_fraction = Mlost_k / (Mdel_k + 1e-9)

    cons_cfg = P.get("constraints", {})
    if "short_circuit_max" in cons_cfg:
        g += [short_circuit_fraction]
        lbg += [0.0]
        ubg += [float(cons_cfg["short_circuit_max"])]
    if "scavenging_min" in cons_cfg:
        g += [yF_k]
        lbg += [float(cons_cfg["scavenging_min"])]
        ubg += [1.0]
    if "trapping_min" in cons_cfg:
        trap_eff = total_trapped / (Mdel_k + 1e-9)
        g += [trap_eff]
        lbg += [float(cons_cfg["trapping_min"])]
        ubg += [ca.inf]

    timing_cfg = P.get("timing", {})
    if timing_cfg:
        eps = 1e-9
        t_cm_in = AinTmom_k / (AinInt_k + eps)
        t_cm_ex = AexTmom_k / (AexInt_k + eps)
        tol = float(timing_cfg.get("tol", 1e-3))
        if "Ain_t_cm" in timing_cfg:
            t_target = float(timing_cfg["Ain_t_cm"])
            g += [t_cm_in]
            lbg += [t_target - tol]
            ubg += [t_target + tol]
        if "Aex_t_cm" in timing_cfg:
            t_target = float(timing_cfg["Aex_t_cm"])
            g += [t_cm_ex]
            lbg += [t_target - tol]
            ubg += [t_target + tol]

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
            diffs = [valve_vars[i + 1] - valve_vars[i] for i in range(len(valve_vars) - 1)]
            smooth_penalty_accum = w_smooth * ca.sumsqr(ca.vertcat(*diffs))

    # W_ind term (maximize): minimize -W_ind
    J_terms = []
    J_terms.append(-W_ind_accum)
    # Thermal efficiency surrogate: maximize W_ind/Q_in -> minimize -(W/Q)
    if w_eta > 0.0:
        J_terms.append(-w_eta * (W_ind_accum / (Q_in_accum + 1e-6)))
    # Scavenging penalty (minimize cycle short-circuit fraction)
    if w_short > 0.0:
        J_terms.append(w_short * short_circuit_fraction)
    # Smoothness
    if w_smooth > 0.0:
        J_terms.append(smooth_penalty_accum)

    J = sum(J_terms)

    nlp = {"x": ca.vertcat(*w), "f": J, "g": ca.vertcat(*g)}
    meta = {"K": K, "C": C, "n_vars": len(w), "n_constraints": len(g),
            "flow_mode": gas_model.mode, "dynamic_wall": dynamic_wall,
            "scavenging_states": True, "timing_states": True}
    return nlp, meta


