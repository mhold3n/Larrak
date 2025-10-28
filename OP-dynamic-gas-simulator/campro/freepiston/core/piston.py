from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class PistonGeometry:
    """Piston geometry and mass properties."""

    # Basic dimensions
    bore: float  # Bore diameter [m]
    stroke: float  # Stroke length [m]
    mass: float  # Piston mass [kg]

    # Connecting rod properties
    rod_length: float  # Connecting rod length [m]
    rod_mass: float  # Connecting rod mass [kg]
    rod_cg_offset: float  # CG offset from big end [m]

    # Piston ring properties
    ring_count: int  # Number of piston rings
    ring_width: float  # Ring width [m]
    ring_gap: float  # Ring gap [m]
    ring_tension: float  # Ring tension force [N]

    # Clearance and tolerances
    clearance_min: float  # Minimum clearance [m]
    clearance_nominal: float  # Nominal clearance [m]


@dataclass
class PistonState:
    """Current state of piston system."""

    # Position and velocity
    x: float  # Piston position [m] (0 = TDC)
    v: float  # Piston velocity [m/s]
    a: float  # Piston acceleration [m/s^2]

    # Forces
    F_gas: float  # Gas pressure force [N]
    F_inertia: float  # Inertia force [N]
    F_friction: float  # Friction force [N]
    F_clearance: float  # Clearance penalty force [N]

    # Thermodynamic state
    p_gas: float  # Gas pressure [Pa]
    T_gas: float  # Gas temperature [K]
    V_chamber: float  # Chamber volume [m^3]


def piston_force_balance(
    *,
    geometry: PistonGeometry,
    state: PistonState,
    p_gas: float,
    T_gas: float,
    V_chamber: float,
    omega: float = 0.0,
) -> Tuple[float, float, float]:
    """Full piston force balance with gas pressure coupling.

    Computes the net force on the piston considering:
    - Gas pressure force
    - Inertia forces (piston + connecting rod)
    - Friction forces
    - Clearance penalty forces

    Parameters
    ----------
    geometry : PistonGeometry
        Piston geometry and mass properties
    state : PistonState
        Current piston state
    p_gas : float
        Gas pressure [Pa]
    T_gas : float
        Gas temperature [K]
    V_chamber : float
        Chamber volume [m^3]
    omega : float
        Engine angular velocity [rad/s]

    Returns
    -------
    F_net : float
        Net force on piston [N]
    F_gas : float
        Gas pressure force [N]
    F_inertia : float
        Inertia force [N]
    """
    # Gas pressure force
    A_piston = math.pi * (geometry.bore / 2.0) ** 2
    F_gas = p_gas * A_piston

    # Inertia forces
    # Piston inertia (simplified - assumes constant acceleration)
    F_piston_inertia = -geometry.mass * state.a

    # Connecting rod inertia (simplified)
    # Assumes connecting rod rotates about big end
    F_rod_inertia = (
        -geometry.rod_mass * state.a * (geometry.rod_cg_offset / geometry.rod_length)
    )

    F_inertia = F_piston_inertia + F_rod_inertia

    # Friction forces (computed separately)
    F_friction = 0.0  # Will be computed by friction model

    # Clearance penalty force
    gap = geometry.clearance_nominal - abs(state.x)
    F_clearance = clearance_penalty(gap=gap, gap_min=geometry.clearance_min, k=1e6)

    # Net force
    F_net = F_gas + F_inertia + F_friction + F_clearance

    return F_net, F_gas, F_inertia


def connecting_rod_kinematics(
    *, x: float, v: float, a: float, rod_length: float, stroke: float,
) -> Tuple[float, float, float]:
    """Connecting rod kinematics for piston motion.

    Computes connecting rod angle, angular velocity, and angular acceleration.

    Parameters
    ----------
    x : float
        Piston position [m]
    v : float
        Piston velocity [m/s]
    a : float
        Piston acceleration [m/s^2]
    rod_length : float
        Connecting rod length [m]
    stroke : float
        Stroke length [m]

    Returns
    -------
    theta : float
        Connecting rod angle [rad]
    omega : float
        Connecting rod angular velocity [rad/s]
    alpha : float
        Connecting rod angular acceleration [rad/s^2]
    """
    # Crank radius (half stroke)
    r = stroke / 2.0

    # Connecting rod angle
    # sin(theta) = (r - x) / rod_length
    sin_theta = (r - x) / rod_length
    theta = math.asin(sin_theta)

    # Angular velocity
    cos_theta = math.sqrt(1.0 - sin_theta**2)
    omega = -v / (rod_length * cos_theta)

    # Angular acceleration
    alpha = -(a + rod_length * omega**2 * sin_theta) / (rod_length * cos_theta)

    return theta, omega, alpha


def piston_slap_force(
    *, geometry: PistonGeometry, state: PistonState, v_lateral: float = 0.0,
) -> float:
    """Piston slap force due to secondary motion.

    Computes the lateral force on the piston due to secondary motion
    and clearance variations.

    Parameters
    ----------
    geometry : PistonGeometry
        Piston geometry
    state : PistonState
        Current piston state
    v_lateral : float
        Lateral velocity [m/s]

    Returns
    -------
    F_slap : float
        Piston slap force [N]
    """
    # Simplified piston slap model
    # Based on clearance and lateral velocity

    gap = geometry.clearance_nominal - abs(state.x)
    if gap <= 0.0:
        return 0.0

    # Stiffness due to clearance
    k_clearance = 1e8  # N/m (simplified)

    # Damping due to lateral motion
    c_damping = 1e3  # Ns/m (simplified)

    F_slap = k_clearance * gap + c_damping * v_lateral

    return F_slap


def piston_ring_dynamics(
    *,
    geometry: PistonGeometry,
    state: PistonState,
    p_gas: float,
    p_crankcase: float = 1e5,
) -> Tuple[float, float]:
    """Piston ring dynamics and blow-by calculation.

    Computes ring friction and blow-by mass flow rate.

    Parameters
    ----------
    geometry : PistonGeometry
        Piston geometry
    state : PistonState
        Current piston state
    p_gas : float
        Gas pressure [Pa]
    p_crankcase : float
        Crankcase pressure [Pa]

    Returns
    -------
    F_ring_friction : float
        Ring friction force [N]
    mdot_blowby : float
        Blow-by mass flow rate [kg/s]
    """
    # Ring friction (simplified)
    # Based on ring tension and gas pressure
    F_ring_tension = geometry.ring_count * geometry.ring_tension
    F_ring_pressure = (
        geometry.ring_count
        * (p_gas - p_crankcase)
        * geometry.ring_width
        * math.pi
        * geometry.bore
    )

    # Total ring force
    F_ring_total = F_ring_tension + F_ring_pressure

    # Friction coefficient (simplified)
    mu_ring = 0.1  # Typical value for piston rings

    F_ring_friction = mu_ring * F_ring_total * math.copysign(1.0, state.v)

    # Blow-by calculation (simplified)
    # Based on pressure difference and ring gap
    if p_gas > p_crankcase:
        # Gas flows from chamber to crankcase
        dp = p_gas - p_crankcase
        A_gap = geometry.ring_count * geometry.ring_gap * geometry.ring_width

        # Simplified orifice flow
        Cd = 0.6  # Discharge coefficient
        rho_gas = 1.0  # kg/m^3 (simplified)

        mdot_blowby = Cd * A_gap * math.sqrt(2.0 * rho_gas * dp)
    else:
        mdot_blowby = 0.0

    return F_ring_friction, mdot_blowby


def clearance_penalty(*, gap: float, gap_min: float, k: float) -> float:
    """Repulsive penalty when piston gap falls below minimum.

    Returns a positive force pushing pistons apart when gap < gap_min.
    """
    if gap >= gap_min:
        return 0.0
    return k * (gap_min - gap)


def piston_energy_balance(
    *, geometry: PistonGeometry, state: PistonState, dt: float,
) -> Dict[str, float]:
    """Piston energy balance for validation.

    Computes energy transfer rates and validates conservation.

    Parameters
    ----------
    geometry : PistonGeometry
        Piston geometry
    state : PistonState
        Current piston state
    dt : float
        Time step [s]

    Returns
    -------
    energy_balance : Dict[str, float]
        Energy balance components
    """
    # Kinetic energy
    E_kinetic = 0.5 * geometry.mass * state.v**2

    # Work done by gas
    dV = state.v * A_piston * dt
    W_gas = state.p_gas * dV

    # Work done by friction
    W_friction = state.F_friction * state.v * dt

    # Work done by clearance penalty
    W_clearance = state.F_clearance * state.v * dt

    # Energy balance
    dE_total = W_gas - W_friction - W_clearance

    return {
        "kinetic_energy": E_kinetic,
        "work_gas": W_gas,
        "work_friction": W_friction,
        "work_clearance": W_clearance,
        "energy_change": dE_total,
    }


def get_piston_force_function(method: str = "full"):
    """Get piston force function by name.

    Parameters
    ----------
    method : str
        Force calculation method:
        - 'full': Full force balance with all components
        - 'simple': Simple gas pressure only
        - 'clearance': Gas pressure + clearance penalty

    Returns
    -------
    force_func : callable
        Piston force function
    """
    if method == "full":
        return piston_force_balance
    if method == "simple":
        return lambda geometry, state, p_gas, T_gas, V_chamber, omega=0.0: (
            p_gas * math.pi * (geometry.bore / 2.0) ** 2,  # F_net
            p_gas * math.pi * (geometry.bore / 2.0) ** 2,  # F_gas
            0.0,  # F_inertia
        )
    if method == "clearance":
        return lambda geometry, state, p_gas, T_gas, V_chamber, omega=0.0: (
            p_gas * math.pi * (geometry.bore / 2.0) ** 2
            + clearance_penalty(
                geometry.clearance_nominal - abs(state.x), geometry.clearance_min, 1e6,
            ),
            p_gas * math.pi * (geometry.bore / 2.0) ** 2,
            0.0,
        )
    raise ValueError(f"Unknown piston force method: {method}")


def piston_dae_residual(
    *,
    geometry: PistonGeometry,
    state: PistonState,
    p_gas: float,
    T_gas: float,
    V_chamber: float,
    omega: float = 0.0,
    dt: float = 1e-6,
) -> Dict[str, float]:
    """
    Piston DAE residual for gas-structure coupling.

    This function provides the residual equations for the differential-algebraic
    system that couples piston dynamics with gas dynamics.

    Parameters
    ----------
    geometry : PistonGeometry
        Piston geometry and mass properties
    state : PistonState
        Current piston state
    p_gas : float
        Gas pressure [Pa]
    T_gas : float
        Gas temperature [K]
    V_chamber : float
        Chamber volume [m^3]
    omega : float
        Engine angular velocity [rad/s]
    dt : float
        Time step [s]

    Returns
    -------
    residuals : Dict[str, float]
        DAE residual components
    """
    # Force balance residual: F_net = m * a
    F_net, F_gas, F_inertia = piston_force_balance(
        geometry=geometry,
        state=state,
        p_gas=p_gas,
        T_gas=T_gas,
        V_chamber=V_chamber,
        omega=omega,
    )

    # Mass-acceleration residual
    m_total = geometry.mass + geometry.rod_mass * (
        geometry.rod_cg_offset / geometry.rod_length
    )
    residual_force = F_net - m_total * state.a

    # Kinematic residual: v = dx/dt
    residual_kinematic = state.v - state.x / dt  # Simplified for now

    # Energy residual: dE/dt = F * v
    E_kinetic = 0.5 * m_total * state.v**2
    residual_energy = F_net * state.v - E_kinetic / dt

    return {
        "force_residual": residual_force,
        "kinematic_residual": residual_kinematic,
        "energy_residual": residual_energy,
        "F_net": F_net,
        "F_gas": F_gas,
        "F_inertia": F_inertia,
    }


def gas_structure_coupling(
    *,
    geometry: PistonGeometry,
    state: PistonState,
    gas_state: Dict[str, float],
    coupling_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Gas-structure coupling for piston-gas interaction.

    This function handles the coupling between gas dynamics and piston dynamics,
    including volume changes, pressure forces, and energy transfer.

    Parameters
    ----------
    geometry : PistonGeometry
        Piston geometry
    state : PistonState
        Current piston state
    gas_state : Dict[str, float]
        Gas state variables (p, T, rho, etc.)
    coupling_params : Dict[str, Any], optional
        Coupling parameters

    Returns
    -------
    coupling_terms : Dict[str, float]
        Gas-structure coupling terms
    """
    if coupling_params is None:
        coupling_params = {}

    # Extract gas state
    p_gas = gas_state.get("p", 1e5)
    T_gas = gas_state.get("T", 300.0)
    rho_gas = gas_state.get("rho", 1.0)

    # Piston area
    A_piston = math.pi * (geometry.bore / 2.0) ** 2

    # Volume change rate
    dV_dt = A_piston * state.v

    # Pressure force on piston
    F_pressure = p_gas * A_piston

    # Work rate (power)
    P_work = F_pressure * state.v

    # Mass flow rate due to piston motion (simplified)
    # This would be more complex in practice
    mdot_piston = rho_gas * dV_dt

    # Heat transfer rate (simplified)
    # Based on piston surface area and temperature difference
    A_surface = math.pi * geometry.bore * geometry.stroke
    h_conv = 100.0  # W/(m^2·K) - simplified
    T_wall = 400.0  # K - simplified
    Q_heat = h_conv * A_surface * (T_gas - T_wall)

    return {
        "dV_dt": dV_dt,
        "F_pressure": F_pressure,
        "P_work": P_work,
        "mdot_piston": mdot_piston,
        "Q_heat": Q_heat,
        "A_piston": A_piston,
    }


def piston_mass_inertia_calculation(
    *,
    geometry: PistonGeometry,
    state: PistonState,
) -> Dict[str, float]:
    """
    Calculate mass and moment of inertia for piston system.

    Parameters
    ----------
    geometry : PistonGeometry
        Piston geometry
    state : PistonState
        Current piston state

    Returns
    -------
    inertia_properties : Dict[str, float]
        Mass and inertia properties
    """
    # Total mass
    m_total = geometry.mass + geometry.rod_mass

    # Piston moment of inertia (simplified - assumes solid cylinder)
    r_piston = geometry.bore / 2.0
    I_piston = 0.5 * geometry.mass * r_piston**2

    # Connecting rod moment of inertia (simplified)
    # Assumes uniform rod
    I_rod = (1.0 / 12.0) * geometry.rod_mass * geometry.rod_length**2

    # Total moment of inertia
    I_total = I_piston + I_rod

    # Effective mass for linear motion
    m_effective = geometry.mass + geometry.rod_mass * (
        geometry.rod_cg_offset / geometry.rod_length
    )

    return {
        "m_total": m_total,
        "m_effective": m_effective,
        "I_piston": I_piston,
        "I_rod": I_rod,
        "I_total": I_total,
    }


def piston_clearance_validation(
    *,
    geometry: PistonGeometry,
    state: PistonState,
) -> Dict[str, float]:
    """
    Validate piston clearance and compute clearance-related forces.

    Parameters
    ----------
    geometry : PistonGeometry
        Piston geometry
    state : PistonState
        Current piston state

    Returns
    -------
    clearance_info : Dict[str, float]
        Clearance validation and forces
    """
    # Current clearance
    clearance_current = geometry.clearance_nominal - abs(state.x)

    # Clearance violation
    clearance_violation = max(0.0, geometry.clearance_min - clearance_current)

    # Clearance penalty force
    F_clearance = clearance_penalty(
        gap=clearance_current,
        gap_min=geometry.clearance_min,
        k=1e6,
    )

    # Clearance safety factor
    safety_factor = clearance_current / geometry.clearance_min

    # Clearance status
    clearance_ok = clearance_current >= geometry.clearance_min

    return {
        "clearance_current": clearance_current,
        "clearance_violation": clearance_violation,
        "F_clearance": F_clearance,
        "safety_factor": safety_factor,
        "clearance_ok": clearance_ok,
    }
