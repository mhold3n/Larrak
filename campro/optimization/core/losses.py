from __future__ import annotations

import math
from dataclasses import dataclass

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class FrictionParameters:
    """Friction model parameters."""

    # Piston ring friction
    mu_ring: float  # Ring friction coefficient
    ring_tension: float  # Ring tension force [N]
    ring_count: int  # Number of rings

    # Bearing friction
    mu_bearing: float  # Bearing friction coefficient
    bearing_diameter: float  # Bearing diameter [m]
    bearing_length: float  # Bearing length [m]

    # Pumping losses
    pumping_coefficient: float  # Pumping loss coefficient

    # Viscous losses
    viscous_coefficient: float  # Viscous loss coefficient


def coulomb_friction(*, v: float, muN: float) -> float:
    """Coulomb friction opposing motion; zero at rest.

    Returns -muN * sign(v).
    """
    if muN <= 0.0 or v == 0.0:
        return 0.0
    return -muN if v > 0.0 else muN


def piston_ring_friction(
    *,
    v: float,
    p_gas: float,
    p_crankcase: float,
    geometry: dict[str, float],
    params: FrictionParameters,
) -> float:
    """Piston ring friction model.

    Computes friction force from piston rings considering:
    - Ring tension
    - Gas pressure differential
    - Velocity-dependent friction

    Parameters
    ----------
    v : float
        Piston velocity [m/s]
    p_gas : float
        Gas pressure [Pa]
    p_crankcase : float
        Crankcase pressure [Pa]
    geometry : Dict[str, float]
        Piston geometry parameters
    params : FrictionParameters
        Friction model parameters

    Returns
    -------
    F_ring : float
        Ring friction force [N]
    """
    if abs(v) < 1e-6:
        return 0.0

    # Ring normal force
    F_ring_tension = params.ring_count * params.ring_tension
    F_ring_pressure = (
        params.ring_count
        * (p_gas - p_crankcase)
        * geometry.get("ring_width", 0.002)
        * math.pi
        * geometry.get("bore", 0.1)
    )

    F_ring_normal = F_ring_tension + F_ring_pressure

    # Velocity-dependent friction coefficient
    # Higher friction at low velocities (boundary lubrication)
    v_ref = 1.0  # m/s reference velocity
    mu_effective = params.mu_ring * (1.0 + 0.5 * math.exp(-abs(v) / v_ref))

    # Friction force
    F_ring = -mu_effective * F_ring_normal * math.copysign(1.0, v)

    return F_ring


def bearing_friction(*, omega: float, load: float, params: FrictionParameters) -> float:
    """Bearing friction model.

    Computes friction torque from main and connecting rod bearings.

    Parameters
    ----------
    omega : float
        Angular velocity [rad/s]
    load : float
        Bearing load [N]
    params : FrictionParameters
        Friction model parameters

    Returns
    -------
    T_bearing : float
        Bearing friction torque [N m]
    """
    if abs(omega) < 1e-6:
        return 0.0

    # Bearing friction torque
    # Simplified model: T = mu * F * r
    r_bearing = params.bearing_diameter / 2.0
    T_bearing = -params.mu_bearing * load * r_bearing * math.copysign(1.0, omega)

    return T_bearing


def pumping_losses(
    *,
    v: float,
    p_gas: float,
    V_chamber: float,
    geometry: dict[str, float],
    params: FrictionParameters,
) -> float:
    """Pumping losses in gas flow.

    Computes pressure losses due to gas flow through valves and passages.

    Parameters
    ----------
    v : float
        Piston velocity [m/s]
    p_gas : float
        Gas pressure [Pa]
    V_chamber : float
        Chamber volume [m^3]
    geometry : Dict[str, float]
        Geometry parameters
    params : FrictionParameters
        Friction model parameters

    Returns
    -------
    F_pumping : float
        Pumping loss force [N]
    """
    if abs(v) < 1e-6:
        return 0.0

    # Pumping loss coefficient (simplified)
    # Based on velocity and pressure
    A_piston = math.pi * (geometry.get("bore", 0.1) / 2.0) ** 2

    # Pressure loss due to flow
    dp_pumping = params.pumping_coefficient * abs(v) * p_gas / 1000.0  # Simplified

    F_pumping = -dp_pumping * A_piston * math.copysign(1.0, v)

    return F_pumping


def viscous_losses(
    *,
    v: float,
    rho: float,
    mu: float,
    geometry: dict[str, float],
    params: FrictionParameters,
) -> float:
    """Viscous losses in gas flow.

    Computes viscous drag forces on piston and gas flow.

    Parameters
    ----------
    v : float
        Piston velocity [m/s]
    rho : float
        Gas density [kg/m^3]
    mu : float
        Dynamic viscosity [Pa s]
    geometry : Dict[str, float]
        Geometry parameters
    params : FrictionParameters
        Friction model parameters

    Returns
    -------
    F_viscous : float
        Viscous drag force [N]
    """
    if abs(v) < 1e-6:
        return 0.0

    # Viscous drag on piston
    A_piston = math.pi * (geometry.get("bore", 0.1) / 2.0) ** 2
    clearance = geometry.get("clearance", 0.001)  # m

    # Simplified viscous drag model
    # Based on Couette flow in clearance
    F_viscous = -params.viscous_coefficient * mu * A_piston * v / clearance

    return F_viscous


def blow_by_losses(
    *, p_gas: float, p_crankcase: float, geometry: dict[str, float],
) -> tuple[float, float]:
    """Blow-by losses through piston rings.

    Computes mass flow rate and energy loss due to blow-by.

    Parameters
    ----------
    p_gas : float
        Gas pressure [Pa]
    p_crankcase : float
        Crankcase pressure [Pa]
    geometry : Dict[str, float]
        Geometry parameters

    Returns
    -------
    mdot_blowby : float
        Blow-by mass flow rate [kg/s]
    E_loss_blowby : float
        Energy loss rate [W]
    """
    if p_gas <= p_crankcase:
        return 0.0, 0.0

    # Blow-by area (simplified)
    ring_gap = geometry.get("ring_gap", 0.0001)  # m
    ring_width = geometry.get("ring_width", 0.002)  # m
    ring_count = geometry.get("ring_count", 3)

    A_blowby = ring_count * ring_gap * ring_width

    # Orifice flow model
    Cd = 0.6  # Discharge coefficient
    rho_gas = 1.0  # kg/m^3 (simplified)

    dp = p_gas - p_crankcase
    mdot_blowby = Cd * A_blowby * math.sqrt(2.0 * rho_gas * dp)

    # Energy loss (simplified)
    # Assume gas temperature is same as chamber temperature
    T_gas = 500.0  # K (simplified)
    cp = 1005.0  # J/(kg K) (simplified)
    E_loss_blowby = mdot_blowby * cp * T_gas

    return mdot_blowby, E_loss_blowby


def total_friction_force(
    *,
    v: float,
    omega: float,
    p_gas: float,
    p_crankcase: float,
    rho: float,
    mu: float,
    V_chamber: float,
    geometry: dict[str, float],
    params: FrictionParameters,
) -> dict[str, float]:
    """Total friction force from all sources.

    Computes combined friction force from all loss mechanisms.

    Parameters
    ----------
    v : float
        Piston velocity [m/s]
    omega : float
        Angular velocity [rad/s]
    p_gas : float
        Gas pressure [Pa]
    p_crankcase : float
        Crankcase pressure [Pa]
    rho : float
        Gas density [kg/m^3]
    mu : float
        Dynamic viscosity [Pa s]
    V_chamber : float
        Chamber volume [m^3]
    geometry : Dict[str, float]
        Geometry parameters
    params : FrictionParameters
        Friction model parameters

    Returns
    -------
    friction_forces : Dict[str, float]
        Individual and total friction forces
    """
    # Individual friction components
    F_ring = piston_ring_friction(
        v=v, p_gas=p_gas, p_crankcase=p_crankcase, geometry=geometry, params=params,
    )

    F_pumping = pumping_losses(
        v=v, p_gas=p_gas, V_chamber=V_chamber, geometry=geometry, params=params,
    )

    F_viscous = viscous_losses(v=v, rho=rho, mu=mu, geometry=geometry, params=params)

    # Bearing friction (converted to linear force)
    load = p_gas * math.pi * (geometry.get("bore", 0.1) / 2.0) ** 2
    T_bearing = bearing_friction(omega=omega, load=load, params=params)
    F_bearing = T_bearing / (
        geometry.get("stroke", 0.1) / 2.0
    )  # Convert to linear force

    # Total friction
    F_total = F_ring + F_pumping + F_viscous + F_bearing

    return {
        "ring_friction": F_ring,
        "pumping_losses": F_pumping,
        "viscous_losses": F_viscous,
        "bearing_friction": F_bearing,
        "total_friction": F_total,
    }


def friction_power_loss(
    *, v: float, omega: float, friction_forces: dict[str, float],
) -> dict[str, float]:
    """Friction power losses.

    Computes power losses from friction forces.

    Parameters
    ----------
    v : float
        Piston velocity [m/s]
    omega : float
        Angular velocity [rad/s]
    friction_forces : Dict[str, float]
        Friction forces

    Returns
    -------
    power_losses : Dict[str, float]
        Power losses from each friction source
    """
    power_losses = {}

    for name, force in friction_forces.items():
        if name == "bearing_friction":
            # Bearing power loss
            power_losses[name] = abs(force * v)  # Simplified conversion
        else:
            # Linear friction power loss
            power_losses[name] = abs(force * v)

    # Total power loss
    power_losses["total_power_loss"] = sum(power_losses.values())

    return power_losses


def get_friction_function(method: str = "full"):
    """Get friction function by name.

    Parameters
    ----------
    method : str
        Friction calculation method:
        - 'full': Full friction model with all components
        - 'simple': Simple Coulomb friction only
        - 'ring': Ring friction only

    Returns
    -------
    friction_func : callable
        Friction function
    """
    if method == "full":
        return total_friction_force
    if method == "simple":
        return lambda v, **kwargs: {"total_friction": coulomb_friction(v=v, muN=100.0)}
    if method == "ring":
        return lambda v, p_gas, p_crankcase, geometry, params, **kwargs: {
            "ring_friction": piston_ring_friction(
                v=v,
                p_gas=p_gas,
                p_crankcase=p_crankcase,
                geometry=geometry,
                params=params,
            ),
            "total_friction": piston_ring_friction(
                v=v,
                p_gas=p_gas,
                p_crankcase=p_crankcase,
                geometry=geometry,
                params=params,
            ),
        }
    raise ValueError(f"Unknown friction method: {method}")
