"""Combustion models for free-piston engines."""

from __future__ import annotations

import math
from dataclasses import dataclass

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class CombustionParameters:
    """Combustion model parameters."""

    # Wiebe function parameters
    m_wiebe: float  # Wiebe shape factor
    a_wiebe: float  # Wiebe efficiency factor

    # Combustion timing
    theta_start: float  # Combustion start angle [deg]
    theta_duration: float  # Combustion duration [deg]

    # Heat release
    Q_total: float  # Total heat release [J]
    LHV_fuel: float  # Lower heating value [J/kg]

    # Ignition delay
    tau_ignition: float  # Ignition delay [s]

    # Flame speed
    S_L0: float  # Laminar flame speed [m/s]
    alpha_turbulence: float  # Turbulence factor


def wiebe_function(
    *, theta: float, theta_start: float, theta_duration: float, m: float, a: float,
) -> float:
    """Wiebe function for heat release rate.

    Computes the cumulative heat release fraction using the Wiebe function:
    x_b = 1 - exp(-a * ((theta - theta_start) / theta_duration)^(m+1))

    Parameters
    ----------
    theta : float
        Current crank angle [deg]
    theta_start : float
        Combustion start angle [deg]
    theta_duration : float
        Combustion duration [deg]
    m : float
        Wiebe shape factor
    a : float
        Wiebe efficiency factor

    Returns
    -------
    x_b : float
        Cumulative burn fraction [0-1]
    """
    if theta < theta_start:
        return 0.0

    if theta > theta_start + theta_duration:
        return 1.0

    # Normalized angle
    theta_norm = (theta - theta_start) / theta_duration

    # Wiebe function
    x_b = 1.0 - math.exp(-a * (theta_norm ** (m + 1.0)))

    return x_b


def wiebe_heat_release_rate(*, theta: float, params: CombustionParameters) -> float:
    """Heat release rate from Wiebe function.

    Computes the instantaneous heat release rate by differentiating
    the Wiebe function.

    Parameters
    ----------
    theta : float
        Current crank angle [deg]
    params : CombustionParameters
        Combustion parameters

    Returns
    -------
    dQ_dtheta : float
        Heat release rate [J/deg]
    """
    if theta < params.theta_start or theta > params.theta_start + params.theta_duration:
        return 0.0

    # Normalized angle
    theta_norm = (theta - params.theta_start) / params.theta_duration

    # Wiebe function derivative
    # d/dtheta[1 - exp(-a * theta_norm^(m+1))]
    # = a * (m+1) * theta_norm^m * exp(-a * theta_norm^(m+1)) / theta_duration

    exponent = -params.a_wiebe * (theta_norm ** (params.m_wiebe + 1.0))
    dQ_dtheta = (
        params.a_wiebe
        * (params.m_wiebe + 1.0)
        * (theta_norm**params.m_wiebe)
        * math.exp(exponent)
        * params.Q_total
        / params.theta_duration
    )

    return dQ_dtheta


def ignition_delay_correlation(
    *, p: float, T: float, phi: float = 1.0, params: CombustionParameters,
) -> float:
    """Ignition delay correlation.

    Computes ignition delay based on pressure, temperature, and equivalence ratio.
    Uses simplified Arrhenius-type correlation.

    Parameters
    ----------
    p : float
        Pressure [Pa]
    T : float
        Temperature [K]
    phi : float
        Equivalence ratio
    params : CombustionParameters
        Combustion parameters

    Returns
    -------
    tau_ignition : float
        Ignition delay [s]
    """
    # Simplified ignition delay correlation
    # tau = A * p^(-n) * exp(E_a / (R * T)) * phi^(-m)

    A = 1e-6  # Pre-exponential factor [s]
    n = 1.0  # Pressure exponent
    E_a = 15000.0  # Activation energy [J/mol]
    R = 8.314  # Gas constant [J/(mol K)]
    m = 0.5  # Equivalence ratio exponent

    tau_ignition = A * (p ** (-n)) * math.exp(E_a / (R * T)) * (phi ** (-m))

    return tau_ignition


def laminar_flame_speed(
    *, phi: float, T: float, p: float, params: CombustionParameters,
) -> float:
    """Laminar flame speed correlation.

    Computes laminar flame speed based on equivalence ratio, temperature, and pressure.

    Parameters
    ----------
    phi : float
        Equivalence ratio
    T : float
        Temperature [K]
    p : float
        Pressure [Pa]
    params : CombustionParameters
        Combustion parameters

    Returns
    -------
    S_L : float
        Laminar flame speed [m/s]
    """
    # Simplified laminar flame speed correlation
    # S_L = S_L0 * (T/T_ref)^alpha * (p/p_ref)^beta * f(phi)

    T_ref = 300.0  # K
    p_ref = 1e5  # Pa
    alpha = 2.0  # Temperature exponent
    beta = -0.5  # Pressure exponent

    # Equivalence ratio dependence (simplified)
    if phi < 0.5 or phi > 1.5:
        f_phi = 0.0
    else:
        f_phi = 4.0 * phi * (1.0 - phi)  # Parabolic dependence

    S_L = params.S_L0 * ((T / T_ref) ** alpha) * ((p / p_ref) ** beta) * f_phi

    return S_L


def turbulent_flame_speed(
    *, S_L: float, u_turb: float, params: CombustionParameters,
) -> float:
    """Turbulent flame speed.

    Computes turbulent flame speed from laminar flame speed and turbulence intensity.

    Parameters
    ----------
    S_L : float
        Laminar flame speed [m/s]
    u_turb : float
        Turbulence intensity [m/s]
    params : CombustionParameters
        Combustion parameters

    Returns
    -------
    S_T : float
        Turbulent flame speed [m/s]
    """
    # Turbulent flame speed correlation
    # S_T = S_L * (1 + alpha * (u_turb / S_L)^n)

    alpha = params.alpha_turbulence
    n = 0.7  # Turbulence exponent

    S_T = S_L * (1.0 + alpha * ((u_turb / S_L) ** n))

    return S_T


def combustion_efficiency(*, theta: float, params: CombustionParameters) -> float:
    """Combustion efficiency from Wiebe function.

    Computes the combustion efficiency based on the Wiebe function.

    Parameters
    ----------
    theta : float
        Current crank angle [deg]
    params : CombustionParameters
        Combustion parameters

    Returns
    -------
    eta_comb : float
        Combustion efficiency [0-1]
    """
    return wiebe_function(
        theta=theta,
        theta_start=params.theta_start,
        theta_duration=params.theta_duration,
        m=params.m_wiebe,
        a=params.a_wiebe,
    )


def heat_release_from_fuel(*, m_fuel: float, params: CombustionParameters) -> float:
    """Heat release from fuel mass.

    Computes heat release based on fuel mass and lower heating value.

    Parameters
    ----------
    m_fuel : float
        Fuel mass [kg]
    params : CombustionParameters
        Combustion parameters

    Returns
    -------
    Q_release : float
        Heat release [J]
    """
    return m_fuel * params.LHV_fuel


def fuel_mass_from_heat(*, Q_release: float, params: CombustionParameters) -> float:
    """Fuel mass from heat release.

    Computes fuel mass based on heat release and lower heating value.

    Parameters
    ----------
    Q_release : float
        Heat release [J]
    params : CombustionParameters
        Combustion parameters

    Returns
    -------
    m_fuel : float
        Fuel mass [kg]
    """
    return Q_release / params.LHV_fuel


def multi_zone_combustion(
    *, zones: dict[str, dict[str, float]], params: CombustionParameters,
) -> dict[str, float]:
    """Multi-zone combustion model.

    Computes heat release for multiple combustion zones.

    Parameters
    ----------
    zones : Dict[str, Dict[str, float]]
        Zone data with 'mass', 'temperature', 'pressure', 'equivalence_ratio'
    params : CombustionParameters
        Combustion parameters

    Returns
    -------
    zone_results : Dict[str, float]
        Heat release rates for each zone
    """
    zone_results = {}

    for zone_name, zone_data in zones.items():
        # Compute heat release for this zone
        phi = zone_data.get("equivalence_ratio", 1.0)
        T = zone_data.get("temperature", 500.0)
        p = zone_data.get("pressure", 1e5)
        m = zone_data.get("mass", 0.001)

        # Heat release from fuel in this zone
        m_fuel = m * phi / 14.7  # Simplified stoichiometric ratio
        Q_zone = heat_release_from_fuel(m_fuel, params)

        zone_results[zone_name] = Q_zone

    return zone_results


def combustion_timing_optimization(
    *, theta_start: float, theta_duration: float, params: CombustionParameters,
) -> dict[str, float]:
    """Combustion timing optimization.

    Optimizes combustion timing for maximum efficiency.

    Parameters
    ----------
    theta_start : float
        Combustion start angle [deg]
    theta_duration : float
        Combustion duration [deg]
    params : CombustionParameters
        Combustion parameters

    Returns
    -------
    optimization_results : Dict[str, float]
        Optimization results
    """
    # Simplified optimization based on Wiebe function
    # Optimal timing depends on engine speed and load

    # Compute burn rate at different angles
    theta_test = [theta_start + i * theta_duration / 10.0 for i in range(11)]
    burn_rates = [wiebe_heat_release_rate(theta, params) for theta in theta_test]

    # Find peak burn rate
    peak_rate = max(burn_rates)
    peak_angle = theta_test[burn_rates.index(peak_rate)]

    # Compute combustion efficiency
    eta_comb = wiebe_function(
        theta=theta_start + theta_duration,
        theta_start=theta_start,
        theta_duration=theta_duration,
        m=params.m_wiebe,
        a=params.a_wiebe,
    )

    return {
        "peak_burn_rate": peak_rate,
        "peak_angle": peak_angle,
        "combustion_efficiency": eta_comb,
        "optimal_start": theta_start,
        "optimal_duration": theta_duration,
    }


def get_combustion_function(method: str = "wiebe"):
    """Get combustion function by name.

    Parameters
    ----------
    method : str
        Combustion calculation method:
        - 'wiebe': Wiebe function heat release
        - 'ignition_delay': Ignition delay correlation
        - 'flame_speed': Flame speed correlation
        - 'multi_zone': Multi-zone combustion

    Returns
    -------
    combustion_func : callable
        Combustion function
    """
    if method == "wiebe":
        return wiebe_heat_release_rate
    if method == "ignition_delay":
        return ignition_delay_correlation
    if method == "flame_speed":
        return laminar_flame_speed
    if method == "multi_zone":
        return multi_zone_combustion
    raise ValueError(f"Unknown combustion method: {method}")


def create_combustion_parameters(fuel_type: str = "gasoline") -> CombustionParameters:
    """Create combustion parameters for different fuel types.

    Parameters
    ----------
    fuel_type : str
        Fuel type: 'gasoline', 'diesel', 'natural_gas', 'hydrogen'

    Returns
    -------
    params : CombustionParameters
        Combustion parameters
    """
    if fuel_type == "gasoline":
        return CombustionParameters(
            m_wiebe=2.0,
            a_wiebe=5.0,
            theta_start=10.0,
            theta_duration=60.0,
            Q_total=1000.0,
            LHV_fuel=44e6,  # J/kg
            tau_ignition=1e-3,
            S_L0=0.4,  # m/s
            alpha_turbulence=2.0,
        )
    if fuel_type == "diesel":
        return CombustionParameters(
            m_wiebe=1.5,
            a_wiebe=6.9,
            theta_start=5.0,
            theta_duration=80.0,
            Q_total=1000.0,
            LHV_fuel=42e6,  # J/kg
            tau_ignition=1e-4,
            S_L0=0.3,  # m/s
            alpha_turbulence=1.5,
        )
    if fuel_type == "natural_gas":
        return CombustionParameters(
            m_wiebe=2.5,
            a_wiebe=4.0,
            theta_start=15.0,
            theta_duration=70.0,
            Q_total=1000.0,
            LHV_fuel=50e6,  # J/kg
            tau_ignition=2e-3,
            S_L0=0.35,  # m/s
            alpha_turbulence=2.5,
        )
    if fuel_type == "hydrogen":
        return CombustionParameters(
            m_wiebe=3.0,
            a_wiebe=3.0,
            theta_start=20.0,
            theta_duration=50.0,
            Q_total=1000.0,
            LHV_fuel=120e6,  # J/kg
            tau_ignition=5e-4,
            S_L0=2.0,  # m/s
            alpha_turbulence=3.0,
        )
    raise ValueError(f"Unknown fuel type: {fuel_type}")
