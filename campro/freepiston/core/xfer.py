from __future__ import annotations

import math

from campro.logging import get_logger

log = get_logger(__name__)


def woschni_h(*, p: float, T: float, B: float, w: float) -> float:
    """Basic Woschni-style convective coefficient h [W/(m^2 K)].

    h = C * p^a * T^b * w^c * B^d with typical exponents.
    Coefficients are nominal and should be calibrated.
    """
    # Nominal exponents (illustrative): a=0.8, b=-0.53, c=0.8, d=-0.2
    # Constant C tuned to get h in ~100-500 W/m^2K range under typical IC conditions
    if p <= 0.0 or T <= 0.0 or B <= 0.0 or w <= 0.0:
        return 0.0
    a, b, c, d = 0.8, -0.53, 0.8, -0.2
    C = 2.8
    return C * (p**a) * (T**b) * (w**c) * (B**d)


def woschni_huber_h(
    *, p: float, T: float, B: float, w: float, C1: float = 130.0, C2: float = 1.4,
) -> float:
    """Calibrated Woschni-Huber correlation for OP engines.

    h = C1 * p^0.8 * T^-0.53 * w^0.8 * B^-0.2 * (1 + C2 * w_m/sw)

    Parameters
    ----------
    p : float
        Pressure [Pa]
    T : float
        Temperature [K]
    B : float
        Bore diameter [m]
    w : float
        Characteristic velocity [m/s]
    C1 : float
        Primary coefficient (default: 130.0 for OP engines)
    C2 : float
        Swirl enhancement factor (default: 1.4)

    Returns
    -------
    h : float
        Heat transfer coefficient [W/(m^2 K)]
    """
    if p <= 0.0 or T <= 0.0 or B <= 0.0 or w <= 0.0:
        return 0.0

    # Base Woschni correlation
    h_base = C1 * (p**0.8) * (T**-0.53) * (w**0.8) * (B**-0.2)

    # Swirl enhancement (simplified - assumes w_m/sw = 1.0 for now)
    swirl_factor = 1.0 + C2 * 1.0  # TODO: Add proper swirl ratio calculation

    return h_base * swirl_factor


def hohenberg_h(
    *, p: float, T: float, B: float, w: float, rho: float, mu: float,
) -> float:
    """Hohenberg correlation for wall heat transfer.

    Nu = 0.130 * Re^0.8 * Pr^0.4 * (V/V_ref)^-0.06

    Parameters
    ----------
    p : float
        Pressure [Pa]
    T : float
        Temperature [K]
    B : float
        Bore diameter [m]
    w : float
        Characteristic velocity [m/s]
    rho : float
        Gas density [kg/m^3]
    mu : float
        Dynamic viscosity [Pa s]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/(m^2 K)]
    """
    if p <= 0.0 or T <= 0.0 or B <= 0.0 or w <= 0.0 or rho <= 0.0 or mu <= 0.0:
        return 0.0

    # Reynolds number
    Re = rho * w * B / mu

    # Prandtl number (assume constant for now)
    Pr = 0.7

    # Nusselt number
    Nu = 0.130 * (Re**0.8) * (Pr**0.4)

    # Volume ratio (simplified - assume V/V_ref = 1.0)
    V_ratio = 1.0
    Nu *= V_ratio**-0.06

    # Heat transfer coefficient
    k = mu * 1005.0 / Pr  # Thermal conductivity from Prandtl number
    h = Nu * k / B

    return h


def compressible_wall_function_h(
    *,
    rho: float,
    u: float,
    mu: float,
    k: float,
    y_plus: float,
    T_wall: float,
    T_gas: float,
    p: float,
    R: float,
) -> float:
    """Compressible law-of-the-wall heat transfer coefficient.

    Accounts for compressibility effects in boundary layer heat transfer.

    Parameters
    ----------
    rho : float
        Gas density [kg/m^3]
    u : float
        Gas velocity [m/s]
    mu : float
        Dynamic viscosity [Pa s]
    k : float
        Thermal conductivity [W/(m K)]
    y_plus : float
        Non-dimensional wall distance
    T_wall : float
        Wall temperature [K]
    T_gas : float
        Gas temperature [K]
    p : float
        Pressure [Pa]
    R : float
        Gas constant [J/(kg K)]

    Returns
    -------
    h : float
        Heat transfer coefficient [W/(m^2 K)]
    """
    if rho <= 0.0 or u <= 0.0 or mu <= 0.0 or k <= 0.0 or y_plus <= 0.0:
        return 0.0

    # Friction velocity
    tau_wall = rho * (u**2) * 0.0225 * (y_plus**-0.25)  # Blasius correlation
    u_tau = math.sqrt(tau_wall / rho)

    # Thermal boundary layer thickness
    Pr = mu * 1005.0 / k  # Prandtl number
    delta_T = mu / (rho * u_tau * Pr)

    # Compressibility correction
    T_ref = 0.5 * (T_wall + T_gas)
    rho_ref = p / (R * T_ref)
    compressibility_factor = (rho / rho_ref) ** 0.5

    # Heat transfer coefficient
    h = k / delta_T * compressibility_factor

    return h


def radiation_heat_transfer(
    *, T_gas: float, T_wall: float, emissivity: float = 0.8,
) -> float:
    """Radiation heat transfer between gas and wall.

    q_rad = sigma * epsilon * (T_gas^4 - T_wall^4)

    Parameters
    ----------
    T_gas : float
        Gas temperature [K]
    T_wall : float
        Wall temperature [K]
    emissivity : float
        Surface emissivity (default: 0.8)

    Returns
    -------
    q_rad : float
        Radiation heat flux [W/m^2]
    """
    sigma = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m^2 K^4)]

    if T_gas <= 0.0 or T_wall <= 0.0:
        return 0.0

    return sigma * emissivity * (T_gas**4 - T_wall**4)


def conjugate_heat_transfer(
    *,
    h_conv: float,
    h_rad: float,
    wall_thickness: float,
    wall_conductivity: float,
    T_gas: float,
    T_wall_inner: float,
    T_wall_outer: float,
) -> float:
    """Conjugate heat transfer across wall with conduction.

    Accounts for heat conduction through wall thickness.

    Parameters
    ----------
    h_conv : float
        Convective heat transfer coefficient [W/(m^2 K)]
    h_rad : float
        Radiative heat transfer coefficient [W/(m^2 K)]
    wall_thickness : float
        Wall thickness [m]
    wall_conductivity : float
        Wall thermal conductivity [W/(m K)]
    T_gas : float
        Gas temperature [K]
    T_wall_inner : float
        Inner wall temperature [K]
    T_wall_outer : float
        Outer wall temperature [K]

    Returns
    -------
    q_total : float
        Total heat flux [W/m^2]
    """
    if wall_thickness <= 0.0 or wall_conductivity <= 0.0:
        return 0.0

    # Convective heat transfer
    q_conv = h_conv * (T_gas - T_wall_inner)

    # Radiative heat transfer
    q_rad = h_rad * (T_gas - T_wall_inner)

    # Conduction through wall
    q_cond = wall_conductivity * (T_wall_inner - T_wall_outer) / wall_thickness

    # Total heat transfer (convective + radiative)
    q_total = q_conv + q_rad

    return q_total


def heat_loss_rate(*, h: float, area: float, T: float, Tw: float) -> float:
    """Basic heat loss rate calculation.

    Parameters
    ----------
    h : float
        Heat transfer coefficient [W/(m^2 K)]
    area : float
        Heat transfer area [m^2]
    T : float
        Gas temperature [K]
    Tw : float
        Wall temperature [K]

    Returns
    -------
    q : float
        Heat loss rate [W]
    """
    if h <= 0.0 or area <= 0.0:
        return 0.0
    return h * area * max(T - Tw, 0.0)


def get_heat_transfer_coefficient(*, method: str, **kwargs) -> float:
    """Unified interface for heat transfer coefficient calculations.

    Parameters
    ----------
    method : str
        Heat transfer correlation method:
        - 'woschni': Basic Woschni correlation
        - 'woschni_huber': Calibrated Woschni-Huber for OP engines
        - 'hohenberg': Hohenberg correlation
        - 'compressible_wall': Compressible law-of-the-wall
    **kwargs
        Method-specific parameters

    Returns
    -------
    h : float
        Heat transfer coefficient [W/(m^2 K)]
    """
    if method == "woschni":
        return woschni_h(**kwargs)
    if method == "woschni_huber":
        return woschni_huber_h(**kwargs)
    if method == "hohenberg":
        return hohenberg_h(**kwargs)
    if method == "compressible_wall":
        return compressible_wall_function_h(**kwargs)
    raise ValueError(f"Unknown heat transfer method: {method}")
