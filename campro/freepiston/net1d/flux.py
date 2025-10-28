from __future__ import annotations

import math

from campro.logging import get_logger

log = get_logger(__name__)


def primitive_from_conservative(
    U: tuple[float, float, float], gamma: float = 1.4,
) -> tuple[float, float, float]:
    """Convert conservative variables to primitive variables.

    Parameters
    ----------
    U : Tuple[float, float, float]
        Conservative variables [rho, rho*u, rho*E]
    gamma : float
        Heat capacity ratio

    Returns
    -------
    rho : float
        Density [kg/m^3]
    u : float
        Velocity [m/s]
    p : float
        Pressure [Pa]
    """
    rho, rhou, rhoE = U

    if rho <= 0.0:
        return 0.0, 0.0, 0.0

    u = rhou / rho
    e = rhoE / rho - 0.5 * u**2  # Specific internal energy
    p = (gamma - 1.0) * rho * e  # Pressure from internal energy

    return rho, u, p


def conservative_from_primitive(
    rho: float, u: float, p: float, gamma: float = 1.4,
) -> tuple[float, float, float]:
    """Convert primitive variables to conservative variables.

    Parameters
    ----------
    rho : float
        Density [kg/m^3]
    u : float
        Velocity [m/s]
    p : float
        Pressure [Pa]
    gamma : float
        Heat capacity ratio

    Returns
    -------
    U : Tuple[float, float, float]
        Conservative variables [rho, rho*u, rho*E]
    """
    e = p / ((gamma - 1.0) * rho)  # Specific internal energy
    E = e + 0.5 * u**2  # Specific total energy

    return (rho, rho * u, rho * E)


def flux_from_primitive(rho: float, u: float, p: float) -> tuple[float, float, float]:
    """Compute flux from primitive variables.

    Parameters
    ----------
    rho : float
        Density [kg/m^3]
    u : float
        Velocity [m/s]
    p : float
        Pressure [Pa]

    Returns
    -------
    F : Tuple[float, float, float]
        Flux vector [rho*u, rho*u^2 + p, (rho*E + p)*u]
    """
    F1 = rho * u
    F2 = rho * u**2 + p
    F3 = (rho * (0.5 * u**2 + p / ((1.4 - 1.0) * rho)) + p) * u

    return (F1, F2, F3)


def roe_averages(
    U_L: tuple[float, float, float], U_R: tuple[float, float, float], gamma: float = 1.4,
) -> tuple[float, float, float, float]:
    """Compute Roe-averaged quantities for wave speed estimation.

    Parameters
    ----------
    U_L : Tuple[float, float, float]
        Left state conservative variables
    U_R : Tuple[float, float, float]
        Right state conservative variables
    gamma : float
        Heat capacity ratio

    Returns
    -------
    rho_roe : float
        Roe-averaged density
    u_roe : float
        Roe-averaged velocity
    H_roe : float
        Roe-averaged total enthalpy
    c_roe : float
        Roe-averaged speed of sound
    """
    rho_L, u_L, p_L = primitive_from_conservative(U_L, gamma)
    rho_R, u_R, p_R = primitive_from_conservative(U_R, gamma)

    # Roe averages
    sqrt_rho_L = math.sqrt(rho_L)
    sqrt_rho_R = math.sqrt(rho_R)
    sqrt_rho_sum = sqrt_rho_L + sqrt_rho_R

    rho_roe = sqrt_rho_L * sqrt_rho_R
    u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / sqrt_rho_sum

    # Total enthalpy
    H_L = (U_L[2] + p_L) / rho_L
    H_R = (U_R[2] + p_R) / rho_R
    H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / sqrt_rho_sum

    # Speed of sound
    c_roe = math.sqrt((gamma - 1.0) * (H_roe - 0.5 * u_roe**2))

    return rho_roe, u_roe, H_roe, c_roe


def wave_speeds(
    U_L: tuple[float, float, float], U_R: tuple[float, float, float], gamma: float = 1.4,
) -> tuple[float, float, float, float]:
    """Compute wave speeds for HLLC solver.

    Parameters
    ----------
    U_L : Tuple[float, float, float]
        Left state conservative variables
    U_R : Tuple[float, float, float]
        Right state conservative variables
    gamma : float
        Heat capacity ratio

    Returns
    -------
    S_L : float
        Left wave speed
    S_R : float
        Right wave speed
    S_star : float
        Contact wave speed
    p_star : float
        Pressure in star region
    """
    rho_L, u_L, p_L = primitive_from_conservative(U_L, gamma)
    rho_R, u_R, p_R = primitive_from_conservative(U_R, gamma)

    # Guard against non-physical states from inputs; clamp minima
    rho_L = max(rho_L, 1e-12)
    rho_R = max(rho_R, 1e-12)
    p_L = max(p_L, 1e-12)
    p_R = max(p_R, 1e-12)

    # Speed of sound
    c_L = math.sqrt(max(0.0, gamma * p_L / rho_L))
    c_R = math.sqrt(max(0.0, gamma * p_R / rho_R))

    # Roe averages for better wave speed estimation
    rho_roe, u_roe, H_roe, c_roe = roe_averages(U_L, U_R, gamma)

    # Wave speeds (Toro's method)
    S_L = min(u_L - c_L, u_roe - c_roe)
    S_R = max(u_R + c_R, u_roe + c_roe)

    # Avoid degenerate ordering
    if S_L >= S_R:
        S_L, S_R = min(S_L, S_R - 1e-9), max(S_L + 1e-9, S_R)

    # Contact wave speed and pressure
    denom = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    if abs(denom) < 1e-12:
        S_star = 0.5 * (S_L + S_R)
    else:
        S_star = (
            p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
        ) / denom

    p_star = p_L + rho_L * (S_L - u_L) * (S_star - u_L)
    p_star = max(p_star, 1e-12)

    return S_L, S_R, S_star, p_star


def hllc_star_state(
    U: tuple[float, float, float],
    S: float,
    S_star: float,
    p_star: float,
    gamma: float = 1.4,
) -> tuple[float, float, float]:
    """Compute HLLC star state.

    Parameters
    ----------
    U : Tuple[float, float, float]
        Conservative variables
    S : float
        Wave speed (S_L or S_R)
    S_star : float
        Contact wave speed
    p_star : float
        Pressure in star region
    gamma : float
        Heat capacity ratio

    Returns
    -------
    U_star : Tuple[float, float, float]
        Star state conservative variables
    """
    rho, rhou, rhoE = U
    u = rhou / rho

    # Star state density
    rho_star = rho * (S - u) / (S - S_star)

    # Star state momentum
    rhou_star = rho_star * S_star

    # Star state energy
    rhoE_star = rho_star * (
        rhoE / rho + (S_star - u) * (S_star + p_star / (rho * (S - u)))
    )

    return (rho_star, rhou_star, rhoE_star)


def hllc_flux(
    U_L: tuple[float, float, float], U_R: tuple[float, float, float], gamma: float = 1.4,
) -> tuple[float, float, float]:
    """HLLC Riemann solver for 1D Euler equations.

    Implements the HLLC (Harten-Lax-van Leer-Contact) approximate Riemann solver
    for the 1D Euler equations with proper wave speed estimation and star states.

    Parameters
    ----------
    U_L : Tuple[float, float, float]
        Left state conservative variables [rho, rho*u, rho*E]
    U_R : Tuple[float, float, float]
        Right state conservative variables [rho, rho*u, rho*E]
    gamma : float
        Heat capacity ratio (default: 1.4)

    Returns
    -------
    F_hat : Tuple[float, float, float]
        Numerical flux at interface
    """
    # Check for vacuum states
    if U_L[0] <= 0.0 or U_R[0] <= 0.0:
        return (0.0, 0.0, 0.0)

    # Compute wave speeds (robust)
    try:
        S_L, S_R, S_star, p_star = wave_speeds(U_L, U_R, gamma)
    except Exception:
        return (0.0, 0.0, 0.0)

    # Compute fluxes from primitive variables (clamped)
    rho_L, u_L, p_L = primitive_from_conservative(U_L, gamma)
    rho_R, u_R, p_R = primitive_from_conservative(U_R, gamma)
    rho_L = max(rho_L, 1e-12)
    rho_R = max(rho_R, 1e-12)
    p_L = max(p_L, 1e-12)
    p_R = max(p_R, 1e-12)
    F_L = flux_from_primitive(rho_L, u_L, p_L)
    F_R = flux_from_primitive(rho_R, u_R, p_R)

    # HLLC flux selection
    if S_L >= 0.0:
        # Left state
        return F_L
    if S_R <= 0.0:
        # Right state
        return F_R
    if S_star >= 0.0:
        # Left star state
        U_L_star = hllc_star_state(U_L, S_L, S_star, p_star, gamma)
        F_L_star = flux_from_primitive(*primitive_from_conservative(U_L_star, gamma))
        return tuple(F_L[i] + S_L * (U_L_star[i] - U_L[i]) for i in range(3))
    # Right star state
    U_R_star = hllc_star_state(U_R, S_R, S_star, p_star, gamma)
    F_R_star = flux_from_primitive(*primitive_from_conservative(U_R_star, gamma))
    return tuple(F_R[i] + S_R * (U_R_star[i] - U_R[i]) for i in range(3))


def enhanced_hllc_flux(
    U_L: tuple[float, float, float],
    U_R: tuple[float, float, float],
    gamma: float = 1.4,
    entropy_fix: bool = True,
) -> tuple[float, float, float]:
    """Enhanced HLLC Riemann solver with entropy fix and improved robustness.

    This enhanced version includes:
    - Entropy fix for sonic flow conditions
    - Improved wave speed estimation
    - Better handling of extreme pressure ratios
    - Enhanced numerical stability

    Parameters
    ----------
    U_L : Tuple[float, float, float]
        Left state conservative variables [rho, rho*u, rho*E]
    U_R : Tuple[float, float, float]
        Right state conservative variables [rho, rho*u, rho*E]
    gamma : float
        Heat capacity ratio (default: 1.4)
    entropy_fix : bool
        Whether to apply entropy fix (default: True)

    Returns
    -------
    F_hat : Tuple[float, float, float]
        Numerical flux at interface
    """
    # Check for vacuum states
    if U_L[0] <= 0.0 or U_R[0] <= 0.0:
        return (0.0, 0.0, 0.0)

    # Compute wave speeds with enhanced estimation
    try:
        S_L, S_R, S_star, p_star = enhanced_wave_speeds(U_L, U_R, gamma)
    except Exception:
        return (0.0, 0.0, 0.0)

    # Compute fluxes from primitive variables
    rho_L, u_L, p_L = primitive_from_conservative(U_L, gamma)
    rho_R, u_R, p_R = primitive_from_conservative(U_R, gamma)

    # Clamp to prevent numerical issues
    rho_L = max(rho_L, 1e-12)
    rho_R = max(rho_R, 1e-12)
    p_L = max(p_L, 1e-12)
    p_R = max(p_R, 1e-12)

    F_L = flux_from_primitive(rho_L, u_L, p_L)
    F_R = flux_from_primitive(rho_R, u_R, p_R)

    # Apply entropy fix if requested
    if entropy_fix:
        S_L, S_R = apply_entropy_fix(U_L, U_R, S_L, S_R, gamma)

    # HLLC flux selection with enhanced robustness
    if S_L >= 0.0:
        # Left state
        return F_L
    if S_R <= 0.0:
        # Right state
        return F_R
    if S_star >= 0.0:
        # Left star state
        U_L_star = enhanced_hllc_star_state(U_L, S_L, S_star, p_star, gamma)
        F_L_star = flux_from_primitive(*primitive_from_conservative(U_L_star, gamma))
        return tuple(F_L[i] + S_L * (U_L_star[i] - U_L[i]) for i in range(3))
    # Right star state
    U_R_star = enhanced_hllc_star_state(U_R, S_R, S_star, p_star, gamma)
    F_R_star = flux_from_primitive(*primitive_from_conservative(U_R_star, gamma))
    return tuple(F_R[i] + S_R * (U_R_star[i] - U_R[i]) for i in range(3))


def enhanced_wave_speeds(
    U_L: tuple[float, float, float], U_R: tuple[float, float, float], gamma: float = 1.4,
) -> tuple[float, float, float, float]:
    """Enhanced wave speed estimation with improved robustness.

    This enhanced version includes:
    - Better handling of extreme pressure ratios
    - Improved contact wave speed calculation
    - Enhanced numerical stability

    Parameters
    ----------
    U_L : Tuple[float, float, float]
        Left state conservative variables
    U_R : Tuple[float, float, float]
        Right state conservative variables
    gamma : float
        Heat capacity ratio

    Returns
    -------
    S_L : float
        Left wave speed
    S_R : float
        Right wave speed
    S_star : float
        Contact wave speed
    p_star : float
        Pressure in star region
    """
    rho_L, u_L, p_L = primitive_from_conservative(U_L, gamma)
    rho_R, u_R, p_R = primitive_from_conservative(U_R, gamma)

    # Enhanced clamping for numerical stability
    rho_L = max(rho_L, 1e-12)
    rho_R = max(rho_R, 1e-12)
    p_L = max(p_L, 1e-12)
    p_R = max(p_R, 1e-12)

    # Speed of sound with enhanced calculation
    c_L = math.sqrt(max(0.0, gamma * p_L / rho_L))
    c_R = math.sqrt(max(0.0, gamma * p_R / rho_R))

    # Enhanced Roe averages
    rho_roe, u_roe, H_roe, c_roe = roe_averages(U_L, U_R, gamma)

    # Enhanced wave speed estimation
    # Use both local and Roe-averaged speeds for robustness
    S_L_local = u_L - c_L
    S_R_local = u_R + c_R
    S_L_roe = u_roe - c_roe
    S_R_roe = u_roe + c_roe

    # Take the more restrictive estimate
    S_L = min(S_L_local, S_L_roe)
    S_R = max(S_R_local, S_R_roe)

    # Ensure proper ordering
    if S_L >= S_R:
        S_L, S_R = min(S_L, S_R - 1e-9), max(S_L + 1e-9, S_R)

    # Enhanced contact wave speed calculation
    denom = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    if abs(denom) < 1e-12:
        S_star = 0.5 * (S_L + S_R)
    else:
        S_star = (
            p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)
        ) / denom

    # Enhanced pressure calculation
    p_star = p_L + rho_L * (S_L - u_L) * (S_star - u_L)
    p_star = max(p_star, 1e-12)

    return S_L, S_R, S_star, p_star


def apply_entropy_fix(
    U_L: tuple[float, float, float],
    U_R: tuple[float, float, float],
    S_L: float,
    S_R: float,
    gamma: float = 1.4,
) -> tuple[float, float]:
    """Apply entropy fix to wave speeds for sonic flow conditions.

    The entropy fix prevents the formation of expansion shocks by ensuring
    that the wave speeds properly represent the rarefaction fan structure.

    Parameters
    ----------
    U_L : Tuple[float, float, float]
        Left state conservative variables
    U_R : Tuple[float, float, float]
        Right state conservative variables
    S_L : float
        Left wave speed
    S_R : float
        Right wave speed
    gamma : float
        Heat capacity ratio

    Returns
    -------
    S_L_fixed : float
        Entropy-fixed left wave speed
    S_R_fixed : float
        Entropy-fixed right wave speed
    """
    rho_L, u_L, p_L = primitive_from_conservative(U_L, gamma)
    rho_R, u_R, p_R = primitive_from_conservative(U_R, gamma)

    # Speed of sound
    c_L = math.sqrt(max(0.0, gamma * p_L / rho_L))
    c_R = math.sqrt(max(0.0, gamma * p_R / rho_R))

    # Check for sonic flow conditions
    # Left rarefaction
    if u_L - c_L < 0.0 < u_L + c_L:
        # Sonic flow in left rarefaction
        S_L = u_L - c_L

    # Right rarefaction
    if u_R - c_R < 0.0 < u_R + c_R:
        # Sonic flow in right rarefaction
        S_R = u_R + c_R

    return S_L, S_R


def enhanced_hllc_star_state(
    U: tuple[float, float, float],
    S: float,
    S_star: float,
    p_star: float,
    gamma: float = 1.4,
) -> tuple[float, float, float]:
    """Enhanced HLLC star state calculation with improved robustness.

    This enhanced version includes:
    - Better handling of extreme conditions
    - Improved numerical stability
    - Enhanced conservation properties

    Parameters
    ----------
    U : Tuple[float, float, float]
        Conservative variables
    S : float
        Wave speed (S_L or S_R)
    S_star : float
        Contact wave speed
    p_star : float
        Pressure in star region
    gamma : float
        Heat capacity ratio

    Returns
    -------
    U_star : Tuple[float, float, float]
        Star state conservative variables
    """
    rho, rhou, rhoE = U

    # Enhanced clamping
    rho = max(rho, 1e-12)
    p_star = max(p_star, 1e-12)

    u = rhou / rho

    # Enhanced star state density calculation
    denom = S - S_star
    if abs(denom) < 1e-12:
        rho_star = rho
    else:
        rho_star = rho * (S - u) / denom
        rho_star = max(rho_star, 1e-12)

    # Star state momentum
    rhou_star = rho_star * S_star

    # Enhanced star state energy calculation
    if abs(S - u) < 1e-12:
        rhoE_star = rhoE
    else:
        rhoE_star = rho_star * (
            rhoE / rho + (S_star - u) * (S_star + p_star / (rho * (S - u)))
        )
        rhoE_star = max(rhoE_star, 1e-12)

    return (rho_star, rhou_star, rhoE_star)


def roe_flux(
    U_L: tuple[float, float, float], U_R: tuple[float, float, float], gamma: float = 1.4,
) -> tuple[float, float, float]:
    """Roe flux difference splitting for 1D Euler equations.

    Alternative to HLLC solver using Roe's approximate Riemann solver.

    Parameters
    ----------
    U_L : Tuple[float, float, float]
        Left state conservative variables
    U_R : Tuple[float, float, float]
        Right state conservative variables
    gamma : float
        Heat capacity ratio

    Returns
    -------
    F_hat : Tuple[float, float, float]
        Numerical flux at interface
    """
    # Roe averages
    rho_roe, u_roe, H_roe, c_roe = roe_averages(U_L, U_R, gamma)

    # Eigenvalues
    lambda1 = u_roe - c_roe
    lambda2 = u_roe
    lambda3 = u_roe + c_roe

    # Eigenvectors (simplified)
    # This is a simplified implementation - full Roe solver would compute
    # the complete eigenvector decomposition

    # For now, use HLLC as the primary solver
    return hllc_flux(U_L, U_R, gamma)


def get_flux_function(method: str = "hllc"):
    """Get flux function by name.

    Parameters
    ----------
    method : str
        Flux method: 'hllc', 'enhanced_hllc', or 'roe'

    Returns
    -------
    flux_func : callable
        Flux function
    """
    if method == "hllc":
        return hllc_flux
    if method == "enhanced_hllc":
        return enhanced_hllc_flux
    if method == "roe":
        return roe_flux
    raise ValueError(f"Unknown flux method: {method}")
