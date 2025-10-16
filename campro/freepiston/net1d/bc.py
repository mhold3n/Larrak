from __future__ import annotations

import math
from typing import Optional, Tuple

from campro.logging import get_logger

log = get_logger(__name__)


def characteristic_variables(rho: float, u: float, p: float, gamma: float = 1.4) -> Tuple[float, float, float]:
    """Compute characteristic variables for 1D Euler equations.
    
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
    R1 : float
        First characteristic variable (entropy)
    R2 : float
        Second characteristic variable (Riemann invariant)
    R3 : float
        Third characteristic variable (Riemann invariant)
    """
    c = math.sqrt(gamma * p / rho)  # Speed of sound

    # Characteristic variables
    R1 = p / (rho ** gamma)  # Entropy
    R2 = u + 2 * c / (gamma - 1.0)  # Right-going Riemann invariant
    R3 = u - 2 * c / (gamma - 1.0)  # Left-going Riemann invariant

    return R1, R2, R3


def primitive_from_characteristics(R1: float, R2: float, R3: float, gamma: float = 1.4) -> Tuple[float, float, float]:
    """Convert characteristic variables back to primitive variables.
    
    Parameters
    ----------
    R1 : float
        First characteristic variable (entropy)
    R2 : float
        Second characteristic variable (Riemann invariant)
    R3 : float
        Third characteristic variable (Riemann invariant)
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
    # Velocity and speed of sound from Riemann invariants
    u = 0.5 * (R2 + R3)
    c = 0.25 * (gamma - 1.0) * (R2 - R3)

    # Pressure and density from entropy and speed of sound
    p = (c ** 2) * (R1 ** (1.0 / gamma)) / gamma
    rho = (p / R1) ** (1.0 / gamma)

    return rho, u, p


def non_reflecting_inlet_bc(U_interior: Tuple[float, float, float],
                           p_target: float, T_target: float,
                           gamma: float = 1.4) -> Tuple[float, float, float]:
    """Non-reflecting inlet boundary condition using characteristics.
    
    Prescribes target pressure and temperature while allowing outgoing waves
    to pass through without reflection.
    
    Parameters
    ----------
    U_interior : Tuple[float, float, float]
        Interior state conservative variables
    p_target : float
        Target pressure [Pa]
    T_target : float
        Target temperature [K]
    gamma : float
        Heat capacity ratio
        
    Returns
    -------
    U_boundary : Tuple[float, float, float]
        Boundary state conservative variables
    """
    # Convert interior state to primitive variables
    rho_int, u_int, p_int = _conservative_to_primitive(U_interior, gamma)

    # Compute interior characteristics
    R1_int, R2_int, R3_int = characteristic_variables(rho_int, u_int, p_int, gamma)

    # Target state (assuming zero velocity at inlet)
    R = 287.0  # Gas constant for air
    rho_target = p_target / (R * T_target)
    u_target = 0.0

    # Compute target characteristics
    R1_target, R2_target, R3_target = characteristic_variables(rho_target, u_target, p_target, gamma)

    # Non-reflecting condition: use incoming characteristic from target,
    # outgoing characteristics from interior
    R1_bc = R1_target  # Entropy from target
    R2_bc = R2_int     # Right-going wave from interior
    R3_bc = R3_target  # Left-going wave from target

    # Convert back to primitive variables
    rho_bc, u_bc, p_bc = primitive_from_characteristics(R1_bc, R2_bc, R3_bc, gamma)

    # Convert to conservative variables
    return _primitive_to_conservative(rho_bc, u_bc, p_bc, gamma)


def non_reflecting_outlet_bc(U_interior: Tuple[float, float, float],
                            p_target: Optional[float] = None,
                            gamma: float = 1.4) -> Tuple[float, float, float]:
    """Non-reflecting outlet boundary condition using characteristics.
    
    Allows outgoing waves to pass through without reflection.
    Optionally prescribes target pressure.
    
    Parameters
    ----------
    U_interior : Tuple[float, float, float]
        Interior state conservative variables
    p_target : float, optional
        Target pressure [Pa]. If None, extrapolates from interior.
    gamma : float
        Heat capacity ratio
        
    Returns
    -------
    U_boundary : Tuple[float, float, float]
        Boundary state conservative variables
    """
    # Convert interior state to primitive variables
    rho_int, u_int, p_int = _conservative_to_primitive(U_interior, gamma)

    # Compute interior characteristics
    R1_int, R2_int, R3_int = characteristic_variables(rho_int, u_int, p_int, gamma)

    if p_target is None:
        # Extrapolate pressure
        p_bc = p_int
    else:
        p_bc = p_target

    # Non-reflecting condition: use outgoing characteristics from interior,
    # incoming characteristic from target pressure
    R1_bc = R1_int  # Entropy from interior
    R2_bc = R2_int  # Right-going wave from interior
    R3_bc = R3_int  # Left-going wave from interior

    # If pressure is prescribed, adjust characteristics to match
    if p_target is not None:
        # Adjust entropy to match target pressure
        c_int = math.sqrt(gamma * p_int / rho_int)
        c_target = math.sqrt(gamma * p_target / rho_int)
        R1_bc = p_target / (rho_int ** gamma)

    # Convert back to primitive variables
    rho_bc, u_bc, p_bc = primitive_from_characteristics(R1_bc, R2_bc, R3_bc, gamma)

    # Convert to conservative variables
    return _primitive_to_conservative(rho_bc, u_bc, p_bc, gamma)


def pressure_velocity_switching_bc(U_interior: Tuple[float, float, float],
                                  bc_type: str, bc_value: float,
                                  gamma: float = 1.4) -> Tuple[float, float, float]:
    """Pressure/velocity switching boundary condition.
    
    Automatically switches between pressure and velocity boundary conditions
    based on flow direction.
    
    Parameters
    ----------
    U_interior : Tuple[float, float, float]
        Interior state conservative variables
    bc_type : str
        Boundary condition type: 'pressure' or 'velocity'
    bc_value : float
        Boundary condition value [Pa] or [m/s]
    gamma : float
        Heat capacity ratio
        
    Returns
    -------
    U_boundary : Tuple[float, float, float]
        Boundary state conservative variables
    """
    rho_int, u_int, p_int = _conservative_to_primitive(U_interior, gamma)

    # Determine flow direction
    if u_int > 0.0:
        # Outflow: prescribe pressure
        p_bc = bc_value if bc_type == "pressure" else p_int
        u_bc = u_int  # Extrapolate velocity
    else:
        # Inflow: prescribe velocity
        u_bc = bc_value if bc_type == "velocity" else u_int
        p_bc = p_int  # Extrapolate pressure

    # Compute density from ideal gas law
    R = 287.0  # Gas constant for air
    T_int = p_int / (R * rho_int)
    rho_bc = p_bc / (R * T_int)

    return _primitive_to_conservative(rho_bc, u_bc, p_bc, gamma)


def _conservative_to_primitive(U: Tuple[float, float, float], gamma: float = 1.4) -> Tuple[float, float, float]:
    """Convert conservative to primitive variables."""
    rho, rhou, rhoE = U

    if rho <= 0.0:
        return 0.0, 0.0, 0.0

    u = rhou / rho
    e = rhoE / rho - 0.5 * u**2
    p = (gamma - 1.0) * rho * e

    return rho, u, p


def _primitive_to_conservative(rho: float, u: float, p: float, gamma: float = 1.4) -> Tuple[float, float, float]:
    """Convert primitive to conservative variables."""
    e = p / ((gamma - 1.0) * rho)
    E = e + 0.5 * u**2

    return (rho, rho * u, rho * E)


def inlet_bc(p_in: float, T_in: float, rho_in: float) -> Tuple[float, float, float]:
    """Simple inlet boundary condition for 1D solver (legacy)."""
    # Conservative variables at inlet
    u_in = 0.0  # placeholder
    rhoE_in = rho_in * (1.5 * 287.0 * T_in)  # simple ideal gas
    return (rho_in, rho_in * u_in, rhoE_in)


def outlet_bc(p_out: float) -> Tuple[float, float, float]:
    """Simple outlet boundary condition for 1D solver (legacy)."""
    # Simple outlet: extrapolate from interior
    return (0.0, 0.0, 0.0)  # placeholder


def get_boundary_condition(method: str = "non_reflecting"):
    """Get boundary condition function by name.
    
    Parameters
    ----------
    method : str
        Boundary condition method:
        - 'non_reflecting': Non-reflecting characteristic-based
        - 'pressure_velocity': Pressure/velocity switching
        - 'simple': Simple inlet/outlet (legacy)
        
    Returns
    -------
    bc_func : callable
        Boundary condition function
    """
    if method == "non_reflecting":
        return non_reflecting_inlet_bc, non_reflecting_outlet_bc
    if method == "pressure_velocity":
        return pressure_velocity_switching_bc
    if method == "simple":
        return inlet_bc, outlet_bc
    raise ValueError(f"Unknown boundary condition method: {method}")
