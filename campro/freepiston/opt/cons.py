"""
Comprehensive path constraints for OP engine optimization.

This module implements comprehensive path constraints for all states and controls
in the opposed-piston engine optimization problem.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from campro.logging import get_logger

log = get_logger(__name__)


def _import_casadi():
    try:
        import casadi as ca  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CasADi is required for constraint building") from exc
    return ca


def comprehensive_path_constraints(
    states: Dict[str, List[Any]],
    controls: Dict[str, List[Any]],
    bounds: Dict[str, float],
    grid: Any,
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Comprehensive path constraints for all states and controls.
    
    Args:
        states: Dictionary of state variables over time
        controls: Dictionary of control variables over time
        bounds: Bounds dictionary
        grid: Collocation grid
        
    Returns:
        g_path, lbg_path, ubg_path: Path constraints and bounds
    """
    ca = _import_casadi()
    g_path = []
    lbg_path = []
    ubg_path = []

    # Pressure constraints
    if "pressure" in states:
        for p in states["pressure"]:
            g_path.append(p)
            lbg_path.append(bounds.get("p_min", 1e3))
            ubg_path.append(bounds.get("p_max", 1e7))

    # Temperature constraints
    if "temperature" in states:
        for T in states["temperature"]:
            g_path.append(T)
            lbg_path.append(bounds.get("T_min", 200.0))
            ubg_path.append(bounds.get("T_max", 2000.0))

    # Piston clearance constraints
    if "x_L" in states and "x_R" in states:
        for xL, xR in zip(states["x_L"], states["x_R"]):
            gap = xR - xL
            g_path.append(gap)
            lbg_path.append(bounds.get("gap_min", 0.0008))
            ubg_path.append(ca.inf)

    # Valve rate constraints
    if "A_in" in controls and len(controls["A_in"]) > 1:
        for i in range(1, len(controls["A_in"])):
            dA_dt = (controls["A_in"][i] - controls["A_in"][i-1]) / grid.h
            g_path.append(dA_dt)
            lbg_path.append(-bounds.get("dA_dt_max", 0.02))
            ubg_path.append(bounds.get("dA_dt_max", 0.02))

    if "A_ex" in controls and len(controls["A_ex"]) > 1:
        for i in range(1, len(controls["A_ex"])):
            dA_dt = (controls["A_ex"][i] - controls["A_ex"][i-1]) / grid.h
            g_path.append(dA_dt)
            lbg_path.append(-bounds.get("dA_dt_max", 0.02))
            ubg_path.append(bounds.get("dA_dt_max", 0.02))

    # Piston velocity constraints
    if "v_L" in states and "v_R" in states:
        for vL, vR in zip(states["v_L"], states["v_R"]):
            g_path.extend([vL, vR])
            lbg_path.extend([-bounds.get("v_max", 50.0), -bounds.get("v_max", 50.0)])
            ubg_path.extend([bounds.get("v_max", 50.0), bounds.get("v_max", 50.0)])

    # Piston acceleration constraints
    if "v_L" in states and "v_R" in states and len(states["v_L"]) > 1:
        for i in range(1, len(states["v_L"])):
            aL = (states["v_L"][i] - states["v_L"][i-1]) / grid.h
            aR = (states["v_R"][i] - states["v_R"][i-1]) / grid.h
            g_path.extend([aL, aR])
            lbg_path.extend([-bounds.get("a_max", 1000.0), -bounds.get("a_max", 1000.0)])
            ubg_path.extend([bounds.get("a_max", 1000.0), bounds.get("a_max", 1000.0)])

    # Density constraints
    if "rho" in states:
        for rho in states["rho"]:
            g_path.append(rho)
            lbg_path.append(bounds.get("rho_min", 0.1))
            ubg_path.append(bounds.get("rho_max", 10.0))

    # Energy constraints
    if "E" in states:
        for E in states["E"]:
            g_path.append(E)
            lbg_path.append(bounds.get("E_min", 0.1))
            ubg_path.append(bounds.get("E_max", 100.0))

    # Velocity constraints
    if "u" in states:
        for u in states["u"]:
            g_path.append(u)
            lbg_path.append(bounds.get("u_min", -100.0))
            ubg_path.append(bounds.get("u_max", 100.0))

    return g_path, lbg_path, ubg_path


def pressure_constraints(
    pressure_states: List[Any],
    bounds: Dict[str, float],
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Pressure path constraints.
    
    Args:
        pressure_states: List of pressure states over time
        bounds: Bounds dictionary
        
    Returns:
        g_pressure, lbg_pressure, ubg_pressure: Pressure constraints and bounds
    """
    g_pressure = []
    lbg_pressure = []
    ubg_pressure = []

    for p in pressure_states:
        g_pressure.append(p)
        lbg_pressure.append(bounds.get("p_min", 1e3))
        ubg_pressure.append(bounds.get("p_max", 1e7))

    return g_pressure, lbg_pressure, ubg_pressure


def temperature_constraints(
    temperature_states: List[Any],
    bounds: Dict[str, float],
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Temperature path constraints.
    
    Args:
        temperature_states: List of temperature states over time
        bounds: Bounds dictionary
        
    Returns:
        g_temp, lbg_temp, ubg_temp: Temperature constraints and bounds
    """
    g_temp = []
    lbg_temp = []
    ubg_temp = []

    for T in temperature_states:
        g_temp.append(T)
        lbg_temp.append(bounds.get("T_min", 200.0))
        ubg_temp.append(bounds.get("T_max", 2000.0))

    return g_temp, lbg_temp, ubg_temp


def piston_clearance_constraints(
    x_L_states: List[Any],
    x_R_states: List[Any],
    bounds: Dict[str, float],
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Piston clearance path constraints.
    
    Args:
        x_L_states: List of left piston position states
        x_R_states: List of right piston position states
        bounds: Bounds dictionary
        
    Returns:
        g_clearance, lbg_clearance, ubg_clearance: Clearance constraints and bounds
    """
    ca = _import_casadi()
    g_clearance = []
    lbg_clearance = []
    ubg_clearance = []

    for xL, xR in zip(x_L_states, x_R_states):
        gap = xR - xL
        g_clearance.append(gap)
        lbg_clearance.append(bounds.get("gap_min", 0.0008))
        ubg_clearance.append(ca.inf)

    return g_clearance, lbg_clearance, ubg_clearance


def valve_rate_constraints(
    valve_states: List[Any],
    bounds: Dict[str, float],
    grid: Any,
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Valve rate path constraints.
    
    Args:
        valve_states: List of valve area states over time
        bounds: Bounds dictionary
        grid: Collocation grid
        
    Returns:
        g_rate, lbg_rate, ubg_rate: Rate constraints and bounds
    """
    g_rate = []
    lbg_rate = []
    ubg_rate = []

    if len(valve_states) > 1:
        for i in range(1, len(valve_states)):
            dA_dt = (valve_states[i] - valve_states[i-1]) / grid.h
            g_rate.append(dA_dt)
            lbg_rate.append(-bounds.get("dA_dt_max", 0.02))
            ubg_rate.append(bounds.get("dA_dt_max", 0.02))

    return g_rate, lbg_rate, ubg_rate


def piston_velocity_constraints(
    v_L_states: List[Any],
    v_R_states: List[Any],
    bounds: Dict[str, float],
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Piston velocity path constraints.
    
    Args:
        v_L_states: List of left piston velocity states
        v_R_states: List of right piston velocity states
        bounds: Bounds dictionary
        
    Returns:
        g_velocity, lbg_velocity, ubg_velocity: Velocity constraints and bounds
    """
    g_velocity = []
    lbg_velocity = []
    ubg_velocity = []

    for vL, vR in zip(v_L_states, v_R_states):
        g_velocity.extend([vL, vR])
        lbg_velocity.extend([-bounds.get("v_max", 50.0), -bounds.get("v_max", 50.0)])
        ubg_velocity.extend([bounds.get("v_max", 50.0), bounds.get("v_max", 50.0)])

    return g_velocity, lbg_velocity, ubg_velocity


def piston_acceleration_constraints(
    v_L_states: List[Any],
    v_R_states: List[Any],
    bounds: Dict[str, float],
    grid: Any,
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Piston acceleration path constraints.
    
    Args:
        v_L_states: List of left piston velocity states
        v_R_states: List of right piston velocity states
        bounds: Bounds dictionary
        grid: Collocation grid
        
    Returns:
        g_accel, lbg_accel, ubg_accel: Acceleration constraints and bounds
    """
    g_accel = []
    lbg_accel = []
    ubg_accel = []

    if len(v_L_states) > 1:
        for i in range(1, len(v_L_states)):
            aL = (v_L_states[i] - v_L_states[i-1]) / grid.h
            aR = (v_R_states[i] - v_R_states[i-1]) / grid.h
            g_accel.extend([aL, aR])
            lbg_accel.extend([-bounds.get("a_max", 1000.0), -bounds.get("a_max", 1000.0)])
            ubg_accel.extend([bounds.get("a_max", 1000.0), bounds.get("a_max", 1000.0)])

    return g_accel, lbg_accel, ubg_accel


def density_constraints(
    density_states: List[Any],
    bounds: Dict[str, float],
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Density path constraints.
    
    Args:
        density_states: List of density states over time
        bounds: Bounds dictionary
        
    Returns:
        g_density, lbg_density, ubg_density: Density constraints and bounds
    """
    g_density = []
    lbg_density = []
    ubg_density = []

    for rho in density_states:
        g_density.append(rho)
        lbg_density.append(bounds.get("rho_min", 0.1))
        ubg_density.append(bounds.get("rho_max", 10.0))

    return g_density, lbg_density, ubg_density


def energy_constraints(
    energy_states: List[Any],
    bounds: Dict[str, float],
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Energy path constraints.
    
    Args:
        energy_states: List of energy states over time
        bounds: Bounds dictionary
        
    Returns:
        g_energy, lbg_energy, ubg_energy: Energy constraints and bounds
    """
    g_energy = []
    lbg_energy = []
    ubg_energy = []

    for E in energy_states:
        g_energy.append(E)
        lbg_energy.append(bounds.get("E_min", 0.1))
        ubg_energy.append(bounds.get("E_max", 100.0))

    return g_energy, lbg_energy, ubg_energy


def velocity_constraints(
    velocity_states: List[Any],
    bounds: Dict[str, float],
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Velocity path constraints.
    
    Args:
        velocity_states: List of velocity states over time
        bounds: Bounds dictionary
        
    Returns:
        g_velocity, lbg_velocity, ubg_velocity: Velocity constraints and bounds
    """
    g_velocity = []
    lbg_velocity = []
    ubg_velocity = []

    for u in velocity_states:
        g_velocity.append(u)
        lbg_velocity.append(bounds.get("u_min", -100.0))
        ubg_velocity.append(bounds.get("u_max", 100.0))

    return g_velocity, lbg_velocity, ubg_velocity


def combustion_timing_constraints(
    Q_comb_states: List[Any],
    bounds: Dict[str, float],
    timing_params: Dict[str, float],
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Combustion timing path constraints.
    
    Args:
        Q_comb_states: List of combustion heat release states
        bounds: Bounds dictionary
        timing_params: Timing parameters
        
    Returns:
        g_combustion, lbg_combustion, ubg_combustion: Combustion constraints and bounds
    """
    g_combustion = []
    lbg_combustion = []
    ubg_combustion = []

    for Q_comb in Q_comb_states:
        # Combustion heat release should be non-negative and within bounds.
        # Avoid Python conditionals on symbolic expressions; encode constraints directly.
        g_combustion.append(Q_comb)
        lbg_combustion.append(0.0)
        ubg_combustion.append(bounds.get("Q_comb_max", 10000.0))

    return g_combustion, lbg_combustion, ubg_combustion


def scavenging_constraints(
    scavenging_states: Dict[str, List[Any]],
    bounds: Dict[str, float],
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Scavenging path constraints.
    
    Args:
        scavenging_states: Dictionary of scavenging state variables
        bounds: Bounds dictionary
        
    Returns:
        g_scavenging, lbg_scavenging, ubg_scavenging: Scavenging constraints and bounds
    """
    g_scavenging = []
    lbg_scavenging = []
    ubg_scavenging = []

    # Fresh charge fraction constraints
    if "y_fresh" in scavenging_states:
        for y_fresh in scavenging_states["y_fresh"]:
            g_scavenging.append(y_fresh)
            lbg_scavenging.append(0.0)
            ubg_scavenging.append(1.0)

    # Mass delivery constraints
    if "m_delivered" in scavenging_states:
        for m_del in scavenging_states["m_delivered"]:
            g_scavenging.append(m_del)
            lbg_scavenging.append(0.0)
            ubg_scavenging.append(bounds.get("m_delivered_max", 1.0))

    # Short-circuit constraints
    if "m_short_circuit" in scavenging_states:
        for m_sc in scavenging_states["m_short_circuit"]:
            g_scavenging.append(m_sc)
            lbg_scavenging.append(0.0)
            ubg_scavenging.append(bounds.get("m_short_circuit_max", 0.1))

    return g_scavenging, lbg_scavenging, ubg_scavenging


def wall_temperature_constraints(
    T_wall_states: List[Any],
    bounds: Dict[str, float],
) -> Tuple[List[Any], List[float], List[float]]:
    """
    Wall temperature path constraints.
    
    Args:
        T_wall_states: List of wall temperature states
        bounds: Bounds dictionary
        
    Returns:
        g_wall, lbg_wall, ubg_wall: Wall temperature constraints and bounds
    """
    g_wall = []
    lbg_wall = []
    ubg_wall = []

    for T_wall in T_wall_states:
        g_wall.append(T_wall)
        lbg_wall.append(bounds.get("T_wall_min", 250.0))
        ubg_wall.append(bounds.get("T_wall_max", 800.0))

    return g_wall, lbg_wall, ubg_wall


def validate_constraint_bounds(
    bounds: Dict[str, float],
) -> bool:
    """
    Validate constraint bounds for consistency.
    
    Args:
        bounds: Bounds dictionary
        
    Returns:
        True if bounds are valid, False otherwise
    """
    # Check pressure bounds
    if "p_min" in bounds and "p_max" in bounds:
        if bounds["p_min"] >= bounds["p_max"]:
            log.error("Invalid pressure bounds: p_min >= p_max")
            return False

    # Check temperature bounds
    if "T_min" in bounds and "T_max" in bounds:
        if bounds["T_min"] >= bounds["T_max"]:
            log.error("Invalid temperature bounds: T_min >= T_max")
            return False

    # Check density bounds
    if "rho_min" in bounds and "rho_max" in bounds:
        if bounds["rho_min"] >= bounds["rho_max"]:
            log.error("Invalid density bounds: rho_min >= rho_max")
            return False

    # Check velocity bounds
    if "v_min" in bounds and "v_max" in bounds:
        if bounds["v_min"] >= bounds["v_max"]:
            log.error("Invalid velocity bounds: v_min >= v_max")
            return False

    # Check acceleration bounds
    if "a_min" in bounds and "a_max" in bounds:
        if bounds["a_min"] >= bounds["a_max"]:
            log.error("Invalid acceleration bounds: a_min >= a_max")
            return False

    return True


def get_default_bounds() -> Dict[str, float]:
    """
    Get default constraint bounds.
    
    Returns:
        Dictionary of default bounds
    """
    return {
        # Pressure bounds [Pa]
        "p_min": 1e3,
        "p_max": 1e7,

        # Temperature bounds [K]
        "T_min": 200.0,
        "T_max": 2000.0,

        # Density bounds [kg/m^3]
        "rho_min": 0.1,
        "rho_max": 10.0,

        # Velocity bounds [m/s]
        "v_min": -50.0,
        "v_max": 50.0,
        "u_min": -100.0,
        "u_max": 100.0,

        # Acceleration bounds [m/s^2]
        "a_min": -1000.0,
        "a_max": 1000.0,

        # Piston position bounds [m]
        "xL_min": -0.1,
        "xL_max": 0.1,
        "xR_min": 0.0,
        "xR_max": 0.2,

        # Clearance bounds [m]
        "gap_min": 0.0008,

        # Valve area bounds [m^2]
        "Ain_max": 0.01,
        "Aex_max": 0.01,

        # Valve rate bounds [m^2/s]
        "dA_dt_max": 0.02,

        # Energy bounds [J/kg]
        "E_min": 0.1,
        "E_max": 100.0,

        # Combustion bounds [W]
        "Q_comb_max": 10000.0,

        # Wall temperature bounds [K]
        "T_wall_min": 250.0,
        "T_wall_max": 800.0,

        # Scavenging bounds
        "m_delivered_max": 1.0,
        "m_short_circuit_max": 0.1,
    }
