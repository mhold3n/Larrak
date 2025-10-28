from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List

from campro.constants import CASADI_PHYSICS_EPSILON
from campro.logging import get_logger

# Import CasADi for domain guards
try:
    import casadi as ca  # type: ignore
except ImportError:
    ca = None  # type: ignore

log = get_logger(__name__)


@dataclass
class WorkCalculationParameters:
    """Parameters for work calculation methods."""

    # Integration method
    integration_method: str  # 'trapezoidal', 'simpson', 'gauss'

    # Work calculation options
    include_heat_transfer: bool  # Include heat transfer work
    include_friction: bool  # Include friction work
    include_blowby: bool  # Include blow-by work

    # Efficiency calculations
    calculate_thermal_efficiency: bool
    calculate_mechanical_efficiency: bool
    calculate_volumetric_efficiency: bool


def smoothness_penalty(*, accel: Iterable[float], weights: Iterable[float]) -> float:
    """Smoothness penalty for acceleration profiles."""
    J = 0.0
    for a, w in zip(accel, weights):
        J += w * (a * a)
    return J


def indicated_work_trapezoidal(
    *, p_series: List[float], V_series: List[float],
) -> float:
    """Indicated work calculation using trapezoidal integration.

    W_ind = ∮ p dV

    Parameters
    ----------
    p_series : List[float]
        Pressure series [Pa]
    V_series : List[float]
        Volume series [m^3]

    Returns
    -------
    W_ind : float
        Indicated work [J]
    """
    if len(p_series) != len(V_series) or len(p_series) < 2:
        return 0.0

    W_ind = 0.0
    for i in range(len(p_series) - 1):
        # Trapezoidal rule: ∫ p dV ≈ 0.5 * (p_i + p_{i+1}) * (V_{i+1} - V_i)
        p_avg = 0.5 * (p_series[i] + p_series[i + 1])
        dV = V_series[i + 1] - V_series[i]
        W_ind += p_avg * dV

    return W_ind


def indicated_work_simpson(*, p_series: List[float], V_series: List[float]) -> float:
    """Indicated work calculation using Simpson's rule.

    More accurate than trapezoidal rule for smooth functions.

    Parameters
    ----------
    p_series : List[float]
        Pressure series [Pa]
    V_series : List[float]
        Volume series [m^3]

    Returns
    -------
    W_ind : float
        Indicated work [J]
    """
    if len(p_series) != len(V_series) or len(p_series) < 3:
        return indicated_work_trapezoidal(p_series, V_series)

    W_ind = 0.0
    n = len(p_series) - 1

    # Simpson's rule requires odd number of intervals
    if n % 2 == 0:
        # Even number of intervals - use Simpson's 3/8 rule for last interval
        for i in range(0, n - 2, 2):
            # Simpson's 1/3 rule
            h = V_series[i + 2] - V_series[i]
            W_ind += (h / 6.0) * (p_series[i] + 4.0 * p_series[i + 1] + p_series[i + 2])

        # Last interval with 3/8 rule
        if n > 2:
            h = V_series[n] - V_series[n - 3]
            W_ind += (h / 8.0) * (
                p_series[n - 3]
                + 3.0 * p_series[n - 2]
                + 3.0 * p_series[n - 1]
                + p_series[n]
            )
    else:
        # Odd number of intervals - use Simpson's 1/3 rule
        for i in range(0, n, 2):
            h = V_series[i + 2] - V_series[i]
            W_ind += (h / 6.0) * (p_series[i] + 4.0 * p_series[i + 1] + p_series[i + 2])

    return W_ind


def indicated_work_gauss(
    *, p_series: List[float], V_series: List[float], n_points: int = 3,
) -> float:
    """Indicated work calculation using Gauss quadrature.

    Most accurate for smooth functions.

    Parameters
    ----------
    p_series : List[float]
        Pressure series [Pa]
    V_series : List[float]
        Volume series [m^3]
    n_points : int
        Number of Gauss points per interval

    Returns
    -------
    W_ind : float
        Indicated work [J]
    """
    if len(p_series) != len(V_series) or len(p_series) < 2:
        return 0.0

    # Gauss-Legendre quadrature weights and points
    if n_points == 2:
        weights = [1.0, 1.0]
        points = [-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]
    elif n_points == 3:
        weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
        points = [-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)]
    else:
        # Fallback to Simpson's rule for other point counts
        return indicated_work_simpson(p_series, V_series)

    W_ind = 0.0
    for i in range(len(p_series) - 1):
        V_start, V_end = V_series[i], V_series[i + 1]
        p_start, p_end = p_series[i], p_series[i + 1]

        # Transform to [-1, 1] interval
        for w, xi in zip(weights, points):
            V_xi = 0.5 * (V_end + V_start) + 0.5 * (V_end - V_start) * xi
            p_xi = p_start + (p_end - p_start) * (V_xi - V_start) / (V_end - V_start)
            W_ind += w * p_xi * (V_end - V_start) * 0.5

    return W_ind


def indicated_work(
    *, p_series: List[float], V_series: List[float], method: str = "trapezoidal",
) -> float:
    """Indicated work calculation with multiple integration methods.

    Parameters
    ----------
    p_series : List[float]
        Pressure series [Pa]
    V_series : List[float]
        Volume series [m^3]
    method : str
        Integration method: 'trapezoidal', 'simpson', 'gauss'

    Returns
    -------
    W_ind : float
        Indicated work [J]
    """
    if method == "trapezoidal":
        return indicated_work_trapezoidal(p_series, V_series)
    if method == "simpson":
        return indicated_work_simpson(p_series, V_series)
    if method == "gauss":
        return indicated_work_gauss(p_series, V_series)
    raise ValueError(f"Unknown integration method: {method}")


def thermal_efficiency(*, W_ind: float, Q_in: float) -> float:
    """Thermal efficiency calculation.

    eta_th = W_ind / Q_in

    Parameters
    ----------
    W_ind : float
        Indicated work [J]
    Q_in : float
        Heat input [J]

    Returns
    -------
    eta_th : float
        Thermal efficiency [0-1]
    """
    if Q_in <= 0.0:
        return 0.0
    return W_ind / ca.fmax(Q_in, CASADI_PHYSICS_EPSILON)


def mechanical_efficiency(*, W_ind: float, W_friction: float) -> float:
    """Mechanical efficiency calculation.

    eta_mech = (W_ind - W_friction) / W_ind

    Parameters
    ----------
    W_ind : float
        Indicated work [J]
    W_friction : float
        Friction work [J]

    Returns
    -------
    eta_mech : float
        Mechanical efficiency [0-1]
    """
    if W_ind <= 0.0:
        return 0.0
    return (W_ind - W_friction) / ca.fmax(W_ind, CASADI_PHYSICS_EPSILON)


def volumetric_efficiency(*, m_actual: float, m_theoretical: float) -> float:
    """Volumetric efficiency calculation.

    eta_vol = m_actual / m_theoretical

    Parameters
    ----------
    m_actual : float
        Actual mass flow rate [kg/s]
    m_theoretical : float
        Theoretical mass flow rate [kg/s]

    Returns
    -------
    eta_vol : float
        Volumetric efficiency [0-1]
    """
    if m_theoretical <= 0.0:
        return 0.0
    return m_actual / ca.fmax(m_theoretical, CASADI_PHYSICS_EPSILON)


def scavenging_efficiency(*, m_fresh: float, m_total: float) -> float:
    """Scavenging efficiency calculation.

    eta_scav = m_fresh / m_total

    Parameters
    ----------
    m_fresh : float
        Fresh charge mass [kg]
    m_total : float
        Total charge mass [kg]

    Returns
    -------
    eta_scav : float
        Scavenging efficiency [0-1]
    """
    if m_total <= 0.0:
        return 0.0
    return m_fresh / ca.fmax(m_total, CASADI_PHYSICS_EPSILON)


def short_circuit_loss(*, m_short_circuit: float, m_total: float) -> float:
    """Short-circuit loss calculation.

    Short-circuit fraction = m_short_circuit / m_total

    Parameters
    ----------
    m_short_circuit : float
        Short-circuit mass [kg]
    m_total : float
        Total mass [kg]

    Returns
    -------
    short_circuit_fraction : float
        Short-circuit fraction [0-1]
    """
    if m_total <= 0.0:
        return 0.0
    return m_short_circuit / ca.fmax(m_total, CASADI_PHYSICS_EPSILON)


def cycle_analysis(
    *,
    p_series: List[float],
    V_series: List[float],
    T_series: List[float],
    m_series: List[float],
    Q_in: float,
    W_friction: float = 0.0,
    params: WorkCalculationParameters,
) -> Dict[str, float]:
    """Comprehensive cycle analysis.

    Computes all relevant efficiency metrics and work terms.

    Parameters
    ----------
    p_series : List[float]
        Pressure series [Pa]
    V_series : List[float]
        Volume series [m^3]
    T_series : List[float]
        Temperature series [K]
    m_series : List[float]
        Mass series [kg]
    Q_in : float
        Heat input [J]
    W_friction : float
        Friction work [J]
    params : WorkCalculationParameters
        Calculation parameters

    Returns
    -------
    analysis : Dict[str, float]
        Cycle analysis results
    """
    # Indicated work
    W_ind = indicated_work(
        p_series=p_series, V_series=V_series, method=params.integration_method,
    )

    # Thermal efficiency
    eta_th = (
        thermal_efficiency(W_ind=W_ind, Q_in=Q_in)
        if params.calculate_thermal_efficiency
        else 0.0
    )

    # Mechanical efficiency
    eta_mech = (
        mechanical_efficiency(W_ind=W_ind, W_friction=W_friction)
        if params.calculate_mechanical_efficiency
        else 0.0
    )

    # Volumetric efficiency (simplified)
    if params.calculate_volumetric_efficiency and len(m_series) > 1:
        m_actual = max(m_series) - min(m_series)
        m_theoretical = max(m_series)  # Simplified
        eta_vol = volumetric_efficiency(m_actual=m_actual, m_theoretical=m_theoretical)
    else:
        eta_vol = 0.0

    # Additional metrics
    p_max = max(p_series) if p_series else 0.0
    p_min = min(p_series) if p_series else 0.0
    T_max = max(T_series) if T_series else 0.0
    T_min = min(T_series) if T_series else 0.0

    # Compression ratio (simplified)
    V_max = max(V_series) if V_series else 0.0
    V_min = min(V_series) if V_series else 0.0
    CR = V_max / ca.fmax(V_min, CASADI_PHYSICS_EPSILON) if V_min > 0.0 else 0.0

    return {
        "indicated_work": W_ind,
        "thermal_efficiency": eta_th,
        "mechanical_efficiency": eta_mech,
        "volumetric_efficiency": eta_vol,
        "pressure_ratio": p_max / ca.fmax(p_min, CASADI_PHYSICS_EPSILON)
        if p_min > 0.0
        else 0.0,
        "temperature_ratio": T_max / ca.fmax(T_min, CASADI_PHYSICS_EPSILON)
        if T_min > 0.0
        else 0.0,
        "compression_ratio": CR,
        "friction_work": W_friction,
        "heat_input": Q_in,
    }


def indicated_work_surrogate(
    *, p_series: Iterable[float], dV_series: Iterable[float],
) -> float:
    """Simple trapezoidal surrogate for W_ind = ∮ p dV (legacy)."""
    import itertools

    J = 0.0
    for p, dV in itertools.zip_longest(p_series, dV_series, fillvalue=0.0):
        J += p * dV
    return J


def scavenging_penalty(*, short_circuit_fraction: float, weight: float) -> float:
    """Scavenging penalty for optimization."""
    if short_circuit_fraction <= 0.0 or weight <= 0.0:
        return 0.0
    return weight * short_circuit_fraction


def multi_objective_scalarization(
    *, objectives: Dict[str, float], weights: Dict[str, float],
) -> float:
    """Multi-objective scalarization.

    Combines multiple objectives into a single scalar objective function.

    Parameters
    ----------
    objectives : Dict[str, float]
        Objective function values
    weights : Dict[str, float]
        Objective weights

    Returns
    -------
    J_scalar : float
        Scalarized objective function
    """
    J_scalar = 0.0
    for name, value in objectives.items():
        weight = weights.get(name, 1.0)
        J_scalar += weight * value

    return J_scalar


def comprehensive_scavenging_objectives(
    states: Dict[str, List[Any]],
    controls: Dict[str, List[Any]],
    geometry: Dict[str, float],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Comprehensive scavenging objectives for two-stroke OP engines.

    Returns:
        Dictionary of objective terms
    """
    ca = _import_casadi()

    objectives = {}

    # 1. Scavenging Efficiency (fresh charge / total trapped)
    if "m_fresh_trapped" in states and "m_total_trapped" in states:
        m_fresh_trapped = states["m_fresh_trapped"][-1]  # At end of cycle
        m_total_trapped = states["m_total_trapped"][-1]
        eta_scav = m_fresh_trapped / (m_total_trapped + 1e-9)
        objectives["scavenging_efficiency"] = weights.get("eta_scav", 1.0) * eta_scav

    # 2. Trapping Efficiency (trapped mass / delivered mass)
    if "m_delivered" in states and "m_total_trapped" in states:
        m_delivered = states["m_delivered"][-1]
        m_total_trapped = states["m_total_trapped"][-1]
        eta_trap = m_total_trapped / (m_delivered + 1e-9)
        objectives["trapping_efficiency"] = weights.get("eta_trap", 1.0) * eta_trap

    # 3. Short-Circuit Loss (minimize fresh charge loss)
    if "m_short_circuit" in states and "m_delivered" in states:
        m_short_circuit = states["m_short_circuit"][-1]
        m_delivered = states["m_delivered"][-1]
        short_circuit_fraction = m_short_circuit / (m_delivered + 1e-9)
        objectives["short_circuit_penalty"] = (
            weights.get("short_circuit", 2.0) * short_circuit_fraction
        )

    # 4. Scavenging Quality (uniformity of fresh charge distribution)
    # This requires 1D model - placeholder for now
    objectives["scavenging_uniformity"] = weights.get("uniformity", 0.5) * 0.0

    # 5. Blow-Down Efficiency (exhaust gas removal)
    if "m_exhaust_removed" in states and "m_exhaust_initial" in states:
        m_exhaust_removed = states["m_exhaust_removed"][-1]
        m_exhaust_initial = states["m_exhaust_initial"][0]
        eta_blowdown = m_exhaust_removed / (m_exhaust_initial + 1e-9)
        objectives["blowdown_efficiency"] = (
            weights.get("eta_blowdown", 1.0) * eta_blowdown
        )

    return objectives


def scavenging_phase_timing_objectives(
    states: Dict[str, List[Any]],
    controls: Dict[str, List[Any]],
    targets: Dict[str, float],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Objectives for optimal scavenging phase timing.

    Args:
        states: State variables over time
        controls: Control variables over time
        targets: Target phase durations
        weights: Objective weights

    Returns:
        Dictionary of timing objective terms
    """
    ca = _import_casadi()

    objectives = {}

    # 1. Intake Phase Duration
    if "A_in" in controls:
        t_intake_start = find_phase_start(controls["A_in"], threshold=0.01)
        t_intake_end = find_phase_end(controls["A_in"], threshold=0.01)
        t_intake_duration = t_intake_end - t_intake_start
        t_intake_target = targets.get("intake_duration", 0.1)
        objectives["intake_timing"] = (
            weights.get("intake_timing", 1.0)
            * (t_intake_duration - t_intake_target) ** 2
        )

    # 2. Exhaust Phase Duration
    if "A_ex" in controls:
        t_exhaust_start = find_phase_start(controls["A_ex"], threshold=0.01)
        t_exhaust_end = find_phase_end(controls["A_ex"], threshold=0.01)
        t_exhaust_duration = t_exhaust_end - t_exhaust_start
        t_exhaust_target = targets.get("exhaust_duration", 0.1)
        objectives["exhaust_timing"] = (
            weights.get("exhaust_timing", 1.0)
            * (t_exhaust_duration - t_exhaust_target) ** 2
        )

    # 3. Overlap Phase (both valves open)
    if "A_in" in controls and "A_ex" in controls:
        overlap_penalty = calculate_overlap_penalty(controls["A_in"], controls["A_ex"])
        objectives["overlap_penalty"] = weights.get("overlap", 2.0) * overlap_penalty

    return objectives


def find_phase_start(valve_areas: List[Any], threshold: float) -> Any:
    """Find the start time of a valve opening phase."""
    ca = _import_casadi()

    # Simplified implementation - in practice, this would be more sophisticated
    # For now, return the first time when valve area exceeds threshold
    for i, area in enumerate(valve_areas):
        if area > threshold:
            return float(i)  # Simplified time representation

    return 0.0


def find_phase_end(valve_areas: List[Any], threshold: float) -> Any:
    """Find the end time of a valve opening phase."""
    ca = _import_casadi()

    # Simplified implementation - in practice, this would be more sophisticated
    # For now, return the last time when valve area exceeds threshold
    for i in range(len(valve_areas) - 1, -1, -1):
        if valve_areas[i] > threshold:
            return float(i)  # Simplified time representation

    return float(len(valve_areas) - 1)


def calculate_overlap_penalty(A_in: List[Any], A_ex: List[Any]) -> Any:
    """Calculate penalty for valve overlap."""
    ca = _import_casadi()

    # Simplified overlap penalty - in practice, this would be more sophisticated
    overlap_penalty = 0.0
    for a_in, a_ex in zip(A_in, A_ex):
        # Penalty increases with both valves being open
        overlap_penalty += a_in * a_ex

    return overlap_penalty


def enhanced_scavenging_efficiency(
    m_fresh_trapped: float,
    m_total_trapped: float,
    m_delivered: float,
    m_short_circuit: float,
) -> Dict[str, float]:
    """
    Enhanced scavenging efficiency calculation with multiple metrics.

    Args:
        m_fresh_trapped: Fresh charge mass trapped [kg]
        m_total_trapped: Total mass trapped [kg]
        m_delivered: Total mass delivered [kg]
        m_short_circuit: Fresh charge short-circuit mass [kg]

    Returns:
        Dictionary of scavenging efficiency metrics
    """
    metrics = {}

    # Scavenging efficiency (fresh charge / total trapped)
    metrics["scavenging_efficiency"] = m_fresh_trapped / (m_total_trapped + 1e-9)

    # Trapping efficiency (trapped mass / delivered mass)
    metrics["trapping_efficiency"] = m_total_trapped / (m_delivered + 1e-9)

    # Short-circuit fraction
    metrics["short_circuit_fraction"] = m_short_circuit / (m_delivered + 1e-9)

    # Fresh charge purity
    metrics["fresh_charge_purity"] = m_fresh_trapped / (
        m_fresh_trapped + m_total_trapped - m_fresh_trapped + 1e-9
    )

    return metrics


def blowdown_efficiency(
    m_exhaust_removed: float,
    m_exhaust_initial: float,
) -> float:
    """
    Blow-down efficiency calculation.

    Args:
        m_exhaust_removed: Exhaust gas mass removed [kg]
        m_exhaust_initial: Initial exhaust gas mass [kg]

    Returns:
        Blow-down efficiency [0-1]
    """
    if m_exhaust_initial <= 0.0:
        return 0.0

    return m_exhaust_removed / ca.fmax(m_exhaust_initial, CASADI_PHYSICS_EPSILON)


def scavenging_quality_index(
    fresh_charge_distribution: List[float],
    target_distribution: List[float] = None,
) -> float:
    """
    Scavenging quality index based on fresh charge distribution uniformity.

    Args:
        fresh_charge_distribution: Fresh charge mass fraction distribution
        target_distribution: Target distribution (uniform if None)

    Returns:
        Quality index [0-1], where 1.0 is perfect uniformity
    """
    if not fresh_charge_distribution:
        return 0.0

    if target_distribution is None:
        # Target uniform distribution
        target_distribution = [
            1.0 / ca.fmax(len(fresh_charge_distribution), CASADI_PHYSICS_EPSILON),
        ] * len(fresh_charge_distribution)

    if len(fresh_charge_distribution) != len(target_distribution):
        return 0.0

    # Calculate coefficient of variation (inverse of uniformity)
    mean_fresh = sum(fresh_charge_distribution) / ca.fmax(
        len(fresh_charge_distribution), CASADI_PHYSICS_EPSILON,
    )
    if mean_fresh <= 0.0:
        return 0.0

    variance = sum((x - mean_fresh) ** 2 for x in fresh_charge_distribution) / ca.fmax(
        len(fresh_charge_distribution), CASADI_PHYSICS_EPSILON,
    )
    std_dev = variance**0.5
    cv = std_dev / ca.fmax(mean_fresh, CASADI_PHYSICS_EPSILON)

    # Quality index (higher is better, 1.0 for perfect uniformity)
    quality_index = 1.0 / (1.0 + cv)

    return quality_index


def get_objective_function(method: str = "indicated_work"):
    """Get objective function by name.

    Parameters
    ----------
    method : str
        Objective function method:
        - 'indicated_work': Indicated work maximization
        - 'thermal_efficiency': Thermal efficiency maximization
        - 'multi_objective': Multi-objective scalarization
        - 'smoothness': Smoothness penalty
        - 'scavenging': Comprehensive scavenging objectives
        - 'timing': Scavenging phase timing objectives

    Returns
    -------
    obj_func : callable
        Objective function
    """
    if method == "indicated_work":
        return indicated_work
    if method == "thermal_efficiency":
        return thermal_efficiency
    if method == "multi_objective":
        return multi_objective_scalarization
    if method == "smoothness":
        return smoothness_penalty
    if method == "scavenging":
        return comprehensive_scavenging_objectives
    if method == "timing":
        return scavenging_phase_timing_objectives
    raise ValueError(f"Unknown objective method: {method}")
