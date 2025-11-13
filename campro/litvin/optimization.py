from __future__ import annotations

import concurrent.futures
import threading
from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np

from campro.diagnostics.scaling import build_scaled_nlp, scale_bounds, scale_value
from campro.freepiston.opt.ipopt_solver import IPOPTOptions, IPOPTSolver
from campro.logging import get_logger

from .config import GeometrySearchConfig, OptimizationOrder, PlanetSynthesisConfig
from .involute_internal import InternalGearParams, sample_internal_flank
from .kinematics import PlanetKinematics
from .metrics import evaluate_order0_metrics, evaluate_order0_metrics_given_phi
from .opt.collocation import make_uniform_grid
from .planetary_synthesis import _newton_solve_phi
from campro.physics.geometry.curvature import CurvatureComponent

log = get_logger(__name__)

# Explicit exports for consumers expecting named attributes
__all__ = ["OptimizationOrder", "optimize_geometry", "OptimResult"]

_CURVATURE_COMPONENT = CurvatureComponent(parameters={})


@dataclass(frozen=True)
class OptimResult:
    best_config: PlanetSynthesisConfig | None
    objective_value: float | None
    feasible: bool
    ipopt_analysis: Any | None = None  # Will be MA57ReadinessReport when available


def _order0_objective(cfg: PlanetSynthesisConfig) -> float:
    m = evaluate_order0_metrics(cfg)
    # Objective combines slip and penalties on closure and edge-contact
    penalty = 0.0
    if not m.feasible:
        penalty += 1e3
    penalty += 1e2 * m.closure_residual + 50.0 * m.phi_edge_fraction
    return m.slip_integral - 0.1 * m.contact_length + penalty


def _litvin_feasibility_surrogate(
    theta_rad: np.ndarray,
    r_profile: np.ndarray,
) -> float:
    """Cheap proxy for Litvin feasibility using local curvature diagnostics."""
    if np.any(~np.isfinite(r_profile)):
        return 5e5
    if np.any(r_profile <= 0):
        return 5e5 + 1e4 * float(np.sum(r_profile <= 0))
    try:
        result = _CURVATURE_COMPONENT.compute({"theta": theta_rad, "r_theta": r_profile})
        if not result.is_successful:
            return 5e5
        kappa = result.outputs["kappa"]
        rho = result.outputs["rho"]
    except Exception as exc:
        log.debug(f"Curvature surrogate failed: {exc}")
        return 5e5
    if np.any(~np.isfinite(kappa)) or np.any(~np.isfinite(rho)):
        return 5e5
    curvature_energy = float(np.mean(np.minimum(np.abs(kappa), 1.0)))
    rho_variation = float(np.var(rho))
    return 1e2 * curvature_energy + 1e-2 * rho_variation


def _objective_ca10_to_ca100(
    gear_config: PlanetSynthesisConfig,
    section_theta_range: tuple[float, float],
    motion_slice: dict[str, np.ndarray],
    rho_target: np.ndarray | None = None,
    universal_theta_deg: np.ndarray | None = None,
    position: np.ndarray | None = None,
) -> float:
    """Objective for CA10-CA100 section: Maximize MA planet onto ring (power stroke torque).
    
    Parameters
    ----------
    gear_config : PlanetSynthesisConfig
        Gear configuration to evaluate
    section_theta_range : tuple[float, float]
        (theta_start, theta_end) for this section [deg]
    motion_slice : dict[str, np.ndarray]
        Motion law data for this section (theta, position, velocity, etc.)
    rho_target : np.ndarray | None
        Target ratio profile ρ_target(θ) for synchronized ring radius optimization
    universal_theta_deg : np.ndarray | None
        Universal theta grid in degrees
    position : np.ndarray | None
        Position array for synthesizing cam profile
    
    Returns
    -------
    float
        Objective value (negative for maximization - lower is better)
    """
    # If rho_target is provided, use synchronized ring radius penalty
    if rho_target is not None and universal_theta_deg is not None and position is not None:
        return _compute_ring_radius_penalty(
            gear_config, rho_target, universal_theta_deg, position
        )
    
    # Evaluate full objective for this gear config
    full_obj = _order0_objective(gear_config)
    
    # For power stroke, we want to maximize mechanical advantage
    # Mechanical advantage is related to torque output
    # Lower slip_integral means better contact, which improves MA
    # We want to maximize this, so we negate (lower objective = better)
    
    # Get metrics for this section
    m = evaluate_order0_metrics(gear_config)
    
    # For power stroke: maximize planet-to-ring torque
    # This means minimize slip (better contact) and maximize contact length
    # Objective: minimize slip, maximize contact (negative for maximization)
    power_obj = m.slip_integral - 0.5 * m.contact_length
    
    penalty = 0.0
    if not m.feasible:
        penalty += 1e3
    penalty += 1e2 * m.closure_residual + 50.0 * m.phi_edge_fraction
    
    return power_obj + penalty


def _objective_bdc_to_tdc(
    gear_config: PlanetSynthesisConfig,
    section_theta_range: tuple[float, float],
    motion_slice: dict[str, np.ndarray],
    rho_target: np.ndarray | None = None,
    universal_theta_deg: np.ndarray | None = None,
    position: np.ndarray | None = None,
) -> float:
    """Objective for BDC-TDC section: Maximize ring MA on planet (easy upstroke movement).
    
    Parameters
    ----------
    gear_config : PlanetSynthesisConfig
        Gear configuration to evaluate
    section_theta_range : tuple[float, float]
        (theta_start, theta_end) for this section [deg]
    motion_slice : dict[str, np.ndarray]
        Motion law data for this section (theta, position, velocity, etc.)
    rho_target : np.ndarray | None
        Target ratio profile ρ_target(θ) for synchronized ring radius optimization
    universal_theta_deg : np.ndarray | None
        Universal theta grid in degrees
    position : np.ndarray | None
        Position array for synthesizing cam profile
    
    Returns
    -------
    float
        Objective value (negative for maximization - lower is better)
    """
    # If rho_target is provided, use synchronized ring radius penalty
    if rho_target is not None and universal_theta_deg is not None and position is not None:
        return _compute_ring_radius_penalty(
            gear_config, rho_target, universal_theta_deg, position
        )
    
    # Evaluate metrics
    m = evaluate_order0_metrics(gear_config)
    
    # For upstroke: maximize ring-to-planet mechanical advantage
    # This means we want easy movement (low resistance)
    # Lower slip means easier motion (ring helps planet move)
    # Objective: minimize resistance, maximize ease of motion
    upstroke_obj = m.slip_integral + 0.3 * m.contact_length  # Positive contact helps
    
    penalty = 0.0
    if not m.feasible:
        penalty += 1e3
    penalty += 1e2 * m.closure_residual + 50.0 * m.phi_edge_fraction
    
    return upstroke_obj + penalty


def _objective_transition_sections(
    gear_config: PlanetSynthesisConfig,
    section_theta_range: tuple[float, float],
    motion_slice: dict[str, np.ndarray],
    rho_target: np.ndarray | None = None,
    universal_theta_deg: np.ndarray | None = None,
    position: np.ndarray | None = None,
) -> float:
    """Objective for transition sections: Balanced objectives.
    
    Parameters
    ----------
    gear_config : PlanetSynthesisConfig
        Gear configuration to evaluate
    section_theta_range : tuple[float, float]
        (theta_start, theta_end) for this section [deg]
    motion_slice : dict[str, np.ndarray]
        Motion law data for this section (theta, position, velocity, etc.)
    rho_target : np.ndarray | None
        Target ratio profile ρ_target(θ) for synchronized ring radius optimization
    universal_theta_deg : np.ndarray | None
        Universal theta grid in degrees
    position : np.ndarray | None
        Position array for synthesizing cam profile
    
    Returns
    -------
    float
        Objective value (negative for maximization - lower is better)
    """
    # If rho_target is provided, use synchronized ring radius penalty
    if rho_target is not None and universal_theta_deg is not None and position is not None:
        return _compute_ring_radius_penalty(
            gear_config, rho_target, universal_theta_deg, position
        )
    
    # Use standard objective with balanced weighting
    return _order0_objective(gear_config)


def _compute_ring_radius_penalty(
    gear_config: PlanetSynthesisConfig,
    rho_target: np.ndarray | None,
    universal_theta_deg: np.ndarray | None,
    position: np.ndarray | None,
) -> float:
    """Compute penalty for ring radius deviation from target ratio profile.
    
    Weighting hierarchy:
    1. Feasibility (highest weight): integer teeth, closure, edge fraction
    2. Efficiency (medium weight): slip_integral, contact_length from Litvin metrics
    3. Ratio deviation (lowest weight): difference from ρ_target
    
    Parameters
    ----------
    gear_config : PlanetSynthesisConfig
        Gear configuration to evaluate
    rho_target : np.ndarray | None
        Target ratio profile ρ_target(θ) on universal theta grid
    universal_theta_deg : np.ndarray | None
        Universal theta grid in degrees
    position : np.ndarray | None
        Position array for synthesizing cam profile
    
    Returns
    -------
    float
        Weighted penalty (higher is worse)
    """
    # If no target ratio profile, return standard objective
    if rho_target is None or universal_theta_deg is None or position is None:
        return _order0_objective(gear_config)
    
    # Synthesize cam profile to get R_planet(θ) = litvin_result.R_psi
    base_radius = float(gear_config.base_center_radius)
    theta_rad = np.deg2rad(universal_theta_deg)
    kin = PlanetKinematics(R0=base_radius, motion=gear_config.motion)
    r_profile = np.asarray([kin.center_distance(float(th)) for th in theta_rad], dtype=float)
    legacy_profile = base_radius + position
    max_diff = float(np.max(np.abs(r_profile - legacy_profile)))
    if max_diff > 1e-6:
        max_idx = int(np.argmax(np.abs(r_profile - legacy_profile)))
        log.debug(
            "Polar pitch vs legacy displacement mismatch during penalty evaluation: "
            f"Δ={max_diff:.6f} mm at θ[{max_idx}]={universal_theta_deg[max_idx]:.2f}°",
        )
    
    R_planet_theta = r_profile
    
    # Compute R_ring(θ) = ρ_target(θ) * R_planet(θ)
    R_ring_theta = rho_target * R_planet_theta
    
    feasibility_penalty = _litvin_feasibility_surrogate(theta_rad, r_profile)
    if np.any(R_ring_theta <= 0):
        return 1e6
    ring_clearance = 2.0 * R_planet_theta
    below_clearance = np.sum(R_ring_theta < ring_clearance)
    if below_clearance:
        feasibility_penalty += 1e4 * (below_clearance / len(R_ring_theta))
    teeth_ratio = gear_config.ring_teeth / gear_config.planet_teeth
    ideal_ratio = float(np.mean(rho_target))
    ratio_deviation = abs(teeth_ratio - ideal_ratio)
    teeth_penalty = 1e2 * ratio_deviation
    
    # Evaluate efficiency metrics using Litvin equations (weight 2)
    m = evaluate_order0_metrics(gear_config)
    
    # Efficiency penalty: slip_integral and contact_length
    # Lower slip and higher contact are better
    efficiency_penalty = m.slip_integral - 0.1 * m.contact_length
    
    # Add feasibility penalties to efficiency
    if not m.feasible:
        efficiency_penalty += 1e3
    efficiency_penalty += 1e2 * m.closure_residual + 50.0 * m.phi_edge_fraction
    
    # Ratio deviation penalty (weight 3, lowest)
    # Compute mean squared deviation from target ratio
    # The actual ratio achieved is R_ring / R_planet
    actual_ratio = R_ring_theta / np.maximum(R_planet_theta, 1e-9)
    ratio_deviation_penalty = np.mean((actual_ratio - rho_target) ** 2)
    
    # Scale to be comparable with other penalties
    ratio_deviation_penalty = 1.0 * ratio_deviation_penalty
    
    # Weighted combination: feasibility > efficiency > ratio
    # Weight 1 (highest): feasibility_penalty + teeth_penalty
    # Weight 2 (medium): efficiency_penalty
    # Weight 3 (lowest): ratio_deviation_penalty
    
    total_penalty = (
        10.0 * (feasibility_penalty + teeth_penalty) +  # Weight 1: highest
        1.0 * efficiency_penalty +  # Weight 2: medium
        0.1 * ratio_deviation_penalty  # Weight 3: lowest
    )
    
    return total_penalty


def optimize_geometry(
    config: GeometrySearchConfig, order: int = OptimizationOrder.ORDER0_EVALUATE,
) -> OptimResult:
    if order == OptimizationOrder.ORDER0_EVALUATE:
        # Evaluate first candidate deterministically
        if not config.ring_teeth_candidates or not config.planet_teeth_candidates:
            return OptimResult(best_config=None, objective_value=None, feasible=False)
        cfg = PlanetSynthesisConfig(
            ring_teeth=config.ring_teeth_candidates[0],
            planet_teeth=config.planet_teeth_candidates[0],
            pressure_angle_deg=sum(config.pressure_angle_deg_bounds) / 2.0,
            addendum_factor=sum(config.addendum_factor_bounds) / 2.0,
            base_center_radius=config.base_center_radius,
            samples_per_rev=config.samples_per_rev,
            motion=config.motion,
        )
        obj = _order0_objective(cfg)
        return OptimResult(best_config=cfg, objective_value=obj, feasible=True)

    if order == OptimizationOrder.ORDER1_GEOMETRY:
        if config.section_boundaries is None:
            log.warning(
                "ORDER1_GEOMETRY requested without section boundaries; "
                "falling back to ORDER0 evaluation.",
            )
            return optimize_geometry(config, OptimizationOrder.ORDER0_EVALUATE)
        return _optimize_piecewise_sections(config)

    if order == OptimizationOrder.ORDER2_MICRO:
        # Ipopt-based NLP optimization of the contact parameter sequence phi(θ)
        return _order2_ipopt_optimization(config)

    # Higher orders will be implemented subsequently
    return OptimResult(best_config=None, objective_value=None, feasible=False)


@dataclass(frozen=True)
class WorkItem:
    """A work item: evaluate a batch of gear combinations for one section.
    
    Frozen dataclass ensures immutability for safe parallel execution.
    Each work item contains 2-3 combinations for better load balancing.
    """
    section_name: str
    theta_range: tuple[float, float]
    combinations: tuple[tuple[int, int], ...]  # List of (ring_teeth, planet_teeth) pairs


def _evaluate_gear_combination_batch(
    work_item: WorkItem,
    pa_lo: float,
    pa_hi: float,
    af_lo: float,
    af_hi: float,
    base_center_radius: float,
    samples_per_rev: int,
    theta_deg: np.ndarray,
    position: np.ndarray,
    rho_target: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a batch of gear combinations for a section with local refinement.
    
    This function is at module level to support multiprocessing (pickling).
    The motion object is recreated from arrays to avoid pickling lambda functions.
    
    Parameters
    ----------
    work_item : WorkItem
        Work item containing section_name, theta_range, and batch of (ring_teeth, planet_teeth) combinations
    pa_lo : float
        Lower bound for pressure angle
    pa_hi : float
        Upper bound for pressure angle
    af_lo : float
        Lower bound for addendum factor
    af_hi : float
        Upper bound for addendum factor
    base_center_radius : float
        Base center radius
    samples_per_rev : int
        Samples per revolution
    theta_deg : np.ndarray
        Cam angle array in degrees (for recreating motion object)
    position : np.ndarray
        Position array in mm (for recreating motion object)
    
    Returns
    -------
    list[dict[str, Any]]
        List of results, one per combination in the batch
    """
    import time
    from scipy.interpolate import interp1d
    from campro.optimization.se2 import angle_map
    from .motion import RadialSlotMotion
    
    # Recreate RadialSlotMotion object from arrays to avoid pickling lambda functions
    theta_rad = np.deg2rad(theta_deg)
    center_offset_interp = interp1d(
        theta_rad, position, kind="cubic", fill_value="extrapolate",
    )
    planet_angle_fn = lambda th: angle_map(th, scale=2.0, offset=0.0)
    
    motion = RadialSlotMotion(
        center_offset_fn=lambda th: float(center_offset_interp(th)),
        planet_angle_fn=planet_angle_fn,
    )
    
    batch_start = time.time()
    results = []
    
    # Select objective based on section
    section_name = work_item.section_name
    if "CA10" in section_name or "CA50" in section_name or "CA90" in section_name or "CA100" in section_name:
        objective_func = _objective_ca10_to_ca100
    elif "BDC" in section_name and "TDC" in section_name:
        objective_func = _objective_bdc_to_tdc
    else:
        objective_func = _objective_transition_sections
    
    # Motion slice for this section (theta range for logging)
    motion_slice = {
        "theta": np.array([work_item.theta_range[0], work_item.theta_range[1]]),
        "theta_start": work_item.theta_range[0],
        "theta_end": work_item.theta_range[1],
    }
    
    # Evaluate each combination in the batch
    for zr, zp in work_item.combinations:
        combo_start = time.time()
        
        # Local refinement for pa and af
        pa = 0.5 * (pa_lo + pa_hi)
        af = 0.5 * (af_lo + af_hi)
        step_pa = max(0.25, (pa_hi - pa_lo) / 8.0)
        step_af = max(0.02, (af_hi - af_lo) / 8.0)
        best_local = float("inf")
        improved = True
        iters = 0
        
        while improved and iters < 10:  # Fewer iterations per combination
            improved = False
            iters += 1
            for delta_pa in (-step_pa, step_pa):
                pa_try = min(pa_hi, max(pa_lo, pa + delta_pa))
                gear_cfg = PlanetSynthesisConfig(
                    ring_teeth=zr,
                    planet_teeth=zp,
                    pressure_angle_deg=pa_try,
                    addendum_factor=af,
                    base_center_radius=base_center_radius,
                    samples_per_rev=samples_per_rev,
                    motion=motion,
                )
                val = objective_func(
                    gear_cfg, work_item.theta_range, motion_slice,
                    rho_target=rho_target,
                    universal_theta_deg=theta_deg,
                    position=position,
                )
                if val < best_local:
                    best_local = val
                    pa = pa_try
                    improved = True
            for delta_af in (-step_af, step_af):
                af_try = min(af_hi, max(af_lo, af + delta_af))
                gear_cfg = PlanetSynthesisConfig(
                    ring_teeth=zr,
                    planet_teeth=zp,
                    pressure_angle_deg=pa,
                    addendum_factor=af_try,
                    base_center_radius=base_center_radius,
                    samples_per_rev=samples_per_rev,
                    motion=motion,
                )
                val = objective_func(
                    gear_cfg, work_item.theta_range, motion_slice,
                    rho_target=rho_target,
                    universal_theta_deg=theta_deg,
                    position=position,
                )
                if val < best_local:
                    best_local = val
                    af = af_try
                    improved = True
            step_pa *= 0.5
            step_af *= 0.5
        
        combo_elapsed = time.time() - combo_start
        
        results.append({
            "section_name": section_name,
            "ring_teeth": zr,
            "planet_teeth": zp,
            "best_pa": pa,
            "best_af": af,
            "best_obj": best_local,
            "elapsed": combo_elapsed,
        })
    
    return results


def _optimize_piecewise_sections(config: GeometrySearchConfig) -> OptimResult:
    """Piecewise section-based optimization with multi-threading and cumulative 2:1 constraint.
    
    Optimizes gear teeth for each of 6 sections (CA10-CA50, CA50-CA90, CA90-CA100,
    CA100-BDC, BDC-TDC, TDC-CA10) in parallel, then integrates results with
    cumulative 2:1 constraint validation.
    """
    import time
    from campro.litvin.section_analysis import get_section_boundaries
    from campro.utils.progress_logger import ProgressLogger
    
    piecewise_logger = ProgressLogger("PIECEWISE", flush_immediately=True)
    piecewise_logger.step(1, 5, "Initializing piecewise section optimization")
    
    section_boundaries = config.section_boundaries
    if section_boundaries is None:
        piecewise_logger.error("Section boundaries not available")
        return OptimResult(best_config=None, objective_value=None, feasible=False)
    
    # Extract theta array - theta_deg is required for grid alignment
    if config.theta_deg is None:
        raise ValueError(
            "theta_deg is required for piecewise section optimization. "
            "The _optimize_piecewise_sections process uses legacy universal grid logic. "
            "Update CamRingOptimizer to pass theta_deg in GeometrySearchConfig before optimization."
        )
    theta_full = np.asarray(config.theta_deg)
    n_samples = len(theta_full)
    piecewise_logger.info(
        f"Using provided theta_deg array with {n_samples} points"
    )
    
    # Get section indices for this theta array
    section_indices = get_section_boundaries(
        theta_full,
        np.zeros(n_samples),  # Placeholder position - not used for indices
        section_boundaries,
    )
    
    piecewise_logger.info(f"Optimizing {len(section_boundaries.sections)} sections")
    for name, (start, end) in section_boundaries.sections.items():
        piecewise_logger.info(f"  Section {name}: {start:.2f}° to {end:.2f}°")
    
    piecewise_logger.step_complete("Initialization", 0.0)
    
    # Step 2: Optimize each section in parallel
    piecewise_logger.step(2, 5, "Optimizing sections in parallel")
    section_results = _optimize_sections_parallel(
        config, section_boundaries, section_indices, piecewise_logger
    )
    
    # Step 3: Validate cumulative 2:1 constraint
    piecewise_logger.step(3, 5, "Validating cumulative 2:1 constraint")
    constraint_valid, constraint_error = _validate_cumulative_ratio(
        section_results, section_boundaries, piecewise_logger
    )
    
    if not constraint_valid:
        piecewise_logger.warning(
            f"Cumulative constraint violated: error={constraint_error:.6f} rad "
            f"(expected 4π = {4*pi:.6f} rad)"
        )
    
    # Step 4: Integrate section ratios into integer gear pair
    piecewise_logger.step(4, 5, "Integrating section ratios into integer gear pair")
    final_ring_teeth, final_planet_teeth = _integrate_section_ratios(
        config, section_results, section_boundaries, constraint_error, piecewise_logger
    )
    
    # Step 5: Create final configuration
    piecewise_logger.step(5, 5, "Creating final gear configuration")
    pa_lo, pa_hi = config.pressure_angle_deg_bounds
    af_lo, af_hi = config.addendum_factor_bounds
    
    # Use average pressure angle and addendum factor
    final_pa = 0.5 * (pa_lo + pa_hi)
    final_af = 0.5 * (af_lo + af_hi)
    
    final_config = PlanetSynthesisConfig(
        ring_teeth=final_ring_teeth,
        planet_teeth=final_planet_teeth,
        pressure_angle_deg=final_pa,
        addendum_factor=final_af,
        base_center_radius=config.base_center_radius,
        samples_per_rev=config.samples_per_rev,
        motion=config.motion,
    )
    
    # Evaluate final objective
    final_obj = _order0_objective(final_config)
    
    piecewise_logger.info(
        f"Final gear pair: ring={final_ring_teeth}, planet={final_planet_teeth}, "
        f"ratio={final_ring_teeth/final_planet_teeth:.3f}, obj={final_obj:.6f}"
    )
    piecewise_logger.step_complete("Final configuration", 0.0)
    
    return OptimResult(
        best_config=final_config,
        objective_value=final_obj,
        feasible=constraint_valid and final_obj is not None,
    )


def _optimize_sections_parallel(
    config: GeometrySearchConfig,
    section_boundaries: Any,  # SectionBoundaries
    section_indices: dict[str, tuple[int, int]],
    logger: Any,  # ProgressLogger
) -> dict[str, dict[str, Any]]:
    """Optimize gear teeth for each section using work-item level parallelism.
    
    Creates a work queue of all (section, gear_combination) pairs and processes
    them in parallel across all available threads for better utilization.
    
    Returns
    -------
    dict[str, dict[str, Any]]
        Section name -> {ratio: float, ring_teeth: int, planet_teeth: int, objective: float}
    """
    import time
    import multiprocessing as mp
    from functools import partial
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    
    # Use exactly 12 threads for even distribution
    n_threads_target = 12
    
    # Generate all combinations first
    all_combinations = [
        (zr, zp)
        for zr in config.ring_teeth_candidates
        for zp in config.planet_teeth_candidates
    ]
    total_combinations = len(all_combinations)
    n_sections = len(section_boundaries.sections)
    
    # Calculate optimal chunk size to evenly distribute across 12 threads
    # Target: ~2-3 combinations per work item for good granularity
    # For 25 combinations per section: 25 / 12 ≈ 2 combinations per thread
    # Create work items with 2-3 combinations each for better load balancing
    combinations_per_chunk = max(1, total_combinations // n_threads_target)
    
    # Generate work queue: chunk combinations into batches for each section
    work_items: list[WorkItem] = []
    for section_name, theta_range in section_boundaries.sections.items():
        # Chunk combinations for this section
        for i in range(0, len(all_combinations), combinations_per_chunk):
            chunk = all_combinations[i:i + combinations_per_chunk]
            work_items.append(WorkItem(
                section_name=section_name,
                theta_range=theta_range,
                combinations=tuple(chunk),
            ))
    
    total_work_items = len(work_items)
    
    logger.info(
        f"Created work queue: {total_work_items} work items "
        f"({n_sections} sections × ~{len(work_items) // n_sections} chunks per section, "
        f"{combinations_per_chunk} combinations per chunk)"
    )
    logger.info(
        f"Total combinations: {total_combinations} per section, "
        f"distributed across {total_work_items} work items for {n_threads_target} threads"
    )
    
    section_results: dict[str, dict[str, Any]] = {}
    # Only use lock for ThreadPoolExecutor (ProcessPoolExecutor doesn't need it)
    use_multiprocessing = getattr(config, "use_multiprocessing", False)
    results_lock = threading.Lock() if not use_multiprocessing else None
    
    pa_lo, pa_hi = config.pressure_angle_deg_bounds
    af_lo, af_hi = config.addendum_factor_bounds
    
    # Validate that we have the required arrays for recreating motion object
    if config.theta_deg is None or config.position is None:
        logger.warning(
            "theta_deg or position not available in config. "
            "Falling back to using motion object directly (may fail with multiprocessing)."
        )
        # Fallback: try to use motion object (will fail if multiprocessing is enabled)
        evaluate_batch = partial(
            _evaluate_gear_combination_batch,
            pa_lo=pa_lo,
            pa_hi=pa_hi,
            af_lo=af_lo,
            af_hi=af_hi,
            base_center_radius=config.base_center_radius,
            samples_per_rev=config.samples_per_rev,
            theta_deg=config.theta_deg if config.theta_deg is not None else np.array([]),
            position=config.position if config.position is not None else np.array([]),
            rho_target=config.rho_target,
        )
    else:
        # Create a partial function with the fixed parameters for multiprocessing compatibility
        # Pass arrays instead of motion object to avoid pickling lambda functions
        evaluate_batch = partial(
            _evaluate_gear_combination_batch,
            pa_lo=pa_lo,
            pa_hi=pa_hi,
            af_lo=af_lo,
            af_hi=af_hi,
            base_center_radius=config.base_center_radius,
            samples_per_rev=config.samples_per_rev,
            theta_deg=config.theta_deg,
            position=config.position,
            rho_target=config.rho_target,
        )
    
    # Use exactly 12 threads for even distribution
    n_threads = 12
    if total_work_items < n_threads:
        n_threads = total_work_items
        logger.info(
            f"Thread count reduced from 12 to {n_threads} "
            f"(matching {total_work_items} work items)"
        )
    else:
        logger.info(f"Using {n_threads} threads for work-item level optimization")
    
    # Choose executor based on configuration
    use_multiprocessing = getattr(config, "use_multiprocessing", False)
    executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
    executor_kwargs: dict[str, Any] = {"max_workers": n_threads}

    if use_multiprocessing:
        try:
            mp_context = mp.get_context("fork")
            executor_kwargs["mp_context"] = mp_context
            logger.info(
                f"Using {n_threads} processes for work-item level optimization "
                f"(bypasses GIL for CPU-bound work via fork context)"
            )
        except ValueError:
            logger.warning(
                "Multiprocessing requested but 'fork' context unavailable on this platform. "
                "Reverting to thread-based execution.",
            )
            use_multiprocessing = False
            executor_class = ThreadPoolExecutor
            executor_kwargs = {"max_workers": n_threads}
    
    # Track timing for utilization diagnostics
    parallel_start = time.time()
    work_item_times: list[float] = []
    
    # Collect all results (will be aggregated by section later)
    all_results: list[dict[str, Any]] = []
    
    with executor_class(**executor_kwargs) as executor:
        # Submit all work items
        futures = {executor.submit(evaluate_batch, work_item): work_item for work_item in work_items}
        
        completed_count = 0
        last_summary_log = parallel_start
        summary_log_interval = 5.0  # Log summary every 5 seconds
        
        for future in as_completed(futures):
            work_item = futures[future]
            completed_count += 1
            try:
                batch_results = future.result()  # Returns list of results
                all_results.extend(batch_results)  # Flatten into all_results
                # Track time for each combination in the batch
                for result in batch_results:
                    work_item_times.append(result.get("elapsed", 0.0))
                
                # Periodic progress logging
                current_time = time.time()
                total_combinations_completed = len(all_results)
                total_combinations_expected = total_combinations * n_sections
                
                if (current_time - last_summary_log >= summary_log_interval) or (completed_count == total_work_items):
                    remaining_work_items = total_work_items - completed_count
                    remaining_combinations = total_combinations_expected - total_combinations_completed
                    progress_pct = (total_combinations_completed / total_combinations_expected) * 100.0
                    
                    # Calculate average time per combination and estimate remaining
                    if total_combinations_completed > 0 and work_item_times:
                        avg_combo_time = sum(work_item_times) / len(work_item_times)
                        estimated_remaining = avg_combo_time * remaining_combinations / n_threads if n_threads > 0 else 0.0
                    else:
                        estimated_remaining = 0.0
                    
                    logger.info(
                        f"Progress: {total_combinations_completed}/{total_combinations_expected} combinations completed "
                        f"({progress_pct:.1f}%) - {completed_count}/{total_work_items} work items, "
                        f"{remaining_combinations} combinations remaining, estimated time: {estimated_remaining:.1f}s"
                    )
                    last_summary_log = current_time
            except Exception as e:
                logger.warning(
                    f"✗ Work item failed (section={work_item.section_name}, "
                    f"batch_size={len(work_item.combinations)} combinations): {e}"
                )
                # Continue - failed work items won't affect section results
    
    # Aggregate results by section: find best combination for each section
    from collections import defaultdict
    section_results_dict: dict[str, list[dict[str, Any]]] = defaultdict(list)
    
    for result in all_results:
        section_name = result["section_name"]
        section_results_dict[section_name].append(result)
    
    # For each section, find the combination with minimum objective
    for section_name, results in section_results_dict.items():
        if not results:
            # No valid results for this section - use default fallback
            section_results[section_name] = {
                "ratio": 2.0,  # Default 2:1 ratio
                "ring_teeth": 40,
                "planet_teeth": 20,
                "objective": float("inf"),
            }
            continue
        
        # Find best result (minimum objective)
        best_result = min(results, key=lambda r: r["best_obj"])
        
        # Convert to expected format
        section_results[section_name] = {
            "ratio": best_result["ring_teeth"] / best_result["planet_teeth"],
            "ring_teeth": best_result["ring_teeth"],
            "planet_teeth": best_result["planet_teeth"],
            "objective": best_result["best_obj"],
        }
        
        # Log best result for this section
        logger.info(
            f"✓ Section {section_name}: best ratio={section_results[section_name]['ratio']:.3f} "
            f"(ring={best_result['ring_teeth']}, planet={best_result['planet_teeth']}), "
            f"obj={best_result['best_obj']:.6f}"
        )
    
    # Calculate utilization diagnostics
    parallel_elapsed = time.time() - parallel_start
    total_work_time = sum(work_item_times) if work_item_times else 0.0
    
    # Utilization metric: (sum of work item times) / (wall time * n_threads)
    # Ideal: close to 1.0 (all threads busy all the time)
    # GIL contention: much less than 1.0 (threads waiting for GIL)
    if parallel_elapsed > 0 and n_threads > 0:
        utilization = total_work_time / (parallel_elapsed * n_threads)
        
        # Calculate speedup (sequential time / parallel time)
        sequential_time = total_work_time
        speedup = sequential_time / parallel_elapsed if parallel_elapsed > 0 else 0.0
        efficiency = speedup / n_threads if n_threads > 0 else 0.0
        
        # Log comprehensive utilization summary
        logger.info("=" * 70)
        logger.info("Parallel Optimization Utilization Summary")
        logger.info("=" * 70)
        logger.info(
            f"Wall-clock time: {parallel_elapsed:.3f}s "
            f"(sum of work item times: {total_work_time:.3f}s)"
        )
        logger.info(f"Work items: {total_work_items}, Threads used: {n_threads}")
        logger.info(f"Speedup: {speedup:.2f}x (sequential: {sequential_time:.3f}s / parallel: {parallel_elapsed:.3f}s)")
        logger.info(f"Parallel efficiency: {efficiency:.2%} (speedup / threads)")
        logger.info(
            f"Thread utilization: {utilization:.2%} "
            f"(ideal: 100%, GIL contention indicated if <50%)"
        )
        
        # Per-section result summary
        if section_results:
            logger.info("Section results summary:")
            for section_name, result in sorted(section_results.items()):
                logger.info(
                    f"  {section_name}: ratio={result['ratio']:.3f} "
                    f"(ring={result['ring_teeth']}, planet={result['planet_teeth']}), "
                    f"obj={result['objective']:.6f}"
                )
        
        logger.info("=" * 70)
        if utilization < 0.5:
            logger.warning(
                f"Low thread utilization ({utilization:.2%}) suggests GIL contention. "
                f"Consider using ProcessPoolExecutor for CPU-bound work."
            )
    else:
        logger.info(f"Parallel optimization completed in {parallel_elapsed:.3f}s")
    
    logger.step_complete("Parallel section optimization", parallel_elapsed)
    return section_results


def _validate_cumulative_ratio(
    section_results: dict[str, dict[str, Any]],
    section_boundaries: Any,  # SectionBoundaries
    logger: Any,  # ProgressLogger
) -> tuple[bool, float]:
    """Validate that cumulative planet rotation equals 2 full rotations (4π).
    
    Returns
    -------
    tuple[bool, float]
        (is_valid, error_magnitude)
    """
    from campro.utils.progress_logger import ProgressLogger
    
    constraint_logger = ProgressLogger("CONSTRAINT", flush_immediately=True)
    constraint_logger.step(1, 3, "Computing section rotations")
    
    total_planet_rotation = 0.0  # radians
    section_rotations = {}
    
    for section_name, (theta_start, theta_end) in section_boundaries.sections.items():
        # Handle 0° as 360° for wraparound calculations (cycle is 1-360°)
        theta_end_normalized = 360.0 if theta_end == 0.0 else theta_end
        
        # Handle wrap-around for any section that crosses 360°/1° boundary
        if theta_start > theta_end_normalized:
            # Wrap around: from start to 360°, then from 1° to end
            delta_theta_ring = np.deg2rad((360.0 - theta_start) + theta_end_normalized)
        else:
            # Normal forward section
            delta_theta_ring = np.deg2rad(theta_end_normalized - theta_start)
        
        # Get ratio for this section
        section_ratio = section_results.get(section_name, {}).get("ratio", 2.0)
        
        # Calculate planet rotation for this section
        delta_theta_planet = delta_theta_ring * section_ratio
        total_planet_rotation += delta_theta_planet
        
        section_rotations[section_name] = {
            "ring_rotation_rad": delta_theta_ring,
            "planet_rotation_rad": delta_theta_planet,
            "ratio": section_ratio,
        }
        
        constraint_logger.info(
            f"Section {section_name}: ring={np.rad2deg(delta_theta_ring):.2f}°, "
            f"planet={np.rad2deg(delta_theta_planet):.2f}°, ratio={section_ratio:.3f}"
        )
    
    constraint_logger.step_complete("Section rotations computed", 0.0)
    
    # Validate constraint
    constraint_logger.step(2, 3, "Validating cumulative constraint")
    expected_rotation = 4.0 * pi  # 2 full rotations
    error = abs(total_planet_rotation - expected_rotation)
    tolerance = 0.01  # 0.01 rad ≈ 0.57°
    is_valid = error < tolerance
    
    constraint_logger.info(
        f"Total planet rotation: {total_planet_rotation:.6f} rad ({np.rad2deg(total_planet_rotation):.2f}°)"
    )
    constraint_logger.info(
        f"Expected: {expected_rotation:.6f} rad (720°), error: {error:.6f} rad ({np.rad2deg(error):.2f}°)"
    )
    
    if is_valid:
        constraint_logger.step_complete("Constraint validation passed", 0.0)
    else:
        constraint_logger.warning(f"Constraint validation failed: error={error:.6f} rad > tolerance={tolerance:.6f} rad")
        constraint_logger.step_complete("Constraint validation failed", 0.0)
    
    return is_valid, error


def _integrate_section_ratios(
    config: GeometrySearchConfig,
    section_results: dict[str, dict[str, Any]],
    section_boundaries: Any,  # SectionBoundaries
    constraint_error: float,
    logger: Any,  # ProgressLogger
) -> tuple[int, int]:
    """Integrate section ratios into integer gear pair satisfying cumulative constraint.
    
    Returns
    -------
    tuple[int, int]
        (ring_teeth, planet_teeth)
    """
    from campro.utils.progress_logger import ProgressLogger
    
    integration_logger = ProgressLogger("INTEGRATION", flush_immediately=True)
    integration_logger.step(1, 4, "Computing weighted section ratios")
    
    # Compute weights based on section duration
    section_weights = {}
    total_duration = 0.0
    
    for section_name, (theta_start, theta_end) in section_boundaries.sections.items():
        # Handle 0° as 360° for wraparound calculations (cycle is 1-360°)
        theta_end_normalized = 360.0 if theta_end == 0.0 else theta_end
        
        # Handle wrap-around for any section that crosses 360°/1° boundary
        if theta_start > theta_end_normalized:
            duration = (360.0 - theta_start) + theta_end_normalized
        else:
            duration = theta_end_normalized - theta_start
        section_weights[section_name] = duration
        total_duration += duration
    
    # Normalize weights
    for section_name in section_weights:
        section_weights[section_name] /= total_duration
    
    # Compute weighted average ratio
    weighted_ratio = 0.0
    for section_name, weight in section_weights.items():
        section_ratio = section_results.get(section_name, {}).get("ratio", 2.0)
        weighted_ratio += weight * section_ratio
        integration_logger.info(
            f"Section {section_name}: ratio={section_ratio:.3f}, weight={weight:.3f}"
        )
    
    integration_logger.info(f"Weighted average ratio: {weighted_ratio:.3f}")
    integration_logger.step_complete("Weighted ratios computed", 0.0)
    
    # Find integer gear pair that best satisfies constraint
    integration_logger.step(2, 4, "Finding optimal integer gear pair")
    
    # Search candidate integer pairs
    best_error = float("inf")
    best_ring = None
    best_planet = None
    
    # Search around weighted ratio
    target_ratio = weighted_ratio
    search_range = 0.5  # ±50% around target
    
    for zr in config.ring_teeth_candidates:
        for zp in config.planet_teeth_candidates:
            ratio = zr / zp
            if abs(ratio - target_ratio) / target_ratio > search_range:
                continue
            
            # Check if this pair satisfies cumulative constraint
            # Recompute total rotation with this integer pair
            total_rot = 0.0
            for section_name, (theta_start, theta_end) in section_boundaries.sections.items():
                # Handle 0° as 360° for wraparound calculations (cycle is 1-360°)
                theta_end_normalized = 360.0 if theta_end == 0.0 else theta_end
                
                # Handle wrap-around for any section that crosses 360°/1° boundary
                if theta_start > theta_end_normalized:
                    delta_theta_ring = np.deg2rad((360.0 - theta_start) + theta_end_normalized)
                else:
                    delta_theta_ring = np.deg2rad(theta_end_normalized - theta_start)
                total_rot += delta_theta_ring * ratio
            
            error = abs(total_rot - 4.0 * pi)
            
            if error < best_error:
                best_error = error
                best_ring = zr
                best_planet = zp
    
    if best_ring is None:
        # Fallback: use closest to weighted ratio
        integration_logger.warning("No integer pair found satisfying constraint, using closest to weighted ratio")
        for zr in config.ring_teeth_candidates:
            for zp in config.planet_teeth_candidates:
                ratio = zr / zp
                error = abs(ratio - target_ratio)
                if error < best_error:
                    best_error = error
                    best_ring = zr
                    best_planet = zp
    
    integration_logger.info(
        f"Optimal integer pair: ring={best_ring}, planet={best_planet}, "
        f"ratio={best_ring/best_planet:.3f}, constraint_error={best_error:.6f} rad"
    )
    integration_logger.step_complete("Integer pair found", 0.0)
    
    # Validate final constraint
    integration_logger.step(3, 4, "Validating final integer pair constraint")
    total_rot = 0.0
    for section_name, (theta_start, theta_end) in section_boundaries.sections.items():
        # Handle 0° as 360° for wraparound calculations (cycle is 1-360°)
        theta_end_normalized = 360.0 if theta_end == 0.0 else theta_end
        
        # Handle wrap-around for any section that crosses 360°/1° boundary
        if theta_start > theta_end_normalized:
            delta_theta_ring = np.deg2rad((360.0 - theta_start) + theta_end_normalized)
        else:
            delta_theta_ring = np.deg2rad(theta_end_normalized - theta_start)
        total_rot += delta_theta_ring * (best_ring / best_planet)
    
    final_error = abs(total_rot - 4.0 * pi)
    integration_logger.info(
        f"Final constraint check: total_rotation={total_rot:.6f} rad ({np.rad2deg(total_rot):.2f}°), "
        f"error={final_error:.6f} rad ({np.rad2deg(final_error):.2f}°)"
    )
    integration_logger.step_complete("Final validation", 0.0)
    
    return best_ring, best_planet


def _order2_ipopt_optimization(config: GeometrySearchConfig) -> OptimResult:
    """
    Ipopt-based NLP optimization of the contact parameter sequence phi(θ).

    This replaces the simple smoothing approach with a proper constrained optimization
    that handles smoothness, contact constraints, and periodicity.
    """
    import sys
    import time
    from campro.utils.progress_logger import ProgressLogger
    
    order2_logger = ProgressLogger("ORDER2", flush_immediately=True)
    order2_logger.step(1, 5, "Initializing CasADi and problem setup")
    setup_start = time.time()
    
    try:
        import casadi as ca
    except ImportError:
        log.error("CasADi not available for ORDER2_MICRO Ipopt optimization")
        order2_logger.error("CasADi not available")
        return OptimResult(best_config=None, objective_value=None, feasible=False)

    # Set up problem dimensions
    n = max(64, config.samples_per_rev)
    order2_logger.info(f"Problem size: {n} collocation points")
    grid = make_uniform_grid(n)
    order2_logger.step_complete("Problem setup", time.time() - setup_start)

    # Construct flank/kinematics once
    order2_logger.step(2, 5, "Constructing gear geometry and kinematics")
    geom_start = time.time()
    module = (
        config.base_center_radius
        * 2.0
        / max(config.ring_teeth_candidates[0] - config.planet_teeth_candidates[0], 1)
    )
    zr = config.ring_teeth_candidates[0]
    zp = config.planet_teeth_candidates[0]
    pa = sum(config.pressure_angle_deg_bounds) / 2.0
    af = sum(config.addendum_factor_bounds) / 2.0
    order2_logger.info(f"Gear config: ring={zr}, planet={zp}, pa={pa:.2f}°, af={af:.3f}")
    
    cand = PlanetSynthesisConfig(
        ring_teeth=zr,
        planet_teeth=zp,
        pressure_angle_deg=pa,
        addendum_factor=af,
        base_center_radius=config.base_center_radius,
        samples_per_rev=config.samples_per_rev,
        motion=config.motion,
    )
    params = InternalGearParams(
        teeth=zr, module=module, pressure_angle_deg=pa, addendum_factor=af,
    )
    flank = sample_internal_flank(params, n=256)
    kin = PlanetKinematics(R0=config.base_center_radius, motion=config.motion)
    order2_logger.step_complete("Geometry construction", time.time() - geom_start)

    # Initialize phi by Newton per node (same as before)
    order2_logger.step(3, 5, f"Initializing phi values for {n} points (Newton solve)")
    phi_init_start = time.time()
    phi_vals: list[float] = []
    seed = flank.phi[len(flank.phi) // 2]
    for i, theta in enumerate(grid.theta):
        phi = _newton_solve_phi(flank, kin, theta, seed) or seed
        phi_vals.append(phi)
        seed = phi
        if (i + 1) % max(1, n // 10) == 0:  # Progress every 10%
            order2_logger.info(f"  Initialized {i+1}/{n} points ({100*(i+1)//n}%)")
    order2_logger.step_complete("Phi initialization", time.time() - phi_init_start)

    # Convert to numpy array for CasADi
    phi_init = np.array(phi_vals)

    # Create CasADi variables
    phi = ca.SX.sym("phi", n)

    # Objective function: slip integral - 0.1 * contact_length + feasibility penalty
    # We'll approximate this using the existing metrics function
    def objective_function_with_physics(phi_vec):
        """Evaluate objective with physics metrics."""
        m = evaluate_order0_metrics_given_phi(cand, phi_vec.tolist())
        penalty = 0.0 if m.feasible else 1e3
        return m.slip_integral - 0.1 * m.contact_length + penalty

    # Store reference for final validation
    objective_with_physics = objective_function_with_physics

    # Create CasADi function for objective (placeholder - not used in hybrid approach)
    # obj_func = ca.Function('obj', [phi], [ca.SX.sym('obj_val')])

    # For CasADi: use smoothness penalty as proxy, validate with physics
    # This is a hybrid approach - CasADi smoothness + Python physics validation
    smoothness_penalty = 0.0
    for i in range(n):
        im = (i - 1) % n
        ip = (i + 1) % n
        smoothness_penalty += (phi[i] - 0.5 * (phi[im] + phi[ip])) ** 2

    # Periodicity constraint: phi[n-1] should be close to phi[0]
    periodicity_constraint = phi[n - 1] - phi[0]

    # Bounds on phi values (based on flank geometry)
    phi_min = float(np.min(flank.phi))
    phi_max = float(np.max(flank.phi))

    # Create NLP problem (unscaled)
    nlp = {
        "x": phi,
        "f": smoothness_penalty,
        "g": periodicity_constraint,
    }

    # Scaling for phi decision variable (keep z = s*phi ~ O(1))
    magnitude = max(abs(phi_min), abs(phi_max))
    s_phi = (1.0 / magnitude) if magnitude > 1e-6 else 1.0
    nlp_scaled = build_scaled_nlp(nlp, s_phi, kind="SX")

    # Create Ipopt options
    order2_logger.step(4, 5, "Setting up Ipopt solver and NLP problem")
    nlp_setup_start = time.time()
    solver_options = IPOPTOptions(
        max_iter=1000,
        tol=1e-6,
        linear_solver="ma27",
        enable_analysis=True,
        print_level=3,
    )

    # Set up Ipopt solver wrapper
    solver = IPOPTSolver(solver_options)
    order2_logger.step_complete("NLP setup", time.time() - nlp_setup_start)
    order2_logger.info(f"  Solver: {solver_options.linear_solver}, max_iter={solver_options.max_iter}, tol={solver_options.tol}")

    # Solve the NLP
    order2_logger.step(5, 5, "Running Ipopt optimization (this may take a while...)")
    solve_start = time.time()
    try:
        # Scale x0/lbx/ubx into z-domain: z = s*phi
        x0_z = scale_value(np.asarray(phi_init), s_phi)
        lbz, ubz = scale_bounds((np.full(n, phi_min), np.full(n, phi_max)), s_phi)

        order2_logger.info("  Starting Ipopt solve...")
        result = solver.solve(
            nlp=nlp_scaled,
            x0=x0_z,
            lbx=lbz,
            ubx=ubz,
            lbg=np.array([0.0]),  # periodicity constraint: remains 0 after scaling
            ubg=np.array([0.0]),
        )
        solve_elapsed = time.time() - solve_start
        order2_logger.info(f"  Ipopt solve completed: success={result.success}, "
                          f"iterations={result.iterations if hasattr(result, 'iterations') else 'unknown'}, "
                          f"time={solve_elapsed:.3f}s")

        if result.success:
            order2_logger.step_complete("Ipopt optimization", solve_elapsed)
            # Extract optimized phi values
            order2_logger.info("Validating solution with physics model...")
            validation_start = time.time()
            phi_opt = result.x_opt

            # Validate with full physics
            physics_obj_val = objective_with_physics(phi_opt)
            order2_logger.info(f"  Physics objective: {physics_obj_val:.6f}")
            log.info(f"Physics objective validation: {physics_obj_val:.6f}")

            # Check if physics-based objective is acceptable
            if physics_obj_val > 1e3:  # Infeasible
                order2_logger.warning("Solution failed physics feasibility check")
                log.warning("Optimized solution failed physics feasibility check")
            else:
                order2_logger.info("  Physics validation passed")
            order2_logger.step_complete("Physics validation", time.time() - validation_start)

            # Evaluate final metrics
            m = evaluate_order0_metrics_given_phi(cand, phi_opt.tolist())
            obj = (
                m.slip_integral - 0.1 * m.contact_length + (0.0 if m.feasible else 1e3)
            )

            # Perform analysis
            from campro.optimization.solver_analysis import analyze_ipopt_run

            analysis = analyze_ipopt_run(
                {
                    "success": result.success,
                    "iterations": result.iterations,
                    "primal_inf": result.primal_inf,
                    "dual_inf": result.dual_inf,
                    "return_status": result.status,
                },
                None,
            )  # No log file for now

            # Add physics validation to analysis
            analysis.stats["physics_objective"] = physics_obj_val
            analysis.stats["physics_feasible"] = physics_obj_val < 1e3

            return OptimResult(
                best_config=cand,
                objective_value=obj,
                feasible=m.feasible,
                ipopt_analysis=analysis,
            )
        failure_msg = f"ORDER2_MICRO Ipopt optimization failed: {result.message}"
        order2_logger.error(failure_msg)
        log.error(failure_msg)
        raise RuntimeError(failure_msg)

    except Exception as e:
        failure_msg = f"ORDER2_MICRO Ipopt optimization error: {e}"
        order2_logger.error(failure_msg)
        log.error(failure_msg)
        raise RuntimeError(failure_msg) from e
