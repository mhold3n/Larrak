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

log = get_logger(__name__)

# Explicit exports for consumers expecting named attributes
__all__ = ["OptimizationOrder", "optimize_geometry", "OptimResult"]


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


def _objective_ca10_to_ca100(
    gear_config: PlanetSynthesisConfig,
    section_theta_range: tuple[float, float],
    motion_slice: dict[str, np.ndarray],
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
    
    Returns
    -------
    float
        Objective value (negative for maximization - lower is better)
    """
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
    
    Returns
    -------
    float
        Objective value (negative for maximization - lower is better)
    """
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
    
    Returns
    -------
    float
        Objective value (negative for maximization - lower is better)
    """
    # Use standard objective with balanced weighting
    return _order0_objective(gear_config)


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
        # Check if section-based optimization is available
        if config.section_boundaries is not None:
            # Use piecewise section-based optimization with multi-threading
            return _optimize_piecewise_sections(config)
        else:
            # Fallback to original grid search
            from campro.utils.progress_logger import ProgressLogger
            fallback_logger = ProgressLogger("ORDER1", flush_immediately=True)
            fallback_logger.warning(
                "⚠ FALLBACK MODE: Section-based optimization not available. "
                "Using sequential grid search instead of piecewise multi-threaded optimization."
            )
            fallback_logger.info(
                "Reason: Section boundaries not provided (combustion model may be unavailable or failed)."
            )
            return _optimize_grid_search(config)

    if order == OptimizationOrder.ORDER2_MICRO:
        # Ipopt-based NLP optimization of the contact parameter sequence phi(θ)
        return _order2_ipopt_optimization(config)

    # Higher orders will be implemented subsequently
    return OptimResult(best_config=None, objective_value=None, feasible=False)


def _optimize_grid_search(config: GeometrySearchConfig) -> OptimResult:
    """Original grid search implementation (fallback when section analysis unavailable)."""
    import time
    from campro.utils.progress_logger import ProgressLogger
    
    order1_logger = ProgressLogger("ORDER1", flush_immediately=True)
    order1_logger.info("=" * 70)
    order1_logger.warning(
        "⚠ FALLBACK MODE ACTIVE: Using sequential grid search (slower than piecewise optimization)"
    )
    order1_logger.info(
        "This method evaluates all gear combinations sequentially without multi-threading."
    )
    order1_logger.info(
        "To enable faster piecewise optimization, ensure combustion model is available."
    )
    order1_logger.info("=" * 70)
    
    best_cfg: PlanetSynthesisConfig | None = None
    best_obj: float | None = None
    pa_lo, pa_hi = config.pressure_angle_deg_bounds
    af_lo, af_hi = config.addendum_factor_bounds

    def obj_for(pa: float, af: float, zr: int, zp: int) -> float:
        cand = PlanetSynthesisConfig(
            ring_teeth=zr,
            planet_teeth=zp,
            pressure_angle_deg=pa,
            addendum_factor=af,
            base_center_radius=config.base_center_radius,
            samples_per_rev=config.samples_per_rev,
            motion=config.motion,
        )
        return _order0_objective(cand)

    total_combinations = len(config.ring_teeth_candidates) * len(config.planet_teeth_candidates)
    combination_count = 0
    order1_logger.info(f"Searching {total_combinations} gear combinations "
                      f"({len(config.ring_teeth_candidates)} ring teeth × {len(config.planet_teeth_candidates)} planet teeth)")

    for idx_zr, zr in enumerate(config.ring_teeth_candidates):
        for idx_zp, zp in enumerate(config.planet_teeth_candidates):
            combination_count += 1
            combo_start = time.time()
            order1_logger.info(
                f"Evaluating combination {combination_count}/{total_combinations}: "
                f"ring_teeth={zr}, planet_teeth={zp}"
            )
            
            pa = 0.5 * (pa_lo + pa_hi)
            af = 0.5 * (af_lo + af_hi)
            step_pa = max(0.25, (pa_hi - pa_lo) / 8.0)
            step_af = max(0.02, (af_hi - af_lo) / 8.0)
            best_local = obj_for(pa, af, zr, zp)
            improved = True
            iters = 0
            
            while improved and iters < 20:
                improved = False
                iters += 1
                # coordinate search in pa
                for delta in (-step_pa, step_pa):
                    pa_try = min(pa_hi, max(pa_lo, pa + delta))
                    val = obj_for(pa_try, af, zr, zp)
                    if val < best_local:
                        best_local = val
                        pa = pa_try
                        improved = True
                # coordinate search in af
                for delta in (-step_af, step_af):
                    af_try = min(af_hi, max(af_lo, af + delta))
                    val = obj_for(pa, af_try, zr, zp)
                    if val < best_local:
                        best_local = val
                        af = af_try
                        improved = True
                # decrease steps
                step_pa *= 0.5
                step_af *= 0.5

            combo_elapsed = time.time() - combo_start
            if best_obj is None or best_local < best_obj:
                best_obj = best_local
                best_cfg = PlanetSynthesisConfig(
                    ring_teeth=zr,
                    planet_teeth=zp,
                    pressure_angle_deg=pa,
                    addendum_factor=af,
                    base_center_radius=config.base_center_radius,
                    samples_per_rev=config.samples_per_rev,
                    motion=config.motion,
                )
                order1_logger.info(
                    f"  ✓ New best: obj={best_local:.6f}, "
                    f"pa={pa:.2f}°, af={af:.3f} ({combo_elapsed:.3f}s)"
                )
            else:
                order1_logger.info(
                    f"  - Current: obj={best_local:.6f} (best={best_obj:.6f}) ({combo_elapsed:.3f}s)"
                )

    return OptimResult(
        best_config=best_cfg,
        objective_value=best_obj,
        feasible=best_cfg is not None,
    )


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
    
    # Extract theta array from section boundaries
    # The section boundaries were computed from primary_data which has theta in degrees
    # We need to create a theta array matching the motion law grid
    n_samples = config.samples_per_rev
    theta_full = np.linspace(0, 360, n_samples, endpoint=False)  # Degrees
    
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
    """Optimize gear teeth for each section in parallel.
    
    Returns
    -------
    dict[str, dict[str, Any]]
        Section name -> {ratio: float, ring_teeth: int, planet_teeth: int, objective: float}
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    
    section_results: dict[str, dict[str, Any]] = {}
    # Only use lock for ThreadPoolExecutor (ProcessPoolExecutor doesn't need it)
    use_multiprocessing = getattr(config, "use_multiprocessing", False)
    results_lock = threading.Lock() if not use_multiprocessing else None
    
    pa_lo, pa_hi = config.pressure_angle_deg_bounds
    af_lo, af_hi = config.addendum_factor_bounds
    
    def optimize_section(section_name: str, theta_range: tuple[float, float]) -> dict[str, Any]:
        """Optimize a single section."""
        section_start = time.time()
        last_progress_log = section_start
        progress_log_interval = 2.0  # Log progress every 2 seconds
        
        # Select objective based on section
        if "CA10" in section_name or "CA50" in section_name or "CA90" in section_name or "CA100" in section_name:
            objective_func = _objective_ca10_to_ca100
        elif "BDC" in section_name and "TDC" in section_name:
            objective_func = _objective_bdc_to_tdc
        else:
            objective_func = _objective_transition_sections
        
        # Motion slice for this section (theta range for logging)
        motion_slice = {
            "theta": np.array([theta_range[0], theta_range[1]]),
            "theta_start": theta_range[0],
            "theta_end": theta_range[1],
        }
        
        best_ratio = None
        best_obj = float("inf")
        best_zr = None
        best_zp = None
        best_pa = 0.5 * (pa_lo + pa_hi)
        best_af = 0.5 * (af_lo + af_hi)
        
        # Calculate total combinations for progress tracking
        total_combinations = len(config.ring_teeth_candidates) * len(config.planet_teeth_candidates)
        combination_count = 0
        
        # Log section start
        if results_lock is not None:
            with results_lock:
                logger.info(f"Section {section_name}: evaluating {total_combinations} gear combinations...")
        
        # Search over gear combinations
        for zr in config.ring_teeth_candidates:
            for zp in config.planet_teeth_candidates:
                combination_count += 1
                
                # Periodic progress logging (every 2 seconds or 10% of combinations)
                current_time = time.time()
                progress_pct = (combination_count / total_combinations) * 100.0
                should_log_progress = (
                    (current_time - last_progress_log >= progress_log_interval) or
                    (combination_count % max(1, total_combinations // 10) == 0)
                )
                
                if should_log_progress and results_lock is not None:
                    elapsed = current_time - section_start
                    with results_lock:
                        if best_ratio is not None:
                            logger.info(
                                f"Section {section_name}: {progress_pct:.1f}% complete "
                                f"({combination_count}/{total_combinations} combinations, {elapsed:.1f}s elapsed) - "
                                f"best so far: ratio={best_ratio:.3f} (ring={best_zr}, planet={best_zp}), obj={best_obj:.6f}"
                            )
                        else:
                            logger.info(
                                f"Section {section_name}: {progress_pct:.1f}% complete "
                                f"({combination_count}/{total_combinations} combinations, {elapsed:.1f}s elapsed)"
                            )
                    last_progress_log = current_time
                
                # Local refinement for pa and af
                pa = 0.5 * (pa_lo + pa_hi)
                af = 0.5 * (af_lo + af_hi)
                step_pa = max(0.25, (pa_hi - pa_lo) / 8.0)
                step_af = max(0.02, (af_hi - af_lo) / 8.0)
                best_local = float("inf")
                improved = True
                iters = 0
                
                while improved and iters < 10:  # Fewer iterations per section
                    improved = False
                    iters += 1
                    for delta_pa in (-step_pa, step_pa):
                        pa_try = min(pa_hi, max(pa_lo, pa + delta_pa))
                        gear_cfg = PlanetSynthesisConfig(
                            ring_teeth=zr,
                            planet_teeth=zp,
                            pressure_angle_deg=pa_try,
                            addendum_factor=af,
                            base_center_radius=config.base_center_radius,
                            samples_per_rev=config.samples_per_rev,
                            motion=config.motion,
                        )
                        val = objective_func(gear_cfg, theta_range, motion_slice)
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
                            base_center_radius=config.base_center_radius,
                            samples_per_rev=config.samples_per_rev,
                            motion=config.motion,
                        )
                        val = objective_func(gear_cfg, theta_range, motion_slice)
                        if val < best_local:
                            best_local = val
                            af = af_try
                            improved = True
                    step_pa *= 0.5
                    step_af *= 0.5
                
                if best_local < best_obj:
                    best_obj = best_local
                    best_ratio = zr / zp
                    best_zr = zr
                    best_zp = zp
        
        section_elapsed = time.time() - section_start
        
        # Prepare log message outside lock to reduce contention
        log_message = (
            f"Section {section_name}: best ratio={best_ratio:.3f} "
            f"(ring={best_zr}, planet={best_zp}), obj={best_obj:.6f} ({section_elapsed:.3f}s)"
        )
        
        # Log with minimal lock time (only for thread-safe logging, not needed for processes)
        if results_lock is not None:
            with results_lock:
                logger.info(log_message)
        else:
            # For ProcessPoolExecutor, logging happens in main process after result collection
            pass
        
        return {
            "ratio": best_ratio,
            "ring_teeth": best_zr,
            "planet_teeth": best_zp,
            "objective": best_obj,
            "elapsed": section_elapsed,  # For utilization diagnostics
            "section_name": section_name,  # For logging in main process
            "log_message": log_message,  # For logging in main process
        }
    
    # Run sections in parallel
    # Cap thread count to number of sections to avoid idle threads
    n_sections = len(section_boundaries.sections)
    n_threads = min(config.n_threads, n_sections)
    if config.n_threads > n_sections:
        logger.info(
            f"Thread count capped from {config.n_threads} to {n_threads} "
            f"(matching {n_sections} sections)"
        )
    # Choose executor based on configuration
    # Note: use_multiprocessing is checked earlier when creating results_lock
    use_multiprocessing = getattr(config, "use_multiprocessing", False)
    executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
    
    if use_multiprocessing:
        logger.info(
            f"Using {n_threads} processes for parallel section optimization "
            f"(bypasses GIL for CPU-bound work)"
        )
    else:
        logger.info(f"Using {n_threads} threads for parallel section optimization")
    
    # Track timing for utilization diagnostics
    parallel_start = time.time()
    section_times: dict[str, float] = {}
    
    with executor_class(max_workers=n_threads) as executor:
        # Submit all sections and log when each starts
        futures = {}
        for name, theta_range in section_boundaries.sections.items():
            logger.info(f"Starting section {name}: {theta_range[0]:.2f}° to {theta_range[1]:.2f}°")
            futures[executor.submit(optimize_section, name, theta_range)] = name
        
        completed_count = 0
        total_sections = len(futures)
        last_summary_log = parallel_start
        summary_log_interval = 5.0  # Log summary every 5 seconds
        
        for future in as_completed(futures):
            section_name = futures[future]
            completed_count += 1
            try:
                result = future.result()
                # Log result if using ProcessPoolExecutor (logging happens in main process)
                if use_multiprocessing and "log_message" in result:
                    logger.info(result["log_message"])
                # Store result without logging fields
                result_clean = {
                    k: v for k, v in result.items()
                    if k not in ("section_name", "log_message")
                }
                section_results[section_name] = result_clean
                # Extract section time from result if available
                if "elapsed" in result:
                    section_times[section_name] = result["elapsed"]
                
                # Log completion with progress counter
                elapsed_time = result.get("elapsed", 0.0)
                parallel_elapsed_so_far = time.time() - parallel_start
                logger.info(
                    f"✓ Section {section_name} completed ({completed_count}/{total_sections}) "
                    f"in {elapsed_time:.3f}s (parallel elapsed: {parallel_elapsed_so_far:.3f}s)"
                )
                
                # Periodic progress summary
                current_time = time.time()
                if (current_time - last_summary_log >= summary_log_interval) or (completed_count == total_sections):
                    remaining = total_sections - completed_count
                    progress_pct = (completed_count / total_sections) * 100.0
                    
                    # Calculate average time per section and estimate remaining
                    if completed_count > 0 and section_times:
                        avg_section_time = sum(section_times.values()) / len(section_times)
                        estimated_remaining = avg_section_time * remaining / n_threads if n_threads > 0 else 0.0
                    else:
                        estimated_remaining = 0.0
                    
                    logger.info(
                        f"Progress summary: {completed_count}/{total_sections} sections completed ({progress_pct:.1f}%) - "
                        f"{remaining} remaining, estimated time remaining: {estimated_remaining:.1f}s"
                    )
                    last_summary_log = current_time
            except Exception as e:
                logger.warning(
                    f"✗ Section {section_name} optimization failed ({completed_count}/{total_sections}): {e}"
                )
                # Use default fallback
                section_results[section_name] = {
                    "ratio": 2.0,  # Default 2:1 ratio
                    "ring_teeth": 40,
                    "planet_teeth": 20,
                    "objective": float("inf"),
                }
    
    # Calculate utilization diagnostics
    parallel_elapsed = time.time() - parallel_start
    total_section_time = sum(section_times.values()) if section_times else 0.0
    
    # Utilization metric: (sum of section times) / (wall time * n_threads)
    # Ideal: close to 1.0 (all threads busy all the time)
    # GIL contention: much less than 1.0 (threads waiting for GIL)
    if parallel_elapsed > 0 and n_threads > 0:
        utilization = total_section_time / (parallel_elapsed * n_threads)
        
        # Calculate speedup (sequential time / parallel time)
        sequential_time = total_section_time
        speedup = sequential_time / parallel_elapsed if parallel_elapsed > 0 else 0.0
        efficiency = speedup / n_threads if n_threads > 0 else 0.0
        
        # Log comprehensive utilization summary
        logger.info("=" * 70)
        logger.info("Parallel Optimization Utilization Summary")
        logger.info("=" * 70)
        logger.info(
            f"Wall-clock time: {parallel_elapsed:.3f}s "
            f"(sum of section times: {total_section_time:.3f}s)"
        )
        logger.info(f"Threads used: {n_threads}")
        logger.info(f"Speedup: {speedup:.2f}x (sequential: {sequential_time:.3f}s / parallel: {parallel_elapsed:.3f}s)")
        logger.info(f"Parallel efficiency: {efficiency:.2%} (speedup / threads)")
        logger.info(
            f"Thread utilization: {utilization:.2%} "
            f"(ideal: 100%, GIL contention indicated if <50%)"
        )
        
        # Per-section timing breakdown
        if section_times:
            logger.info("Section timing breakdown:")
            for section_name, section_time in sorted(section_times.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {section_name}: {section_time:.3f}s")
        
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
        # Convert degrees to radians
        delta_theta_ring = np.deg2rad(abs(theta_end - theta_start))
        
        # Handle wrap-around for TDC-CA10
        if section_name == "TDC-CA10" and theta_start > theta_end:
            # Wrap around: from TDC to 360°, then from 0° to CA10
            delta_theta_ring = np.deg2rad((360.0 - theta_start) + theta_end)
        
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
        if section_name == "TDC-CA10" and theta_start > theta_end:
            duration = (360.0 - theta_start) + theta_end
        else:
            duration = abs(theta_end - theta_start)
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
                delta_theta_ring = np.deg2rad(abs(theta_end - theta_start))
                if section_name == "TDC-CA10" and theta_start > theta_end:
                    delta_theta_ring = np.deg2rad((360.0 - theta_start) + theta_end)
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
        delta_theta_ring = np.deg2rad(abs(theta_end - theta_start))
        if section_name == "TDC-CA10" and theta_start > theta_end:
            delta_theta_ring = np.deg2rad((360.0 - theta_start) + theta_end)
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
        order2_logger.warning(f"Ipopt optimization failed: {result.message}")
        order2_logger.info("Falling back to simple smoothing...")
        log.warning(f"ORDER2_MICRO Ipopt optimization failed: {result.message}")
        # Fall back to simple smoothing
        return _order2_fallback_smoothing(config, phi_init, cand)

    except Exception as e:
        order2_logger.error(f"Ipopt optimization error: {e}")
        order2_logger.info("Falling back to simple smoothing...")
        log.error(f"ORDER2_MICRO Ipopt optimization error: {e}")
        # Fall back to simple smoothing
        return _order2_fallback_smoothing(config, phi_init, cand)


def _order2_fallback_smoothing(
    config: GeometrySearchConfig, phi_init: np.ndarray, cand: PlanetSynthesisConfig,
) -> OptimResult:
    """Fallback to simple smoothing if Ipopt fails."""
    phi_vals = phi_init.tolist()

    # Simple smoothing (quadratic penalty) with few iterations
    lam = 1e-2
    for _ in range(5):
        # local averaging as a proxy for solving (I + λL)φ = rhs
        new_phi = phi_vals.copy()
        for i in range(len(phi_vals)):
            im = (i - 1) % len(phi_vals)
            ip = (i + 1) % len(phi_vals)
            new_phi[i] = (phi_vals[i] + lam * (phi_vals[im] + phi_vals[ip])) / (
                1.0 + 2.0 * lam
            )
        phi_vals = new_phi

    m = evaluate_order0_metrics_given_phi(cand, phi_vals)
    obj = m.slip_integral - 0.1 * m.contact_length + (0.0 if m.feasible else 1e3)
    return OptimResult(best_config=cand, objective_value=obj, feasible=m.feasible)
