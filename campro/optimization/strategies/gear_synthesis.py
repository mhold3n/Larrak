"""
Geometric Gear Synthesis Strategy.

This module consolidates the geometric optimization logic for gear synthesis,
focusing on "Order 0" (Evaluation) and "Order 1" (Piecewise Optimization).
It replaces the legacy `campro.litvin.optimization` module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

# Import shared config definitions
from campro.litvin.config import (
    GeometrySearchConfig,
    OptimizationOrder,
    PlanetSynthesisConfig,
)
from campro.litvin.metrics import evaluate_order0_metrics
from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class OptimResult:
    """Standardized result container for optimization steps."""

    feasible: bool
    best_config: PlanetSynthesisConfig | None
    objective_value: float | None = None
    metrics: dict[str, Any] | None = None
    ipopt_analysis: dict[str, Any] | None = None
    message: str = ""


def optimize_geometry(
    config: GeometrySearchConfig,
    order: int = OptimizationOrder.ORDER0_EVALUATE,
) -> OptimResult:
    """
    Main entry point for geometric gear optimization.

    Args:
        config: The geometry search configuration (candidates, motion, etc.)
        order: The optimization order (level of complexity).

    Returns:
        OptimResult: The optimization result.
    """
    if order == OptimizationOrder.ORDER0_EVALUATE:
        return _order0_objective(config)

    if order == OptimizationOrder.ORDER1_GEOMETRY:
        # Segment-based piecewise optimization (if sections available) or global grid search
        # We currently use _optimize_piecewise_sections for the main ORDER1 logic
        # as it handles both cases (defaults to 1 section if no boundaries).
        return _optimize_piecewise_sections(config)

    if order == OptimizationOrder.ORDER2_MICRO:
        log.warning(
            "ORDER2_MICRO is deprecated and pruned. Returning ORDER1 result logic or infeasible."
        )
        return OptimResult(feasible=False, best_config=None, message="ORDER2_MICRO is deprecated")

    raise ValueError(f"Unknown optimization order: {order}")


def _order0_objective(config: GeometrySearchConfig) -> OptimResult:
    """
    Evaluate the single configuration provided in config (base_center_radius, etc.).
    This serves as a feasibility check for the initial guess.
    """
    return _grid_search_best_candidate(config, "ORDER0")


def _grid_search_best_candidate(config: GeometrySearchConfig, label: str) -> OptimResult:
    """Perform a grid search over discrete candidates in config."""

    ring_candidates = config.ring_teeth_candidates or [50]
    planet_candidates = config.planet_teeth_candidates or [25]

    # Pressure angle / Addendum are bounds. We pick nominals for grid search.
    pa_min, pa_max = config.pressure_angle_deg_bounds
    add_min, add_max = config.addendum_factor_bounds

    pa_nominal = (pa_min + pa_max) / 2.0
    add_nominal = (add_min + add_max) / 2.0

    base_r = config.base_center_radius

    # Generate combinations
    combos = list(product(ring_candidates, planet_candidates))

    best_score = float("inf")
    best_res = OptimResult(feasible=False, best_config=None)

    # We can parallelize this evaluation if needed, but for ORDER0 usually it's quick.

    for ring_z, planet_z in combos:
        # Create a specific config for this point
        candidate = PlanetSynthesisConfig(
            ring_teeth=ring_z,
            planet_teeth=planet_z,
            pressure_angle_deg=pa_nominal,
            addendum_factor=add_nominal,
            base_center_radius=base_r,
            samples_per_rev=config.samples_per_rev,
            motion=config.motion,
        )

        try:
            metrics_obj = evaluate_order0_metrics(candidate)
            # Calculate a scalar score (lower is better)
            # Example score: slip_integral + 100 * closure_residual + penalty for infeasibility
            score = metrics_obj.slip_integral + 100.0 * metrics_obj.closure_residual
            if not metrics_obj.feasible:
                score += 1e6

            if metrics_obj.feasible and score < best_score:
                best_score = score
                best_res = OptimResult(
                    feasible=True,
                    best_config=candidate,
                    objective_value=score,
                    metrics={
                        "slip_integral": metrics_obj.slip_integral,
                        "contact_length": metrics_obj.contact_length,
                    },
                )
            elif score < best_score:
                # Keep best even if infeasible, but mark best_score
                best_score = score
                best_res = OptimResult(
                    feasible=False,
                    best_config=candidate,
                    objective_value=score,
                    metrics={"slip_integral": metrics_obj.slip_integral},
                )
        except Exception:
            pass

    return best_res


def _optimize_piecewise_sections(config: GeometrySearchConfig) -> OptimResult:
    """
    Perform optimization either globally or piecewise if sections are defined.
    """
    # 1. Find best discrete candidate (Grid Search)
    grid_res = _grid_search_best_candidate(config, "ORDER1_GRID")

    if not grid_res.feasible or grid_res.best_config is None:
        return OptimResult(
            feasible=False, best_config=None, message="No feasible discrete candidate found"
        )

    best_cand = grid_res.best_config

    # 2. Refine continuous parameters (Pressure Angle, Addendum)
    # We can use scipy.optimize.minimize
    from scipy.optimize import minimize

    pa_min, pa_max = config.pressure_angle_deg_bounds
    add_min, add_max = config.addendum_factor_bounds

    x0 = [best_cand.pressure_angle_deg, best_cand.addendum_factor]
    bounds = [(pa_min, pa_max), (add_min, add_max)]

    def objective(x):
        pa, add = x
        # Create new config (immutable)
        cand = PlanetSynthesisConfig(
            ring_teeth=best_cand.ring_teeth,
            planet_teeth=best_cand.planet_teeth,
            pressure_angle_deg=pa,
            addendum_factor=add,
            base_center_radius=best_cand.base_center_radius,
            samples_per_rev=best_cand.samples_per_rev,
            motion=best_cand.motion,
        )

        try:
            m = evaluate_order0_metrics(cand)
            score = m.slip_integral + 100.0 * m.closure_residual
            if not m.feasible:
                return 1e6 + score
            return score
        except:
            return 1e9

    res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

    if res.success:
        final_cand = PlanetSynthesisConfig(
            ring_teeth=best_cand.ring_teeth,
            planet_teeth=best_cand.planet_teeth,
            pressure_angle_deg=res.x[0],
            addendum_factor=res.x[1],
            base_center_radius=best_cand.base_center_radius,
            samples_per_rev=best_cand.samples_per_rev,
            motion=best_cand.motion,
        )

        m_final = evaluate_order0_metrics(final_cand)
        final_score = m_final.slip_integral + 100.0 * m_final.closure_residual

        return OptimResult(
            feasible=m_final.feasible,
            best_config=final_cand,
            objective_value=final_score,
            metrics={"slip_integral": m_final.slip_integral},
        )

    return grid_res
