from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from campro.logging import get_logger
from .motion import RadialSlotMotion
from .planetary_synthesis import PlanetSynthesisConfig, synthesize_planet_from_motion
from ..opt.collocation import CollocationGrid, make_uniform_grid
from .metrics import evaluate_order0_metrics, evaluate_order0_metrics_given_phi
from .planetary_synthesis import _newton_solve_phi
from .involute_internal import InternalGearParams, sample_internal_flank
from .kinematics import PlanetKinematics


log = get_logger(__name__)


@dataclass(frozen=True)
class GeometrySearchConfig:
    ring_teeth_candidates: Sequence[int]
    planet_teeth_candidates: Sequence[int]
    pressure_angle_deg_bounds: Tuple[float, float]
    addendum_factor_bounds: Tuple[float, float]
    base_center_radius: float
    samples_per_rev: int
    motion: RadialSlotMotion


class OptimizationOrder:
    ORDER0_EVALUATE = 0
    ORDER1_GEOMETRY = 1
    ORDER2_MICRO = 2
    ORDER3_CO_MOTION = 3


@dataclass(frozen=True)
class OptimResult:
    best_config: PlanetSynthesisConfig | None
    objective_value: float | None
    feasible: bool


def _order0_objective(cfg: PlanetSynthesisConfig) -> float:
    m = evaluate_order0_metrics(cfg)
    # Objective combines slip and penalties on closure and edge-contact
    penalty = 0.0
    if not m.feasible:
        penalty += 1e3
    penalty += 1e2 * m.closure_residual + 50.0 * m.phi_edge_fraction
    return m.slip_integral - 0.1 * m.contact_length + penalty


def optimize_geometry(config: GeometrySearchConfig, order: int = OptimizationOrder.ORDER0_EVALUATE) -> OptimResult:
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
        # Coarse grid + local refinement (Powell-like coordinate search)
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

        for zr in config.ring_teeth_candidates:
            for zp in config.planet_teeth_candidates:
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

        return OptimResult(best_config=best_cfg, objective_value=best_obj, feasible=best_cfg is not None)

    if order == OptimizationOrder.ORDER2_MICRO:
        # Collocation-based refinement of the contact parameter sequence phi(θ)
        n = max(64, config.samples_per_rev)
        grid = make_uniform_grid(n)
        # Construct flank/kinematics once
        module = config.base_center_radius * 2.0 / max(config.ring_teeth_candidates[0] - config.planet_teeth_candidates[0], 1)
        zr = config.ring_teeth_candidates[0]
        zp = config.planet_teeth_candidates[0]
        pa = sum(config.pressure_angle_deg_bounds) / 2.0
        af = sum(config.addendum_factor_bounds) / 2.0
        cand = PlanetSynthesisConfig(
            ring_teeth=zr,
            planet_teeth=zp,
            pressure_angle_deg=pa,
            addendum_factor=af,
            base_center_radius=config.base_center_radius,
            samples_per_rev=config.samples_per_rev,
            motion=config.motion,
        )
        params = InternalGearParams(teeth=zr, module=module, pressure_angle_deg=pa, addendum_factor=af)
        flank = sample_internal_flank(params, n=256)
        kin = PlanetKinematics(R0=config.base_center_radius, motion=config.motion)

        # Initialize phi by Newton per node
        phi_vals: list[float] = []
        seed = flank.phi[len(flank.phi) // 2]
        for theta in grid.theta:
            phi = _newton_solve_phi(flank, kin, theta, seed) or seed
            phi_vals.append(phi)
            seed = phi

        # Simple smoothing (quadratic penalty) with few iterations
        lam = 1e-2
        for _ in range(5):
            # local averaging as a proxy for solving (I + λL)φ = rhs
            new_phi = phi_vals.copy()
            for i in range(len(phi_vals)):
                im = (i - 1) % len(phi_vals)
                ip = (i + 1) % len(phi_vals)
                new_phi[i] = (phi_vals[i] + lam * (phi_vals[im] + phi_vals[ip])) / (1.0 + 2.0 * lam)
            phi_vals = new_phi

        m = evaluate_order0_metrics_given_phi(cand, phi_vals)
        obj = m.slip_integral - 0.1 * m.contact_length + (0.0 if m.feasible else 1e3)
        return OptimResult(best_config=cand, objective_value=obj, feasible=m.feasible)

    # Higher orders will be implemented subsequently
    return OptimResult(best_config=None, objective_value=None, feasible=False)


