from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import pi, sqrt

from campro.constants import PROFILE_CLOSURE_TOL
from campro.logging import get_logger

from .config import PlanetSynthesisConfig
from .involute_internal import InternalGearParams, sample_internal_flank
from .kinematics import PlanetKinematics
from .planetary_synthesis import (
    _newton_solve_phi,
    _partial_theta,
    _planet_coords,
)

log = get_logger(__name__)


@dataclass(frozen=True)
class Order0Metrics:
    slip_integral: float
    contact_length: float
    closure_residual: float
    phi_edge_fraction: float
    samples: int
    feasible: bool


def evaluate_order0_metrics(config: PlanetSynthesisConfig) -> Order0Metrics:
    # Build flank and kinematics identically to synthesis
    module = (
        config.base_center_radius
        * 2.0
        / max(config.ring_teeth - config.planet_teeth, 1)
    )
    params = InternalGearParams(
        teeth=config.ring_teeth,
        module=module,
        pressure_angle_deg=config.pressure_angle_deg,
        addendum_factor=config.addendum_factor,
    )
    flank = sample_internal_flank(params, n=256)
    kin = PlanetKinematics(R0=config.base_center_radius, motion=config.motion)

    n_theta = max(64, int(config.samples_per_rev))
    theta_vals = [2.0 * pi * i / n_theta for i in range(n_theta + 1)]
    h = 1e-4

    pts: list[tuple[float, float]] = []
    slips: list[float] = []
    phi_hits: list[float] = []

    prev_phi: float | None = None
    for theta_r in theta_vals:
        # seed from previous for continuity
        if prev_phi is None:
            # pick mid-flank as initial seed
            phi_seed = flank.phi[len(flank.phi) // 2]
        else:
            phi_seed = prev_phi
        phi_contact = _newton_solve_phi(flank, kin, theta_r, phi_seed) or phi_seed
        (pt, tangent) = _planet_coords(flank, kin, phi_contact, theta_r)
        dtheta_vec = _partial_theta(flank, kin, phi_contact, theta_r, h)

        # sliding metric = |dC/dθ · t̂| Δθ (Δθ normalized by uniform grid spacing)
        t_norm = sqrt(max(1e-16, tangent[0] * tangent[0] + tangent[1] * tangent[1]))
        tx = tangent[0] / t_norm
        ty = tangent[1] / t_norm
        slip = abs(dtheta_vec[0] * tx + dtheta_vec[1] * ty)

        pts.append(pt)
        slips.append(slip)
        phi_hits.append(phi_contact)
        prev_phi = phi_contact

    # Closure residual
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    closure = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))

    # Contact length along the path (polyline length)
    length = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        length += sqrt(dx * dx + dy * dy)

    # Edge contact fraction
    phi_min = flank.phi[0]
    phi_max = flank.phi[-1]
    span = max(1e-12, phi_max - phi_min)
    edge_eps = 0.05 * span
    edge_hits = sum(
        1 for p in phi_hits if (p - phi_min) < edge_eps or (phi_max - p) < edge_eps
    )
    edge_frac = edge_hits / max(1, len(phi_hits))

    feasible = (closure <= PROFILE_CLOSURE_TOL) and (edge_frac < 0.2)

    # Normalize slip integral by grid spacing (Δθ = 2π/n)
    dtheta = 2.0 * pi / n_theta
    slip_integral = sum(slips) * dtheta

    return Order0Metrics(
        slip_integral=slip_integral,
        contact_length=length,
        closure_residual=closure,
        phi_edge_fraction=edge_frac,
        samples=len(theta_vals),
        feasible=feasible,
    )


def evaluate_order0_metrics_given_phi(
    config: PlanetSynthesisConfig,
    phi_values: Sequence[float],
) -> Order0Metrics:
    module = (
        config.base_center_radius
        * 2.0
        / max(config.ring_teeth - config.planet_teeth, 1)
    )
    params = InternalGearParams(
        teeth=config.ring_teeth,
        module=module,
        pressure_angle_deg=config.pressure_angle_deg,
        addendum_factor=config.addendum_factor,
    )
    flank = sample_internal_flank(params, n=256)
    kin = PlanetKinematics(R0=config.base_center_radius, motion=config.motion)

    n_theta = len(phi_values) - 1
    if n_theta < 4:
        return Order0Metrics(0.0, 0.0, 1e9, 1.0, 0, False)
    theta_vals = [2.0 * pi * i / n_theta for i in range(n_theta + 1)]
    h = 1e-4

    pts: list[tuple[float, float]] = []
    slips: list[float] = []
    phi_hits: list[float] = []

    for theta_r, phi in zip(theta_vals, phi_values):
        (pt, tangent) = _planet_coords(flank, kin, phi, theta_r)
        dtheta_vec = _partial_theta(flank, kin, phi, theta_r, h)
        # sliding metric = |dC/dθ · t̂|
        t_norm = sqrt(max(1e-16, tangent[0] * tangent[0] + tangent[1] * tangent[1]))
        tx = tangent[0] / t_norm
        ty = tangent[1] / t_norm
        slip = abs(dtheta_vec[0] * tx + dtheta_vec[1] * ty)
        pts.append(pt)
        slips.append(slip)
        phi_hits.append(phi)

    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    closure = sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))

    length = 0.0
    for i in range(len(pts) - 1):
        dx = pts[i + 1][0] - pts[i][0]
        dy = pts[i + 1][1] - pts[i][1]
        length += sqrt(dx * dx + dy * dy)

    # Edge contact fraction
    phi_min = flank.phi[0]
    phi_max = flank.phi[-1]
    span = max(1e-12, phi_max - phi_min)
    edge_eps = 0.05 * span
    edge_hits = sum(
        1 for p in phi_hits if (p - phi_min) < edge_eps or (phi_max - p) < edge_eps
    )
    edge_frac = edge_hits / max(1, len(phi_hits))

    feasible = (closure <= PROFILE_CLOSURE_TOL) and (edge_frac < 0.2)

    dtheta = 2.0 * pi / n_theta
    slip_integral = sum(slips) * dtheta

    return Order0Metrics(
        slip_integral=slip_integral,
        contact_length=length,
        closure_residual=closure,
        phi_edge_fraction=edge_frac,
        samples=len(theta_vals),
        feasible=feasible,
    )
