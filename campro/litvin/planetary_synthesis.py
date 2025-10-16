from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, pi, sin
from typing import List, Tuple

from campro.constants import PROFILE_CLOSURE_TOL
from campro.logging import get_logger

from .config import PlanetSynthesisConfig
from .involute_internal import InternalGearParams, InvoluteFlank, sample_internal_flank
from .kinematics import PlanetKinematics

log = get_logger(__name__)


@dataclass(frozen=True)
class PlanetToothProfile:
    points: Sequence[Tuple[float, float]]


 # Config imported from campro.litvin.config


def _rotate(theta: float, x: float, y: float) -> Tuple[float, float]:
    c = cos(theta)
    s = sin(theta)
    return c * x - s * y, s * x + c * y


def _interpolate_flank(flank: InvoluteFlank, phi: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    phis = flank.phi
    if phi <= phis[0]:
        return flank.points[0], flank.tangents[0]
    if phi >= phis[-1]:
        return flank.points[-1], flank.tangents[-1]
    idx = bisect_left(phis, phi)
    phi0 = phis[idx - 1]
    phi1 = phis[idx]
    t = (phi - phi0) / (phi1 - phi0)
    x0, y0 = flank.points[idx - 1]
    x1, y1 = flank.points[idx]
    tx0, ty0 = flank.tangents[idx - 1]
    tx1, ty1 = flank.tangents[idx]
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)
    tx = tx0 + t * (tx1 - tx0)
    ty = ty0 + t * (ty1 - ty0)
    return (x, y), (tx, ty)


def _planet_coords(flank: InvoluteFlank, kin: PlanetKinematics, phi: float, theta_r: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    (x_ring, y_ring), (tx_ring, ty_ring) = _interpolate_flank(flank, phi)

    x_world, y_world = _rotate(theta_r, x_ring, y_ring)
    tx_world, ty_world = _rotate(theta_r, tx_ring, ty_ring)

    d = kin.center_distance(theta_r)
    theta_p = kin.planet_angle(theta_r)

    x_rel = x_world - d
    y_rel = y_world
    x_planet, y_planet = _rotate(-theta_p, x_rel, y_rel)
    tx_planet, ty_planet = _rotate(-theta_p, tx_world, ty_world)
    return (x_planet, y_planet), (tx_planet, ty_planet)


def _partial_theta(flank: InvoluteFlank, kin: PlanetKinematics, phi: float, theta_r: float, h: float) -> Tuple[float, float]:
    x1, _ = _planet_coords(flank, kin, phi, theta_r + h)
    x0, _ = _planet_coords(flank, kin, phi, theta_r - h)
    return ((x1[0] - x0[0]) / (2 * h), (x1[1] - x0[1]) / (2 * h))


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _newton_solve_phi(
    flank: InvoluteFlank,
    kin: PlanetKinematics,
    theta_r: float,
    phi_init: float,
    max_iter: int = 12,
    tol: float = 1e-8,
) -> float | None:
    h = 1e-4
    phi = phi_init
    for _ in range(max_iter):
        (_, tangent) = _planet_coords(flank, kin, phi, theta_r)
        dtheta = _partial_theta(flank, kin, phi, theta_r, h)
        f = _cross(tangent[0], tangent[1], dtheta[0], dtheta[1])
        # numerical derivative df/dphi via central differences
        (_, t_p) = _planet_coords(flank, kin, phi + h, theta_r)
        dtheta_p = _partial_theta(flank, kin, phi + h, theta_r, h)
        f_p = _cross(t_p[0], t_p[1], dtheta_p[0], dtheta_p[1])
        (_, t_m) = _planet_coords(flank, kin, phi - h, theta_r)
        dtheta_m = _partial_theta(flank, kin, phi - h, theta_r, h)
        f_m = _cross(t_m[0], t_m[1], dtheta_m[0], dtheta_m[1])
        df_dphi = (f_p - f_m) / (2 * h)
        if abs(df_dphi) < 1e-12:
            break
        step = -f / df_dphi
        phi = phi + step
        if abs(step) < tol:
            return phi
    return None


def synthesize_planet_from_motion(config: PlanetSynthesisConfig) -> PlanetToothProfile:
    params = InternalGearParams(
        teeth=config.ring_teeth,
        module=config.base_center_radius * 2.0 / max(config.ring_teeth - config.planet_teeth, 1),
        pressure_angle_deg=config.pressure_angle_deg,
        addendum_factor=config.addendum_factor,
    )
    flank = sample_internal_flank(params, n=256)
    kin = PlanetKinematics(R0=config.base_center_radius, motion=config.motion)

    n_theta = max(32, int(config.samples_per_rev))
    theta_vals = [2.0 * pi * i / n_theta for i in range(n_theta + 1)]
    h = 1e-4
    pts: List[Tuple[float, float]] = []

    # predictor-corrector: start from minimal |w|, then use Newton with previous phi
    prev_phi: float | None = None
    for theta_r in theta_vals:
        crossings: List[Tuple[float, float]] = []
        prev_w = None
        prev_phi = None
        for phi in flank.phi:
            (point, tangent) = _planet_coords(flank, kin, phi, theta_r)
            dtheta = _partial_theta(flank, kin, phi, theta_r, h)
            w = _cross(tangent[0], tangent[1], dtheta[0], dtheta[1])
            if prev_w is not None and prev_phi is not None:
                if w == 0.0:
                    crossings.append((phi, w))
                elif prev_w == 0.0:
                    crossings.append((prev_phi, prev_w))
                elif w * prev_w < 0.0:
                    # Linear interpolation
                    t = abs(prev_w) / (abs(prev_w) + abs(w))
                    phi_star = prev_phi + t * (phi - prev_phi)
                    crossings.append((phi_star, 0.0))
            prev_w = w
            prev_phi = phi

        phi_seed = crossings[0][0] if crossings else None
        if phi_seed is None:
            best_phi = flank.phi[0]
            best_val = float("inf")
            for phi in flank.phi:
                (_, tangent) = _planet_coords(flank, kin, phi, theta_r)
                dtheta = _partial_theta(flank, kin, phi, theta_r, h)
                w = abs(_cross(tangent[0], tangent[1], dtheta[0], dtheta[1]))
                if w < best_val:
                    best_val = w
                    best_phi = phi
            phi_seed = best_phi

        if prev_phi is not None:
            # Use previous phi as a better seed (continuation)
            phi_seed = prev_phi

        phi_newton = _newton_solve_phi(flank, kin, theta_r, phi_seed)
        phi_contact = phi_newton if phi_newton is not None else phi_seed

        (pt, _) = _planet_coords(flank, kin, phi_contact, theta_r)
        pts.append(pt)
        prev_phi = phi_contact

    # Closure check
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    if dist > PROFILE_CLOSURE_TOL:
        log.warning("Planet profile closure residual %.3e exceeds tolerance %.3e", dist, PROFILE_CLOSURE_TOL)

    # Replicate single-contact path into tooth profile over z_p teeth by rotating
    # The current pts trace one flank path over θ_r∈[0,2π]. Use z_p copies spaced by 2π/z_p.
    if config.planet_teeth <= 1:
        return PlanetToothProfile(points=pts)

    replicated: List[Tuple[float, float]] = []
    for k in range(config.planet_teeth):
        ang = 2.0 * pi * k / config.planet_teeth
        c = cos(ang)
        s = sin(ang)
        for (x, y) in pts:
            replicated.append((c * x - s * y, s * x + c * y))

    return PlanetToothProfile(points=replicated)



