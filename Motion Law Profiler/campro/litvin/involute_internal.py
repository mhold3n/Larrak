from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, sin, sqrt
from typing import List, Tuple

from campro.constants import DEG_TO_RAD


@dataclass(frozen=True)
class InternalGearParams:
    teeth: int
    module: float
    pressure_angle_deg: float
    addendum_factor: float


@dataclass(frozen=True)
class InvoluteFlank:
    phi: Sequence[float]
    points: Sequence[Tuple[float, float]]
    tangents: Sequence[Tuple[float, float]]


def base_radius(params: InternalGearParams) -> float:
    alpha = params.pressure_angle_deg * DEG_TO_RAD
    rb = params.module * params.teeth / 2.0 * cos(alpha)
    return rb


def pitch_radius(params: InternalGearParams) -> float:
    return params.module * params.teeth / 2.0


def involute_xy(rb: float, phi: float) -> Tuple[float, float]:
    # Parametric involute of a circle of radius rb
    x = rb * (cos(phi) + phi * sin(phi))
    y = rb * (sin(phi) - phi * cos(phi))
    return x, y


def sample_internal_flank(params: InternalGearParams, n: int = 200) -> InvoluteFlank:
    # Returns a nominal involute flank (one side) in the ring local frame.
    rb = base_radius(params)
    if rb <= 0.0:
        raise ValueError("base radius must be positive")

    rp = pitch_radius(params)
    # For internal gear, tooth tips are inside pitch circle
    ra = rp - params.addendum_factor * params.module
    if ra <= 0.0:
        raise ValueError("tip radius must be positive")

    # Solve ra = rb * sqrt(1 + phi^2) => phi_max = sqrt((ra/rb)^2 - 1)
    ratio = max(1.0, (ra / rb) ** 2)
    phi_max = sqrt(ratio - 1.0)

    phi_min = max(1e-4, phi_max / (n * 2.0))
    phis: List[float] = []
    pts: List[Tuple[float, float]] = []
    tangents: List[Tuple[float, float]] = []
    for i in range(n):
        phi = phi_min + (phi_max - phi_min) * i / max(1, n - 1)
        x, y = involute_xy(rb, phi)
        # Tangent vector d/dphi
        tx = rb * phi * cos(phi)
        ty = rb * phi * sin(phi)
        phis.append(phi)
        pts.append((x, y))
        tangents.append((tx, ty))

    return InvoluteFlank(phi=phis, points=pts, tangents=tangents)
