from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple

from .motion import RadialSlotMotion


@dataclass(frozen=True)
class PlanetSynthesisConfig:
    ring_teeth: int
    planet_teeth: int
    pressure_angle_deg: float
    addendum_factor: float
    base_center_radius: float  # R0 at TDC
    samples_per_rev: int
    motion: RadialSlotMotion


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
