from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    pressure_angle_deg_bounds: tuple[float, float]
    addendum_factor_bounds: tuple[float, float]
    base_center_radius: float
    samples_per_rev: int
    motion: RadialSlotMotion
    section_boundaries: Any | None = None  # SectionBoundaries from section_analysis
    n_threads: int = 4
    use_multiprocessing: bool = True  # Use ProcessPoolExecutor instead of ThreadPoolExecutor for CPU-bound work
    # Actual theta arrays - when provided, these override samples_per_rev for grid generation
    theta_deg: np.ndarray | None = field(default=None, compare=False)
    theta_rad: np.ndarray | None = field(default=None, compare=False)
    # Position array for recreating motion object in worker processes (avoids pickling lambda functions)
    position: np.ndarray | None = field(default=None, compare=False)
    # Target ratio profile ρ_target(θ) for synchronized ring radius optimization
    rho_target: np.ndarray | None = field(default=None, compare=False)


class OptimizationOrder:
    ORDER0_EVALUATE = 0
    ORDER1_GEOMETRY = 1
    ORDER2_MICRO = 2
    ORDER3_CO_MOTION = 3
