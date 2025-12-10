from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RadialSlotMotion:
    center_offset_fn: Callable[[float], float]
    planet_angle_fn: Callable[[float], float]
    d_center_offset_fn: Callable[[float], float] | None = None
    d2_center_offset_fn: Callable[[float], float] | None = None
