from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RadialSlotMotion:
    center_offset_fn: Callable[[float], float]
    planet_angle_fn: Callable[[float], float]
    d_center_offset_fn: Callable[[float], float] | None = None
    d2_center_offset_fn: Callable[[float], float] | None = None


