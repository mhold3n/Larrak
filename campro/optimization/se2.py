"""
Minimal SE(2) utilities for angle/frame transforms on the universal grid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SE2:
    theta: float  # rotation (radians)
    tx: float = 0.0  # translation x
    ty: float = 0.0  # translation y

    def R(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        c = math.cos(self.theta)
        s = math.sin(self.theta)
        return ((c, -s), (s, c))

    def apply(self, x: float, y: float) -> Tuple[float, float]:
        R = self.R()
        xr = R[0][0] * x + R[0][1] * y + self.tx
        yr = R[1][0] * x + R[1][1] * y + self.ty
        return xr, yr


def angle_map(theta_source: float, scale: float = 1.0, offset: float = 0.0) -> float:
    """Map an angle by scale and offset: theta_target = scale*theta_source + offset."""
    return scale * theta_source + offset


