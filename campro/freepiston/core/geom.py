from __future__ import annotations

import math

from campro.logging import get_logger

log = get_logger(__name__)


def piston_area(B: float) -> float:
    return math.pi * (B * 0.5) ** 2


def chamber_volume(*, B: float, Vc: float, x_L: float, x_R: float) -> float:
    """Opposed-piston chamber volume.

    V = Vc + A_p * (x_R - x_L)
    """
    return Vc + piston_area(B) * (x_R - x_L)
