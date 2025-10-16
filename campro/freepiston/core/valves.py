from __future__ import annotations

from campro.logging import get_logger

log = get_logger(__name__)


def clamp_area(A: float, A_max: float) -> float:
    if A <= 0.0:
        return 0.0
    if A_max <= A:
        return A_max
    return A


def effective_area_linear(*, lift: float, A_max: float) -> float:
    """Map normalized lift in [0,1] to effective area with clamping."""
    return clamp_area(A_max * max(0.0, min(1.0, lift)), A_max)


