from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CollocationGrid:
    theta: Sequence[float]


def make_uniform_grid(n: int) -> CollocationGrid:
    import math

    n = max(8, int(n))
    theta = [2.0 * math.pi * i / n for i in range(n)]
    return CollocationGrid(theta=theta)


def central_diff(values: Sequence[float], h: float) -> Tuple[Sequence[float], Sequence[float]]:
    n = len(values)
    d = [0.0] * n
    d2 = [0.0] * n
    for i in range(n):
        ip = (i + 1) % n
        im = (i - 1) % n
        d[i] = (values[ip] - values[im]) / (2.0 * h)
        d2[i] = (values[ip] - 2.0 * values[i] + values[im]) / (h * h)
    return d, d2


@dataclass(frozen=True)
class MicroRelief:
    control_points: Sequence[float]


def relief_value(relief: MicroRelief, s: float) -> float:
    n = max(2, len(relief.control_points))
    s = max(0.0, min(1.0, s))
    t = s * (n - 1)
    i = int(t)
    if i >= n - 1:
        return relief.control_points[-1]
    w = t - i
    return (1.0 - w) * relief.control_points[i] + w * relief.control_points[i + 1]


