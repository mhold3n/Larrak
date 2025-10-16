from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class CollocationGrid:
    nodes: List[float]
    weights: List[float]
    a: List[List[float]]  # coefficients a_cj


def make_grid(K: int, C: int, kind: Literal["radau", "gauss"] = "radau") -> CollocationGrid:
    """Return a minimal collocation grid skeleton.

    This is a placeholder returning evenly spaced nodes and simple weights.
    Replace with Radau IIA / Gauss–Legendre sets and validated coefficients.
    """
    if K <= 0 or C <= 0:
        raise ValueError("K and C must be positive integers")
    if kind == "radau":
        # Implement s=1 Radau IIA (implicit Euler equivalence)
        if C == 1:
            return CollocationGrid(nodes=[1.0], weights=[1.0], a=[[1.0]])
        if C == 3:
            # 3-stage Radau IIA (order 5) on [0,1]
            nodes = [
                0.155051025721682,  # c1
                0.644948974278318,  # c2
                1.0,                # c3
            ]
            a = [
                [
                    0.196815477223660,
                    -0.065535425850198,
                    0.023770974348220,
                ],
                [
                    0.394424314739087,
                    0.292073411665228,
                    -0.041548752125998,
                ],
                [
                    0.376403062700467,
                    0.512485826188421,
                    0.111111111111111,
                ],
            ]
            weights = [
                0.376403062700467,
                0.512485826188421,
                0.111111111111111,
            ]
            return CollocationGrid(nodes=nodes, weights=weights, a=a)
        raise NotImplementedError("Radau IIA implemented only for C=1 in this draft")
    # Gauss–Legendre
    if kind == "gauss":
        if C == 2:
            # 2-stage Gauss (order 4) on [0,1]; nodes shifted from [-1,1]
            # c = 0.5 ± sqrt(3)/6; weights = 0.5 each
            import math

            c1 = 0.5 - math.sqrt(3.0) / 6.0
            c2 = 0.5 + math.sqrt(3.0) / 6.0
            nodes = [c1, c2]
            weights = [0.5, 0.5]
            # Butcher A for Gauss s=2
            a = [
                [
                    0.25, 0.25 - math.sqrt(3.0) / 6.0,
                ],
                [
                    0.25 + math.sqrt(3.0) / 6.0, 0.25,
                ],
            ]
            return CollocationGrid(nodes=nodes, weights=weights, a=a)
        raise NotImplementedError("Gauss–Legendre implemented only for C=2 in this draft")
    # Fallback (should not reach)
    raise NotImplementedError(f"Unknown collocation kind: {kind}")


