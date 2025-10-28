from __future__ import annotations

import math

from campro.freepiston.core.geom import chamber_volume, piston_area


def test_piston_area_and_volume_linear_gap():
    B = 0.082
    Vc = 3.2e-5
    x_L, x_R = 0.02, 0.16
    A = piston_area(B)
    V = chamber_volume(B=B, Vc=Vc, x_L=x_L, x_R=x_R)
    assert abs(A - math.pi * (B * 0.5) ** 2) < 1e-12
    assert abs(V - (Vc + A * (x_R - x_L))) < 1e-12
