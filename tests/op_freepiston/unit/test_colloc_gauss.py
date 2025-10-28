from __future__ import annotations

import math

from campro.freepiston.opt.colloc import make_grid


def test_gauss_legendre_s2_nodes_weights():
    g = make_grid(6, 2, kind="gauss")
    c1 = 0.5 - math.sqrt(3.0) / 6.0
    c2 = 0.5 + math.sqrt(3.0) / 6.0
    assert abs(g.nodes[0] - c1) < 1e-12
    assert abs(g.nodes[1] - c2) < 1e-12
    assert abs(sum(g.weights) - 1.0) < 1e-12
    assert len(g.a) == 2 and len(g.a[0]) == 2
