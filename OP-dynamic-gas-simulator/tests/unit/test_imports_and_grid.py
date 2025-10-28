from __future__ import annotations

from campro.freepiston.opt.colloc import make_grid


def test_make_grid_basic():
    grid = make_grid(5, 3, kind="radau")
    assert len(grid.nodes) == 3
    assert len(grid.weights) == 3
    assert len(grid.a) == 3
    assert len(grid.a[0]) == 3
