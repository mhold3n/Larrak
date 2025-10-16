from __future__ import annotations

from campro.freepiston.opt.colloc import make_grid


def test_colloc_grid_convergence():
    """Test that collocation grids have expected properties."""
    # Radau s=1 should have nodes summing to 1
    g1 = make_grid(5, 1, kind="radau")
    assert abs(sum(g1.weights) - 1.0) < 1e-12

    # Radau s=3 should have nodes summing to 1
    g3 = make_grid(5, 3, kind="radau")
    assert abs(sum(g3.weights) - 1.0) < 1e-12

    # Gauss s=2 should have nodes summing to 1
    g2 = make_grid(5, 2, kind="gauss")
    assert abs(sum(g2.weights) - 1.0) < 1e-12
