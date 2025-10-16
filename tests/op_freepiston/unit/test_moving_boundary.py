from __future__ import annotations

from campro.freepiston.net1d.mesh import moving_boundary_mesh


def test_moving_boundary_mesh_basic():
    """Test moving boundary mesh creation."""
    mesh = moving_boundary_mesh(x_L=0.05, x_R=0.15, n_cells=10)
    assert mesh.n_cells == 10
    # Length should be x_R - x_L = 0.1
    expected_length = 0.1
    actual_length = sum(mesh.dx)
    assert abs(actual_length - expected_length) < 1e-12


def test_moving_boundary_mesh_zero_length():
    """Test handling of zero/negative length."""
    mesh = moving_boundary_mesh(x_L=0.1, x_R=0.1, n_cells=5)
    assert mesh.n_cells == 5
    # Should avoid zero length
    assert sum(mesh.dx) > 0.0
