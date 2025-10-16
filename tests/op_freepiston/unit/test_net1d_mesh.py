from __future__ import annotations

from campro.freepiston.net1d.mesh import uniform_mesh


def test_uniform_mesh_basic():
    mesh = uniform_mesh(length=1.0, n_cells=10)
    assert mesh.n_cells == 10
    assert len(mesh.x) == 10
    assert len(mesh.dx) == 10
    assert abs(mesh.dx[0] - 0.1) < 1e-12
