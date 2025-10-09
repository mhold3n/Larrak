from __future__ import annotations

import numpy as np

from campro.freepiston.net1d.mesh import MovingBoundaryMesh
from campro.freepiston.net1d.stepper import TimeStepParameters, ale_conservative_update


def test_ale_update_conservation_no_flux_no_sources():
    # Domain 1.0 length, 10 cells, unit area -> volumes = dx
    n_cells = 10
    mesh = MovingBoundaryMesh(n_cells=n_cells, x_left=0.0, x_right=1.0)

    # Uniform state: rho=1, u=0, E=2.5 (arbitrary), so U = [1, 0, 2.5]
    rho = np.ones(n_cells)
    u = np.zeros(n_cells)
    E = 2.5 * np.ones(n_cells)
    U = np.stack([rho, rho * u, rho * E], axis=0)

    # No sources, no boundary motion -> conservation over a step
    mesh.update(x_left=0.0, x_right=1.0, v_left=0.0, v_right=0.0)

    dt = 1e-3
    params = TimeStepParameters(dt_initial=dt)

    U_new = ale_conservative_update(U, mesh, dt)

    # Check mass, momentum, energy conservation to numerical tolerance
    def totals(U_arr: np.ndarray, m: MovingBoundaryMesh) -> tuple[float, float, float]:
        V = m.cell_volumes()
        mass = float(np.sum(U_arr[0] * V))
        mom = float(np.sum(U_arr[1] * V))
        ener = float(np.sum(U_arr[2] * V))
        return mass, mom, ener

    M0, P0, E0 = totals(U, mesh)
    M1, P1, E1 = totals(U_new, mesh)

    assert abs(M1 - M0) < 1e-12
    assert abs(P1 - P0) < 1e-12
    assert abs(E1 - E0) < 1e-12

