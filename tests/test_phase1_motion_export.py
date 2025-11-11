from __future__ import annotations

import numpy as np

from campro.optimization.freepiston_phase1_adapter import FreePistonPhase1Adapter
from campro.constraints.cam import CamMotionConstraints


class DummySolution:
    def __init__(self, x_opt: np.ndarray, K: int, C: int, nodes: np.ndarray, omega_deg_per_s: float = 360.0):
        self.meta = {
            "optimization": {
                "success": True,
                "x_opt": x_opt,
                "iterations": 5,
                "cpu_time": 0.01,
                "f_opt": 1.0,
                "message": "ok",
            },
            "grid": type("Grid", (), {"nodes": nodes})(),  # minimal grid holder
            "meta": {
                "K": K,
                "C": C,
                "variable_groups": {
                    # positions and velocities: [xL0, xR0] + K*C pairs
                    "positions": list(range(0, 2 + 2 * K * C)),
                    "velocities": list(range(2 + 2 * K * C, 2 + 4 * K * C + 2)),
                    # densities and temperatures for pressure: initial + K*C
                    "densities": list(range(2 + 4 * K * C + 2, 3 + 4 * K * C + 2 + K * C)),
                    "temperatures": list(range(3 + 4 * K * C + 2 + K * C, 4 + 4 * K * C + 2 + 2 * K * C)),
                },
                "combustion_model": {"omega_deg_per_s": omega_deg_per_s},
            },
        }
        self.data = {}


def _build_dummy_solution(K: int = 4, C: int = 2) -> DummySolution:
    # Build x_opt vector consistent with variable_groups indexing
    n_pos = 2 + 2 * K * C
    n_vel = 2 + 2 * K * C
    n_den = 1 + K * C
    n_tmp = 1 + K * C
    # Simple ramps for positions and velocities; densities and temps constant
    x_pos = np.linspace(0.0, 0.01, n_pos)  # meters
    x_vel = np.linspace(0.0, 0.2, n_vel)   # m/s
    rho = np.ones(n_den) * 1.2             # kg/m^3
    T = np.ones(n_tmp) * 300.0             # K
    x_opt = np.concatenate([x_pos, x_vel, rho, T])
    # Radau-like nodes (monotone within element)
    nodes = np.linspace(0.25, 1.0, C)
    return DummySolution(x_opt=x_opt, K=K, C=C, nodes=nodes)


def test_freepiston_adapter_exports_motion_and_pressure():
    adapter = FreePistonPhase1Adapter()
    constraints = CamMotionConstraints(
        stroke=50.0,  # mm
        upstroke_duration_percent=50.0,
        vmax=1000.0,
        amax=1e4,
        jmax=1e6,
    )
    sol = _build_dummy_solution(K=3, C=2)
    # Use private method for focused unit test
    result = adapter._convert_solution_to_result(
        solution=sol,
        cam_constraints=constraints,
        cycle_time=1.0,
        n_points=360,
    )
    assert result is not None
    assert result.solution is not None
    # Authoritative series
    assert "time_s" in result.solution
    assert "position_mm" in result.solution
    assert "pressure_pa" in result.solution
    t = np.asarray(result.solution["time_s"])
    x = np.asarray(result.solution["position_mm"])
    p = np.asarray(result.solution["pressure_pa"])
    assert t.ndim == 1 and x.ndim == 1 and p.ndim == 1
    assert len(t) == len(x) == len(p)
    # Back-compat series
    assert "cam_angle" in result.solution and "position" in result.solution
    # Metadata pressure mappings
    pr = result.metadata.get("pressure", {})
    assert isinstance(pr, dict)
    assert "vs_time" in pr and "vs_theta" in pr

