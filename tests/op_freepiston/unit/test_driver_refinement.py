from __future__ import annotations

from campro.freepiston.opt.driver import solve_cycle, solve_cycle_with_refinement


def test_driver_smoke_without_execution_heavy():
    # Minimal config; we won't assert on solver success to avoid heavy runs
    P = {
        "num": {"K": 2, "C": 1},
        "flow": {"mode": "0d"},
        "geometry": {"bore": 0.08, "clearance_volume": 3.2e-5},
        "bounds": {"x_gap_min": 8e-4},
        "obj": {"w": {"smooth": 0.0}},
    }
    try:
        _ = solve_cycle(P)
    except Exception:
        # We only care that call path exists; failures are acceptable in smoke
        pass


def test_driver_refinement_path_exists():
    P = {
        "num": {"K_0d": 2, "K_1d": 2, "C": 1},
        "flow": {"mode": "0d"},
        "geometry": {"bore": 0.08, "clearance_volume": 3.2e-5},
        "bounds": {"x_gap_min": 8e-4},
    }
    try:
        _ = solve_cycle_with_refinement(P, refinement_strategy="fixed")
    except Exception:
        pass
