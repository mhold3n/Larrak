"""Phase 2 IPOPT integration test against golden references."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Ensure project root on path for standalone execution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from campro.optimization.cam_ring_optimizer import CamRingOptimizer

try:
    from .common import (
        arrays_close,
        heavy_env_enabled,
        load_golden_json,
        require_golden_initialized,
        save_golden_json,
        serialize_array,
    )
except ImportError:  # pragma: no cover - script execution
    from tests.heavy.common import (
        arrays_close,
        heavy_env_enabled,
        load_golden_json,
        require_golden_initialized,
        save_golden_json,
        serialize_array,
    )

PHASE1_GOLDEN = "phase1_ipopt_reference.json"
PHASE2_GOLDEN = "phase2_ipopt_reference.json"


def _build_primary_data_from_phase1() -> dict[str, Any]:
    phase1 = load_golden_json(PHASE1_GOLDEN)
    require_golden_initialized(phase1, PHASE1_GOLDEN)
    arrays = phase1.get("arrays", {})
    theta = np.asarray(arrays.get("primary_theta"), dtype=float)
    position = np.asarray(arrays.get("primary_position"), dtype=float)
    velocity = np.asarray(arrays.get("primary_velocity"), dtype=float)
    if theta.size == 0 or position.size == 0:
        raise RuntimeError(
            "Phase 1 golden reference has no theta/position data; run phase1 integration with --update",
        )
    acceleration = np.gradient(velocity, theta) if velocity.size else np.zeros_like(theta)
    return {
        "cam_angle": theta,
        "theta_deg": theta,
        "theta_rad": np.deg2rad(theta),
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
    }


def _run_phase2_optimization() -> dict[str, Any]:
    primary_data = _build_primary_data_from_phase1()
    optimizer = CamRingOptimizer(enable_order2_micro=True)
    result = optimizer.optimize(primary_data=primary_data)
    if result.status != result.status.CONVERGED:
        raise RuntimeError(f"CamRingOptimizer failed: {result.metadata}")

    solution = result.solution or {}
    ring_profile = solution.get("ring_profile", {})
    cam_profile = solution.get("cam_profile", {})
    gear_geometry = solution.get("gear_geometry", {})

    return {
        "initialized": True,
        "schema_version": 1,
        "metadata": {
            "optimized_gear_config": result.metadata.get("optimized_gear_config"),
            "objective_value": result.objective_value,
            "order_results": result.metadata.get("order_results"),
        },
        "arrays": {
            "cam_theta": serialize_array(cam_profile.get("theta", [])),
            "cam_profile_radius": serialize_array(cam_profile.get("profile_radius", [])),
            "ring_psi": serialize_array(ring_profile.get("psi", [])),
            "ring_radius": serialize_array(ring_profile.get("R_ring", [])),
            "planet_radius": serialize_array(ring_profile.get("R_planet", [])),
            "gear_base_circle_cam": float(gear_geometry.get("base_circle_cam", 0.0)),
            "gear_base_circle_ring": float(gear_geometry.get("base_circle_ring", 0.0)),
        },
    }


def _compare_against_golden(update: bool) -> None:
    if update:
        payload = _run_phase2_optimization()
        save_golden_json(PHASE2_GOLDEN, payload)
        print(f"Updated golden reference at {PHASE2_GOLDEN}")
        return

    try:
        golden = load_golden_json(PHASE2_GOLDEN)
        require_golden_initialized(golden, PHASE2_GOLDEN)
    except (FileNotFoundError, RuntimeError):
        print("Phase 2 golden reference missing/uninitialized; regenerating now...")
        payload = _run_phase2_optimization()
        save_golden_json(PHASE2_GOLDEN, payload)
        print(f"Created golden reference at {PHASE2_GOLDEN}")
        return
    actual = _run_phase2_optimization()

    gold_arrays = golden.get("arrays", {})
    act_arrays = actual.get("arrays", {})

    for key in ("cam_theta", "cam_profile_radius", "ring_psi", "ring_radius", "planet_radius"):
        arrays_close(key, np.asarray(act_arrays[key]), np.asarray(gold_arrays[key]))

    for key in ("gear_base_circle_cam", "gear_base_circle_ring"):
        arrays_close(key, np.asarray([act_arrays[key]]), np.asarray([gold_arrays[key]]), rtol=1e-6, atol=1e-9)


@pytest.mark.ipopt_phase2
def test_phase2_ipopt_against_golden() -> None:
    if not heavy_env_enabled():
        pytest.skip(
            "CAMPRO_RUN_HEAVY_TESTS not set – skipping Phase 2 IPOPT integration test",
        )
    _compare_against_golden(update=False)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 IPOPT integration harness")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Run the optimization and refresh the golden reference",
    )
    return parser.parse_args(argv or sys.argv[1:])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not heavy_env_enabled():
        print(
            "CAMPRO_RUN_HEAVY_TESTS is not enabled. Set it to 1 to run this script.",
            file=sys.stderr,
        )
        return 2

    try:
        _compare_against_golden(update=args.update)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
