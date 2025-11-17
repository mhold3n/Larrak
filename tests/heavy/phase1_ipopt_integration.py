"""Phase 1 IPOPT integration test against golden references."""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Ensure project root is importable when running as a standalone script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from campro.optimization.unified_framework import (
    UnifiedOptimizationFramework,
    UnifiedOptimizationSettings,
)

try:
    from .common import (
        GOLDEN_DIR,
        arrays_close,
        heavy_env_enabled,
        load_golden_json,
        require_golden_initialized,
        save_golden_json,
        serialize_array,
    )
except ImportError:  # pragma: no cover - script execution from repo root
    from tests.heavy.common import (
        GOLDEN_DIR,
        arrays_close,
        heavy_env_enabled,
        load_golden_json,
        require_golden_initialized,
        save_golden_json,
        serialize_array,
    )

GOLDEN_FILE = "phase1_ipopt_reference.json"
CANONICAL_INPUT = {
    "stroke": 20.0,
    "cycle_time": 1.0,
    "upstroke_duration_percent": 60.0,
    "zero_accel_duration_percent": 0.0,
    "motion_type": "minimum_jerk",
}

GOLDEN_PRINT_LEVEL = 2


def _get_relaxed_settings() -> UnifiedOptimizationSettings:
    """Return a relaxed optimization settings profile for golden regeneration."""
    settings = UnifiedOptimizationSettings()
    settings.max_iterations = 50
    settings.tolerance = 1e-4
    return settings


@contextlib.contextmanager
def _ipopt_print_level_guard(enable: bool, level: int = GOLDEN_PRINT_LEVEL):
    """
    Temporarily lower IPOPT print level when generating goldens.

    The driver reads CAMPRO_IPOPT_PRINT_LEVEL and applies it to IPOPT options.
    Also enables FREE_PISTON_DEBUG for verbose scaling output.
    """
    if not enable:
        yield
        return

    previous_print_level = os.environ.get("CAMPRO_IPOPT_PRINT_LEVEL")
    previous_debug = os.environ.get("FREE_PISTON_DEBUG")
    
    os.environ["CAMPRO_IPOPT_PRINT_LEVEL"] = str(level)
    os.environ["FREE_PISTON_DEBUG"] = "1"  # Enable debug output for scaling
    
    try:
        yield
    finally:
        if previous_print_level is None:
            os.environ.pop("CAMPRO_IPOPT_PRINT_LEVEL", None)
        else:
            os.environ["CAMPRO_IPOPT_PRINT_LEVEL"] = previous_print_level
        
        if previous_debug is None:
            os.environ.pop("FREE_PISTON_DEBUG", None)
        else:
            os.environ["FREE_PISTON_DEBUG"] = previous_debug


def _run_phase1_optimization(*, use_relaxed_settings: bool = False) -> dict[str, Any]:
    """Execute the full primary optimization flow via UnifiedOptimizationFramework."""
    settings = _get_relaxed_settings() if use_relaxed_settings else None
    framework = UnifiedOptimizationFramework(settings=settings)
    data = framework.optimize_cascaded(CANONICAL_INPUT)

    if data.primary_theta is None or data.primary_position is None:
        raise RuntimeError("Primary optimization did not produce theta/position outputs")

    summary = framework.get_optimization_summary()
    primary_meta = summary.get("convergence_info", {}).get("primary", {})

    return {
        "initialized": True,
        "schema_version": 1,
        "metadata": {
            "motion_type": CANONICAL_INPUT["motion_type"],
            "stroke_mm": CANONICAL_INPUT["stroke"],
            "samples": int(len(data.primary_theta)),
            "assumptions": primary_meta.get("assumptions", {}),
        },
        "arrays": {
            "primary_theta": serialize_array(data.primary_theta),
            "primary_position": serialize_array(data.primary_position),
            "primary_velocity": serialize_array(
                getattr(data, "primary_velocity", np.zeros_like(data.primary_theta)),
            ),
            "primary_load_profile": serialize_array(
                getattr(data, "primary_load_profile", np.zeros_like(data.primary_theta)),
            ),
        },
    }


def _compare_against_golden(update: bool) -> None:
    if update:
        with _ipopt_print_level_guard(True):
            payload = _run_phase1_optimization(use_relaxed_settings=True)
            save_golden_json(GOLDEN_FILE, payload)
            print(f"Updated golden reference at {GOLDEN_DIR / GOLDEN_FILE}")
            return

    try:
        golden = load_golden_json(GOLDEN_FILE)
        require_golden_initialized(golden, GOLDEN_FILE)
    except (FileNotFoundError, RuntimeError):
        print("Golden reference missing/uninitialized; regenerating now...")
        with _ipopt_print_level_guard(True):
            payload = _run_phase1_optimization(use_relaxed_settings=True)
            save_golden_json(GOLDEN_FILE, payload)
            print(f"Created golden reference at {GOLDEN_FILE}")
            return

    actual = _run_phase1_optimization()

    gold_meta = golden.get("metadata", {})
    act_meta = actual.get("metadata", {})
    if gold_meta.get("samples") != act_meta.get("samples"):
        raise AssertionError(
            f"Sample count mismatch: {act_meta.get('samples')} != {gold_meta.get('samples')}",
        )

    gold_arrays = golden.get("arrays", {})
    act_arrays = actual.get("arrays", {})
    for key in ("primary_theta", "primary_position", "primary_velocity", "primary_load_profile"):
        arrays_close(key, np.asarray(act_arrays[key]), np.asarray(gold_arrays[key]))


@pytest.mark.ipopt_phase1
def test_phase1_ipopt_against_golden() -> None:
    """Pytest entry point for heavy IPOPT validation."""
    if not heavy_env_enabled():
        pytest.skip(
            "CAMPRO_RUN_HEAVY_TESTS not set – skipping IPOPT integration test",
        )
    _compare_against_golden(update=False)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 IPOPT integration harness")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Run the optimization and refresh the golden reference",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if not heavy_env_enabled():
        print(
            "CAMPRO_RUN_HEAVY_TESTS is not enabled. Set it to 1 to run this script.",
            file=sys.stderr,
        )
        return 2

    try:
        _compare_against_golden(update=args.update)
    except Exception as exc:  # pragma: no cover - informative CLI errors
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())