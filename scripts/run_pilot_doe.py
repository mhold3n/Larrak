#!/usr/bin/env python3
"""Pilot DOE Run: Generate training data with live solvers.

Runs a small DOE (5 samples) to verify solver integration and
generate initial training data for surrogates.

Usage:
    python scripts/run_pilot_doe.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parents[1]))

import logging

from Simulations.hifi import ConjugateHTAdapter, StructuralFEAAdapter
from Simulations.hifi.doe_runner import (
    DOEConfig,
    ParameterRange,
    generate_lhs_samples,
    save_results,
)
from Simulations.hifi.example_inputs import create_simulation_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# Narrower ranges for pilot (faster execution)
PILOT_RANGES = [
    ParameterRange("bore_mm", 80, 90, "mm"),
    ParameterRange("stroke_mm", 85, 95, "mm"),
    ParameterRange("compression_ratio", 11, 13, ""),
    ParameterRange("rpm", 2000, 4000, "rpm"),
    ParameterRange("load_fraction", 0.6, 0.9, ""),
]


def run_structural_case(case_id: int, params: dict) -> dict:
    """Run single CalculiX structural case."""
    log.info(f"[Case {case_id}] Starting structural FEA...")

    result = {
        "case_id": case_id,
        "solver": "structural",
        "params": params,
        "success": False,
        "outputs": {},
    }

    try:
        adapter = StructuralFEAAdapter()
        sim_input = create_simulation_input(
            run_id=f"pilot_struct_{case_id:03d}",
            bore_mm=params["bore_mm"],
            stroke_mm=params["stroke_mm"],
            rpm=params["rpm"],
            load_fraction=params["load_fraction"],
            compression_ratio=params["compression_ratio"],
        )

        adapter.load_input(sim_input)
        output = adapter.solve_steady_state()

        result["success"] = output.success
        result["outputs"]["von_mises_max"] = output.max_von_mises
        result["outputs"]["calibration"] = output.calibration_params
        log.info(f"[Case {case_id}] Structural complete: success={output.success}")

    except Exception as e:
        result["error"] = str(e)
        log.error(f"[Case {case_id}] Structural failed: {e}")

    return result


def run_thermal_case(case_id: int, params: dict) -> dict:
    """Run single OpenFOAM thermal case."""
    log.info(f"[Case {case_id}] Starting thermal CFD...")

    result = {
        "case_id": case_id,
        "solver": "thermal",
        "params": params,
        "success": False,
        "outputs": {},
    }

    try:
        adapter = ConjugateHTAdapter()
        sim_input = create_simulation_input(
            run_id=f"pilot_thermal_{case_id:03d}",
            bore_mm=params["bore_mm"],
            stroke_mm=params["stroke_mm"],
            rpm=params["rpm"],
            load_fraction=params["load_fraction"],
            compression_ratio=params["compression_ratio"],
        )

        adapter.load_input(sim_input)
        output = adapter.solve_steady_state()

        result["success"] = output.success
        result["outputs"]["T_crown_max"] = output.T_crown_max
        result["outputs"]["calibration"] = output.calibration_params
        log.info(f"[Case {case_id}] Thermal complete: success={output.success}")

    except Exception as e:
        result["error"] = str(e)
        log.error(f"[Case {case_id}] Thermal failed: {e}")

    return result


def main():
    n_samples = 5
    output_dir = Path("data/pilot_doe")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== Pilot DOE: {n_samples} samples ===")
    log.info(f"Output: {output_dir}")

    # Generate samples
    samples = generate_lhs_samples(PILOT_RANGES, n_samples, seed=42)
    param_names = [r.name for r in PILOT_RANGES]

    all_results = []

    for i, sample in enumerate(samples):
        params = dict(zip(param_names, sample))
        log.info(f"\n--- Case {i + 1}/{n_samples} ---")
        log.info(
            f"Params: bore={params['bore_mm']:.1f}mm, stroke={params['stroke_mm']:.1f}mm, "
            f"CR={params['compression_ratio']:.1f}, RPM={params['rpm']:.0f}"
        )

        # Run structural first (usually faster)
        struct_result = run_structural_case(i, params)
        all_results.append(struct_result)

        # Then thermal
        thermal_result = run_thermal_case(i, params)
        all_results.append(thermal_result)

        # Save intermediate checkpoint
        checkpoint_path = output_dir / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = output_dir / f"pilot_results_{timestamp}.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary
    struct_success = sum(1 for r in all_results if r["solver"] == "structural" and r["success"])
    thermal_success = sum(1 for r in all_results if r["solver"] == "thermal" and r["success"])

    log.info(f"\n=== Pilot DOE Complete ===")
    log.info(f"Structural: {struct_success}/{n_samples} successful")
    log.info(f"Thermal: {thermal_success}/{n_samples} successful")
    log.info(f"Results: {final_path}")

    return 0 if (struct_success + thermal_success) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
