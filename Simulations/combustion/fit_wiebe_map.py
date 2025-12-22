"""
Wiebe Map Fitter.
Fits Wiebe 'm' and 'a' parameters to Prechamber Simulation Results.
"""

import json
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# PATCH: Explicitly add Conda Library bin for MKL/Numpy
try:
    conda_prefix = sys.prefix
    conda_paths = [
        os.path.join(conda_prefix, "Library", "bin"),
        os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
        os.path.join(conda_prefix, "Library", "usr", "bin"),
    ]

    current_path = os.environ.get("PATH", "")
    for p in conda_paths:
        if os.path.exists(p) and p not in current_path:
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(p)
                except OSError:
                    pass
except Exception as e:
    print(f"Warning: DLL Patch failed: {e}")

from pathlib import Path

from Simulations.common.io_schema import SimulationOutput
from Simulations.common.surrogates import WiebeParameterSurrogate

CALIBRATION_DIR = Path(__file__).parent.parent.parent / "thermo" / "calibration"
CALIBRATION_DIR.mkdir(exist_ok=True, parents=True)


def fit_wiebe_map(run_dirs: list[Path], output_file: str = "combustion_map.v1.json"):
    """
    Reads outputs, fits model, saves JSON.
    Model: [a, m, start, duration] = f(rpm, load, etc)
    """
    print(f"Fitting combustion map from {len(run_dirs)} runs...")

    data_points = []

    for r in run_dirs:
        out_file = r / "outputs.json"
        if not out_file.exists():
            continue

        try:
            with open(out_file) as f:
                content = f.read()
            out = SimulationOutput.model_validate_json(content)

            if not out.success:
                continue

            params = out.calibration_params
            if not params:
                continue

            # Combine operating point into the record for features
            # Assuming 'rpm' and 'p_tm' (manifold pressure) or 'load' are in cal params?
            # Or in Input?
            # The cal_params usually contain results like 'wiebe_m'.
            # Operating conditions might need to be fetched from Input if not in cal_params.
            # But earlier code used cal_params.get("rpm", 0.0). Let's trust they are there.

            data_points.append(params)
        except Exception as e:
            print(f"Error processing {r}: {e}")
            continue

    print(f"Collected {len(data_points)} valid data points.")

    if not data_points:
        print("No valid data points found.")
        return

    # Fit Model
    # Features: rpm, p_max_bar usually? Or jet_intensity?
    # Prechamber sim usually correlates with 'jet_intensity' or 'lambda'.
    # For now, let's use 'rpm' and 'load' (if available) or 'jet_intensity' as before.
    # Previous code used ["jet_intensity"]. Let's stick to that if available.

    model = WiebeParameterSurrogate(feature_names=["jet_intensity"])
    res = model.fit(data_points)

    # Construct Artifact (Multi-Model)
    artifact = {"model_type": "wiebe_surrogate_v1", "features": model.feature_names, "models": {}}

    for param, fit_res in res.items():
        artifact["models"][param] = {
            "coeffs": fit_res.get("coeffs", {}),
            "r2": fit_res.get("r2", 0.0),
        }

    out_path = CALIBRATION_DIR / output_file
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Saved calibration to {out_path}")


if __name__ == "__main__":
    # Test Mode: Scan _runs
    run_root = Path(__file__).parent.parent / "_runs"
    all_runs = [x for x in run_root.iterdir() if x.is_dir()]
    fit_wiebe_map(all_runs)
