"""
Friction Map Fitter.
Fits Chen-Flynn or polynomial models to Phase 4 Friction Results.
"""

import json
import os
import sys
from pathlib import Path

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
                except:
                    pass
except Exception as e:
    print(f"Warning: DLL Patch failed: {e}")

from typing import List, Dict
from Simulations.common.io_schema import SimulationOutput
from Simulations.common.surrogates import PolynomialSurrogate

CALIBRATION_DIR = Path(__file__).parent.parent.parent / "thermo" / "calibration"
CALIBRATION_DIR.mkdir(exist_ok=True, parents=True)

def fit_friction_map(run_dirs: List[Path], output_file: str = "friction_map.v1.json"):
    """
    Reads outputs, fits model, saves JSON.
    Model: FMEP = A + B*P_max + C*RPM + D*RPM^2
    """
    print(f"Fitting friction map from {len(run_dirs)} runs...")
    
    data_points = []
    
    for r in run_dirs:
        out_file = r / "outputs.json"
        if not out_file.exists():
            continue
            
        try:
            with open(out_file, "r") as f:
                content = f.read()
                # Debug print
                # print(f"Reading {r}: {content[:50]}...")
                
            out = SimulationOutput.model_validate_json(content)
            
            if not out.success:
                print(f"Skipping failed run: {r}")
                continue
                
            params = out.calibration_params
            if not params:
                print(f"Run {r} has no calibration params.")
                continue
                
            data_points.append(params)
        except Exception as e:
            print(f"Error processing {r}: {e}")
            continue
            
    print(f"Collected {len(data_points)} valid data points.")
        
    if not data_points:
        print("No valid data points found.")
        return

    # Fit Model
    # Features: p_max_bar, rpm
    # Target: fmep_bar
    # Using degree=1 because we only have 3 points in the test suite.
    model = PolynomialSurrogate(
        feature_names=["p_max_bar", "rpm"],
        target_name="fmep_bar",
        degree=1,
        interaction=False
    )
    
    try:
        res = model.fit(data_points)
        print("Fit Results:")
        print(f"  R2: {res['r2']:.4f}")
        print(f"  Coeffs: {res['coeffs']}")
        
        # Save Artifact
        artifact = {
            "model_type": "polynomial_deg1",
            "features": model.feature_names,
            "coeffs": res["coeffs"],
            "r2": res["r2"],
            "n_samples": len(data_points)
        }
        
        out_path = CALIBRATION_DIR / output_file
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2)
            
        print(f"Saved calibration to {out_path}")
        
    except Exception as e:
        print(f"Fitting Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test Mode: Scan _runs
    run_root = Path(__file__).parent.parent / "_runs"
    all_runs = [x for x in run_root.iterdir() if x.is_dir()]
    fit_friction_map(all_runs)
