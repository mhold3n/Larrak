"""
Friction Map Fitter.
Reads simulation runs and fits a polynomial surrogate for FMEP.
Target: FMEP = f(p_max, rpm)
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# PATCH DLL for MKL
try:
    conda_prefix = sys.prefix
    conda_paths = [os.path.join(conda_prefix, "Library", "bin")]
    current_path = os.environ.get("PATH", "")
    for p in conda_paths:
        if os.path.exists(p) and p not in current_path:
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                try: 
                    os.add_dll_directory(p) 
                except: 
                    pass
except: pass

from typing import List, Dict
from Simulations.common.io_schema import SimulationOutput
from Simulations.common.surrogates import PolynomialSurrogate

def fit_friction_map(run_dirs: List[Path], output_path: Path = None):
    """
    Scan run directories, extract Friction results, fit surrogate.
    """
    print(f"Fitting friction map from {len(run_dirs)} runs...")
    
    # 1. Collect Data
    data_records = []
    
    for run_dir in run_dirs:
        try:
            out_file = run_dir / "outputs.json"
            if not out_file.exists(): continue
            
            with open(out_file, "r") as f:
                res_dict = json.load(f)
                
                # Check for friction data
                if "friction_fmep" in res_dict and res_dict["friction_fmep"] is not None:
                     cal_params = res_dict.get("calibration_params", {})
                     
                     # Flatten record for Surrogate
                     record = {
                         "friction_fmep": res_dict["friction_fmep"],
                         "p_max_bar": cal_params.get("p_max_bar", 0.0),
                         "rpm": cal_params.get("rpm", 0.0)
                     }
                     
                     if record["p_max_bar"] > 0:
                         data_records.append(record)
                         
        except Exception as e:
            print(f"Skipping run {run_dir}: {e}")
            
    if len(data_records) < 3:
        print("Not enough data to fit friction map (need > 3 points).")
        return
        
    # 2. Fit Surrogate
    feature_names = ["p_max_bar", "rpm"]
    target_name = "friction_fmep"
    
    surrogate = PolynomialSurrogate(feature_names=feature_names, target_name=target_name, degree=1)
    
    try:
        fit_results = surrogate.fit(data_records)
    except Exception as e:
        print(f"Fitting failed: {e}")
        return
    
    # 3. Save Artifact
    if output_path is None:
        output_path = Path("thermo/calibration/friction_map.v1.json")
        
    # Construct Artifact
    artifact = {
        "model_type": "polynomial_deg1",
        "features": feature_names,
        "coeffs": fit_results["coeffs"],
        "r2": fit_results["r2"],
        "n_samples": len(data_records)
    }
    
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)
        
    print(f"Saved friction calibration to {output_path}")

if __name__ == "__main__":
    # Test Mode
    runs_dir = Path("Simulations/_runs")
    if runs_dir.exists():
        fit_friction_map([x for x in runs_dir.iterdir() if x.is_dir()])
