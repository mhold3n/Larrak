"""
Manual Integration Test for Phase 4 Exporter.
"""
import sys
import os
import numpy as np
import json
import casadi as ca

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# PATCH: Explicitly add Conda Library bin and HSL bin to PATH for CasADi
conda_prefix = sys.prefix
conda_paths = [
    os.path.join(conda_prefix, "Library", "bin"),
    os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
    os.path.join(conda_prefix, "Library", "usr", "bin"),
]

# HSL Path (Hardcoded based on detected location)
hsl_dll = r"c:\Users\maxed\OneDrive\Desktop\Github Projects\Larrak\Libraries\CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin\libcoinhsl.dll"
hsl_dir = os.path.dirname(hsl_dll)
conda_paths.insert(0, hsl_dir)

current_path = os.environ.get("PATH", "")
for p in conda_paths:
    if os.path.exists(p) and p not in current_path:
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(p)
            except:
                pass


from thermo.nlp import build_thermo_nlp
from thermo.export_candidate import export_candidate
from Simulations.common.io_schema import SimulationInput

def test_export_flow():
    print("1. Building NLP...")
    res = build_thermo_nlp(n_coll=20, debug_feasibility=True)
    if isinstance(res, tuple):
        nlp, meta = res
    else:
        print("Error: NLP builder did not return tuple.")
        return

    print("2. Solving Trivial Problem...")
    # Just run 1 iter to get a valid-ish w vector
    opts = {"ipopt.max_iter": 1, "ipopt.print_level": 0, "print_time":0}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    sol = solver(
        x0=meta["w0"],
        lbx=meta["lbw"],
        ubx=meta["ubw"],
        lbg=meta["lbg"],
        ubg=meta["ubg"]
    )
    w_opt = sol["x"].full().flatten()
    
    print("3. Exporting Candidate...")
    geo_params = {"bore": 0.1, "stroke": 0.1, "conrod": 0.2, "compression_ratio": 15.0}
    ops_params = {"rpm": 3000, "p_int": 2e5, "T_int": 350.0}
    
    json_str = export_candidate(
        run_id="test_export_001",
        meta=meta,
        w_opt=w_opt,
        geo_params=geo_params,
        ops_params=ops_params
    )
    
    print("4. validating Output...")
    # Parse back
    model = SimulationInput.model_validate_json(json_str)
    print(f"Success! Generated Run ID: {model.run_id}")
    print(f"BC Length: {len(model.boundary_conditions.pressure_gas)}")
    
    # Write to file for inspection
    with open("test_candidate.json", "w") as f:
        f.write(json_str)
    print("Written to test_candidate.json")

if __name__ == "__main__":
    test_export_flow()
