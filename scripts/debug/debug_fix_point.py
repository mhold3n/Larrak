
import os
import sys
import json
import pathlib
import casadi as ca
import numpy as np
import traceback

# Path Patch
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# CasADi Patch
conda_prefix = sys.prefix
conda_paths = [
    os.path.join(conda_prefix, "Library", "bin"),
    os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
    os.path.join(conda_prefix, "Library", "usr", "bin"),
]
hsl_dll = r"c:\Users\maxed\OneDrive\Desktop\Github Projects\Larrak\Libraries\CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin\libcoinhsl.dll"
conda_paths.insert(0, os.path.dirname(hsl_dll))
current_path = os.environ.get("PATH", "")
for p in conda_paths:
    if os.path.exists(p) and p not in current_path:
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try: os.add_dll_directory(p)
            except: pass
os.environ["HSLLIB_PATH"] = hsl_dll

from campro.optimization.nlp.thermo_nlp import build_thermo_nlp

def test_point():
    # Targets: RPM=1000, p=1.0, fuel=20
    rpm = 1000.0
    p_int_bar = 1.0
    fuel_mass_mg = 20.0
    
    print(f"Testing Point: RPM={rpm}, Boost={p_int_bar}, Fuel={fuel_mass_mg}", flush=True)
    
    try:
        # Load Maps
        maps = {}
        root = pathlib.Path(PROJECT_ROOT)
        f_map = root / "thermo/calibration/friction_map.v1.json"
        w_map = root / "thermo/calibration/combustion_map.v1.json"
        if f_map.exists():
            with open(f_map) as f: maps["friction"] = json.load(f)
        if w_map.exists():
            with open(w_map) as f: maps["combustion"] = json.load(f)
            
        print("Maps loaded.", flush=True)
        
        # Build
        omega = rpm * 2 * np.pi / 60.0
        n_coll = 50
        
        print("Building NLP...", flush=True)
        nlp_res = build_thermo_nlp(
            n_coll=n_coll,
            Q_total=1000.0,
            p_int=2.0e5,
            T_int=300.0,
            omega_val=omega,
            debug_feasibility=False,
            calibration_map=maps
        )
        
        if isinstance(nlp_res, tuple):
            nlp_dict, meta = nlp_res
        else:
            nlp_dict, meta = nlp_res, {}
            
        print("Creating Solver...", flush=True)
        solver = ca.nlpsol("solver", "ipopt", nlp_dict, {"ipopt.max_iter": 5, "ipopt.print_level": 5})
        
        # Solve
        p_int = p_int_bar * 1e5
        fuel = fuel_mass_mg * 1e-6
        lhv = 44.0e6
        q_total = fuel * lhv
        t_int = 300.0
        
        p_vec = [
            0.0,
            float(np.radians(40.0)),
            0.05,
            omega,
            p_int,
            t_int,
            q_total
        ]
        
        print(f"p_vec: {p_vec}", flush=True)
        
        # Check w0 size
        w0 = meta.get("w0")
        print(f"w0 shape: {w0.shape}", flush=True)
        
        print("Starting Solve...", flush=True)
        res = solver(
            x0=w0,
            lbx=meta.get("lbw"),
            ubx=meta.get("ubw"),
            lbg=meta.get("lbg"),
            ubg=meta.get("ubg"),
            p=p_vec
        )
        
        print("Solve status:", solver.stats()["return_status"], flush=True)

        
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_point()
