
import os
import sys
import json
import pathlib
import casadi as ca
import numpy as np
import pandas as pd
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Robust Path Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"DEBUG: CWD={os.getcwd()}")
print(f"DEBUG: PROJECT_ROOT={PROJECT_ROOT}")

# --- CASADI WINDOWS PATCH START ---
conda_prefix = sys.prefix
conda_paths = [
    os.path.join(conda_prefix, "Library", "bin"),
    os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
    os.path.join(conda_prefix, "Library", "usr", "bin"),
]

# HSL Path - auto-detected for cross-platform support
from campro.environment.resolve import hsl_path as resolve_hsl_path
try:
    hsl_dll = str(resolve_hsl_path())
    conda_paths.insert(0, os.path.dirname(hsl_dll))
    os.environ["HSLLIB_PATH"] = hsl_dll
except RuntimeError:
    print("Warning: HSL library not detected, solver may fail")
    hsl_dll = None

current_path = os.environ.get("PATH", "")
for p in conda_paths:
    if os.path.exists(p) and p not in current_path:
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try: os.add_dll_directory(p)
            except: pass
# --- CASADI WINDOWS PATCH END ---

from campro.optimization.nlp.thermo_nlp import build_thermo_nlp

# Config
CSV_PATH = r"tests/goldens/phase1/doe_output/phase1_physical_results.csv"
OUTPUT_PATH = r"tests/goldens/phase1/doe_output/phase1_physical_results_fixed.csv"

# Global Cache (Worker Reuse)
PROCESS_CACHE = {}

def load_calibration_maps():
    maps = {}
    try:
        root = pathlib.Path(PROJECT_ROOT)
        f_map_path = root / "thermo" / "calibration" / "friction_map.v1.json"
        w_map_path = root / "thermo" / "calibration" / "combustion_map.v1.json"
        
        if f_map_path.exists():
            with open(f_map_path, "r") as f: maps["friction"] = json.load(f)
        if w_map_path.exists():
            with open(w_map_path, "r") as f: maps["combustion"] = json.load(f)
    except Exception as e:
        print(f"Warning: Map load failed: {e}")
    return maps

def get_solver(N=50):
    if "solver" in PROCESS_CACHE:
        return PROCESS_CACHE["solver"], PROCESS_CACHE["meta"]
    
    maps = load_calibration_maps()
    omega_default = 2000.0 * 2 * np.pi / 60.0
    
    nlp_res = build_thermo_nlp(
        n_coll=N,
        Q_total=1000.0,
        p_int=2.0e5,
        T_int=300.0,
        omega_val=omega_default,
        debug_feasibility=False,
        calibration_map=maps
    )
    
    if isinstance(nlp_res, tuple):
        nlp_dict, meta = nlp_res
    else:
        nlp_dict, meta = nlp_res, {}

    opts = {
        "ipopt.max_iter": 5000,
        "ipopt.tol": 1e-4,
        "ipopt.accept_after_max_steps": 5,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0
    }
    
    solver = ca.nlpsol("solver", "ipopt", nlp_dict, opts)
    PROCESS_CACHE["solver"] = solver
    PROCESS_CACHE["meta"] = meta
    return solver, meta

def solve_point(row):
    try:
        rpm = float(row["rpm"])
        p_int_bar = float(row["p_int_bar"])
        fuel_mass_mg = float(row["fuel_mass_mg"])
        
        solver, meta = get_solver()
        
        # Physics Inputs
        p_int = p_int_bar * 1e5
        fuel = fuel_mass_mg * 1e-6
        lhv = 44.0e6
        q_total = fuel * lhv
        t_int = 300.0
        omega = rpm * 2 * np.pi / 60.0
        
        p_vec = [
            0.0,
            float(np.radians(40.0)),
            0.05,
            omega,
            p_int,
            t_int,
            q_total
        ]
        
        w0 = meta.get("w0") if meta else None
        lbw = meta.get("lbw") if meta else None
        ubw = meta.get("ubw") if meta else None
        lbg = meta.get("lbg") if meta else None
        ubg = meta.get("ubg") if meta else None

        res = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=p_vec)
        
        stats = solver.stats()
        status = stats["return_status"]
        valid = stats["success"] or "Acceptable" in status
        
        result_updates = {
            "status": "Optimal" if valid else f"Fixed_{status}",
            "objective": float(res["f"]),
            "iter_count": stats["iter_count"],
            "solver_status": status
        }
        
        return row.name, result_updates, None

    except Exception as e:
        print(f"EXCEPTION in solve_point for RPM={row.get('rpm')}: {e}")
        # traceback.print_exc() # Be verbose in logs
        return row.name, {"status": "Fix_Failed", "error": str(e)}, None


def fix_failures():
    print(f"Loading {CSV_PATH}...")
    if os.path.exists(OUTPUT_PATH):
        print(f"Loading existing fixed file from {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH)
    else:
        df = pd.read_csv(CSV_PATH)
    
    # Retry previous failures too
    mask = df["status"].isin(["Maximum_Iterations_Exceeded", "Fix_Failed"])
    candidates = df[mask]
    print(f"Found {len(candidates)} candidates to fix.")
    
    if len(candidates) == 0:
        print("No failures to fix.")
        return

    print("Starting Recovery Run (Parallel)...")
    
    df_fixed = df.copy()
    success_count = 0
    processed_count = 0
    
    max_workers = min(8, os.cpu_count() or 4)
    print(f"Using {max_workers} workers.")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(solve_point, row): idx for idx, row in candidates.iterrows()}
        
        for future in as_completed(futures):
            try:
                # Add timeout
                idx, updates, _ = future.result(timeout=45)
                
                if "Optimal" in updates.get("status", ""):
                    success_count += 1
                
                for k, v in updates.items():
                    if k in df_fixed.columns:
                        df_fixed.at[idx, k] = v
                    elif k == "error":
                        pass # Don't save error text to maintain CSV schema? 
                        # Or maybe save it to solver_status if we want visibility?
                        # df_fixed.at[idx, "solver_status"] = v
                
                processed_count += 1
                if processed_count % 20 == 0:
                    print(f"Processed {processed_count}/{len(candidates)} (Success: {success_count})")
                    df_fixed.to_csv(OUTPUT_PATH, index=False)
                    
            except Exception as e: # Timeout or executor error
                print(f"Error getting future result: {e}")
                
    print(f"Recovery Complete. Recovered {success_count}/{len(candidates)} points.")
    df_fixed.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved fixed dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    fix_failures()
