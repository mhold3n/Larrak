
import os
import sys
import json
import casadi as ca
import numpy as np
import pandas as pd
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Robust Path Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    hsl_lib = str(resolve_hsl_path())
    conda_paths.insert(0, os.path.dirname(hsl_lib))
    os.environ["HSLLIB_PATH"] = hsl_lib
except RuntimeError:
    print("Warning: HSL library not detected, solver may fail")
    hsl_lib = None

current_path = os.environ.get("PATH", "")
for p in conda_paths:
    if os.path.exists(p) and p not in current_path:
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try: os.add_dll_directory(p)
            except: pass
# --- CASADI WINDOWS PATCH END ---

from campro.optimization.nlp.thermo_nlp import build_thermo_nlp

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests/goldens/phase4/doe_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "phase4_doe_results.csv")

def load_calibration_maps():
    maps = {}
    try:
        f_map_path = os.path.join(PROJECT_ROOT, "thermo", "calibration", "friction_map.v1.json")
        w_map_path = os.path.join(PROJECT_ROOT, "thermo", "calibration", "combustion_map.v1.json")
        if os.path.exists(f_map_path):
            with open(f_map_path, "r") as f: maps["friction"] = json.load(f)
        if os.path.exists(w_map_path):
            with open(w_map_path, "r") as f: maps["combustion"] = json.load(f)
    except: pass
    return maps

def solve_point(params, maps=None):
    try:
        if maps is None: maps = load_calibration_maps()
        
        # Unpack 7 Variables
        rpm = params["rpm"]
        fuel_mg = params["fuel_mass_mg"]
        p_int_bar = params["p_int_bar"]
        
        # Valve Timing (Degrees)
        int_open_deg = params["intake_open_deg"]
        int_dur_deg = params["intake_dur_deg"]
        exh_open_deg = params["exhaust_open_deg"]
        exh_dur_deg = params["exhaust_dur_deg"]
        
        # Convert to SI/Rad
        omega = rpm * 2 * np.pi / 60.0
        p_int = p_int_bar * 1e5
        fuel = fuel_mg * 1e-6
        lhv = 44.0e6
        q_total = fuel * lhv
        t_int = 300.0
        
        int_open_rad = float(np.radians(int_open_deg))
        int_dur_rad = float(np.radians(int_dur_deg))
        exh_open_rad = float(np.radians(exh_open_deg))
        exh_dur_rad = float(np.radians(exh_dur_deg))
        
        # Build NLP
        # n_coll=30 for speed in large DOE
        nlp_res = build_thermo_nlp(
            n_coll=30,
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

        # Muted solver
        solver = ca.nlpsol("solver", "ipopt", nlp_dict, {
            "ipopt.print_level": 0, 
            "ipopt.max_iter": 1000, # Faster timeout
            "ipopt.tol": 1e-4
        })
        
        # Parameter Vector
        # [theta_start, theta_dur, f_pc, omega, p_int, T_int, Q_total, 
        #  intake_open, intake_dur, exhaust_open, exhaust_dur]
        # Order depends on nlp definition! 
        # We need to construct dict if possible, or ensure order matches add_parameter calls.
        
        # Check nlp.py param order:
        # theta_start, theta_dur, f_pc, omega, p_int, T_int, Q_total, int_open, int_dur, exh_open, exh_dur
        p_vec = [
            0.0, float(np.radians(40.0)), 0.05, 
            omega, p_int, t_int, q_total,
            int_open_rad, int_dur_rad,
            exh_open_rad, exh_dur_rad
        ]
        
        res = solver(
            x0=meta.get("w0"),
            lbx=meta.get("lbw"),
            ubx=meta.get("ubw"),
            lbg=meta.get("lbg"),
            ubg=meta.get("ubg"),
            p=p_vec
        )
        
        stats = solver.stats()
        status = stats["return_status"]
        success = stats["success"]
        
        # Extract Results
        # For DOE, we care about Objective (Efficiency)
        # Obj is usually negative work or fuel specific
        # thermo/nlp.py sets J = -Work_net / Q_total (actually -Efficiency?)
        # Let's check J definition. Usually J = -Efficiency.
        J_val = float(res["f"])
        eff = -J_val # Assuming minimization of negative efficiency
        
        # Also helpful to get Net Work
        # Need to reconstruct?
        work_j = eff * q_total
        
        result = params.copy()
        result["status"] = status
        result["thermal_efficiency"] = eff
        result["work_net_j"] = work_j
        result["solver_time"] = stats.get("t_wall_total", 0.0)
        
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        # print(f"Error: {e}")
        r = params.copy()
        r["status"] = "Error"
        return r

def generate_doe():
    print("Generating High-Fidelity Simulation DOE (7 Variables)...")
    
    # 7 variables
    # 1. RPM: 1000 - 6000
    # 2. Fuel: 20 - 300
    # 3. Boost: 1.0 - 4.0
    # 4. Int Open: 100 - 180 (Before BDC)
    # 5. Int Dur: 40 - 100
    # 6. Exh Open: 100 - 180
    # 7. Exh Dur: 60 - 140
    
    bounds = [
        [1000, 6000], # RPM
        [20, 300],    # Fuel
        [1.0, 4.0],   # Boost
        [100, 180],   # Int Open Deg (Start)
        [40, 100],    # Int Dur Deg
        [80, 160],    # Exh Open Deg (Earlier than Intake)
        [60, 140]     # Exh Dur Deg
    ]
    
    sampler = qmc.LatinHypercube(d=7, seed=42)
    sample = sampler.random(n=500) # 500 points for a decent map
    # Let's do 500 points to verify pipeline
    
    scaled_sample = qmc.scale(sample, [b[0] for b in bounds], [b[1] for b in bounds])
    
    tasks = []
    cols = ["rpm", "fuel_mass_mg", "p_int_bar", "intake_open_deg", "intake_dur_deg", "exhaust_open_deg", "exhaust_dur_deg"]
    
    for row in scaled_sample:
        task = {k: v for k, v in zip(cols, row)}
        tasks.append(task)
        
    print(f"Created {len(tasks)} design points.")
    
    # Execute
    print("Starting Parallel Execution...")
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        futures = {executor.submit(solve_point, t): t for t in tasks}
        
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            results.append(res)
            if i % 10 == 0:
                print(f"Completed {i}/{len(tasks)}...", end="\r")
                
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved DOE results to {OUTPUT_CSV}")
    print(df["status"].value_counts())

if __name__ == "__main__":
    generate_doe()
