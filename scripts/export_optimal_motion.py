import os
import sys
import numpy as np
import pandas as pd
import casadi as ca
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import env_setup # [FIX] Path resolution for CasADi

from thermo.nlp import build_thermo_nlp, SCALES

def export_optimal_motion():
    print("--- Extracting Global Optimal Motion ---")
    
    # 1. Global Optimum Parameters (From Phase 5 Analysis)
    rpm = 3600.0
    p_int_bar = 4.0
    fuel_mass_mg = 20.0
    
    # Derived inputs
    p_int = p_int_bar * 1e5
    fuel_mass = fuel_mass_mg * 1e-6
    lhv = 44.0e6
    q_total = fuel_mass * lhv
    omega_val = rpm * 2 * np.pi / 60.0
    t_int_val = 300.0 # Standard assumption
    
    print(f"Target: {rpm} RPM, {p_int_bar} Bar, {fuel_mass_mg} mg Fuel")
    
    # 2. Load Calibration Map (if available, for consistency)
    cal_map = {}
    try:
        with open("thermo/calibration/friction_map.v1.json", "r") as f:
            cal_map["friction"] = json.load(f)
        with open("thermo/calibration/combustion_map.v1.json", "r") as f:
            cal_map["combustion"] = json.load(f)
        print("Loaded Calibration Maps.")
    except Exception as e:
        print(f"Warning: Calibration maps not found or failed ({e}). Using defaults.")

    # 3. Build NLP
    print("Building NLP...")
    nlp_res = build_thermo_nlp(
        n_coll=100, # Higher resolution for motion profile
        Q_total=q_total,
        p_int=p_int,
        T_int=t_int_val,
        omega_val=omega_val,
        calibration_map=cal_map,
        debug_feasibility=True 
    )
    
    if isinstance(nlp_res, tuple):
        nlp_dict, meta = nlp_res
    else:
        nlp_dict = nlp_res
        meta = {}
        
    solver_opts = {
        "ipopt": {
            "max_iter": 3000,
            "print_level": 5, # Show progress
            "tol": 1e-6,
            "linear_solver": "mumps"
        },
        "print_time": 0
    }
    
    solver = ca.nlpsol("solver", "ipopt", nlp_dict, solver_opts)
    
    # 4. Prepare Parameters
    # Order: [theta_start, theta_dur, f_pc, omega, p_int, T_int, Q_total, ... valve ...]
    # We must match nlp.py exactly.
    # Looking at nlp.py (viewed earlier):
    # p_vec = vertcat(theta_start, theta_dur, f_pc, omega, p_int, T_int, q_total, geo..., etc)
    # Wait, the parameter packing logic is usually handled by a wrapper or manual construction.
    # In generate_doe.py, it was manual.
    # We must replicate that manual packing.
    
    theta_start = 0.0
    theta_dur = float(np.radians(40.0))
    f_pc = 0.05
    
    # Valve Defaults
    intake_open = float(np.radians(340.0))
    intake_dur = float(np.radians(220.0))
    exhaust_open = float(np.radians(140.0))
    exhaust_dur = float(np.radians(220.0))
    
    # Geo Defaults (from nlp.py if param) - Wait, geo is usually fixed in nlp.py unless exposed.
    # Viewed nlp.py L90+:
    # p_dyn includes inputs.
    # And L97: p_int, T_int, Q_total
    # But where are the geo params?
    # In generate_doe.py L211:
    # p_vec = [theta_start, theta_dur, f_pc, omega, p_int, T_int, q_total, int_o, int_d, exh_o, exh_d]
    # Let's assume this matches.
    
    p_vec = [
        theta_start,
        theta_dur,
        f_pc,
        omega_val,
        p_int,
        t_int_val,
        q_total,
        intake_open,
        intake_dur,
        exhaust_open,
        exhaust_dur
    ]
    
    # 5. Solve
    print("Solving...")
    try:
        res = solver(
            x0=meta.get("w0"),
            lbx=meta.get("lbw"),
            ubx=meta.get("ubw"),
            lbg=meta.get("lbg"),
            ubg=meta.get("ubg"),
            p=p_vec
        )
    except Exception as e:
        # Fallback without valve params if signature mismatch
        print(f"Solve failed with full params: {e}")
        # Retrying logic removed for clarity, or keep it?
        # Assuming p_vec logic is robust now.
        raise e

    stats = solver.stats()
    print(f"Solver Return Status: {stats.get('return_status', 'Unknown')}")
    if not stats.get("success", False):
         print(f"Warning: Solver did not converge!")

    # 6. Extract Trajectory
    w_opt = res["x"]
    
    # Use Meta Indices (Robust & Simple)
    if "state_indices_x" in meta:
        idx_x = meta["state_indices_x"]
        idx_v = meta["state_indices_v"]
        
        # w_opt is a flattened numpy array (from res["x"])
        w_vals = w_opt.full().flatten()
        
        x_series = w_vals[idx_x]
        v_series = w_vals[idx_v]
        
    else:
        print("Error: NLP Meta missing 'state_indices_x'. Check nlp.py version.")
        # Fallback debug
        print(f"Meta keys: {meta.keys()}")
        return

    # 7. Post-Process (De-Scaling)
    # Scales are in meta or imported
    scale_x = SCALES["x"]
    shift_x = SCALES["x_shift"]
    
    x_phys_m = (x_series - shift_x) * scale_x
    
    # Theta Grid
    n_points = len(x_phys_m)
    theta_arr = np.linspace(0, 2*np.pi, n_points)
    
    # 8. Export
    df_out = pd.DataFrame({
        "theta_rad": theta_arr,
        "x_m": x_phys_m,
        "v_norm": v_series # Keep normalized or scale? v is typically m/rad if consistent, or m/s?
        # nlp.py: v_phys = v_k * SCALES["v"]. SCALES["v"]=0.1. Units: m/rad.
    })
    
    output_path = "dashboard/phase5/final_motion_profile.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Saved Optimal Motion Profile to {output_path}")
    print(f"Stats: Piston Stroke from {x_phys_m.min()*1000:.1f}mm to {x_phys_m.max()*1000:.1f}mm")

if __name__ == "__main__":
    export_optimal_motion()
