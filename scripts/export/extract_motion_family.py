
import os
import sys
import json
import casadi as ca
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor, as_completed
# Robust Path Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- CASADI PLATFORM PATCH (Cross-Platform) ---
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
# --- CASADI PLATFORM PATCH END ---

from campro.optimization.nlp.thermo_nlp import build_thermo_nlp
from campro.optimization.nlp.geometry import StandardSliderCrankGeometry

RESULTS_CSV = os.path.join(PROJECT_ROOT, "tests/goldens/phase1/doe_output/phase1_physical_results_fixed.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests/goldens/phase1/doe_output")

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

def solve_point(row_dict, maps=None):
    """
    Re-solves the NLP for a single point to extract the trajectory.
    Returns the smoothed x_opt array on a standardized 0-2pi grid.
    """
    if maps is None:
        maps = load_calibration_maps()
        
    rpm = float(row_dict["rpm"])
    p_int_bar = float(row_dict["p_int_bar"])
    fuel_mass_mg = float(row_dict["fuel_mass_mg"])
    
    n_coll = 60
    omega = rpm * 2 * np.pi / 60.0
    
    try:
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

        # Muted solver
        solver = ca.nlpsol("solver", "ipopt", nlp_dict, {
            "ipopt.print_level": 0, 
            "ipopt.max_iter": 3000,
            "ipopt.tol": 1e-5
        })
        
        # Physics
        p_int = p_int_bar * 1e5
        fuel = fuel_mass_mg * 1e-6
        lhv = 44.0e6
        q_total = fuel * lhv
        t_int = 300.0
        
        p_vec = [0.0, float(np.radians(40.0)), 0.05, omega, p_int, t_int, q_total]
        
        res = solver(
            x0=meta.get("w0"),
            lbx=meta.get("lbw"),
            ubx=meta.get("ubw"),
            lbg=meta.get("lbg"),
            ubg=meta.get("ubg"),
            p=p_vec
        )
        
        status = solver.stats()["return_status"]
        if status not in ["Solve_Succeeded", "Solved_To_Acceptable_Level", "Search_Direction_Becomes_Too_Small", "Infeasible_Problem_Detected"]:
            print(f"Skipping {rpm} RPM / {fuel_mass_mg} mg: Solver failed with status '{status}'")
            return None

        w_opt = res["x"].full().flatten()
        x_idxs = meta.get("state_indices_x", [])
        if not x_idxs: 
            print("Error: state_indices_x missing.")
            return None
        
        x_scaled = w_opt[x_idxs]
        scales = meta.get("scales", {})
        x_phys = (x_scaled - scales.get("x_shift", 1.0)) * scales.get("x", 0.2)
        
        # Post-Processing: Smooth -> Clamp -> Pin
        total_pts = len(x_phys)
        theta_arr = np.linspace(0, 2*np.pi, total_pts)
        
        # Smoothing
        try:
            from scipy.signal import savgol_filter
            from scipy.interpolate import make_interp_spline
            
            win_len = min(51, total_pts if total_pts % 2 != 0 else total_pts - 1)
            if win_len > 3:
                x_filt = savgol_filter(x_phys, window_length=win_len, polyorder=3, mode='wrap')
            else:
                x_filt = x_phys
                
            spl = make_interp_spline(theta_arr, x_filt, k=3)
            x_smooth = spl(theta_arr)
            
            # Clamp and Pin
            x_final = np.maximum(0.0, x_smooth)
            x_final[0] = 0.0
            x_final[-1] = 0.0
            
            print(f"Success: {rpm} RPM / {fuel_mass_mg} mg")
            
            return {
                "rpm": rpm,
                "fuel_mg": fuel_mass_mg,
                "load": fuel_mass_mg, # Proxy for load
                "theta": theta_arr,
                "x_opt": x_final
            }
            
        except ImportError:
            print("ImportError in smoothing")
            return None

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error solving point {rpm}/{fuel_mass_mg}: {e}")
        return None

def analyze_motion_families():
    print("Loading Dataset...")
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["status"] == "Optimal"]
    
    # Filter for REALISTIC points (Phase 1 had artifacts > 100% eff)
    # Also ignore very low fuel (unstable)
    valid_df = df[
        (df["thermal_efficiency"] > 0.3) & 
        (df["thermal_efficiency"] < 0.6) & 
        (df["fuel_mass_mg"] > 50.0) # Ensure robust combustion
    ]
    
    if valid_df.empty:
        print("Warning: No points meet realistic criteria. Using raw Optimal set.")
        valid_df = df
    else:
        print(f"Filtered to {len(valid_df)} realistic points.")
        df = valid_df # Use valid subset
    
    # 1. RPM Sweep Selection (Constant Fuel, Max RPMs)
    # Group by Fuel (rounded) and count unique RPMs
    df["fuel_round"] = df["fuel_mass_mg"].round(1)
    
    # Find fuel with most unique RPMs
    fuel_counts = df.groupby("fuel_round")["rpm"].nunique()
    best_fuel = fuel_counts.idxmax()
    print(f"Selected RPM Sweep Fuel: {best_fuel} mg ({fuel_counts[best_fuel]} RPMs)")
    
    rpm_sweep_df = df[df["fuel_round"] == best_fuel].sort_values("rpm")
    # Take a subset if too many (e.g. Low, Med, High + Intermediate)
    unique_rpms = sorted(rpm_sweep_df["rpm"].unique())
    # Subsample: min, q1, med, q3, max
    selected_rpms = sorted(list(set([
        unique_rpms[0], 
        unique_rpms[len(unique_rpms)//4], 
        unique_rpms[len(unique_rpms)//2], 
        unique_rpms[3*len(unique_rpms)//4], 
        unique_rpms[-1]
    ])))
    print(f"Target RPMs: {selected_rpms}")
    
    rpm_points = []
    for r in selected_rpms:
        # Get row for this RPM
        row = rpm_sweep_df[rpm_sweep_df["rpm"] == r].iloc[0]
        rpm_points.append(row.to_dict())

    # 2. Load Sweep Selection (Constant RPM, Max Fuels)
    # Use most common RPM
    best_rpm = df["rpm"].mode()[0]
    print(f"Selected Load Sweep RPM: {best_rpm}")
    
    load_sweep_df = df[df["rpm"] == best_rpm].sort_values("fuel_mass_mg")
    unique_fuels = sorted(load_sweep_df["fuel_mass_mg"].unique())
     # Subsample: min, q1, med, q3, max
    selected_fuels = sorted(list(set([
        unique_fuels[0], 
        unique_fuels[len(unique_fuels)//4], 
        unique_fuels[len(unique_fuels)//2], 
        unique_fuels[3*len(unique_fuels)//4], 
        unique_fuels[-1]
    ])))
    print(f"Target Fuels: {[round(f,1) for f in selected_fuels]}")
    
    load_points = []
    for f in selected_fuels:
        row = load_sweep_df[load_sweep_df["fuel_mass_mg"] == f].iloc[0]
        load_points.append(row.to_dict())
        
    # Execute Batch
    all_tasks = rpm_points + load_points
    print(f"solving {len(all_tasks)} points...")
    
    families = {"rpm_sweep": [], "load_sweep": []}
    
    # Run sequentially or parallel? Parallel is safer for time.
    # But need maps loaded inside function due to pickling?
    # Actually, maps pickle fine usually.
    
    maps = load_calibration_maps()
    
    # RPM Sweep
    print("Processing RPM Sweep...")
    for p in rpm_points:
        res = solve_point(p, maps)
        if res: families["rpm_sweep"].append(res)
        
    # Load Sweep
    print("Processing Load Sweep...")
    for p in load_points:
        res = solve_point(p, maps)
        if res: families["load_sweep"].append(res)

    # Plot RPM Sweep (Verify Independence)
    fig_rpm = go.Figure()
    for res in families["rpm_sweep"]:
        fig_rpm.add_trace(go.Scatter(
            x=np.degrees(res["theta"]), 
            y=res["x_opt"], 
            mode='lines',
            name=f"{int(res['rpm'])} RPM"
        ))
    fig_rpm.update_layout(
        title=f"Motion Law Independence Check (Fuel={best_fuel} mg)",
        xaxis_title="Crank Angle (deg)",
        yaxis_title="Piston Position (m)",
        template="plotly_white"
    )
    fig_rpm.write_html(os.path.join(OUTPUT_DIR, "motion_family_rpm_sweep.html"))
    
    # Plot Load Sweep (Show Family)
    fig_load = go.Figure()
    for res in families["load_sweep"]:
        fig_load.add_trace(go.Scatter(
            x=np.degrees(res["theta"]), 
            y=res["x_opt"], 
            mode='lines',
            name=f"{res['fuel_mg']:.1f} mg"
        ))
    fig_load.update_layout(
        title=f"Motion Law Family (RPM={best_rpm})",
        xaxis_title="Crank Angle (deg)",
        yaxis_title="Piston Position (m)",
        template="plotly_white"
    )
    fig_load.write_html(os.path.join(OUTPUT_DIR, "motion_family_load_sweep.html"))
    
    print("Generation Complete.")

if __name__ == "__main__":
    analyze_motion_families()
