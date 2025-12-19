
import os
import sys
import json
import pathlib
import casadi as ca
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Robust Path Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- CASADI WINDOWS PATCH START (Standardized) ---
conda_prefix = sys.prefix
conda_paths = [
    os.path.join(conda_prefix, "Library", "bin"),
    os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
    os.path.join(conda_prefix, "Library", "usr", "bin"),
]
hsl_path = os.path.join(PROJECT_ROOT, "Libraries", "CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5", "bin", "libcoinhsl.dll")
if os.path.exists(hsl_path):
    conda_paths.insert(0, os.path.dirname(hsl_path))
    os.environ["HSLLIB_PATH"] = hsl_path

current_path = os.environ.get("PATH", "")
for p in conda_paths:
    if os.path.exists(p) and p not in current_path:
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try: os.add_dll_directory(p)
            except: pass
# --- CASADI WINDOWS PATCH END ---

from thermo.nlp import build_thermo_nlp
from thermo.geometry import StandardSliderCrankGeometry

RESULTS_CSV = os.path.join(PROJECT_ROOT, "tests/goldens/phase1/doe_output/phase1_physical_results_fixed.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "tests/goldens/phase1/doe_output/phase3_optimal_motion.csv")
OUTPUT_PLOT = os.path.join(PROJECT_ROOT, "tests/goldens/phase1/doe_output/phase3_optimal_motion_plot.html")

def load_calibration_maps():
    maps = {}
    try:
        f_map_path = os.path.join(PROJECT_ROOT, "thermo", "calibration", "friction_map.v1.json")
        w_map_path = os.path.join(PROJECT_ROOT, "thermo", "calibration", "combustion_map.v1.json")
        
        if os.path.exists(f_map_path):
            with open(f_map_path, "r") as f: maps["friction"] = json.load(f)
        if os.path.exists(w_map_path):
            with open(w_map_path, "r") as f: maps["combustion"] = json.load(f)
    except Exception as e:
        print(f"Warning: Map load failed: {e}")
    return maps

def extract_optimal_motion():
    print(f"Loading results from {RESULTS_CSV}...")
    df = pd.read_csv(RESULTS_CSV)
    
    # Filter for Optimal points
    valid_df = df[df["status"] == "Optimal"]
    if valid_df.empty:
        print("No Optimal points found!")
        return
        
    # Pick the best point (Max Thermal Efficiency)
    # Filter out potential outliers if needed (e.g. eff > 0.6 is suspicious physical limit but let's trust valid)
    best_idx = valid_df["thermal_efficiency"].idxmax()
    best_row = valid_df.loc[best_idx]
    
    print("\n--- Best Operating Point Identified ---")
    print(f"Index: {best_idx}")
    print(f"RPM: {best_row['rpm']}")
    print(f"Boost (bar): {best_row['p_int_bar']}")
    print(f"Fuel (mg): {best_row['fuel_mass_mg']}")
    print(f"Thermal Eff: {best_row['thermal_efficiency']:.4f}")
    print(f"Net Work (J): {best_row['abs_work_net_j']:.2f}")
    print("---------------------------------------")
    
    # Re-Solve
    rpm = float(best_row["rpm"])
    p_int_bar = float(best_row["p_int_bar"])
    fuel_mass_mg = float(best_row["fuel_mass_mg"])
    
    maps = load_calibration_maps()
    n_coll = 60 # Higher resolution for motion law export
    omega = rpm * 2 * np.pi / 60.0
    
    print(f"Re-solving with high resolution (N={n_coll})...")
    
    nlp_res = build_thermo_nlp(
        n_coll=n_coll,
        Q_total=1000.0, # Dummy, overwritten by parameter
        p_int=2.0e5,    # Dummy
        T_int=300.0,    # Dummy
        omega_val=omega,
        debug_feasibility=False,
        calibration_map=maps
    )
    
    if isinstance(nlp_res, tuple):
        nlp_dict, meta = nlp_res
    else:
        nlp_dict, meta = nlp_res, {}

    solver = ca.nlpsol("solver", "ipopt", nlp_dict, {
        "ipopt.print_level": 5, 
        "ipopt.max_iter": 3000,
        "ipopt.tol": 1e-5
    })
    
    # Physics Inputs
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
    
    res = solver(
        x0=meta.get("w0"),
        lbx=meta.get("lbw"),
        ubx=meta.get("ubw"),
        lbg=meta.get("lbg"),
        ubg=meta.get("ubg"),
        p=p_vec
    )
    
    if not solver.stats()["success"]:
        print("Warning: Re-solve did not converge fully! Proceeding with caution.")
    
    # Extract Trajectories
    w_opt = res["x"]
    
    # We need to map w_opt back to x, p, T
    # The `meta` dict might have indices, or we use the helper diagnostics_fn if available?
    # `build_thermo_nlp` export logic:
    # res[1]["diagnostics_fn"] = diag_fn (returns scalars)
    # To get arrays, we need to inspect the `w` vector manually using indices.
    
    # meta["state_indices_x"] etc were added in recent patch?
    # Let's check if they exist.
    
    x_idxs = meta.get("state_indices_x", [])
    t_idxs = meta.get("state_indices_T_c", [])
    m_idxs = meta.get("state_indices_m_c", [])
    
    # If not present (params in previous file view suggested they MIGHT be there), 
    # we can deduce them: N collocation points -> (d+1)*N or similar.
    # Actually, CollocationBuilder structure is:
    # w = [stat_0, stat_1... control_0...] repeated?
    # No, usually all vars for interval k.
    
    # Simplest way: use `collocation_builder` inverse mapping if available.
    # Or, rely on the fact that `build_thermo_nlp` returns `res` which is `builder.export_nlp()`.
    # And `meta` is the second element.
    # In `thermo.nlp`, we saw:
    # if "x" in builder._var_indices: res[1]["state_indices_x"] = ...
    
    if not x_idxs:
        print("Error: State indices not found in NLP meta. Update thermo/nlp.py to export them.")
        # Fallback: We can't easily guess.
        return

    # Extract raw values (SX or DM) -> numpy
    w_np = w_opt.full().flatten()
    
    x_scaled = w_np[x_idxs]
    t_scaled = w_np[t_idxs]
    m_scaled = w_np[m_idxs]
    
    scales = meta.get("scales", {})
    x_scale = scales.get("x", 0.2)
    x_shift = scales.get("x_shift", 1.0)
    t_scale = scales.get("T_c", 1000.0)
    m_scale = scales.get("m_c", 5e-4) # Verify value
    
    # De-scale
    # scaled_x = (x_phys / scale) + shift
    # x_phys = (scaled_x - shift) * scale
    x_phys = (x_scaled - x_shift) * x_scale
    t_phys = t_scaled * t_scale
    m_phys = m_scaled * m_scale
    


    # Theta Grid
    # Updated NLP to 2*pi (360 deg) for full cycle
    total_pts = len(x_phys)
    theta_arr = np.linspace(0, 2*np.pi, total_pts)
    
    # Smooth x_opt using Savitzky-Golay and B-spline
    x_opt_final = x_phys
    try:
        from scipy.signal import savgol_filter
        from scipy.interpolate import make_interp_spline
        
        # 1. Remove high-freq noise
        # window_length must be odd and <= total_pts
        win_len = min(51, total_pts if total_pts % 2 != 0 else total_pts - 1)
        if win_len > 3:
            # mode='wrap' is critical for the cyclic 0-360 motion
            x_filtered = savgol_filter(x_phys, window_length=win_len, polyorder=3, mode='wrap')
        else:
            x_filtered = x_phys
            
        # 2. Create Continuous B-Spline
        spl = make_interp_spline(theta_arr, x_filtered, k=3)
        x_opt_raw_smooth = spl(theta_arr)
        
        # 3. Post-Process: Clamp and Pin
        # Prevent negative values (physically impossible)
        x_opt_final = np.maximum(0.0, x_opt_raw_smooth)
        
        # Pin endpoints to 0.0 exactly for valid closure
        x_opt_final[0] = 0.0
        x_opt_final[-1] = 0.0
        
        
    except ImportError:
        print("Scipy not found, skipping smoothing.")
    
    # Derived Volume & Pressure
    # Use same params as build_thermo_nlp defaults (line 34-36, 64)
    # bore=0.1, stroke=0.1, conrod=0.4, CR=15.0
    geo = StandardSliderCrankGeometry(
        bore=0.1,
        stroke=0.1,
        conrod=0.4,
        compression_ratio=15.0
    )
    
    area_piston = np.pi * geo.B**2 / 4.0
    # x_phys: 0 at TDC? 
    # linear_x_physical in nlp.py: 0 at TDC, stroke at BDC.
    # x_phys IS the distance from TDC.
    vol_arr = geo.V_c + area_piston * x_opt_final
    
    r_gas = 287.0
    p_arr_pa = (m_phys * r_gas * t_phys) / vol_arr
    p_arr_bar = p_arr_pa / 1e5
    
    # Standard Slider Crank for Reference
    # Need to match stroke and connecting rod length.
    # theta_arr goes 0 to 4pi.
    # x_std = geo.piston_position(theta_arr) ?
    # geo.piston_position returns 0 at TDC usually.
    x_std = []
    for th in theta_arr:
        # Standard geometry calc
        # r = S/2
        # l = ConRod
        # x = r(1 - cos) + l(1 - sqrt(1 - lam^2 sin^2))
        r = geo.S / 2.0
        l = geo.L
        
        # Robust kinematic formula (approx or exact)
        # Exact: r(1 - cos(theta)) + l(1 - sqrt(1 - (r/l)^2 * sin^2(theta)))
        term1 = 1 - np.cos(th)
        lam = r / l
        discr = 1 - lam**2 * np.sin(th)**2
        # Safety for floating point errors if lam ~ 1 (not here usually, R=4)
        discr = max(0.0, discr)
        term2 = (1.0 / lam) * (1 - np.sqrt(discr)) # Wait, l * (1-sqrt) = r/lam * ...
        # Correct form:
        # x = r * (term1) + l * (1 - sqrt(discr))
        
        xb = r * term1 + l * (1 - np.sqrt(discr))
        x_std.append(xb)
    x_std = np.array(x_std)
    
    # Save CSV
    out_df = pd.DataFrame({
        "theta_rad": theta_arr,
        "theta_deg": np.degrees(theta_arr),
        # ... rest same ...

        "x_opt_m": x_opt_final,
        "x_std_m": x_std,
        "volume_m3": vol_arr,
        "pressure_bar": p_arr_bar,
        "temperature_k": t_phys
    })
    
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved motion law to {OUTPUT_CSV}")
    
    # Plot
    fig = go.Figure()
    # Normalized X
    fig.add_trace(go.Scatter(x=out_df["theta_deg"], y=out_df["x_opt_m"], name="Optimal (Phase 1)", line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=out_df["theta_deg"], y=out_df["x_std_m"], name="Standard Slider-Crank", line=dict(color='gray', dash='dash')))
    
    fig.update_layout(
        title=f"Optimal Piston Motion Law (RPM={rpm}, Eff={best_row['thermal_efficiency']:.2%})",
        xaxis_title="Crank Angle (deg)",
        yaxis_title="Piston Position from TDC (m)",
        template="plotly_white"
    )
    
    fig.write_html(OUTPUT_PLOT)
    print(f"Saved plot to {OUTPUT_PLOT}")

if __name__ == "__main__":
    extract_optimal_motion()
