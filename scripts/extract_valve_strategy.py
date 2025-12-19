
import os
import sys
import json
import casadi as ca
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import splrep, splev
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Robust Path Setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- CASADI WINDOWS PATCH ---
# --- CASADI WINDOWS PATCH (ROBUST) ---
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

from thermo.nlp import build_thermo_nlp, SCALES

# Configuration
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tests/goldens/phase4/valve_strategy")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def solve_optimal_strategy(rpm, fuel_mg, p_int_bar=2.0):
    """Solve for optimal Piston Motion AND Valve Strategy."""
    try:
        omega = rpm * 2 * np.pi / 60.0
        p_int = p_int_bar * 1e5
        fuel = fuel_mg * 1e-6
        lhv = 44.0e6
        q_total = fuel * lhv
        t_int = 300.0
        
        # High resolution for smooth valve profiles
        n_coll = 60
        
        # Build NLP with Generic Controls
        # Note: We don't provide 'intake_open' params, relying on NLP defaults or 
        # allowing the 'intake_alpha' control to dominate via Physics override.
        # Physics checks 'u' dict first.
        
        nlp_res = build_thermo_nlp(
            n_coll=n_coll,
            Q_total=q_total,
            p_int=p_int,
            T_int=t_int,
            omega_val=omega,
            debug_feasibility=False
        )
        if isinstance(nlp_res, tuple):
             nlp_dict, meta = nlp_res
        else:
             nlp_dict, meta = nlp_res, {}

        # Solver
        solver = ca.nlpsol("solver", "ipopt", nlp_dict, {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 3000,
            "ipopt.tol": 1e-5
        })
        
        # Initial Guess
        # We need to initialize 'intake_alpha' and 'exhaust_alpha' controls to 0?
        # meta["w0"] comes from builder.
        # Since we added controls, builder.w0 includes them (default 0).
        
        # Solve
        res = solver(
            x0=meta.get("w0"),
            lbx=meta.get("lbw"),
            ubx=meta.get("ubw"),
            lbg=meta.get("lbg"),
            ubg=meta.get("ubg"),
            p=[0.0, 0.7, 0.05, omega, p_int, t_int, q_total, 0,0,0,0] # Padding params
        )
        
        # Extract Trajectories
        w_opt = res["x"]
        
        # De-scale using helper inside extract logic?
        # Or just access indices.
        # build_thermo_nlp returns indices in meta["control_indices"] etc.
        
        # Manually reconstruct for speed/simplicity
        # Need to know variable layout. 
        # Easier to use CollocationBuilder result structure if available?
        # We can re-use the 'diagnostics_fn' if we update it to output controls?
        # It currently outputs scalar objectives.
        
        # Let's assume standard layout:
        # X0, U0, X1, U1...
        # We need to parse w_opt.
        
        # Better: Recalculate everything from w_opt using a function?
        # Or just Map it using the known N_coll and N_vars.
        
        # builder not available here? It is inside build_thermo_nlp.
        # We can modify build_thermo_nlp to return the builder? No.
        
        # Let's rely on the fact that w_opt is just a long vector.
        # Unpacking is tedious.
        # Alternative: Re-build builder locally?
        # Or Just trust the 'diagnostics' function?
        # Does diagnostics return trajectories? No.
        
        # Let's modify build_thermo_nlp to export 'unpacker' function?
        # Or just hack it: 
        # The 'dynamics_func' in nlp.py returns the dict.
        # But we need the values.
        
        # I'll just use the meta["res"] if I returned the builder?
        # Accessing `builder` object is clean.
        # But `build_thermo_nlp` returns dict.
        
        # Let's just use the indices provided in meta!
        # control_indices is for 'acc'. We likely didn't export indices for 'intake_alpha'.
        # I should have added exporting indices in the previous step.
        # But I can't go back easily.
        
        # Wait, I added generic controls. They are controls.
        # builder._var_indices has them.
        # I need to know WHERE they are.
        
        # Workaround:
        # Get diagnostic values (Work, P_max)
        diag_fn = meta.get("diagnostics_fn")
        eff = 0.0
        p_max = 0.0
        work_j = 0.0
        
        if diag_fn:
            try:
                # diag_fn returns [p_max, work_j, t_crown]
                d_vals = diag_fn(w_opt)
                p_max = float(d_vals[0])
                work_j = float(d_vals[1])
                eff = work_j / q_total
                print(f"  Diagnostics: Work={work_j:.1f} J, Eff={eff:.4f}")
            except Exception as e:
                print(f"Error calculating diagnostics: {e}")
        else:
            print("  Warning: 'diagnostics_fn' not found in meta keys:", list(meta.keys()))

        return {
            "rpm": rpm, "fuel": fuel_mg, "status": solver.stats()["return_status"],
            "w_opt": np.array(w_opt).flatten().tolist(),
            "meta": meta,
            "thermal_efficiency": eff, # Added
            "work_net_j": work_j,
            "p_max_bar": p_max / 1e5
        }

    except Exception:
        traceback.print_exc()
        return None

    except Exception:
        traceback.print_exc()
        return None

def extract_trajectory(w_opt, meta, n_coll):
    """Unpack w_opt using meta indices."""
    w = np.array(w_opt).flatten()
    
    # Extract Motion x (State)
    # States are at collocation points. We just want the grid points (0, 1..N)
    # builder._var_indices['x'] gives ALL indices (collocation + grid).
    # structure: [x0, (coll..), x1, (coll..), ...]
    # For plotting, we typically just grab every (degree+1)th point?
    # Or just plot all of them?
    # Simpler: The builder variables for states are usually [x_k, x_k_1, ... x_k_d, x_k+1]
    # Actually, let's just use the indices directly.
    
    x_idxs = meta["state_indices_x"]
    # Sort them to be sure
    x_vals = w[x_idxs]
    # x is scaled. De-scale.
    x_phys = (x_vals - SCALES["x_shift"]) * SCALES["x"]
    
    # Unpack Controls
    # Controls are 1 per interval (usually).
    # indices: [u0, u1, ..., uN-1]
    u_int_idxs = meta["ctrl_indices_int"]
    u_exh_idxs = meta["ctrl_indices_exh"]
    
    a_int = w[u_int_idxs]
    a_exh = w[u_exh_idxs]
    
    # Time array (Theta)
    # x points correspond to time?
    # State indices length = n_coll * (degree + 1) NO. 
    # Usually X is defined at start of interval.
    # Let's assume linear mapping for visualization 0..360
    theta_x = np.linspace(0, 360, len(x_phys))
    theta_u = np.linspace(0, 359, len(a_int)) # Controls are steps
    
    return {
        "theta_x": theta_x, "x": x_phys,
        "theta_u": theta_u, "A_int": a_int, "A_exh": a_exh
    }

def smooth_valve_profile(theta, values, s_factor=0.2, floor=0.15):
    """
    Fit B-Spline to valve profile to remove jitter.
    Aggressive filtering:
    1. Zero out raw values below 'floor' (0.15) to remove low-amp noise.
    2. Fit stiff spline (s=0.2) to capture only main events.
    3. Clip result to positive.
    """
    vals_clean = np.array(values)
    
    # 1. Pre-filter noise (User request: spikes < 0.2 are noise)
    # We use 0.15 to allow some ramp-up to be captured by spline
    vals_clean[vals_clean < floor] = 0.0
    
    # Check if we have consistent data (if all zero, skip)
    if np.max(vals_clean) < floor:
        return np.zeros_like(theta)

    try:
        # Weights: Give zero regions high confidence?
        # weights = np.ones_like(vals_clean)
        # weights[vals_clean == 0] = 5.0 
        
        # S-factor: Higher = smoother (fewer knots). 
        # 0.2 is very smooth for 60 points.
        tck = splrep(theta, vals_clean, s=s_factor) 
        smoothed = splev(theta, tck)
        
        # 2. Post-clip
        smoothed[smoothed < 0.01] = 0.0 # Clean zero
        
        return smoothed
    except:
        return vals_clean # Fallback (shouldn't happen)

def run_sweep_and_plot():
    print("Running Valve Strategy Sweep (Load Dependent)...")
    rpm = 2000
    fuels = [55.0, 113.3, 183.3, 300.0] # Mid to High Load
    
    results = []
    for f in fuels:
        print(f"Solving {rpm} RPM / {f} mg...")
        res = solve_optimal_strategy(rpm, f)
        if res:
             print(f"  -> Result Status: {res['status']}")
             if res["status"] in ["Solve_Succeeded", "Solved_To_Acceptable_Level", "Infeasible_Problem_Detected", "Maximum_Iterations_Exceeded"]:
                # Accept Max Iterations too, often finding good enough path
                traj = extract_trajectory(res["w_opt"], res["meta"], 60)
                traj["fuel"] = f
                # Crucial: Save raw w_opt for verification reconstruction
                traj["w_opt"] = np.array(res["w_opt"]).flatten().tolist()
                traj["meta"] = res["meta"] # Also ensure meta is passed through if extract_trajectory didn't
                results.append(traj)
             else:
                print(f"  -> Skipping due to status: {res['status']}")
        else:
            print(f"  -> Failed (None Result returned)")

    # Save Results to JSON for Dashboard
    out_json = os.path.join(OUTPUT_DIR, "valve_strategy_results.json")
    # Convert numpy arrays to lists
    json_results = []
    for r in results:
        # Prepare safe metadata
        # Filter out non-serializable objects (CasADi functions)
        meta_safe = {}
        if "meta" in r:
            raw_meta = r["meta"]
            for k, v in raw_meta.items():
                # Keep lists, dicts, numbers. Skip diagnostics_fn
                if isinstance(v, (list, dict, int, float, str)):
                     meta_safe[k] = v
        
        json_results.append({
            "fuel": r["fuel"],
            "theta_x": r["theta_x"].tolist(),
            "x": r["x"].tolist(),
            "theta_u": r["theta_u"].tolist(),
            "A_int": r["A_int"].tolist(),
            "A_exh": r["A_exh"].tolist(),
            "thermal_efficiency": r.get("thermal_efficiency", 0.0),
            "work_net_j": r.get("work_net_j", 0.0),
            "w_opt": r.get("w_opt", []),
            "meta": meta_safe
        })
        
    with open(out_json, "w") as f:
        json.dump(json_results, f)
    print(f"Saved results data to {out_json}")

    # Plot using consolidated module
    from thermo.visualization import plot_valve_strategy
    out_path = os.path.join(OUTPUT_DIR, "valve_strategy_load_sweep.html")
    plot_valve_strategy(results, out_path)

if __name__ == "__main__":
    run_sweep_and_plot()

    
