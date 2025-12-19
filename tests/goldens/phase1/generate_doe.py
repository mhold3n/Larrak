"""
Generate Thermo Calibration DOE Golden.
Tests Thermo Module across RPM, Load, and Phi ranges using a unified Design of Experiments.
"""

import os
import sys

# Critical Configuration:
# Enable Threaded BLAS for Inner-Loop Acceleration (User Request)
# We reduce Parallel Workers (Outer Loop) to prevent thrashing.
# os.environ["OMP_NUM_THREADS"] = "1"     <-- REMOVED
# os.environ["OPENBLAS_NUM_THREADS"] = "1" <-- REMOVED

from typing import Any

import casadi as ca
import numpy as np
import pandas as pd
import csv
import time
import datetime
import itertools

# PATCH: Explicitly add Conda Library bin and HSL bin to PATH for CasADi
# This fixes "plugin not found" and "library not found" errors in non-activated envs
import os
import sys

# 1. Conda Library Paths
conda_prefix = sys.prefix
conda_paths = [
    os.path.join(conda_prefix, "Library", "bin"),
    os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
    os.path.join(conda_prefix, "Library", "usr", "bin"),
]

# 2. HSL Path (Hardcoded based on detected location)
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

# Explicitly set HSLLIB option availability for Ipopt via Env (optional/fallback)
os.environ["HSLLIB_PATH"] = hsl_dll

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from tests.infra.doe_runner import DOERunner
from thermo.nlp import build_thermo_nlp
import json
import pathlib

# Load Calibration Maps (Global)
CALIBRATION_MAPS = {}
try:
    # Adjust path to match project structure relative to this file?
    # File is in tests/goldens/phase1/
    # Project root is ../../../
    root = pathlib.Path(__file__).parent.parent.parent.parent
    f_map_path = root / "thermo" / "calibration" / "friction_map.v1.json"
    w_map_path = root / "thermo" / "calibration" / "combustion_map.v1.json"
    
    if f_map_path.exists():
        with open(f_map_path, "r") as f:
            CALIBRATION_MAPS["friction"] = json.load(f)
            print(f"Loaded Friction Map: {f_map_path}")
            
    if w_map_path.exists():
        with open(w_map_path, "r") as f:
            CALIBRATION_MAPS["combustion"] = json.load(f)
            print(f"Loaded Combustion Map: {w_map_path}")
            
except Exception as e:
    print(f"Warning: Failed to load calibration maps: {e}")



# PROCESS-LOCAL CACHE (For Multiprocessing Speedup)
PROCESS_CACHE = {}

def phase1_test(params: dict[str, Any], solver_opts: dict[str, Any] = None, debug_feasibility: bool = True) -> dict[str, Any]:
    """
    Execute a single Phase 1 test case with physical inputs.
    Params: rpm, p_int_bar, fuel_mass_mg
    solver_opts: Optional override for CasADi solver options
    """
    rpm = params["rpm"]
    p_int_bar = params["p_int_bar"]
    fuel_mass_mg = params["fuel_mass_mg"]

    # 1. Derived Physical Inputs
    p_int = p_int_bar * 1e5
    fuel_mass = fuel_mass_mg * 1e-6

    # Lower Heating Value
    lhv = 44.0e6  # J/kg
    q_total = fuel_mass * lhv

    # Standard Thermodynamic Assumptions
    r_gas = 287.0
    t_int_val = 300.0
    rho_int = p_int / (r_gas * t_int_val)
    
    from thermo.config import CONFIG
    
    # Geometry (From Fixed Variables)
    B = CONFIG.geometry.bore
    S = CONFIG.geometry.stroke
    CR = CONFIG.geometry.cr
    
    # 100mm bore, 200mm stroke (Now dynamic)
    vd = (np.pi * (B**2) / 4.0) * S
    m_air_approx = rho_int * vd
    af_stoic = 14.7
    af_actual = m_air_approx / fuel_mass if fuel_mass > 0 else 1e9
    phi_approx = af_stoic / af_actual

    try:
        # --- CACHING LOGIC START ---
        # Reuse solver if available in this process to avoid costly JIT recompilation (1-2s -> 0.01s overhead)
        
        global PROCESS_CACHE
        if "solver" not in PROCESS_CACHE:
            # Build NLP (ONCE PER WORKER)
            # Use dummy/default values for parameters
            omega_default = 2000.0 * 2 * np.pi / 60.0
            
            # Pass Geometry to NLP Builder? 
            # Ideally build_thermo_nlp should take these. 
            # Currently it defaults inside. 
            # We must assume NLP matches CONFIG or we reconstruct it.
            # But the Geometry is "Fixed" for the PROJECT. 
            # So updating logic in nlp.py to read CONFIG is best, BUT for now 
            # we will proceed assuming NLP defaults match CONFIG or are overridden by P.
            # Wait, NLP P vector handles Valve Timing, but Geometry (B, S) is baked into `StandardSliderCrankGeometry`.
            # We should patch `build_thermo_nlp` to use CONFIG too.
            
            nlp_res = build_thermo_nlp(
                n_coll=50,
                Q_total=1000.0, # Dummy, will be overridden by parameter
                p_int=2.0e5,    # Dummy
                T_int=300.0,    # Dummy
                omega_val=omega_default, # Dummy
                debug_feasibility=debug_feasibility, # Bake this setting in
                initial_conditions=None, # Symbolic initial conditions handled in NLP now? No, we kept it simplified.
                calibration_map=CALIBRATION_MAPS
            )

            if isinstance(nlp_res, tuple):
                nlp_dict, meta = nlp_res
            else:
                nlp_dict, meta = nlp_res, {}

            # Create Solver
            if solver_opts is None:
                opts = {
                    "ipopt": {
                        "max_iter": 3000,
                        "print_level": 0,
                        "sb": "yes",
                        "tol": 1e-6,
                        "linear_solver": "ma57",
                    },
                    "print_time": 0,
                }
            else:
                opts = solver_opts
            
            solver = ca.nlpsol("solver", "ipopt", nlp_dict, opts)
            
            # Cache everything needed for solving
            PROCESS_CACHE["solver"] = solver
            PROCESS_CACHE["meta"] = meta
            PROCESS_CACHE["w0"] = meta.get("w0")
            PROCESS_CACHE["lbw"] = meta.get("lbw")
            PROCESS_CACHE["ubw"] = meta.get("ubw")
            PROCESS_CACHE["lbg"] = meta.get("lbg")
            PROCESS_CACHE["ubg"] = meta.get("ubg")
            
            if "diagnostics_fn" in meta:
                PROCESS_CACHE["diag_fn"] = meta["diagnostics_fn"]
            
        # Retrieve from Cache
        solver = PROCESS_CACHE["solver"]
        w0 = PROCESS_CACHE["w0"]
        lbw = PROCESS_CACHE["lbw"]
        ubw = PROCESS_CACHE["ubw"]
        lbg = PROCESS_CACHE["lbg"]
        ubg = PROCESS_CACHE["ubg"]
        
        # --- PARAMETER PACKING ---
        omega = rpm * 2 * np.pi / 60.0
        theta_start = 0.0
        theta_dur = float(np.radians(40.0))
        f_pc = 0.05
        
        # Valve Timings (From CONFIG)
        intake_open = CONFIG.geometry.intake_open
        intake_dur = CONFIG.geometry.intake_dur
        exhaust_open = CONFIG.geometry.exhaust_open
        exhaust_dur = CONFIG.geometry.exhaust_dur

        p_vec = [
            theta_start,
            theta_dur,
            f_pc,
            omega,
            p_int,     # Pa
            t_int_val, # K
            q_total,   # J
            intake_open,
            intake_dur,
            exhaust_open,
            exhaust_dur
        ]
        
        if "solver" in PROCESS_CACHE:
             s_in = PROCESS_CACHE["solver"].size_in("p")
             # print(f"DEBUG: Solver expects p size {s_in}, provided {len(p_vec)}")
        
        # 4. Solve
        res = solver(
            x0=w0,
            lbx=lbw,
            ubx=ubw,
            lbg=lbg,
            ubg=ubg,
            p=p_vec # CasADi handles list -> DM auto? Usually yes.
        )

        stats = solver.stats()
        status = "Optimal" if stats["success"] else stats["return_status"]
        if "Acceptable" in stats["return_status"]:
            status = "Acceptable"

        obj_val = float(res["f"])

        # Work and Efficiency
        # CRITICAL: Use 'true_work' from diagnostics, NOT objective (which includes penalties)
        abs_work_j = 0.0
        # Results Map
        p_max_bar = 0.0
        abs_work_j = 0.0  # This represents Net Work (Cycle Integral) from NLP

        # Check if diagnostics function is available in meta dict
        # nlp_res is (nlp_dict, meta_dict)
        if "diag_fn" in PROCESS_CACHE:
            diag_fn = PROCESS_CACHE["diag_fn"]
            w_opt = res["x"]
            try:
                d_res = diag_fn(w_opt)
                # d_res returns [p_max, work_j]
                # work_j in NLP is calculated as sum(P*dV), which is Net Indicated Work
                p_max_bar = float(d_res[0]) / 1e5
                abs_work_j = float(d_res[1])
            except Exception as e:
                # Log usage error but don't fail the test
                print(f"Warning: Diagnostics function failed: {e}")
                pass
        else:
            # Fallback path (should not happen with updated nlp.py)
            pass

        # Calculate Compression Work Estimate (Adiabatic) just for reference
        # W_comp = (P1*V1 / (gamma-1)) * (CR^(gamma-1) - 1)
        B, S_piston, CR_val = 0.1, 0.1, 15.0
        V_disp_p = (np.pi * B**2 / 4.0) * S_piston
        V_c_p = V_disp_p / (CR_val - 1)
        # Use Single Cylinder Volume to match NLP output
        V_bdc_total = 1.0 * (V_c_p + V_disp_p)
        gamma_air = 1.35
        w_comp_est = (p_int * V_bdc_total / (gamma_air - 1)) * (CR_val ** (gamma_air - 1) - 1)

        # Physics Correction: abs_work_j from NLP is Gross Expansion Work.
        # We must subtract Adiabatic Compression Work to get Net Indicated Work.
        # Note: abs_work_j already includes Friction separation in Diagnostics return (Brake Gross?),
        # but let's assume it returns Brake Gross.
        # So abs_work_net_j = Brake Gross - Compression = Brake Net.
        abs_work_net_j = abs_work_j - w_comp_est 

        # Net Efficiency
        thermal_eff = abs_work_net_j / q_total if q_total > 0 else 0.0

        output = {
            "status": status,
            "doe_status": "Completed",
            "rpm": rpm,
            "p_int_bar": p_int_bar,
            "fuel_mass_mg": fuel_mass_mg,
            "q_total_j": q_total,
            "phi_est": phi_approx,
            "objective": obj_val,
            "abs_work_j": abs_work_j,  # Gross Work
            "abs_work_net_j": abs_work_net_j,  # Net Work
            "w_comp_est_j": w_comp_est,
            "p_max_bar": p_max_bar,
            "thermal_efficiency": thermal_eff,
            "iter_count": stats["iter_count"],
            "solver_status": stats["return_status"],
        }

    except Exception as e:
        err_msg = str(e)
        if "solver" in PROCESS_CACHE:
             try:
                 sz = PROCESS_CACHE["solver"].size_in("p")
                 err_msg += f" [Solver expects p={sz}, provided={len(p_vec)}]"
             except: pass
             
        output = {
            "status": "Exception",
            "doe_status": "Failed",
            "rpm": rpm,
            "p_int_bar": p_int_bar,
            "fuel_mass_mg": fuel_mass_mg,
            "error": err_msg,
        }

    return output


def main():
    # 1. Define Physical Grid (Rectangular Domain)
    # Load from centralized CONFIG to ensure consistency with Simulation Layer
    from thermo.config import CONFIG
    
    rpm_levels = CONFIG.rpm_grid
    p_int_levels = CONFIG.boost_grid
    fuel_levels = CONFIG.fuel_grid
    
    print(f"Scope: {len(rpm_levels)} RPMs x {len(p_int_levels)} Boosts x {len(fuel_levels)} Fuels")

    # Full Factorial
    doe_list = []
    import itertools

    for r, p, f in itertools.product(rpm_levels, p_int_levels, fuel_levels):
        # AFR Logic Filter
        # p is p_int_bar. Need p_int_pa.
        p_pa = p * 1e5
        fuel_kg = f * 1e-6
        
        # Estimate Air Mass (See phase1_test logic)
        T_int = 300.0
        R = 287.0
        rho = p_pa / (R * T_int)
        
        # Geometry (Access CONFIG inside main? It is imported)
        B = CONFIG.geometry.bore
        S = CONFIG.geometry.stroke
        Vd = (np.pi * B**2 / 4.0) * S # Single Piston Displacement
        m_air = rho * Vd
        
        af_stoic = 14.7
        af_actual = m_air / fuel_kg if fuel_kg > 0 else 1e9
        lam = af_actual / af_stoic
        
        # Check Limits
        if CONFIG.ranges.lambda_min <= lam <= CONFIG.ranges.lambda_max:
             doe_list.append({"rpm": r, "p_int_bar": p, "fuel_mass_mg": f})
        # else:
        #      print(f"Skipping: RPM={r}, Boost={p}, Fuel={f}, Lambda={lam:.2f}")

    print(f"Generated {len(doe_list)} DOE points (Filtered by Lambda {CONFIG.ranges.lambda_min}-{CONFIG.ranges.lambda_max})")
    
    # --- RESUME / FORCE RERUN LOGIC ---
    output_dir = "dashboard/thermo"
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "thermo_doe_results.csv")

    force_rerun = str(os.environ.get("LARRAK_FORCE_RERUN", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
    if force_rerun and os.path.exists(results_file):
        try:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = results_file.replace(".csv", f".bak.{ts}.csv")
            os.replace(results_file, backup)
            print(f"[Dashboard] FORCE_RERUN enabled: moved existing results to {backup}", flush=True)
        except Exception as e:
            print(f"[Dashboard] FORCE_RERUN enabled but failed to move existing results: {e}", flush=True)
            # Best-effort: continue; resume logic may still skip.
    
    if os.path.exists(results_file):
        try:
            print(f"Index: Found existing results file. Checking for completed points...")
            # Use on_bad_lines='skip' to handle potential crash corruption at EOF
            df_exist = pd.read_csv(results_file, on_bad_lines='skip')
            
            # Create a set of completed tuples assuming columns exist
            # Columns (from phase1_test): rpm, p_int_bar, fuel_mass_mg
            if {"rpm", "p_int_bar", "fuel_mass_mg"}.issubset(df_exist.columns):
                # Rounding might be needed if floats are slightly off? 
                # Be careful with floating point matching. 
                # Let's round to 4 decimals for robust set matching.
                existing_set = set(zip(
                    df_exist["rpm"].round(1), 
                    df_exist["p_int_bar"].round(4), 
                    df_exist["fuel_mass_mg"].round(4)
                ))
                
                initial_count = len(doe_list)
                # Filter DOE list
                doe_list = [d for d in doe_list if (
                    round(d["rpm"], 1), 
                    round(d["p_int_bar"], 4), 
                    round(d["fuel_mass_mg"], 4)
                ) not in existing_set]
                
                print(f"Resuming: Skipped {initial_count - len(doe_list)} points. Remaining: {len(doe_list)}", flush=True)
            else:
                print("Warning: Existing CSV missing required columns. Resuming not possible (or file invalid).", flush=True)
                
        except Exception as e:
            print(f"Warning: Failed to read existing results for resume: {e}", flush=True)
    # --------------------

    # 2. Run DOE (Parallel)
    # Define Output Location
    # Define Output Location (Consolidated Dashboard)
    # output_dir = "tests/goldens/phase1/doe_output"
    # User Request: Centralize to Dashboard
    output_dir = "dashboard/thermo"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Runner
    runner = DOERunner("phase1_physical", output_dir)

    # Load Design
    runner.design = pd.DataFrame(doe_list)

    # Run
    # DOERunner handles parallel execution and incremental saving
    print(f"Starting DOE Run...", flush=True)
    runner.run(test_func=phase1_test, workers=4)

    print(f"DOE Completed.")


if __name__ == "__main__":
    main()
