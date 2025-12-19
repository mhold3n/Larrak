import os
import sys

# [DEBUG] Immediate start log
print(f"[INFO] Process {os.getpid()} starting: run_adaptive_gear_design.py", flush=True)

import pandas as pd
import numpy as np
# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- ENVIRONMENT SETUP ---
import env_setup
# -------------------------

def run_adaptive_gear_loop():
    
    # Lazy Imports to prevent load lock
    import casadi as ca
    from surrogate.gear_config import GEAR_CONFIG
    from surrogate.conjugate_nlp import build_gear_nlp

    print("--- Gear Kinematics Adaptive Loop: User -> NLP -> Kinematics -> Calibration ---")
    
    # 1. Inputs (Target)
    target_path = "dashboard/final_motion_profile.csv"
    if not os.path.exists(target_path):
        print(f"Error: Target {target_path} not found. Run Gear Profile Synthesis first.")
        # Fallback for dev: Generate dummy sine
        target_x = np.sin(np.linspace(0, 2*np.pi, 360))
    else:
        df = pd.read_csv(target_path)
        if "x_m" in df.columns:
            target_x = df["x_m"].values * 1000.0 # Convert m to mm
        elif "x_opt" in df.columns:
            target_x = df["x_opt"].values
        else:
             # Just take first column if unnamed
             target_x = df.iloc[:,1].values 
             # Ensure length matches Config
             # Resample if needed
             
    print(f"[Core] Target Profile Length: {len(target_x)}")
    print(f"[Core] Gear Ratio Target: {GEAR_CONFIG.gear.ratio}")
    
    # 2. Adaptive Loop
    max_loops = 3
    loop_count = 0
    converged = False
    
    # Import FEA Check (Geometric Contact Analysis)
    from surrogate.fea_check import analyze_gear_set
    
    while loop_count < max_loops and not converged:
        print(f"\n=== Loop {loop_count + 1} / {max_loops} ===")
        
        # A. Construction (NLP)
        print("[S2] Building Conjugate Gear NLP...")
        opti, vars_dict = build_gear_nlp(target_x, n_points=len(target_x))
        
        # B. Solve
        print("[S2] Solving (this may take time)...")
        # Enable Solver Logging
        opti.solver('ipopt', {'ipopt.print_level': 5, 'print_time': 1, 'ipopt.sb': 'no'})
        success = False
        try:
            sol = opti.solve()
            Rp_opt = sol.value(vars_dict["Rp"])
            Rr_opt = sol.value(vars_dict["Rr"])
            C_opt = sol.value(vars_dict["C"])
            
            # C might be scalar or vector depending on NLP version. Ensure vector.
            if np.isscalar(C_opt) or C_opt.ndim == 0:
                 C_opt = np.full_like(Rp_opt, C_opt)
                 
            print(f"    Optimal Center Distance: {np.mean(C_opt):.4f} mm (Range: {np.min(C_opt):.2f}-{np.max(C_opt):.2f})")
            success = True
        except Exception as e:
            print(f"    Optimization Failed: {e}")
            try:
                # Attempt to retrieve debug values
                Rp_opt = opti.debug.value(vars_dict["Rp"])
                Rr_opt = opti.debug.value(vars_dict["Rr"])
                C_opt = opti.debug.value(vars_dict["C"])
                if np.isscalar(C_opt) or C_opt.ndim == 0:
                     C_opt = np.full_like(Rp_opt, C_opt)
            except:
                Rp_opt = np.full(len(target_x), np.nan)
                Rr_opt = np.full(len(target_x), np.nan)
                C_opt = np.full(len(target_x), np.nan)

        # C. Validation (FEA & Kinematics)
        print("[S3] Validating with Geometric FEA (Binding/NVH)...")
        
        metrics = {'interference': True, 'tracking_rmse': 999.0, 'curvature_score': 999.0, 'jerk_score': 999.0}
        feasible = False
        
        if success:
             feasible, metrics = analyze_gear_set(Rp_opt, Rr_opt, C_opt, target_x)
             
             print(f"    RMSE (Motion): {metrics['tracking_rmse']:.4f} mm")
             print(f"    Curvature Score: {metrics['curvature_score']:.2f}")
             print(f"    Jerk/NVH Score: {metrics['jerk_score']:.2f}")
             print(f"    Interference/Binding: {metrics['interference']}")
        else:
             print("    [!] Optimization Failed. Triggering Unsolvable Handler.")

        # D. Feedback Loop (Recalibration)
        if feasible and metrics['tracking_rmse'] <= 0.5:
             print("    [✓] Convergence Reached. FEA Passed.")
             converged = True
        else:
             print("    [!] Validation Failed. Analyzing Failure Mode...")
             # 1. Unsolvable / Solver Crash
             if not success:
                 print("    [Case: Unsolvable] Relaxing Center Distance Bounds...")
                 # Directly modify NLP constraint logic in next loop? 
                 # Currently Config doesn't expose strict C bounds, but envelope.
                 GEAR_CONFIG.gear.min_radius *= 0.8 # Allow smaller gears
                 GEAR_CONFIG.gear.max_radius *= 1.2 # Allow larger gears
                 
             # 2. Interference / Binding Detected
             elif metrics['interference']:
                 print("    [Case: Binding/Interference] Increasing Smoothing (Curvature) and Slip penalty...")
                 GEAR_CONFIG.gear.w_smooth *= 5.0
                 GEAR_CONFIG.gear.w_slip *= 2.0
                 
             # 3. High Motion Error
             elif metrics['tracking_rmse'] > 0.5:
                 print("    [Case: High Motion Error] Increasing Tracking Fidelity...")
                 GEAR_CONFIG.gear.w_tracking *= 5.0
             
             print(f"    [P5] Recalibrated Weights: Smooth={GEAR_CONFIG.gear.w_smooth}, Track={GEAR_CONFIG.gear.w_tracking}")
             loop_count += 1
             
    # 4. Output (Save if converged OR valid best effort)
    if converged or (feasible and loop_count >= max_loops):
        if not converged:
            print("\n[!] Max Loops Reached. Saving Best Effort Result.")
            
        print("\n--- Gear Kinematics Complete: Optimized Conjugate Gears ---")
        out_df = pd.DataFrame({
            "theta": np.linspace(0, 360, len(Rp_opt)),
            "Rp": Rp_opt,
            "Rr": Rr_opt,
            "C": C_opt
        })
        out_path = "dashboard/phase3_optimized_gears.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    run_adaptive_gear_loop()
