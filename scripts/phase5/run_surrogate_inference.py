import os
import sys
import json
import numpy as np
import plotly.graph_objects as go

# Setup Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from truthmaker.surrogates.adapters.kinematics import InverseKinematics

def run_interpreter():
    # Load Data
    json_path = os.path.join(PROJECT_ROOT, "tests/goldens/phase4/valve_strategy/valve_strategy_results.json")
    if not os.path.exists(json_path):
        print("Error: JSON results not found.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    # Initialize Kinematics (OP Geometry)
    # Bore/Stroke must match Phase 1 Setup
    # Stroke=0.1, Conrod=0.4
    ik = InverseKinematics(stroke=0.1, conrod=0.4)
    
    fig = go.Figure()
    
    print(f"{'Fuel':<10} | {'Max Ratio':<10} | {'Min Ratio':<10} | {'RMSE (mm)':<10}")
    print("-" * 50)

    for res in results:
        fuel = res["fuel"]
        # Phase 1 Theta (Cycle Clock)
        # Check units. x in meters.
        theta_deg = np.array(res["theta_x"])
        theta_rad = np.radians(theta_deg)
        x_phys = np.array(res["x"])
        
        # Determine Cycle Phase (0-360)
        # Phase 1 usually outputs 0-360.
        
        try:
            phi, ratio = ik.solve_trajectory(x_phys, theta_rad)
            
            # Normalize Phi to 0-360 for plotting
            phi_deg = np.degrees(phi) % 360
            
            # Plot Ratio(theta)
            # ...
            
            # Forward Verification
            x_recon = ik.verify_forward_kinematics(phi)
            rmse = np.sqrt(np.mean((x_phys - x_recon)**2))
            
            fig.add_trace(go.Scatter(
                x=theta_deg, 
                y=ratio,
                mode='lines',
                name=f"{fuel}mg Ratio"
            ))
            
            print(f"{fuel:<10.1f} | {np.max(ratio):<10.2f} | {np.min(ratio):<10.2f} | {rmse*1000:<10.4f}")
            
            # Litvin Check
            from truthmaker.surrogates.adapters.litvin import LitvinVerifier
            verifier = LitvinVerifier()
            litvin_res = verifier.verify_ratio_profile(theta_rad, ratio)
            
            if not litvin_res["sign_consistent"]:
                print(f"  [FAIL] Litvin: Mixed Sign (Reversal). Range: [{litvin_res['min_ratio']:.2f}, {litvin_res['max_ratio']:.2f}]")
            else:
                print(f"  [PASS] Litvin: Sign Consistent.")
                
            if litvin_res["singularity_internal"]:
                print(f"  [FAIL] Litvin: Internal Singularity (Ratio crosses 1.0).")
                
            if not litvin_res["is_closed"]:
                print(f"  [FAIL] Litvin: Closure Error {litvin_res['closure_error_deg']:.2f} deg.")

            
        except Exception as e:
            print(f"{fuel:<10} | Error: {e}")

    fig.update_layout(
        title="Required Gear Ratio Function (Transmission Profile)",
        xaxis_title="Cycle Clock Angle (deg)",
        yaxis_title="Gear Ratio (d_crank / d_clock)",
        template="plotly_white"
    )
    
    out_path = os.path.join(PROJECT_ROOT, "output/gear_ratio_plot.html")
    fig.write_html(out_path)
    print(f"\nSaved Gear Ratio Plot to {out_path}")

if __name__ == "__main__":
    run_interpreter()
