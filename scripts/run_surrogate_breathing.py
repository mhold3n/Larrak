import os
import sys
import json
import numpy as np
import plotly.graph_objects as go

# Setup Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from surrogate.breathing import BreathingKinematics

def run_breathing_check():
    json_path = os.path.join(PROJECT_ROOT, "tests/goldens/phase4/valve_strategy/valve_strategy_results.json")
    if not os.path.exists(json_path):
        print("Error: JSON results not found.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    # Stroke 0.1m
    bk = BreathingKinematics(stroke=0.1)
    
    fig = go.Figure()
    
    print(f"{'Fuel':<10} | {'Max Ratio':<10} | {'Min Ratio':<10} | {'Planet R(mm)':<15}")
    print("-" * 60)

    for res in results:
        fuel = res["fuel"]
        theta_deg = np.array(res["theta_x"])
        theta_rad = np.radians(theta_deg)
        x_phys = np.array(res["x"])
        
        try:
            C_arr, psi, ratio = bk.solve_trajectory(x_phys, theta_rad)
            
            # Plot Ratio
            fig.add_trace(go.Scatter(
                x=theta_deg, 
                y=ratio,
                mode='lines',
                name=f"{fuel}mg Ratio"
            ))
            
            print(f"{fuel:<10.1f} | {np.max(ratio):<10.2f} | {np.min(ratio):<10.2f} | {bk.r_p*1000:<15.2f}")
            
        except Exception as e:
            print(f"{fuel:<10} | Error: {e}")

    fig.update_layout(
        title="Breathing Gear Ratio (d_psi/d_theta)",
        xaxis_title="Cycle Angle (deg)",
        yaxis_title="Gear Ratio (Planet Spin / Cycle)",
        template="plotly_white"
    )
    
    out_path = os.path.join(PROJECT_ROOT, "dashboard/breathing_ratio_plot.html")
    fig.write_html(out_path)
    print(f"\nSaved Breathing Plot to {out_path}")

if __name__ == "__main__":
    run_breathing_check()
