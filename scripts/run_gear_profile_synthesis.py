import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate.breathing import BreathingKinematics
from provenance.context import run_context
from provenance import hooks

def run_gear_profile_synthesis():
    # Start Provenance Run
    with run_context(module_id="gear_profile_synthesis"):
        hooks.install()
        try:
            print("--- Running Gear Profile Synthesis ---")
            
            # 1. Load Motion Profile
            csv_path = "dashboard/orchestration/target_motion_profile.csv"
            # Ensure folder exists for demo purposes if it doesn't - user likely has it, but good to be safe or check existance
            if not os.path.exists(csv_path):
                print(f"Error: Motion profile not found at {csv_path}")
                # For demo, creating a dummy file if missing, to prove the flow
                # In real scenario we return, but I want to force a trace
                return

            df = pd.read_csv(csv_path)
            theta_arr = df["theta_rad"].values
            x_arr = df["x_m"].values
            
            # 2. Analyze Motion
            x_min = np.min(x_arr)
            x_max = np.max(x_arr)
            stroke_eff = x_max - x_min
            
            print(f"Effective Stroke: {stroke_eff*1000:.2f} mm")
            
            # Normalize Position to 0..S for the solver
            x_norm = x_arr - x_min
            
            # 3. Solze Kinematics
            bk = BreathingKinematics(stroke=stroke_eff)
            c_arr, psi_arr, ratio_arr = bk.solve_trajectory(x_norm, theta_arr)
            
            print(f"Planet Radius required: {bk.r_p*1000:.2f} mm")
            
            # 4. Visualization
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=("Motion Decomposition", "Gear Angle (Psi)", "Gear Ratio (dPsi/dTheta)")
            )
            
            # Plot 1: Motion
            fig.add_trace(go.Scatter(x=theta_arr, y=x_norm, name="Target x(theta)", line=dict(color="blue", width=3)), row=1, col=1)
            fig.add_trace(go.Scatter(x=theta_arr, y=c_arr, name="Breathing C(theta)", line=dict(color="green", dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=theta_arr, y=x_norm - c_arr, name="Gear Residual", line=dict(color="red")), row=1, col=1)
            
            # Plot 2: Psi
            fig.add_trace(go.Scatter(x=theta_arr, y=np.degrees(psi_arr), name="Psi (deg)", line=dict(color="orange")), row=2, col=1)
            
            # Plot 3: Ratio
            fig.add_trace(go.Scatter(x=theta_arr, y=ratio_arr, name="Gear Ratio", line=dict(color="purple")), row=3, col=1)
            
            fig.update_layout(height=900, title_text=f"Breathing Gear Design - Stroke {stroke_eff*1000:.1f}mm")
            fig.update_xaxes(title_text="Crank Angle (rad)", row=3, col=1)
            
            out_path = "dashboard/final_gear_set.html"
            fig.write_html(out_path)
            print(f"Dashboard saved to {out_path}")
            
            # Save Gear Data
            df_gear = pd.DataFrame({
                "theta": theta_arr,
                "c_breathing": c_arr,
                "psi_planet": psi_arr,
                "ratio": ratio_arr
            })
            df_gear.to_csv("dashboard/final_gear_data.csv", index=False)
            print("Gear data saved to dashboard/final_gear_data.csv")

        finally:
            hooks.uninstall()

if __name__ == "__main__":
    run_gear_profile_synthesis()
