import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import glob
from plotly.subplots import make_subplots

# Setup Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def plot_profiles():
    profile_dir = os.path.join(PROJECT_ROOT, "output/profiles")
    files = glob.glob(os.path.join(profile_dir, "breathing_profile_*mg.csv"))
    
    if not files:
        print("No profiles found.")
        return

    # Sort files by fuel mass
    # Ensure robustness if filename format changes
    try:
        files.sort(key=lambda x: float(os.path.basename(x).split('_')[2].replace('mg.csv','')))
    except:
        pass # Fallback to default sort order

    # Create Subplots (One per profile)
    rows = len(files)
    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=[os.path.basename(f).replace("breathing_profile_","").replace(".csv","") for f in files],
        vertical_spacing=0.08
    )
    
    # Geometry Estimates (for visualization context)
    # Assumes R_center ~ 0.05m, Ratio 2:1 -> R_ring ~ 0.10m
    R_ring_est = 0.10 
    
    # Ring Circle Points
    theta_circ = np.linspace(0, 2*np.pi, 200)
    ring_x = R_ring_est * np.cos(theta_circ)
    ring_y = R_ring_est * np.sin(theta_circ)

    for i, fpath in enumerate(files):
        row = i + 1
        fname = os.path.basename(fpath)
        fuel_str = fname.replace("breathing_profile_", "").replace("mg.csv", "")
        
        df = pd.read_csv(fpath)
        
        # 1. Plot Ring (Output/Stator Reference)
        fig.add_trace(go.Scatter(
            x=ring_x, y=ring_y,
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name=f"Ring" if i==0 else None # Legend only once
        ), row=row, col=1)

        # 2. Plot Planet (Generated Profile)
        fig.add_trace(go.Scatter(
            x=df["x"], y=df["y"],
            mode='lines',
            line=dict(color='red'),
            name=f"Planet" if i==0 else None
        ), row=row, col=1)
        
        # 3. Plot Sun/Center (Proxy)
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(color='yellow', size=10, line=dict(color='orange', width=2)),
            name=f"Sun (Center)" if i==0 else None
        ), row=row, col=1)
        
        # Enforce aspect ratio per subplot
        # x and y should be 1:1
        fig.update_yaxes(
            scaleanchor=f"x{row}",
            scaleratio=1,
            row=row, col=1
        )

    fig.update_layout(
        title="Breathing Gear Sets (Planet Profiles vs Ring)",
        height=350 * rows,
        template="plotly_white",
        showlegend=True
    )
    
    out_path = os.path.join(PROJECT_ROOT, "output/breathing_gear_sets.html")
    fig.write_html(out_path)
    print(f"Saved Set Plot to {out_path}")

if __name__ == "__main__":
    plot_profiles()
