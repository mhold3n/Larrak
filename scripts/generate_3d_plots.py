
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Configuration
INPUT_CSV = r"tests/goldens/phase1/doe_output/phase1_physical_results.csv"
OUTPUT_HTML = r"tests/goldens/DOE_3D_Visuals.html"

def generate_plots():
    print(f"Loading data from {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print("Error: CSV file not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows.")

    # 1. Filter Data (Valid Physics Only)
    # Status must be Optimal or Acceptable (Max Iterations Exceeded is often usable)
    valid_statuses = ["Optimal", "Maximum_Iterations_Exceeded"]
    df_clean = df[df["status"].isin(valid_statuses)].copy()
    
    # Filter physical outliers (Efficiency must be 0-80%)
    # Some points might be numerical noise > 1.0 or < 0
    if "thermal_efficiency" in df_clean.columns:
        df_clean = df_clean[
            (df_clean["thermal_efficiency"] > 0.0) & 
            (df_clean["thermal_efficiency"] < 0.80)
        ]
    
    print(f"Filtered to {len(df_clean)} valid physical points.")

    # 2. Define Plots to Generate
    # We want Objective (or Key Outputs) on Z, Controls on X/Y
    
    # Controls: rpm, p_int_bar, fuel_mass_mg, phi_est
    # Outputs: thermal_efficiency, p_max_bar, objective
    
    figures = []
    
    # Plot 1: Efficiency Map (RPM vs Manifold Pressure)
    fig1 = create_3d_scatter(
        df_clean, 
        x="rpm", y="p_int_bar", z="thermal_efficiency", 
        color="fuel_mass_mg",
        title="Thermal Efficiency Map (RPM vs Boost)"
    )
    figures.append(fig1)

    # Plot 2: Peak Pressure (RPM vs Fueling)
    fig2 = create_3d_scatter(
        df_clean, 
        x="rpm", y="fuel_mass_mg", z="p_max_bar", 
        color="p_int_bar",
        title="Peak Cylinder Pressure (RPM vs Fuel)"
    )
    figures.append(fig2)
    
    # Plot 3: Objective Function Landscape (Boost vs Fuel)
    fig3 = create_3d_scatter(
        df_clean, 
        x="p_int_bar", y="fuel_mass_mg", z="objective", 
        color="rpm",
        title="Optimizer Objective Landscape (Boost vs Fuel)"
    )
    figures.append(fig3)

    # 3. Save to HTML
    print(f"Generating HTML at {OUTPUT_HTML}...")
    with open(OUTPUT_HTML, 'w') as f:
        f.write("<html><head><title>DOE 3D Results</title></head><body>")
        f.write("<h1 style='font-family:sans-serif; text-align:center;'>Phase 5 DOE Results: 3D Visualization</h1>")
        
        for i, fig in enumerate(figures):
            # Convert to HTML div
            fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn' if i==0 else False)
            f.write(f"<div style='margin-bottom: 50px;'>{fig_html}</div>")
            
        f.write("</body></html>")
    
    print("Done!")

def create_3d_scatter(df, x, y, z, color, title):
    fig = go.Figure(data=[go.Scatter3d(
        x=df[x],
        y=df[y],
        z=df[z],
        mode='markers',
        marker=dict(
            size=3,
            color=df[color],                # set color to an array/list of desired values
            colorscale='Viridis',   # choose a colorscale
            opacity=0.8,
            showscale=True,
            colorbar=dict(title=color)
        )
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z,
        ),
        height=700,
        margin=dict(r=0, b=0, l=0, t=50)
    )
    return fig

if __name__ == "__main__":
    generate_plots()
