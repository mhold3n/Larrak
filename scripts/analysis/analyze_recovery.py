
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Path to fixed results
CSV_PATH = r"tests/goldens/phase1/doe_output/phase1_physical_results_fixed.csv"

def analyze():
    if not os.path.exists(CSV_PATH):
        print(f"File {CSV_PATH} not found.")
        return

    print(f"Loading {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV (might be writing): {e}")
        return

    total = len(df)
    status_counts = df["status"].value_counts()
    
    print("\n--- Status Summary ---")
    print(status_counts)
    
    # Filter for intended fix targets (originally failed)
    # This requires knowing which ones were failures. 
    # Whatever has "Fixed_" prefix or "Maximum_Iterations_Exceeded" is relevant.
    
    mask_fixed = df["status"].str.contains("Fixed_", na=False) | (df["status"] == "Maximum_Iterations_Exceeded")
    df_fixed = df[mask_fixed]
    
    print(f"\nSubset of originally failed/processing points: {len(df_fixed)}")
    print(df_fixed["status"].value_counts())
    
    # 3D Scatter of Status
    # We want to see WHERE the Infeasible points are.
    # Color by Status.
    
    fig = px.scatter_3d(
        df,
        x="rpm",
        y="p_int_bar",
        z="fuel_mass_mg",
        color="status",
        title="DOE Results Status Distribution (Live)",
        opacity=0.6
    )
    
    # Save chart
    output_html = "tests/goldens/phase1/doe_output/recovery_status_3d.html"
    fig.write_html(output_html)
    print(f"\nSaved 3D plot to {output_html}")
    
    # Check for "Optimal" recoveries
    recovered = df[df["status"] == "Fixed_Optimal"] # Or if I logic maps them to just "Optimal"?
    # My script sets status "Optimal" if valid.
    # But how do I distinguish ORIGINALLY Optimal from RECOVERED Optimal?
    # I didn't verify this.
    # The script: "status": "Optimal" if valid else ...
    # So they merge into "Optimal".
    # But I can check `solver_status`.
    # Original Optimal has `solver_status` = "Solve_Succeeded".
    # Recovered Optimal will also have "Solve_Succeeded".
    # But maybe `iter_count` is different?
    
    pass

if __name__ == "__main__":
    analyze()
