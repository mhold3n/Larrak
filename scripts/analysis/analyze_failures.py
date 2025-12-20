
import pandas as pd
import plotly.express as px
import os

CSV_PATH = r"tests/goldens/phase1/doe_output/phase1_physical_results.csv"

def analyze_failures():
    if not os.path.exists(CSV_PATH):
        print("CSV not found")
        return

    df = pd.read_csv(CSV_PATH)
    total = len(df)
    failed = df[df["status"] == "Maximum_Iterations_Exceeded"]
    
    print(f"Total Points: {total}")
    print(f"Max Iters Exceeded: {len(failed)} ({len(failed)/total*100:.1f}%)")
    
    # 3D Scatter of Failures vs Success
    df["is_failed"] = df["status"] == "Maximum_Iterations_Exceeded"
    
    fig = px.scatter_3d(
        df, 
        x="rpm", 
        y="p_int_bar", 
        z="fuel_mass_mg",
        color="is_failed",
        title="Distribution of Solver Failures (Yellow=Failed)",
        opacity=0.6,
        size_max=2
    )
    
    output_html = "tests/goldens/DOE_Failures.html"
    fig.write_html(output_html)
    print(f"Visualization saved to {output_html}")
    
    # Text Analysis
    print("\nFailure Clusters (counts by RPM):")
    print(failed["rpm"].value_counts().sort_index())

if __name__ == "__main__":
    analyze_failures()
