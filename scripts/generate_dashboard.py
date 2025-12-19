import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
import ast

# Setup Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from thermo.visualization import (
    plot_valve_strategy, 
    plot_motion_family, 
    plot_efficiency_map
)

# Configuration
GOLDENS_DIR = os.path.join(PROJECT_ROOT, "tests/goldens")
DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")
os.makedirs(DASHBOARD_DIR, exist_ok=True)

def generate_doe_plots():
    print("Generating DOE Plots...")
    # 1. Phase 1 DOE (Physical)
    p1_csv = os.path.join(GOLDENS_DIR, "phase1/doe_output/phase1_physical_results_fixed.csv")
    if os.path.exists(p1_csv):
        df = pd.read_csv(p1_csv)
        # Filter realistic (0 < eff < 0.6)
        # Remove negatives (physical impossibility) and super-high artifacts
        df = df[(df["thermal_efficiency"] > 0.0) & (df["thermal_efficiency"] < 0.6)] 
        
        # Original simple map
        plot_efficiency_map(
            df, "rpm", "fuel_mass_mg", "thermal_efficiency",
            os.path.join(DASHBOARD_DIR, "doe_efficiency_map_basic.html")
        )
        
        # Requested specific 3D Surface (RPM, Air, Fuel)
        from thermo.visualization.doe import plot_efficiency_surface
        plot_efficiency_surface(
            df, 
            os.path.join(DASHBOARD_DIR, "doe_efficiency_surface_rpm_air_fuel.html")
        )

def generate_motion_plots():
    print("Generating Motion Family Plots...")
    # Loading Standard Motion Family Results
    # This usually comes from extract_motion_family.py which generates a CSV or HTML.
    # Let's assume we can re-use the Phase 1 DOE results which contain x_opt?
    # No, Phase 1 only has scalar results.
    # The motion family script runs its own optimization.
    # If JSON not available, skip.
    print("  (Skipping Motion plots - requires extract_motion_family.py to save JSON first)")

def generate_valve_plots():
    print("Generating Valve Strategy Plots...")
    json_path = os.path.join(GOLDENS_DIR, "phase4/valve_strategy/valve_strategy_results.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Convert list back to numpy? Not strictly needed for Plotly, but nice.
        # But our viz function expects dicts.
        
        from thermo.visualization import plot_valve_strategy
        out_path = os.path.join(DASHBOARD_DIR, "valve_strategy_load_sweep.html")
        plot_valve_strategy(data, out_path)
    else:
        print(f"  Warning: {json_path} not found. Run scripts/extract_valve_strategy.py first.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Project Dashboard")
    parser.add_argument("--component", type=str, choices=["all", "valve", "doe", "motion"], default="all")
    args = parser.parse_args()
    
    if args.component in ["all", "doe"]:
        generate_doe_plots()
    if args.component in ["all", "motion"]:
        generate_motion_plots()
    if args.component in ["all", "valve"]:
        generate_valve_plots()
        
    print(f"Dashboard generated in {DASHBOARD_DIR}")
