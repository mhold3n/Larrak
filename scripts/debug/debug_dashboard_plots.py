
import sys
import os
import pandas as pd
import traceback

# Add project root
sys.path.insert(0, os.getcwd())

try:
    from tests.infra.sensitivity_dashboard import generate_phase_report, load_matrix_data
    from tests.infra.interpretation import DOEAnalyzer
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

csv_path = r"tests/goldens/phase1/doe_output/phase1_physical_results.csv"
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows.")

try:
    print("Initializing Analyzer...")
    analyzer = DOEAnalyzer(df=df)
    
    print("Fitting Model...")
    # inputs=["rpm", "p_int_bar", "fuel_mass_mg"]
    # response="objective" (or thermal_efficiency)
    
    # Filter valid rows for fit
    inputs = ["rpm", "p_int_bar", "fuel_mass_mg"]
    response = "objective"
    analyzer.fit_model(inputs, response)
    print("Model Fitted.")
    
    print("Generating Structure Figures...")
    eff = analyzer.get_standardized_effects()
    print("Standardized Effects calculated.")
    
    print("Computing Sensitivity...")
    bounds = {c: (df[c].min(), df[c].max()) for c in inputs}
    sens = analyzer.compute_sensitivity(bounds)
    print("Sensitivity Computed.")
    
    print("Generating Phase Report...")
    report = generate_phase_report(df, "Phase 1")
    if "error" in report:
        print(f"Report Generation Error: {report['error']}")
    else:
        print("Report Generated Successfully keys:", report.keys())
        
except Exception:
    traceback.print_exc()
