"""
Phase 1 Interpretation Script.
Analyzes Thermo DOE data (Efficiency, etc.)
"""

import glob
import os
import sys

import pandas as pd

# Add project root
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from tests.infra.interpretation import DOEAnalyzer


def run_analysis() -> None:
    # 1. Load Data
    data_dir = os.path.join(os.path.dirname(__file__), "doe_output")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print("No DOE data found.")
        return

    # Prefer combined adaptive file if exists
    combined_files = [f for f in csv_files if "adaptive_combined" in f]
    if combined_files:
        latest_csv = max(combined_files, key=os.path.getmtime)
    else:
        latest_csv = max(csv_files, key=os.path.getmtime)

    print(f"Analyzing {latest_csv}...")

    analyzer = DOEAnalyzer(data_path=latest_csv)

    # 2. Define Problem
    # Inputs: rpm, q_total, phi
    # Response: efficiency (calculated)

    # Pre-calc efficiency if needed
    if "efficiency" not in analyzer.df.columns:
        # eff = -objective / q_total
        # Ensure q_total is numeric
        analyzer.df["objective"] = pd.to_numeric(analyzer.df["objective"])
        analyzer.df["q_total"] = pd.to_numeric(analyzer.df["q_total"])
        analyzer.df["efficiency"] = (
            -1.0 * analyzer.df["objective"] / (analyzer.df["q_total"] + 1e-9)
        )

    inputs = ["rpm", "q_total", "phi"]
    response = "efficiency"

    bounding_box = {"rpm": (800, 8000), "q_total": (1000, 7000), "phi": (0.7, 1.3)}

    # 3. Fit RSM
    print("\n--- Response Surface Model ---")
    analyzer.fit_model(inputs, response, degree=2, interactions=True)

    # 4. ANOVA
    print("\n--- ANOVA Screening ---")
    anova_table = analyzer.run_anova()
    print(anova_table)

    # 5. Sensitivity Analysis
    print("\n--- Global Sensitivity (Derivative-based) ---")
    sens = analyzer.compute_sensitivity(bounding_box)
    print(sens)

    # 6. Physics Check
    # Efficiency should peak around Phi=1.0 or lean (phi<1.0) depending on model?
    # Usually lean is efficient until misfire.
    # Check slope at nominal point
    nominal = pd.DataFrame({"rpm": [2000], "q_total": [3000], "phi": [1.0]})
    pred_nominal = analyzer.predict(nominal)[0]

    lean = pd.DataFrame({"rpm": [2000], "q_total": [3000], "phi": [0.8]})
    pred_lean = analyzer.predict(lean)[0]

    print("\n--- Physics Checks ---")
    print(f"Eff @ Phi=1.0: {pred_nominal:.4f}")
    print(f"Eff @ Phi=0.8: {pred_lean:.4f}")

    if pred_lean > pred_nominal:
        print("PASS: Leaner mix improves efficiency (expected for ideal Otto/Diesel).")
    else:
        print("NOTE: Leaner mix did not improve efficiency. Check specific heat ratio model.")

    # Save artifact
    report_path = os.path.join(os.path.dirname(__file__), "INTERPRETATION_REPORT.txt")
    with open(report_path, "w") as f:
        f.write(analyzer.generate_report_text())
        f.write("\n\nPhysics Checks:\n")
        f.write(f"Eff(Phi=1.0)={pred_nominal:.4f}, Eff(Phi=0.8)={pred_lean:.4f}\n")

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    run_analysis()
