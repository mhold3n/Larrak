"""
Phase 3 Interpretation Script.
Analyzes Mechanical/Campro DOE data (Tracking Cost, Efficiency).
"""

import sys
import os
import glob
import pandas as pd

# Add project root
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from tests.infra.interpretation import DOEAnalyzer


def run_analysis():
    # 1. Load Data
    data_dir = os.path.join(os.path.dirname(__file__), "doe_output")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print("No DOE data found.")
        return

    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"Analyzing {latest_csv}...")

    analyzer = DOEAnalyzer(data_path=latest_csv)

    # 2. Config
    inputs = ["r_max", "load_scale", "rpm"]

    # Response: Tracking Cost usually (Objective)
    response = "objective"

    bounding_box = {"r_max": (0.1, 0.5), "load_scale": (0.1, 5.0), "rpm": (800, 8000)}

    # 3. Fit RSM
    print("\n--- Response Surface Model ---")
    analyzer.fit_model(inputs, response, degree=2, interactions=True)

    # 4. ANOVA
    print("\n--- ANOVA Screening ---")
    print(analyzer.run_anova())

    # 5. Sensitivity
    print("\n--- Sensitivity ---")
    print(analyzer.compute_sensitivity(bounding_box))

    # 6. Physics Checks
    print("\n--- Physics Checks ---")

    # Load Sensitivity: Does high load destroy tracking (increase cost)?
    low_load = pd.DataFrame({"r_max": [0.3], "load_scale": [0.1], "rpm": [2000]})
    high_load = pd.DataFrame({"r_max": [0.3], "load_scale": [5.0], "rpm": [2000]})

    cost_low = analyzer.predict(low_load)[0]
    cost_high = analyzer.predict(high_load)[0]

    print(f"Cost @ Load=0.1: {cost_low:.6f}")
    print(f"Cost @ Load=5.0: {cost_high:.6f}")

    if cost_high > cost_low * 1.5:
        print("PASS: High load significantly increases tracking cost/error.")
    else:
        print("NOTE: Tracking cost robust to load? Or load sensitivity masked by other factors.")

    # Save artifact
    report_path = os.path.join(os.path.dirname(__file__), "INTERPRETATION_REPORT.txt")
    with open(report_path, "w") as f:
        f.write(analyzer.generate_report_text())
        f.write(f"\nCost(Load=0.1)={cost_low:.6f}, Cost(Load=5.0)={cost_high:.6f}\n")

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    run_analysis()
