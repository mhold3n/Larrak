"""
Phase 2 Interpretation Script.
Analyzes Interpreter DOE data (Fitting Error vs Geometry).
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
    inputs = ["mean_ratio", "stroke", "conrod"]

    # Response: Fitting Error (RMSE)
    # The DOE runner might output 'fit_error' or 'objective' (which was set to fit_error)
    if "fit_error" in analyzer.df.columns:
        response = "fit_error"
    else:
        response = "objective"

    bounding_box = {"mean_ratio": (1.5, 3.0), "stroke": (0.1, 0.3), "conrod": (0.3, 0.5)}

    # 3. Fit RSM
    print("\n--- Response Surface Model ---")
    analyzer.fit_model(inputs, response, degree=2, interactions=True)

    # 4. ANOVA
    print("\n--- ANOVA Screening ---")
    print(analyzer.run_anova())

    # 5. Sensitivity
    print("\n--- Sensitivity ---")
    print(analyzer.compute_sensitivity(bounding_box))

    # 6. Physics/Geometric Checks
    # Error should increase with Scaling (Stroke) or Difficulty (Ratio)
    print("\n--- Physics Checks ---")

    # Check partial derivative sign for Stroke
    sens = analyzer.compute_sensitivity(bounding_box)  # Recalc or lookup
    # Heuristic: Predict low vs high stroke
    low_stroke = pd.DataFrame({"mean_ratio": [2.0], "stroke": [0.1], "conrod": [0.4]})
    high_stroke = pd.DataFrame({"mean_ratio": [2.0], "stroke": [0.3], "conrod": [0.4]})

    err_low = analyzer.predict(low_stroke)[0]
    err_high = analyzer.predict(high_stroke)[0]

    print(f"Error @ Stroke=0.1: {err_low:.6f}")
    print(f"Error @ Stroke=0.3: {err_high:.6f}")

    if err_high > err_low:
        print("PASS: Larger stroke increases fitting error (expected).")
    else:
        print(
            "NOTE: Larger stroke has less/equal error? Might be within noise or scaling worked well."
        )

    # Save artifact
    report_path = os.path.join(os.path.dirname(__file__), "INTERPRETATION_REPORT.txt")
    with open(report_path, "w") as f:
        f.write(analyzer.generate_report_text())
        f.write(f"\nError(Stroke=0.1)={err_low:.6f}, Error(Stroke=0.3)={err_high:.6f}\n")

    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    run_analysis()
