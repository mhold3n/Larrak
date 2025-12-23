import pandas as pd
import glob
import os


def analyze_doe():
    # Find latest CSV
    pattern = "tests/goldens/phase1/doe_output/phase1_sensitivity_doe_*.csv"
    files = glob.glob(pattern)
    if not files:
        print("No DOE output files found.")
        return

    latest_file = max(files, key=os.path.getmtime)
    print(f"Analyzing: {latest_file}")

    try:
        df = pd.read_csv(latest_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    total = len(df)
    print(f"Total Cases: {total}")

    if "status" not in df.columns:
        print("Column 'status' not found.")
        return

    stats = df["status"].value_counts()
    print("\n=== Status Distribution ===")
    print(stats)

    success = stats.get("Optimal", 0) + stats.get("Solve_Succeeded", 0)
    # Check for custom statuses
    if "solver_status" in df.columns:
        solver_stats = df["solver_status"].value_counts()
        print("\n=== Solver Status ===")
        print(solver_stats)

        # Refine Success count
        success_solver = solver_stats.get("Solve_Succeeded", 0)
        print(f"\nSolver Successes: {success_solver}")

    print(f"\nSuccess Rate (Status): {success / total * 100:.2f}%")

    # Analyze Boost Skipping
    if "doe_status" in df.columns:
        doe_stats = df["doe_status"].value_counts()
        print("\n=== DOE Status ===")
        print(doe_stats)

    # Analyze Infeasible Problem Detected
    infeasible = df[df["status"] == "Infeasible_Problem_Detected"]
    if not infeasible.empty:
        print(f"\nInfeasible Problems: {len(infeasible)}")
        # breakdown by rpm/load?
        print("Sample Infeasible:")
        print(infeasible[["rpm", "q_total", "phi"]].head())


if __name__ == "__main__":
    analyze_doe()
