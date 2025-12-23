import glob
import os

import pandas as pd


def verify():
    # 1. Find latest Phase 1 CSV
    # Matches logic in sensitivity_dashboard.py:load_matrix_data
    csv_files = glob.glob("tests/goldens/phase1/doe_output/*.csv")
    if not csv_files:
        print("No CSV found")
        return

    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"Loading: {latest_csv}")

    # 2. Load
    df = pd.read_csv(latest_csv)

    # 3. Check columns
    print(f"Columns: {df.columns.tolist()}")

    # 4. Show sample of 'objective'
    if "objective" in df.columns:
        print("\nFirst 5 rows of 'objective' column (Raw Data):")
        print(df["objective"].head().to_string())

        print("\nStats for 'objective':")
        print(df["objective"].describe().to_string())
    else:
        print("CRITICAL: 'objective' column missing!")


if __name__ == "__main__":
    verify()
