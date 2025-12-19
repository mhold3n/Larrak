import pandas as pd
import os

f = "dashboard/phase1/phase1_physical_results.csv"
print(f"Checking {f}...")
if os.path.exists(f):
    print(f"Size: {os.path.getsize(f)} bytes")
    try:
        df = pd.read_csv(f, on_bad_lines='skip')
        print(f"Read success: {len(df)} rows")
        print(df.head())
    except Exception as e:
        print(f"Read failed: {e}")
else:
    print("File not found.")
