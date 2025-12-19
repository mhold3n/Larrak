
import sys
import os

try:
    import pandas as pd
    print("Pandas imported")
except ImportError as e:
    print(f"Pandas failed: {e}")

try:
    import sklearn
    from sklearn.linear_model import LinearRegression
    print("Sklearn imported")
except ImportError as e:
    print(f"Sklearn failed: {e}")

try:
    from tests.infra.interpretation import DOEAnalyzer
    print("DOEAnalyzer imported")
except ImportError as e:
    print(f"DOEAnalyzer failed: {e}")

csv_path = r"C:\Users\maxed\OneDrive\Desktop\Github Projects\Larrak\tests\goldens\phase1\doe_output\phase1_physical_results.csv"
if os.path.exists(csv_path):
    print(f"CSV found: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")
        
        analyzer = DOEAnalyzer(df=df)
        print("Analyzer instantiated")
    except Exception as e:
        print(f"Data loading failed: {e}")
else:
    print("CSV not found")
