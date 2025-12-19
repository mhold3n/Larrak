import pandas as pd
import os

results_file = "dashboard/phase1/phase1_physical_results.csv"

def analyze():
    if not os.path.exists(results_file):
        print("Results file not found.")
        return

    try:
        df = pd.read_csv(results_file, on_bad_lines='skip')
        
        print(f"--- Run Status ---")
        print(f"Total Samples: {len(df)}")
        print(f"Solver Status Breakdown:")
        print(df['status'].value_counts())
        
        # Filter for valid numerical results for physical analysis
        valid_df = df[pd.to_numeric(df['thermal_efficiency'], errors='coerce').notnull()]
        valid_df = valid_df[valid_df['status'].isin(['Optimal', 'Acceptable', 'Maximum_Iterations_Exceeded'])]
        
        if len(valid_df) == 0:
            print("\nNo valid numerical data yet.")
            return

        print(f"\n--- Physical Validation (N={len(valid_df)}) ---")
        
        print("\nThermal Efficiency (Target: 0.30 - 0.45):")
        print(valid_df['thermal_efficiency'].describe())
        
        print("\nPeak Pressure [bar] (Expect: 50 - 200):")
        print(valid_df['p_max_bar'].describe())
        
        print("\nNet Work [J] (Expect: Positive):")
        print(valid_df['abs_work_net_j'].describe())
        
        # Check for non-physical outliers
        neg_work = len(valid_df[valid_df['abs_work_net_j'] < 0])
        super_eff = len(valid_df[valid_df['thermal_efficiency'] > 0.60])
        
        print(f"\n--- Outlier Check ---")
        print(f"Negative Work Points: {neg_work} ({(neg_work/len(valid_df))*100:.1f}%)")
        print(f"Suspiciously High Efficiency (>60%): {super_eff} ({(super_eff/len(valid_df))*100:.1f}%)")

        # Top Performer
        best_pt = valid_df.loc[valid_df['thermal_efficiency'].idxmax()]
        print(f"\n--- Current Best Design ---")
        print(best_pt[['rpm', 'p_int_bar', 'fuel_mass_mg', 'thermal_efficiency', 'p_max_bar']])

    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    analyze()
