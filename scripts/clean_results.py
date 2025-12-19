import pandas as pd

file_path = "dashboard/phase1/phase1_physical_results.csv"

def clean():
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"Original Rows: {len(df)}")
        
        # Drop duplicates based on inputs
        cols = ['rpm', 'p_int_bar', 'fuel_mass_mg']
        df_clean = df.drop_duplicates(subset=cols, keep='last') # Keep last to prefer newer/resumed results? Or first?
        # Actually 'last' is better if I debugged/fixed something, but here it's just a resume. Last is fine.
        
        print(f"Cleaned Rows: {len(df_clean)}")
        
        if len(df_clean) != len(df):
            print("Saving cleaned file...")
            df_clean.to_csv(file_path, index=False)
            print("Done.")
        else:
            print("No duplicates to remove.")
            
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    clean()
