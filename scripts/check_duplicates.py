import pandas as pd

file_path = "dashboard/phase1/phase1_physical_results.csv"
try:
    df = pd.read_csv(file_path, on_bad_lines='skip')
    print(f"Total Rows: {len(df)}")
    
    # Check for exact duplicates of inputs
    cols = ['rpm', 'p_int_bar', 'fuel_mass_mg']
    dupes = df[df.duplicated(subset=cols, keep=False)]
    
    if len(dupes) > 0:
        print(f"Found {len(dupes)} duplicate entries.")
        print("Sample Duplicates:")
        print(dupes.head(10))
        
        # Check if values are identical or if they were reresolved
        print("\nChecking consistency of results for duplicates...")
        sample_pt = dupes.iloc[0]
        match = df[(df['rpm'] == sample_pt['rpm']) & 
                   (df['p_int_bar'] == sample_pt['p_int_bar']) & 
                   (df['fuel_mass_mg'] == sample_pt['fuel_mass_mg'])]
        print(match)
    else:
        print("No duplicates found. The grid size might be larger than expected.")
        
except Exception as e:
    print(f"Error: {e}")
