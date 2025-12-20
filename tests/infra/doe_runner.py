
import os
import pandas as pd
import time
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

class DOERunner:
    def __init__(self, name, output_dir):
        self.name = name
        # Normalize path separators for cross-platform compatibility
        self.output_dir = str(Path(output_dir).resolve())
        self.design = None
        
    def run(self, test_func, workers=1):
        print(f"Starting execution of {len(self.design)} tests with {workers} workers.")
        
        results = []

        # Prepare Output File - use Path for consistent cross-platform paths
        output_file = str(Path(self.output_dir) / f"{self.name}_results.csv")
        
        # Convert dataframe to list of dicts
        doe_dicts = self.design.to_dict('records')

        # Defer header writing until we have results
        # PATCH: Check if file exists to support RESUME (Append Mode)
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"Resuming run: Appending to existing file {output_file}")
            header_written = True
        else:
            header_written = False
        
        if workers > 1:
            # Use ProcessPoolExecutor to bypass GIL for CPU-bound CasADi tasks
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Map returns order, but we might want to track progress
                futures = [executor.submit(test_func, params) for params in doe_dicts]
                
                # Buffer for batch writes
                batch_results = []
                
                for i, future in enumerate(futures):
                    try:
                        res = future.result()
                        merged_res = doe_dicts[i].copy()
                        merged_res.update(res)
                        
                        results.append(merged_res)
                        batch_results.append(merged_res)
                        
                        if i % 100 == 0:
                            print(f"Completed {i}/{len(doe_dicts)}")
                            
                        # Incremental Write every 10 items
                        if len(batch_results) >= 10:
                            df_batch = pd.DataFrame(batch_results)
                            mode = 'w' if not header_written else 'a'
                            df_batch.to_csv(output_file, mode=mode, header=not header_written, index=False)
                            header_written = True
                            batch_results = []
                            
                    except Exception as e:
                        print(f"Test failed: {e}")
                        err_res = doe_dicts[i].copy()
                        err_res.update({"status": "Failed", "error": str(e)})
                        results.append(err_res)
                        batch_results.append(err_res)
                            
                    except Exception as e:
                        print(f"Test failed: {e}")
                        err_res = doe_dicts[i].copy()
                        err_res.update({"status": "Failed", "error": str(e)})
                        results.append(err_res)
                        batch_results.append(err_res)
                
                # Flush remaining
                # Flush remaining
                if batch_results:
                     df_batch = pd.DataFrame(batch_results)
                     mode = 'w' if not header_written else 'a'
                     df_batch.to_csv(output_file, mode=mode, header=not header_written, index=False)
                     header_written = True

        else:
            batch_results = []
            for i, params in enumerate(doe_dicts):
                try:
                    res = test_func(params)
                    merged_res = params.copy()
                    merged_res.update(res)
                    results.append(merged_res)
                    batch_results.append(merged_res)
                except Exception as e:
                    print(f"Test {i} failed: {e}")
                    err_res = params.copy()
                    err_res.update({"status": "Failed", "error": str(e)})
                    results.append(err_res)
                    batch_results.append(err_res)
                
                if i % 10 == 0:
                   print(f"Completed {i}/{len(doe_dicts)}")
                
                
                if len(batch_results) >= 10:
                    df_batch = pd.DataFrame(batch_results)
                    mode = 'w' if not header_written else 'a'
                    df_batch.to_csv(output_file, mode=mode, header=not header_written, index=False)
                    header_written = True
                    batch_results = []
            
            # Flush remaining batch after loop completes
            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                mode = 'w' if not header_written else 'a'
                df_batch.to_csv(output_file, mode=mode, header=not header_written, index=False)
                header_written = True

        print(f"Results saved to {output_file}")
        
        # Return results - handle case where file may not exist (0 tests run with no prior data)
        if os.path.exists(output_file):
            return pd.read_csv(output_file)
        else:
            print(f"Warning: Output file not found. Returning empty DataFrame.")
            return pd.DataFrame()


