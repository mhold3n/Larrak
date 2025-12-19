import time
import pandas as pd
import numpy as np
import os

def bench_io(rows, cols, workers):
    print(f"Benchmarking CSV I/O: {rows} rows, {cols} cols, simulated {workers} parallel sources...")
    
    # Simulate data
    data = [{f"col_{c}": np.random.rand() for c in range(cols)} for _ in range(rows)]
    df = pd.DataFrame(data)
    
    output_file = "bench_io.csv"
    if os.path.exists(output_file):
        os.remove(output_file)
        
    start = time.time()
    
    # Simulate the DOERunner write pattern (Sequential Append in Main Thread)
    # Since DOERunner writes in the main thread (thread-safe), the contention is purely disk speed
    # NOT locking between workers.
    # We will just write chunks to see raw speed.
    
    chunk_size = 10
    for i in range(0, rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        mode = 'w' if i == 0 else 'a'
        header = True if i == 0 else False
        chunk.to_csv(output_file, mode=mode, header=header, index=False)
        
    end = time.time()
    duration = end - start
    rate = rows / duration
    
    print(f"Time: {duration:.4f}s")
    print(f"Rate: {rate:.1f} rows/sec")
    print("-" * 20)

if __name__ == "__main__":
    # Test small row (typical DOE point)
    bench_io(1000, 15, 8) 
