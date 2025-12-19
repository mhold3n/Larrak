
import time
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_bound_task(n):
    # Simulate heavy Python work (Holds GIL)
    s = 0
    for i in range(n):
        s += i * i
    return s

def bench_parallel():
    N_TASK = 10000000
    WORKERS = 8
    TASKS = 32
    
    print(f"Benchmarking Parallelism: {TASKS} tasks, {WORKERS} workers")
    
    # 1. ThreadPoolExecutor
    start = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        list(executor.map(cpu_bound_task, [N_TASK]*TASKS))
    end = time.time()
    t_threads = end - start
    print(f"Threads (GIL): {t_threads:.4f}s")
    
    # 2. ProcessPoolExecutor
    start = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        list(executor.map(cpu_bound_task, [N_TASK]*TASKS))
    end = time.time()
    t_procs = end - start
    print(f"Processes (No GIL): {t_procs:.4f}s")
    
    speedup = t_threads / t_procs
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    bench_parallel()
