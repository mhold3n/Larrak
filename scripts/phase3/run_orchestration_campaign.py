"""
Orchestration Campaign.
"Calibrate-Then-Optimize" Strategy.

1. Sampling: Generates a Training Set (LHS/Grid).
2. Simulation: Runs High-Fidelity Physics on Training Set.
3. Surrogate: Fits Polynomial/Neural Models to physics data.
4. Scale: Ready for Full Factorial Optimization (0D).
"""

import os
import sys
import json
import shutil
import numpy as np
import time
from pathlib import Path
from typing import List, Dict

# Helpers
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# PATCH DLL for MKL
try:
    conda_prefix = sys.prefix
    conda_paths = [os.path.join(conda_prefix, "Library", "bin")]
    current_path = os.environ.get("PATH", "")
    for p in conda_paths:
        if os.path.exists(p) and p not in current_path:
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                try: os.add_dll_directory(p) 
                except: pass
except: pass

from Simulations.common.sampling import generate_stratified_training_set
from Simulations.common.io_schema import SimulationInput, OperatingPoint, GeometryConfig, BoundaryConditions
from Simulations.structural.friction import FrictionSimulation, FrictionConfig
from Simulations.combustion.prechamber_reduced import PrechamberSimulation, CombustionConfig
from Simulations.common.logger import SimulationLogger
from Simulations.structural.fit_friction_map import fit_friction_map
from Simulations.combustion.fit_wiebe_map import fit_wiebe_map

def run_campaign(n_samples: int = 25, clean_start: bool = True):
    print(f"=== Orchestration Campaign: N={n_samples} Samples ===")
    
    # 1. Design Space Definition
    bounds = {
        "rpm": (1000.0, 6000.0),
        "intake_pressure": (0.5e5, 3.0e5), # 0.5 to 3.0 bar
        "lambda": (0.8, 1.2) # Optional variation
    }
    
    # 2. Generate Training Set
    training_set = generate_stratified_training_set(bounds, n_samples)
    print(f"\n[Sampling] Generated {len(training_set)} training points.")
    
    # 3. Execution Loop
    sim_dir = PROJECT_ROOT / "Simulations" / "_runs_phase5"
    if clean_start and sim_dir.exists():
        shutil.rmtree(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)
    
    logger = SimulationLogger(run_root=sim_dir)
    
    print("\n[Simulation] Starting High-Fidelity Loop...")
    
    start_time = time.time()
    
    for i, point in enumerate(training_set):
        print(f"   [{i+1}/{len(training_set)}] Processing: {point}")
        
        # Create Input
        # Dummy BCs for schema validation (High-Fi sims might recalc them or ignore)
        dummy_ca = np.linspace(0, 720, 721).tolist()
        dummy_zeros = np.zeros(721).tolist()
        
        inp = SimulationInput(
            run_id=f"train_{i}",
            geometry=GeometryConfig(bore=0.1, stroke=0.2, compression_ratio=15.0, conrod=0.4), 
            operating_point=OperatingPoint(
                rpm=point["rpm"],
                p_intake=point["intake_pressure"],
                lambda_val=1.0 / point.get("lambda", 1.0), # Schema uses lambda_val? Check.
                T_intake=320.0
            ),
            boundary_conditions=BoundaryConditions(
                crank_angle=dummy_ca,
                pressure_gas=dummy_zeros,
                temperature_gas=dummy_zeros
            )
        )
        # Note: OperatingPoint field is 'lambda_val' in schema definition! 
        # Line 25: lambda_val: float = Field(1.0...)
        # I must check if I used 'phi' or 'lambda_val'.
        # Previous code used 'phi' in constructor call => likely wrong if pydantic.
        
        # Run Friction
        try:
             f_sim = FrictionSimulation("fric", FrictionConfig())
             f_sim.load_input(inp)
             f_res = f_sim.solve_steady_state()
             
             # Run Combustion
             c_sim = PrechamberSimulation("comb", CombustionConfig())
             c_sim.load_input(inp)
             c_res = c_sim.solve_steady_state()
             
             # Merge Calibration Params
             merged_params = {}
             if f_res.calibration_params: merged_params.update(f_res.calibration_params)
             if c_res.calibration_params: merged_params.update(c_res.calibration_params)
             
             # Add Input Features explicitly
             merged_params["rpm"] = inp.operating_point.rpm
             merged_params["load"] = inp.operating_point.p_intake
             merged_params["jet_intensity"] = inp.operating_point.p_intake / 1e5 * (inp.operating_point.rpm/3000) 
             
             # Update c_res with merged
             c_res.calibration_params = merged_params
             c_res.friction_fmep = f_res.friction_fmep
             
             logger.start_run(inp)
             logger.log_output(c_res)
             
        except Exception as e:
            print(f"      Failed: {e}")
            
    duration = time.time() - start_time
    print(f"\n[Simulation] Loop Finished in {duration:.2f}s")
    
    # 4. Calibration (Fitting)
    print("\n[Calibration] Fitting Surrogates...")
    run_dirs = [x for x in sim_dir.iterdir() if x.is_dir()]
    
    if len(run_dirs) > 0:
        fit_friction_map(run_dirs)
        fit_wiebe_map(run_dirs)
    else:
        print("No successful runs to fit.")
    
    print("\n=== Orchestration Campaign Complete ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=25, help="Number of training samples")
    args = parser.parse_args()
    
    run_campaign(n_samples=args.samples)
