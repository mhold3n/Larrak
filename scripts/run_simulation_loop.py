"""
High-Fidelity Simulation Loop Orchestrator.
Automates the Surrogate-Based Optimization (SBO) Cycle:
1. Optimize 0D Engine (using current calibration).
2. Export Candidate Design.
3. Run High-Fidelity Simulations (Friction, Combustion).
4. Update Calibration Maps (Fit Surrogates).
5. Repeat (Future).
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# PATCH: Explicitly add Conda Library bin for MKL/Numpy (Common Persistence)
try:
    conda_prefix = sys.prefix
    conda_paths = [
        os.path.join(conda_prefix, "Library", "bin"),
        os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
        os.path.join(conda_prefix, "Library", "usr", "bin"),
    ]
    current_path = os.environ.get("PATH", "")
    for p in conda_paths:
        if os.path.exists(p) and p not in current_path:
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                try: os.add_dll_directory(p)
                except: pass
except: pass

from thermo.nlp import build_thermo_nlp
from thermo.export_candidate import export_candidate
from Simulations.common.io_schema import SimulationInput

# Simulation Solvers
from Simulations.structural.friction import FrictionSimulation, FrictionConfig
from Simulations.combustion.prechamber_reduced import PrechamberSimulation, CombustionConfig
from Simulations.common.logger import SimulationLogger

# Fitters
import Simulations.structural.fit_friction_map as friction_fitter
import Simulations.combustion.fit_wiebe_map as combustion_fitter

def run_phase4_cycle():
    print("=== High-Fidelity Simulation Loop Orchestrator ===")
    
    # 1. Run 0D Optimization (Single Point)
    print("\n[Step 1] Running 0D Optimization...")
    # Target: 3000 RPM, High Load
    rpm_target = 3000.0
    p_int_target = 2.0e5 # 2 bar
    T_int_target = 320.0
    
    # Build NLP
    nlp, meta = build_thermo_nlp(n_coll=40)
    
    # Solve
    import casadi as ca
    opts = {"ipopt": {"max_iter": 100, "print_level": 0}, "print_time": 0}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    
    # Solve
    res = solver(x0=meta["w0"], lbx=meta["lbw"], ubx=meta["ubw"], lbg=meta["lbg"], ubg=meta["ubg"])
    w_opt = res["x"]
    print(f"   Optimization Solved: Status={solver.stats()['return_status']}")
    
    # 2. Export Candidate
    print("\n[Step 2] Exporting Candidate Design...")
    
    # Define Geometry & OP Params derived from optimization or fixed
    # In Phase 1 these are fixed, usually. 
    # Let's match typical Larrak demo: 100mm bore, 200mm stroke
    geo_params = {
        "bore": 0.1,
        "stroke": 0.2, # Long stroke
        "conrod": 0.4,
        "compression_ratio": 15.0
    }
    
    ops_params = {
        "rpm": rpm_target,
        "p_int": p_int_target, # Pa
        "T_int": T_int_target, # K
        "lambda": 1.0
    }
    
    run_id = f"cycle_demo_{int(rpm_target)}"
    
    # Generate JSON String
    json_str = export_candidate(run_id, meta, np.array(w_opt), geo_params, ops_params)
    
    # Save to File
    out_dir = Path("Simulations/candidates")
    out_dir.mkdir(exist_ok=True, parents=True)
    json_path = out_dir / f"{run_id}.json"
    
    with open(json_path, "w") as f:
        f.write(json_str)
        
    print(f"   Candidate saved to: {json_path}")
    
    # Load it back to modify logic if needed
    inp = SimulationInput.model_validate_json(json_str)
        
    # FORCE Operating Point to Test Condition (since NLP might be generic)
    inp.operating_point.rpm = rpm_target
    inp.operating_point.p_intake = 2.0e5 # 2 bar boost
    
    # 3. Run High-Fidelity Simulations
    print("\n[Step 3] Running High-Fidelity Simulations...")
    logger = SimulationLogger()
    
    # 3.1 Friction
    print("   Running Friction Simulation...")
    f_sim = FrictionSimulation("friction_loop", FrictionConfig())
    f_sim.load_input(inp)
    f_res = f_sim.solve_steady_state()
    logger.start_run(inp) # Start new run dir
    logger.log_output(f_res, meta={"domain": "friction", "cycle": 1})
    print(f"      FMEP: {f_res.friction_fmep:.4f} bar")
    
    print(f"      FMEP: {f_res.friction_fmep:.4f} bar")
    
    # Force distinct timestamp for logger
    import time
    time.sleep(2)
    
    # 3.2 Combustion
    print("   Running Combustion Simulation...")
    c_sim = PrechamberSimulation("combustion_loop", CombustionConfig())
    c_sim.load_input(inp)
    c_res = c_sim.solve_steady_state()
    # Log to same run dir? Logger.start_run creates new dir. 
    # Usually we want separate runs or same run?
    # Phase 4 DB structure: One folder per "Simulation Run". 
    # If we run both sims on same input, maybe they share folder?
    # For now, separate folders is safer for fitters scanning.
    logger.start_run(inp) 
    logger.log_output(c_res, meta={"domain": "combustion", "cycle": 1})
    print(f"      Wiebe M: {c_res.calibration_params.get('wiebe_m', 0.0):.4f}")
    
    # 4. Update Calibration Maps
    print("\n[Step 4] Updating Calibration Maps...")
    
    # Scan all runs
    run_root = Path("Simulations/_runs")
    all_runs = [x for x in run_root.iterdir() if x.is_dir()]
    
    # Friction Fit
    friction_fitter.fit_friction_map(all_runs)
    
    # Combustion Fit
    combustion_fitter.fit_wiebe_map(all_runs)
    
    print("\n=== Loop Complete ===")

if __name__ == "__main__":
    run_phase4_cycle()
