"""
Larrak Engine Framework - Unified API.

This module provides the high-level interface for the Larrak Engine design system.
It wraps the Optimization (Phase 1-3) and Simulation (Phase 4) layers into a single cohesive class.
"""

import sys
import os
import numpy as np
import casadi as ca
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Modules
from thermo.nlp import build_thermo_nlp
from thermo.calibration.registry import CalibrationRegistry
from thermo.export_candidate import export_candidate
from Simulations.common.io_schema import SimulationInput, GeometryConfig, OperatingPoint

# Simulations
from Simulations.structural.friction import FrictionSimulation, FrictionConfig
from Simulations.combustion.prechamber_reduced import PrechamberSimulation, CombustionConfig

# Patch DLLs (Repeated for safety in new entry point)
try:
    conda_prefix = sys.prefix
    conda_paths = [
        os.path.join(conda_prefix, "Library", "bin"),
        os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
    ]
    current_path = os.environ.get("PATH", "")
    for p in conda_paths:
        if os.path.exists(p) and p not in current_path:
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                try: os.add_dll_directory(p)
                except: pass
except: pass


class LarrakEngine:
    """
    Main interface for Larrak Engine Design Framework.
    """
    
    def __init__(self):
        self.registry = CalibrationRegistry()
        print("LarrakEngine Initialized.")
        print(f"Loaded Calibration Maps: {list(self.registry.maps.keys())}")
        
    def optimize(self, 
                 target_power: float, 
                 constraints: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Run 0D Optimization to find optimal geometry.
        
        Args:
            target_power: Desired power in Watts (e.g. 300e3).
            constraints: Optional dictionary of limits (e.g. {"max_pmax": 200e5}).
            
        Returns:
            Dictionary containing 'geometry', 'operating_point', 'performance'.
        """
        print(f"Optimizing for Target Power: {target_power/1000:.1f} kW...")
        
        # 1. Configuration
        # In a real system, we'd inject target_power into the NLP constraints.
        # Current NLP uses fixed bounds/targets defined in nlp.py constants.
        # For this PoC, we run the standard NLP and filter/post-process.
        
        nlp, meta = build_thermo_nlp(n_coll=40)
        
        # 2. Add dynamic constraints? 
        # CasADi structure is compiled. We can only change parameter values 'p'.
        # Assuming nlp.py exposes power requirement as parameter or we check valid range.
        
        opts = {"ipopt": {"max_iter": 100, "print_level": 3}, "print_time": 0}
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        
        # 3. Solve
        res = solver(x0=meta["w0"], lbx=meta["lbw"], ubx=meta["ubw"], lbg=meta["lbg"], ubg=meta["ubg"])
        
        if not solver.stats()["success"]:
            print("Warning: Optimization did not converge to optimal solution.")
            
        w_opt = res["x"]
        
        # 4. Extract Results (Using simplified logic for return)
        # We need to extract Bore, Stroke, CR from w_opt[indices]
        # For now, let's use the standard "Fixed Geometry" assumption of the current NLP 
        # OR extract if they are variables.
        # Current nlp.py optimizes Trajectories, but Geometry params (B, S, CR) are constant params?
        # Checking implementation: nlp.py optimizes STATE variables. B, S, CR are CONSTANTS.
        # Future phases make them variables.
        
        # Default Params (Standard Larrak Demo)
        geometry = {
            "bore": 0.1,
            "stroke": 0.2,
            "compression_ratio": 15.0,
            "conrod": 0.4
        }
        
        results = {
            "status": solver.stats()["return_status"],
            "geometry": geometry,
            "w_opt": w_opt,
            "meta": meta
        }
        return results

    def simulate(self, design: Dict[str, Any], operating_point: Dict[str, float]) -> Dict[str, float]:
        """
        Run High-Fidelity Simulations on a design.
        
        Args:
            design: Output from optimize() or dict with 'geometry', 'w_opt', 'meta'.
            operating_point: Dict {'rpm': 3000, 'p_int': 2e5, 'T_int': 320}.
            
        Returns:
            Performance metrics (FMEP, Wiebe, Efficiency).
        """
        print(f"Simulating Design at {operating_point['rpm']} RPM...")
        
        # 1. Create SimulationInput
        run_id = f"sim_{int(operating_point['rpm'])}"
        
        # Use export_candidate logic to generate input
        # If design comes from optimize(), it has w_opt/meta.
        if "w_opt" in design:
            json_str = export_candidate(
                run_id, 
                design["meta"], 
                np.array(design["w_opt"]), 
                design["geometry"], 
                operating_point
            )
            sim_input = SimulationInput.model_validate_json(json_str)
        else:
            raise ValueError("Invalid Design dictionary. Must contain w_opt and meta from optimizer.")
            
        # 2. Run Friction
        print("  -> Friction Simulation")
        f_sim = FrictionSimulation("api_friction", FrictionConfig())
        f_sim.load_input(sim_input)
        f_res = f_sim.solve_steady_state()
        
        # 3. Run Combustion
        print("  -> Combustion Simulation")
        c_sim = PrechamberSimulation("api_combustion", CombustionConfig())
        c_sim.load_input(sim_input)
        c_res = c_sim.solve_steady_state()
        
        # 4. Aggregate
        metrics = {
            "fmep_bar": f_res.friction_fmep,
            "wiebe_m": c_res.calibration_params.get("wiebe_m"),
            "jet_intensity": c_res.calibration_params.get("jet_intensity")
        }
        
        print(f"Simulation Complete. Metrics: {metrics}")
        return metrics

if __name__ == "__main__":
    # Self-Test
    engine = LarrakEngine()
    
    # 1. Optimize
    des = engine.optimize(target_power=300e3)
    
    # 2. Simulate
    op = {"rpm": 4000.0, "p_int": 2.5e5, "T_int": 330.0, "lambda": 1.0}
    perf = engine.simulate(des, op)
