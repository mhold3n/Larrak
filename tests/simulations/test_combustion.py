"""
Tests for Phase 4 Combustion Solver.
"""
import sys
import os
import pytest
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Simulations.combustion.prechamber_reduced import PrechamberSimulation, CombustionConfig
from Simulations.common.io_schema import SimulationInput, GeometryConfig, OperatingPoint, BoundaryConditions
from Simulations.common.logger import SimulationLogger
from Simulations.combustion.fit_wiebe_map import fit_wiebe_map

def test_combustion_pipeline():
    """Verify Prechamber Solver and Map Fitting."""
    
    cfg = CombustionConfig()
    sim = PrechamberSimulation("test_comb", cfg)
    
    geo = GeometryConfig(bore=0.1, stroke=0.1, conrod=0.2, compression_ratio=15)
    
    # Run 3 cases to get points for line fitting
    configs = [
        (2000, 1.0e-6), # Low RPM, Small Vol
        (4000, 1.0e-6), # Med RPM
        (6000, 2.0e-6), # High RPM, Large Vol
    ]
    
    runs_dir = []
    
    for rpm, v_pre in configs:
        ops = OperatingPoint(rpm=rpm, p_intake=1e5, T_intake=300, T_oil=373.0)
        
        # Override config
        cfg.V_pre = v_pre
        sim = PrechamberSimulation("test_comb", cfg)
        
        # Dummy BCs
        bcs = BoundaryConditions(
            crank_angle=[0.0, 180.0],
            pressure_gas=[1e5, 1e5],
            temperature_gas=[300.0, 300.0],
            piston_speed=[0.0, 0.0]
        )
        
        inp = SimulationInput(run_id=f"comb_{rpm}_{v_pre}", geometry=geo, operating_point=ops, boundary_conditions=bcs)
        sim.load_input(inp)
        out = sim.solve_steady_state()
        
        # Log
        logger = SimulationLogger()
        path = logger.start_run(inp)
        logger.log_output(out, meta={"test": "comb_physics"})
        runs_dir.append(path)
        
        print(f"Run {rpm} RPM: Jet={out.calibration_params['jet_intensity']:.4f}, m={out.calibration_params['wiebe_m']:.4f}")
        assert out.calibration_params['wiebe_m'] > 0.0
        
    # Test Fitting
    # We call the function directly on the paths we just created
    import Simulations.combustion.fit_wiebe_map as fitter
    
    # Patch Path for DLLs again? 
    # Not strictly needed if test runner environment is correct, 
    # BUT fit_wiebe_map imports surrogates which imports numpy.
    # If run in same process, should be fine.
    
    fitter.fit_wiebe_map(runs_dir, output_file="test_combustion_map.json")
    
    # Check artifact
    artifact_path = fitter.CALIBRATION_DIR / "test_combustion_map.json"
    assert artifact_path.exists()
    
    with open(artifact_path, "r") as f:
        import json
        data = json.load(f)
        assert data["r2"] > 0.9, "Fit should be good for synthetic correlation"

if __name__ == "__main__":
    test_combustion_pipeline()
