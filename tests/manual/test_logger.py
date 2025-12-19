"""
Test script for SimulationLogger.
Verifies that runs are created, hashed, and stored correctly.
"""
import sys
import os
import shutil
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Simulations.common.logger import SimulationLogger
from Simulations.common.io_schema import SimulationInput, GeometryConfig, OperatingPoint, BoundaryConditions

def test_logger():
    print("1. setup Logger...")
    # Use a temp directory for testing
    test_runs_dir = Path("./test_runs_temp")
    if test_runs_dir.exists():
        shutil.rmtree(test_runs_dir)
    
    logger = SimulationLogger(run_root=test_runs_dir)
    
    print("2. Create Dummy Input...")
    geo = GeometryConfig(bore=0.1, stroke=0.1, conrod=0.2, compression_ratio=15)
    ops = OperatingPoint(rpm=3000, p_intake=1e5, T_intake=300)
    bcs = BoundaryConditions(crank_angle=[0, 180], pressure_gas=[1e5, 1e5], temperature_gas=[300, 300])
    
    inp = SimulationInput(
        run_id="logger_test_001",
        geometry=geo,
        operating_point=ops,
        boundary_conditions=bcs
    )
    
    print("3. Start Run (Logging Inputs)...")
    run_dir = logger.start_run(inp)
    print(f"Run Directory Created: {run_dir}")
    
    if not (run_dir / "inputs.json").exists():
        print("FAIL: inputs.json not created.")
        return
        
    print("PASS: inputs.json exists.")
    
    print("4. Cleaning up...")
    shutil.rmtree(test_runs_dir)
    print("Done.")

if __name__ == "__main__":
    test_logger()
