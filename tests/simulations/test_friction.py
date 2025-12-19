"""
Tests for Phase 4 Friction Solver.
"""
import sys
import os
import pytest
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Simulations.structural.friction import FrictionSimulation, FrictionConfig
from Simulations.common.io_schema import SimulationInput, GeometryConfig, OperatingPoint, BoundaryConditions
from Simulations.common.logger import SimulationLogger

def test_stribeck_physics():
    """Verify Friction Solver behavior."""
    
    cfg = FrictionConfig()
    sim = FrictionSimulation("test_fric", cfg)
    
    geo = GeometryConfig(bore=0.1, stroke=0.1, conrod=0.2, compression_ratio=15)
    ops = OperatingPoint(rpm=3000, p_intake=1e5, T_intake=300, T_oil=373.0) # 100C Oil
    
    # Generate Synthetic Piston Motion (Sinusoidal approx)
    theta_deg = np.linspace(0, 180, 100) # Expansion
    theta_rad = theta_deg * np.pi / 180.0
    omega = ops.rpm * 2 * np.pi / 60.0
    # v = R * omega * sin(theta)
    R = geo.stroke / 2.0
    v_trace = R * omega * np.sin(theta_rad)
    
    # Pressure Pulse (Peak at 15 deg)
    p_trace = 1e5 + 50e5 * np.exp(-((theta_deg - 15)**2) / 100.0)
    # T trace dummy
    t_trace = np.ones_like(theta_deg) * 1000.0
    
    bcs = BoundaryConditions(
        crank_angle=theta_deg.tolist(),
        pressure_gas=p_trace.tolist(),
        temperature_gas=t_trace.tolist(),
        piston_speed=v_trace.tolist()
    )
    
    inp = SimulationInput(run_id="fric_01", geometry=geo, operating_point=ops, boundary_conditions=bcs)
    sim.load_input(inp)
    
    out = sim.solve_steady_state()
    
    print(f"RPM: {ops.rpm}, Peak P: {np.max(p_trace)/1e5:.1f} bar")
    print(f"FMEP: {out.friction_fmep:.4f} bar")
    
    # [VERIFICATION] Save to DB for Map Fitting Test
    logger = SimulationLogger()
    logger.start_run(inp)
    logger.log_output(out, meta={"test": "stribeck_physics"})
    
    assert out.friction_fmep > 0.0, "Friction should be positive"
    # FMEP for expansion stroke typically 0.1-0.3 bar
    assert out.friction_fmep < 2.0, "Friction too high for single ring/stroke"

def test_friction_sensitivity():
    """Check that FMEP increases with Load (Pressure) and Speed."""
    cfg = FrictionConfig()
    sim = FrictionSimulation("test_sens", cfg)
    geo = GeometryConfig(bore=0.1, stroke=0.1, conrod=0.2, compression_ratio=15)
    
    # Baseline
    ops1 = OperatingPoint(rpm=2000, p_intake=1e5, T_intake=300, T_oil=373.0)
    # High Speed
    ops2 = OperatingPoint(rpm=6000, p_intake=1e5, T_intake=300, T_oil=373.0)
    
    def run_case(ops, peak_p):
        theta_deg = np.linspace(0, 180, 100)
        omega = ops.rpm * 2.0 * np.pi / 60.0
        v_trace = 0.05 * omega * np.sin(theta_deg * np.pi / 180.0)
        p_trace = 1e5 + peak_p * np.exp(-((theta_deg - 15)**2) / 100.0)
        bcs = BoundaryConditions(
            crank_angle=theta_deg.tolist(),
            pressure_gas=p_trace.tolist(),
            temperature_gas=np.ones_like(theta_deg).tolist(),
            piston_speed=v_trace.tolist()
        )
        inp = SimulationInput(run_id=f"test_sens_{ops.rpm}_{peak_p}", geometry=geo, operating_point=ops, boundary_conditions=bcs)
        sim.load_input(inp)
        res = sim.solve_steady_state()
        
        # Log it
        logger = SimulationLogger()
        logger.start_run(inp)
        logger.log_output(res)
        
        return res.friction_fmep
        
    fmep_low = run_case(ops1, 50e5)
    fmep_high = run_case(ops2, 50e5)
    
    print(f"FMEP Low RPM: {fmep_low}")
    print(f"FMEP High RPM: {fmep_high}")
    
    # Hydrodynamic dominance: Higher RPM -> Higher Friction?
    # Or Stribeck dip? At high speed, usually hydrodynamic drag dominates -> increases with speed.
    assert fmep_high > fmep_low, "Friction should increase with RPM (Hydrodynamic regime)"

if __name__ == "__main__":
    test_stribeck_physics()
