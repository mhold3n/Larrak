"""
Tests for Phase 4 Thermal Solver.
"""
import sys
import os
import pytest
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Simulations.thermal.steady_transient import ThermalSimulation, ThermalConfig
from Simulations.common.io_schema import SimulationInput, GeometryConfig, OperatingPoint, BoundaryConditions

def test_1d_thermal_sanity():
    """Verify 1D solver returns physically bounded temperatures."""
    
    # Setup
    cfg = ThermalConfig(method="1D_Resistance")
    sim = ThermalSimulation("test_thermal", cfg)
    
    # Input: Hot Gas (1500K), Cold Oil (400K)
    # Piston should be somewhat in between
    geo = GeometryConfig(bore=0.1, stroke=0.1, conrod=0.2, compression_ratio=15)
    ops = OperatingPoint(rpm=3000, p_intake=1e5, T_intake=300, T_oil=400.0)
    
    # Mock Trace
    theta = np.linspace(0, 720, 100)
    # Rectangular pulse for heat?
    p_trace = np.ones_like(theta) * 50e5 # 50 bar
    t_trace = np.ones_like(theta) * 1500.0 # 1500 K
    v_trace = np.ones_like(theta) * 10.0
    
    bcs = BoundaryConditions(
        crank_angle=theta.tolist(),
        pressure_gas=p_trace.tolist(),
        temperature_gas=t_trace.tolist(),
        piston_speed=v_trace.tolist()
    )
    
    inp = SimulationInput(run_id="therm_01", geometry=geo, operating_point=ops, boundary_conditions=bcs)
    
    sim.load_input(inp)
    out = sim.solve_steady_state()
    
    print(f"T_crown: {out.T_crown_max}")
    
    assert out.T_crown_max < 1500.0, "Piston hotter than gas!"
    assert out.T_crown_max > 400.0, "Piston colder than oil!"
    assert out.success is True

if __name__ == "__main__":
    test_1d_thermal_sanity()
