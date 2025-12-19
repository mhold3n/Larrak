"""
Unit tests for IO Schema contract.
Runs with pytest.
"""

import pytest
import json
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Simulations.common.io_schema import SimulationInput, GeometryConfig, OperatingPoint, BoundaryConditions

def test_valid_input_creation():
    """Test creating a valid input object."""
    geo = GeometryConfig(bore=0.1, stroke=0.1, conrod=0.2, compression_ratio=15)
    ops = OperatingPoint(rpm=3000, p_intake=1e5, T_intake=300)
    bcs = BoundaryConditions(
        crank_angle=[0, 180, 360],
        pressure_gas=[1e5, 5e6, 1e5],
        temperature_gas=[300, 1500, 500]
    )
    
    sim_input = SimulationInput(
        run_id="test_run_001",
        geometry=geo,
        operating_point=ops,
        boundary_conditions=bcs
    )
    
    assert sim_input.run_id == "test_run_001"
    assert sim_input.geometry.bore == 0.1

def test_array_length_mismatch():
    """Test that BCs raise error if arrays are different lengths."""
    with pytest.raises(ValueError):
        BoundaryConditions(
            crank_angle=[0, 180], # Length 2
            pressure_gas=[1e5, 5e6, 1e5], # Length 3
            temperature_gas=[300, 1500, 500]
        )

def test_negative_values():
    """Test constraints on physical values."""
    with pytest.raises(ValueError):
        GeometryConfig(bore=-0.1, stroke=0.1, conrod=0.1, compression_ratio=10)
