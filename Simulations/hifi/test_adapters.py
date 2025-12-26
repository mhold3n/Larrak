#!/usr/bin/env python
"""Test script for high-fidelity solver adapters.

Demonstrates the full workflow:
1. Create SimulationInput with geometry and BCs
2. Load into adapter
3. Execute solver (mesh → run → parse)
4. Get SimulationOutput

Usage:
    cd /path/to/Larrak
    PYTHONPATH=. python Simulations/hifi/test_adapters.py
"""

import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Simulations.hifi import (
    CombustionCFDAdapter,
    ConjugateHTAdapter,
    GearContactFEAAdapter,
    GmshMesher,
    PortFlowCFDAdapter,
    StructuralFEAAdapter,
)
from Simulations.hifi.example_inputs import (
    create_simulation_input,
    part_load_cruise,
    wot_high_speed,
)


def test_mesh_generation():
    """Test Gmsh mesh creation."""
    print("\n" + "=" * 60)
    print("TEST: Mesh Generation")
    print("=" * 60)

    mesher = GmshMesher()

    # 2D thermal mesh
    result = mesher.create_cylinder_2d(
        bore=0.085, height=0.015, output_path="/tmp/test_thermal.msh"
    )
    print(f"2D Mesh: {result['n_nodes']} nodes, {result['n_elements']} elements")

    # 3D piston mesh
    result = mesher.create_piston_3d(
        bore=0.085, crown_thickness=0.015, skirt_length=0.050, output_path="/tmp/test_piston.msh"
    )
    print(f"3D Mesh: {result['n_nodes']} nodes, {result['n_elements']} elements")

    print("✓ Mesh generation passed")
    return True


def test_structural_adapter():
    """Test StructuralFEAAdapter with CalculiX."""
    print("\n" + "=" * 60)
    print("TEST: Structural FEA Adapter (CalculiX)")
    print("=" * 60)

    adapter = StructuralFEAAdapter()
    sim_input = wot_high_speed()  # High thermal/mechanical load

    print(f"Input: {sim_input.run_id}")
    print(f"  Geometry: {sim_input.geometry.bore * 1000:.1f}mm bore")
    print(f"  Operating: {sim_input.operating_point.rpm:.0f} RPM")
    print(f"  P_max: {max(sim_input.boundary_conditions.pressure_gas) / 1e5:.1f} bar")

    adapter.load_input(sim_input)

    print("\nExecuting adapter...")
    try:
        output = adapter.solve_steady_state()
        print(f"✓ Success: {output.success}")
        print(f"  Results: {adapter.results}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    return True


def test_combustion_adapter():
    """Test CombustionCFDAdapter with OpenFOAM."""
    print("\n" + "=" * 60)
    print("TEST: Combustion CFD Adapter (OpenFOAM)")
    print("=" * 60)

    adapter = CombustionCFDAdapter()
    sim_input = create_simulation_input(run_id="combustion_test", rpm=3000, load_fraction=1.0)

    print(f"Input: {sim_input.run_id}")
    adapter.load_input(sim_input)

    print("\nExecuting adapter...")
    try:
        output = adapter.solve_steady_state()
        print(f"✓ Success: {output.success}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    return True


def test_all_adapters_instantiate():
    """Verify all adapters can be instantiated."""
    print("\n" + "=" * 60)
    print("TEST: Adapter Instantiation")
    print("=" * 60)

    adapters = [
        ("StructuralFEA", StructuralFEAAdapter()),
        ("CombustionCFD", CombustionCFDAdapter()),
        ("ConjugateHT", ConjugateHTAdapter()),
        ("PortFlowCFD", PortFlowCFDAdapter()),
        ("GearContactFEA", GearContactFEAAdapter()),
    ]

    for name, adapter in adapters:
        print(f"✓ {name}: {adapter.name} ({adapter.config.backend})")

    print("\n✓ All adapters instantiated successfully")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("HIGH-FIDELITY ADAPTER TEST SUITE")
    print("=" * 60)

    tests = [
        ("Instantiation", test_all_adapters_instantiate),
        ("Mesh Generation", test_mesh_generation),
        # ("Structural FEA", test_structural_adapter),  # Requires Docker
        # ("Combustion CFD", test_combustion_adapter),  # Requires OpenFOAM case setup
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(p for _, p in results)
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
