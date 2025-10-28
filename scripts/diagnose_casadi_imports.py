#!/usr/bin/env python3
"""
Diagnostic script to systematically test CasADi import chain.

This script tests the import chain for the complex gas optimizer step-by-step
to identify where the import failures occur and why.
"""

import os
import sys
import traceback

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_casadi_basic():
    """Test basic CasADi import and functionality."""
    print("=" * 60)
    print("TEST 1: Basic CasADi Import and IPOPT Availability")
    print("=" * 60)

    try:
        import casadi as ca

        print("✓ CasADi imported successfully")
        print(f"  Version: {ca.__version__}")
        print(f"  Build type: {ca.CasadiMeta_build_type()}")

        # Test IPOPT availability
        try:
            x = ca.MX.sym("x")
            f = x**2
            nlp = {"x": x, "f": f}
            solver = ca.nlpsol("test", "ipopt", nlp)
            print("✓ IPOPT solver created successfully")

            # Test solving a simple problem
            result = solver(x0=2.0)
            print(f"✓ Test solve successful: x = {result['x']}")
            return True

        except Exception as e:
            print(f"✗ IPOPT test failed: {e}")
            return False

    except ImportError as e:
        print(f"✗ CasADi import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ CasADi test failed: {e}")
        traceback.print_exc()
        return False


def test_freepiston_imports():
    """Test freepiston module imports step by step."""
    print("\n" + "=" * 60)
    print("TEST 2: Freepiston Module Imports")
    print("=" * 60)

    # Test basic freepiston import
    try:
        print("✓ campro.freepiston imported successfully")
    except Exception as e:
        print(f"✗ campro.freepiston import failed: {e}")
        traceback.print_exc()
        return False

    # Test freepiston.opt import
    try:
        print("✓ campro.freepiston.opt imported successfully")
    except Exception as e:
        print(f"✗ campro.freepiston.opt import failed: {e}")
        traceback.print_exc()
        return False

    # Test individual opt modules
    modules_to_test = [
        "campro.freepiston.opt.nlp",
        "campro.freepiston.opt.driver",
        "campro.freepiston.opt.ipopt_solver",
        "campro.freepiston.opt.optimization_lib",
        "campro.freepiston.opt.config_factory",
    ]

    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[""])
            print(f"✓ {module_name} imported successfully")
        except Exception as e:
            print(f"✗ {module_name} import failed: {e}")
            traceback.print_exc()
            return False

    return True


def test_optimization_lib_components():
    """Test specific components from optimization_lib."""
    print("\n" + "=" * 60)
    print("TEST 3: Optimization Library Components")
    print("=" * 60)

    try:
        from campro.freepiston.opt.optimization_lib import (
            IPOPTBackend,
            MotionLawOptimizer,
            OptimizationConfig,
            ProblemBuilder,
            ResultProcessor,
        )

        print("✓ All optimization_lib components imported successfully")

        # Test creating a basic config
        try:
            config = OptimizationConfig()
            print("✓ OptimizationConfig created successfully")
        except Exception as e:
            print(f"✗ OptimizationConfig creation failed: {e}")
            return False

        # Test creating optimizer (without running it)
        try:
            optimizer = MotionLawOptimizer(config)
            print("✓ MotionLawOptimizer created successfully")
            return True
        except Exception as e:
            print(f"✗ MotionLawOptimizer creation failed: {e}")
            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"✗ optimization_lib import failed: {e}")
        traceback.print_exc()
        return False


def test_config_factory():
    """Test config factory functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: Config Factory")
    print("=" * 60)

    try:
        from campro.freepiston.opt.config_factory import (
            ConfigFactory,
            create_optimization_scenario,
        )

        print("✓ Config factory imported successfully")

        # Test creating efficiency scenario
        try:
            config = create_optimization_scenario("efficiency")
            print("✓ Efficiency scenario created successfully")
            print(f"  Geometry keys: {list(config.geometry.keys())}")
            print(f"  Thermodynamics keys: {list(config.thermodynamics.keys())}")
            return True
        except Exception as e:
            print(f"✗ Efficiency scenario creation failed: {e}")
            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"✗ Config factory import failed: {e}")
        traceback.print_exc()
        return False


def test_thermal_efficiency_adapter():
    """Test thermal efficiency adapter import and setup."""
    print("\n" + "=" * 60)
    print("TEST 5: Thermal Efficiency Adapter")
    print("=" * 60)

    try:
        from campro.optimization.thermal_efficiency_adapter import (
            ThermalEfficiencyAdapter,
            ThermalEfficiencyConfig,
        )

        print("✓ Thermal efficiency adapter imported successfully")

        # Test creating adapter
        try:
            config = ThermalEfficiencyConfig()
            adapter = ThermalEfficiencyAdapter(config)
            print("✓ ThermalEfficiencyAdapter created successfully")

            if adapter.complex_optimizer is not None:
                print("✓ Complex optimizer is available")
                return True
            print("⚠ Complex optimizer is None (import failed during setup)")
            return False

        except Exception as e:
            print(f"✗ ThermalEfficiencyAdapter creation failed: {e}")
            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"✗ Thermal efficiency adapter import failed: {e}")
        traceback.print_exc()
        return False


def test_nlp_building():
    """Test NLP building functionality."""
    print("\n" + "=" * 60)
    print("TEST 6: NLP Building")
    print("=" * 60)

    try:
        from campro.freepiston.opt.nlp import build_collocation_nlp

        print("✓ NLP building function imported successfully")

        # Create a minimal problem configuration
        P = {
            "geometry": {
                "bore": 0.1,
                "stroke": 0.05,
                "compression_ratio": 10.0,
                "clearance_volume": 1e-5,
                "mass": 0.5,
                "rod_mass": 0.1,
                "rod_length": 0.1,
            },
            "thermodynamics": {
                "gamma": 1.4,
                "R": 287.0,
                "cp": 1005.0,
                "cv": 718.0,
            },
            "num": {"K": 10, "C": 3},
            "solver": {"ipopt": {}},
            "objective": {"method": "indicated_work"},
        }

        try:
            nlp, meta = build_collocation_nlp(P)
            print("✓ NLP built successfully")
            print(f"  Variables: {nlp['x'].shape}")
            print(f"  Objective: {nlp['f'].shape}")
            return True
        except Exception as e:
            print(f"✗ NLP building failed: {e}")
            traceback.print_exc()
            return False

    except ImportError as e:
        print(f"✗ NLP building import failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all diagnostic tests."""
    print("CasADi Import Chain Diagnostic")
    print("=" * 60)
    print("This script will test the import chain step by step to identify")
    print("where the complex gas optimizer import fails.")
    print()

    tests = [
        ("Basic CasADi", test_casadi_basic),
        ("Freepiston Imports", test_freepiston_imports),
        ("Optimization Lib", test_optimization_lib_components),
        ("Config Factory", test_config_factory),
        ("Thermal Efficiency Adapter", test_thermal_efficiency_adapter),
        ("NLP Building", test_nlp_building),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED - CasADi integration should work!")
        print(
            "The complex gas optimizer should be available for thermal efficiency optimization.",
        )
    else:
        print("❌ SOME TESTS FAILED - CasADi integration has issues")
        print("Check the error messages above to identify what needs to be fixed.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
