#!/usr/bin/env python3
"""
Verification script for CasADi with the MA27 HSL solver.
Run this script to verify that CasADi can access the MA27 linear solver.
"""

import os
import sys


def test_casadi_import():
    """Test if CasADi can be imported."""
    try:
        import casadi as ca

        print(f"[OK] CasADi version: {ca.__version__}")
        print(f"[OK] CasADi path: {ca.__file__}")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import CasADi: {e}")
        return False


def test_hsl_solvers():
    """Test if the MA27 HSL solver is available."""
    try:
        import time

        import casadi as ca

        x = ca.MX.sym("x")
        nlp = {"x": x, "f": x**2, "g": x}

        hsl_solvers = ["ma27"]
        available_solvers = []
        solver_times = {}

        print("Testing HSL solver (MA27 only):")
        for solver_name in hsl_solvers:
            try:
                start_time = time.time()
                solver = ca.nlpsol(
                    f"solver_{solver_name}",
                    "ipopt",
                    nlp,
                    {
                        "ipopt.linear_solver": solver_name,
                        "ipopt.print_level": 0,
                        "ipopt.sb": "yes",
                    },
                )
                end_time = time.time()

                print(
                    f"[OK] {solver_name.upper()}: Available (creation time: {end_time - start_time:.3f}s)",
                )
                available_solvers.append(solver_name)
                solver_times[solver_name] = end_time - start_time

            except Exception as e:
                print(f"[FAIL] {solver_name.upper()}: Failed - {e}")

        print(
            f"\nSummary: {len(available_solvers)}/{len(hsl_solvers)} HSL solvers available",
        )
        return len(available_solvers) == len(hsl_solvers)

    except Exception as e:
        print(f"[FAIL] HSL solver test failed: {e}")
        return False


def test_hsl_libraries():
    """Test if HSL libraries are accessible."""
    hsl_path = r"C:\Users\maxed\OneDrive\Desktop\Github Projects\Larrak\CoinHSL-archive.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin"

    required_dlls = [
        "libcoinhsl.dll",
        "libhsl.dll",
        "libgfortran-5.dll",
        "libopenblas.dll",
    ]

    all_found = True
    for dll in required_dlls:
        dll_path = os.path.join(hsl_path, dll)
        if os.path.exists(dll_path):
            print(f"[OK] {dll}: Found")
        else:
            print(f"[FAIL] {dll}: Not found at {dll_path}")
            all_found = False

    return all_found


def main():
    """Main verification function."""
    print("CasADi HSL Installation Verification")
    print("=" * 50)

    # Test 1: CasADi import
    print("\n1. Testing CasADi import...")
    casadi_ok = test_casadi_import()

    # Test 2: HSL libraries
    print("\n2. Testing HSL libraries...")
    hsl_libs_ok = test_hsl_libraries()

    # Test 3: HSL solvers
    print("\n3. Testing HSL solvers...")
    hsl_solvers_ok = test_hsl_solvers()

    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY:")
    print(f"CasADi import: {'[OK] PASS' if casadi_ok else '[FAIL] FAIL'}")
    print(f"HSL libraries: {'[OK] PASS' if hsl_libs_ok else '[FAIL] FAIL'}")
    print(f"HSL solvers:   {'[OK] PASS' if hsl_solvers_ok else '[FAIL] FAIL'}")

    if casadi_ok and hsl_libs_ok and hsl_solvers_ok:
        print(
            "\n[SUCCESS] All tests passed! CasADi with HSL solvers is properly configured.",
        )
        return 0
    print("\n[ERROR] Some tests failed. Please check the installation.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
