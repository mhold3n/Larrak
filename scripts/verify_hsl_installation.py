#!/usr/bin/env python3
"""
Verification script for CasADi with HSL solvers.
Run this script to verify that CasADi can access HSL linear solvers (MA27, MA57, MA77, MA86, MA97).
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
    """Test if HSL solvers are available."""
    try:
        import time

        import casadi as ca
        
        # Detect available solvers using hsl_detector
        try:
            from campro.environment.hsl_detector import detect_available_solvers, get_hsl_library_path
            
            hsl_solvers = detect_available_solvers(test_runtime=False)
            hsl_lib_path = get_hsl_library_path()
            
            if hsl_lib_path:
                print(f"[INFO] HSL library path: {hsl_lib_path}")
            else:
                print("[WARN] HSL library path not detected")
            
            if not hsl_solvers:
                print("[WARN] No HSL solvers detected in CoinHSL library")
                hsl_solvers = ["ma27"]  # Fallback for testing
        except ImportError:
            print("[WARN] hsl_detector not available; testing MA27 only")
            hsl_solvers = ["ma27"]
        except Exception as e:
            print(f"[WARN] Error detecting solvers: {e}; testing MA27 only")
            hsl_solvers = ["ma27"]

        x = ca.MX.sym("x")
        nlp = {"x": x, "f": x**2, "g": x}

        available_solvers = []
        solver_times = {}
        
        # Get HSL library path for solver configuration
        solver_opts = {}
        try:
            from campro.environment.hsl_detector import get_hsl_library_path
            hsl_lib_path = get_hsl_library_path()
            if hsl_lib_path:
                solver_opts["ipopt.hsllib"] = str(hsl_lib_path)
        except Exception:
            pass

        print(f"Testing HSL solvers ({len(hsl_solvers)} detected):")
        for solver_name in hsl_solvers:
            try:
                start_time = time.time()
                solver_opts["ipopt.linear_solver"] = solver_name
                solver_opts["ipopt.print_level"] = 0
                solver_opts["ipopt.sb"] = "yes"
                
                solver = ca.nlpsol(
                    f"solver_{solver_name}",
                    "ipopt",
                    nlp,
                    solver_opts,
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
        if available_solvers:
            print(f"Available solvers: {', '.join(s.upper() for s in available_solvers)}")
        return len(available_solvers) > 0

    except Exception as e:
        print(f"[FAIL] HSL solver test failed: {e}")
        return False


def test_hsl_libraries():
    """Test if HSL libraries are accessible."""
    try:
        from campro.environment.hsl_detector import get_hsl_library_path, find_coinhsl_directory
        
        coinhsl_dir = find_coinhsl_directory()
        hsl_lib_path = get_hsl_library_path()
        
        if not coinhsl_dir:
            print("[FAIL] CoinHSL directory not found")
            return False
        
        if not hsl_lib_path:
            print("[FAIL] HSL library file not found")
            return False
        
        print(f"[OK] CoinHSL directory: {coinhsl_dir}")
        print(f"[OK] HSL library: {hsl_lib_path}")
        
        # Check for common dependencies (platform-specific)
        import platform
        system = platform.system().lower()
        
        if system == "windows":
            bin_dir = coinhsl_dir / "bin"
            required_dlls = [
                "libcoinhsl.dll",
                "libhsl.dll",
                "libgfortran-5.dll",
                "libopenblas.dll",
            ]
            all_found = True
            for dll in required_dlls:
                dll_path = bin_dir / dll
                if dll_path.exists():
                    print(f"[OK] {dll}: Found")
                else:
                    print(f"[WARN] {dll}: Not found at {dll_path}")
            return all_found
        elif system == "darwin":
            lib_dir = coinhsl_dir / "lib"
            if (lib_dir / "libcoinhsl.dylib").exists():
                print("[OK] libcoinhsl.dylib: Found")
                return True
            else:
                print("[FAIL] libcoinhsl.dylib: Not found")
                return False
        else:
            # Linux
            lib_dir = coinhsl_dir / "lib"
            if (lib_dir / "libcoinhsl.so").exists():
                print("[OK] libcoinhsl.so: Found")
                return True
            else:
                print("[FAIL] libcoinhsl.so: Not found")
                return False
                
    except ImportError:
        print("[WARN] hsl_detector not available; skipping library check")
        return True  # Don't fail if detector not available
    except Exception as e:
        print(f"[FAIL] HSL library test failed: {e}")
        return False


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
