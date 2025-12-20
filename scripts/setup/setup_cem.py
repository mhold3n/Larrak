"""
Setup script for CEM development environment.

Usage:
    python scripts/setup_cem.py check    # Check prerequisites
    python scripts/setup_cem.py build    # Build CEM
    python scripts/setup_cem.py test     # Test client
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthmaker.cem import cem_runtime, is_cem_available, CEMClient
import numpy as np


def check_prerequisites():
    """Check all CEM prerequisites."""
    print("=" * 60)
    print("CEM Prerequisites Check")
    print("=" * 60)
    
    # Platform
    try:
        from campro.environment.context import ctx
        print(f"Platform: {ctx.platform}")
    except ImportError:
        import sys
        platform = "windows" if sys.platform == "win32" else sys.platform
        print(f"Platform: {platform}")
    
    # .NET SDK
    dotnet_version = cem_runtime.dotnet_version
    if dotnet_version:
        print(f".NET SDK: {dotnet_version} ✓")
    else:
        print(".NET SDK: NOT FOUND ✗")
        print("  Install from: https://dotnet.microsoft.com/download/dotnet/8.0")
    
    # CEM Project
    if cem_runtime.cem_project_dir.exists():
        print(f"CEM Project: {cem_runtime.cem_project_dir} ✓")
    else:
        print(f"CEM Project: {cem_runtime.cem_project_dir} ✗")
    
    # CEM Build
    if cem_runtime.is_built:
        print(f"CEM Built: {cem_runtime.executable_path} ✓")
    else:
        print("CEM Built: NO")
    
    # Overall status
    print()
    if is_cem_available():
        print("STATUS: CEM is available")
        if not cem_runtime.is_built:
            print("  Run 'python scripts/setup_cem.py build' to build")
    else:
        print("STATUS: CEM is NOT available (install .NET SDK)")
    
    return is_cem_available()


def build_cem():
    """Build the CEM project."""
    print("=" * 60)
    print("Building CEM")
    print("=" * 60)
    
    if not cem_runtime.is_dotnet_available:
        print("ERROR: .NET SDK not found")
        print("Install from: https://dotnet.microsoft.com/download/dotnet/8.0")
        return False
    
    return cem_runtime.build(force=True)


def test_client():
    """Test the CEM client."""
    print("=" * 60)
    print("Testing CEM Client (Mock Mode)")
    print("=" * 60)
    
    with CEMClient(mock=True) as cem:
        # Test motion validation
        theta = np.linspace(0, 2 * np.pi, 360)
        x_profile = 50 + 40 * np.sin(theta)
        
        report = cem.validate_motion(x_profile, theta)
        
        print(f"Validation: {'PASS' if report.is_valid else 'FAIL'}")
        print(f"CEM Version: {report.cem_version}")
        print(f"Violations: {len(report.violations)}")
        
        for v in report.violations:
            print(f"  [{v.code.name}] {v.message}")
        
        # Test initial guess
        guess = cem.get_gear_initial_guess(x_profile, theta)
        print(f"\nInitial Guess:")
        print(f"  Mean Rp: {np.mean(guess.Rp):.1f} mm")
        print(f"  Mean C: {np.mean(guess.C):.1f} mm")
        
        # Test envelope
        envelope = cem.get_thermo_envelope(bore=0.1, stroke=0.2, cr=15.0, rpm=3000)
        print(f"\nThermo Envelope:")
        print(f"  Boost: {envelope.boost_range[0]:.1f} - {envelope.boost_range[1]:.1f} bar")
        print(f"  Fuel: {envelope.fuel_range[0]:.1f} - {envelope.fuel_range[1]:.1f} mg")
    
    print("\nTest PASSED ✓")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    cmd = sys.argv[1]
    
    if cmd == "check":
        check_prerequisites()
    elif cmd == "build":
        if check_prerequisites():
            build_cem()
    elif cmd == "test":
        test_client()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
