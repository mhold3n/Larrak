"""
End-to-end test of CEM gRPC client with live service.

Usage:
    # In terminal 1: Start CEM service
    dotnet run --project Larrak.CEM/src/Larrak.CEM.API
    
    # In terminal 2: Run this test
    python scripts/test_cem_grpc.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import subprocess
import numpy as np
from truthmaker.cem import CEMClient, cem_runtime


def test_mock_mode():
    """Test client in mock mode."""
    print("=" * 60)
    print("Testing Mock Mode")
    print("=" * 60)
    
    with CEMClient(mock=True) as cem:
        theta = np.linspace(0, 2 * np.pi, 360)
        x_profile = 50 + 40 * np.sin(theta)
        
        report = cem.validate_motion(x_profile, theta)
        print(f"Mock Validation: {'PASS' if report.is_valid else 'FAIL'}")
        print(f"CEM Version: {report.cem_version}")
    
    return True


def test_grpc_mode():
    """Test client with live gRPC service."""
    print("=" * 60)
    print("Testing gRPC Mode")
    print("=" * 60)
    
    # Check if service is running
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 50051))
    sock.close()
    
    if result != 0:
        print("[!] CEM service not running on port 50051")
        print("    Start with: dotnet run --project Larrak.CEM/src/Larrak.CEM.API")
        return False
    
    # Test connection
    with CEMClient(mock=False) as cem:
        if cem.mock:
            print("[!] Client fell back to mock mode")
            return False
        
        print(f"[✓] Connected to CEM service v{cem.cem_version}")
        
        # Test motion validation
        theta = np.linspace(0, 2 * np.pi, 360)
        x_profile = 50 + 40 * np.sin(theta)
        
        start = time.perf_counter()
        report = cem.validate_motion(x_profile, theta)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"\nMotion Validation (gRPC):")
        print(f"  Valid: {report.is_valid}")
        print(f"  Violations: {len(report.violations)}")
        print(f"  Latency: {elapsed:.1f} ms")
        
        # Test thermo envelope
        start = time.perf_counter()
        envelope = cem.get_thermo_envelope(bore=0.1, stroke=0.2, cr=15.0, rpm=3000)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"\nThermo Envelope (gRPC):")
        print(f"  Boost: {envelope.boost_range[0]:.1f} - {envelope.boost_range[1]:.1f} bar")
        print(f"  Fuel: {envelope.fuel_range[0]:.1f} - {envelope.fuel_range[1]:.1f} mg")
        print(f"  Latency: {elapsed:.1f} ms")
        
        # Test gear initial guess
        start = time.perf_counter()
        guess = cem.get_gear_initial_guess(x_profile, theta)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"\nGear Initial Guess (gRPC):")
        print(f"  Mean Rp: {np.mean(guess.Rp):.1f} mm")
        print(f"  Mean C: {np.mean(guess.C):.1f} mm")
        print(f"  Latency: {elapsed:.1f} ms")
    
    print("\n" + "=" * 60)
    print("gRPC Test PASSED")
    print("=" * 60)
    return True


def benchmark_mock_vs_grpc():
    """Benchmark mock vs gRPC latency."""
    print("=" * 60)
    print("Latency Benchmark: Mock vs gRPC")
    print("=" * 60)
    
    theta = np.linspace(0, 2 * np.pi, 360)
    x_profile = 50 + 40 * np.sin(theta)
    n_iterations = 100
    
    # Mock timing
    with CEMClient(mock=True) as cem:
        start = time.perf_counter()
        for _ in range(n_iterations):
            cem.validate_motion(x_profile, theta)
        mock_elapsed = (time.perf_counter() - start) * 1000 / n_iterations
    
    print(f"Mock:  {mock_elapsed:.3f} ms/call")
    
    # gRPC timing (if available)
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 50051))
    sock.close()
    
    if result == 0:
        with CEMClient(mock=False) as cem:
            if not cem.mock:
                start = time.perf_counter()
                for _ in range(n_iterations):
                    cem.validate_motion(x_profile, theta)
                grpc_elapsed = (time.perf_counter() - start) * 1000 / n_iterations
                
                print(f"gRPC:  {grpc_elapsed:.3f} ms/call")
                print(f"Ratio: {grpc_elapsed / mock_elapsed:.1f}x")
    else:
        print("gRPC:  (service not running)")


if __name__ == "__main__":
    # Always test mock mode
    test_mock_mode()
    print()
    
    # Try gRPC mode
    test_grpc_mode()
    print()
    
    # Benchmark if service is running
    benchmark_mock_vs_grpc()
