"""Quick test of CEM client."""
import sys
sys.path.insert(0, '.')

import numpy as np
from truthmaker.cem import CEMClient, ViolationCode

print("Testing CEM Client (Mock Mode)")
print("=" * 50)

# Create client
cem = CEMClient(mock=True)
print(f"CEM Version: {cem.cem_version}")
print(f"Config Hash: {cem.config_hash}")

# Test motion validation
theta = np.linspace(0, 2 * np.pi, 360)
x_profile = 50 + 40 * np.sin(theta)  # 80mm stroke

with cem:
    report = cem.validate_motion(x_profile, theta)
    print(f"\nMotion Validation:")
    print(f"  Valid: {report.is_valid}")
    print(f"  Violations: {len(report.violations)}")
    
    for v in report.violations:
        print(f"  [{v.code.name}] {v.message}")
        if v.margin is not None:
            print(f"    Margin: {v.margin:.2f}")
        print(f"    Suggested: {v.suggested_action.name}")

    # Test initial guess
    guess = cem.get_gear_initial_guess(x_profile, theta)
    print(f"\nGear Initial Guess:")
    print(f"  Mean Rp: {np.mean(guess.Rp):.1f} mm")
    print(f"  Mean C: {np.mean(guess.C):.1f} mm")
    print(f"  Phase Offset: {np.degrees(guess.phase_offset):.1f} deg")

    # Test thermo envelope
    envelope = cem.get_thermo_envelope(bore=0.1, stroke=0.2, cr=15.0, rpm=3000)
    print(f"\nThermo Envelope:")
    print(f"  Boost: {envelope.boost_range[0]:.1f} - {envelope.boost_range[1]:.1f} bar")
    print(f"  Fuel: {envelope.fuel_range[0]:.1f} - {envelope.fuel_range[1]:.1f} mg")

print("\n" + "=" * 50)
print("CEM Client test PASSED")
