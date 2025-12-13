import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.goldens.phase1.generate_doe import phase1_test


def test_efficiency():
    print("Running Efficiency Verification Test...")
    params = {"rpm": 1000.0, "p_int_bar": 1.0, "fuel_mass_mg": 20.0}
    result = phase1_test(params)

    print("\n--- Results ---")
    print(f"Status: {result.get('status')}")
    print(f"Objective (Cost): {result.get('objective'):.2f}")
    print(f"Abs Work (J): {result.get('abs_work_j'):.2f}")
    print(f"Q_total (J): {result.get('q_total_j'):.2f}")
    print(f"Thermal Efficiency: {result.get('thermal_efficiency') * 100:.2f}%")
    print(f"Peak Pressure: {result.get('p_max_bar'):.2f} bar")

    # Calculate Compression Work Estimate (Adiabatic)
    import numpy as np

    # Geometry (Hardcoded from nlp.py defaults for now, verifying consistency)
    B = 0.1
    S = 0.1  # Per piston
    CR = 15.0
    V_c = (np.pi * B**2 / 4) * S / (CR - 1)
    V_disp = (np.pi * B**2 / 4) * S
    # OP Symmetric Volume
    V_bdc = 2.0 * (V_c + V_disp)
    V_tdc = 2.0 * V_c

    gamma = 1.35  # From nlp.py
    p_int = params["p_int_bar"] * 1e5

    # W_comp = (P1*V1 / (gamma-1)) * (CR^(gamma-1) - 1)
    # Note: Work is Done ON gas, so we subtract this positive quantity from Expansion Work
    w_comp = (p_int * V_bdc / (gamma - 1)) * (CR ** (gamma - 1) - 1)

    w_gross = result.get("abs_work_j")
    w_net = w_gross - w_comp

    eff_net = w_net / result.get("q_total_j")

    print(f"Gross Work (Expansion): {w_gross:.2f} J")
    print(f"Est. Compression Work: {w_comp:.2f} J")
    print(f"Net Work: {w_net:.2f} J")
    print(f"Net Thermal Efficiency: {eff_net * 100:.2f}%")

    if 0.35 <= eff_net <= 0.60:
        print("\nSUCCESS: Net Efficiency is within realistic range (35-60%).")
    else:
        print(f"\nFAILURE: Net Efficiency {eff_net * 100:.2f}% checks failed.")


if __name__ == "__main__":
    test_efficiency()
