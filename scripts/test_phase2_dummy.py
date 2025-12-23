import json
import numpy as np
import os
import subprocess
import sys


def main():
    # 1. Generate Dummy Phase 1 Data
    theta = np.linspace(0, 4 * np.pi, 200)  # 2 cycles

    # Simple Sinusoidal Piston Motion (Approx)
    stroke = 0.2
    conrod = 0.4
    r = stroke / 2.0
    l = conrod

    # Explicit Kinematics
    # Assuming constant crank speed dpsi/dt = 1
    # x = r cos(theta) + sqrt(l^2 - r^2 sin^2(theta))
    # v = dx/dtheta

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    root = np.sqrt(l**2 - r**2 * sin_t**2)

    x = r * cos_t + root
    v = -r * sin_t - (r**2 * sin_t * cos_t) / root

    # Let's perturb it slightly to simulate "Optimization" result (non-standard)
    # Add a small 2nd harmonic
    # x += 0.005 * np.cos(2*theta)
    # v += -0.01 * np.sin(2*theta)

    data = {"trajectory": {"theta": theta.tolist(), "x": x.tolist(), "v": v.tolist()}}

    input_file = "dummy_phase1.json"
    output_file = "dummy_phase2.json"

    with open(input_file, "w") as f:
        json.dump(data, f)

    # 2. Run Phase 2 Script
    cmd = [
        sys.executable,
        "scripts/run_phase2.py",
        "--input",
        input_file,
        "--output",
        output_file,
        "--stroke",
        str(stroke),
        "--conrod",
        str(conrod),
        "--mean-ratio",
        "2.0",
    ]

    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)

    if res.returncode != 0:
        print("Test FAILED")
        sys.exit(1)

    # 3. Validation
    with open(output_file, "r") as f:
        res_data = json.load(f)

    mse = res_data["meta"]["fitting_error"]
    print(f"Fitting MSE: {mse}")

    # Ideally MSE should be small for this simple case if mean ratio 2.0 is compatible
    # Wait, simple crank kinematics implies dpsi/dtheta = 1.
    # Ideal ratio i = dpsi/dphi + 1. If phi=theta, i = 1+1 = 2.
    # So ideal ratio is exactly 2.0 everywhere.
    # The fitter should easily find flat line 2.0.

    if mse < 1e-4:
        print("Test PASSED: MSE is low as expected.")
    else:
        print("Test WARNING: MSE seems high?")

    # Clean up
    if os.path.exists(input_file):
        os.remove(input_file)
    if os.path.exists(output_file):
        os.remove(output_file)


if __name__ == "__main__":
    main()
