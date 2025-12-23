"""
Generate Phase 2 DOE Golden.
Tests Interpreter Module across geometric and ratio ranges using a unified Design of Experiments.
"""

import os
import sys
from typing import Any

import numpy as np

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from interpreter.interface import Interpreter
from tests.infra.doe_runner import DOERunner


def phase2_test(params: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a single Phase 2 test case.
    Params: degree, mean_ratio, stroke, conrod
    """
    interp = Interpreter()

    # 1. Generate Input Trajectory (Synthetic Phase 1)
    # Based on the requested geometry but perturbed to require fitting
    stroke = params["stroke"]
    conrod = params["conrod"]

    cycle_angle = np.linspace(0, 4 * np.pi, 200)  # 2 cycles

    target_mean = params["mean_ratio"]
    # implied psi = (target_mean - 1) * cycle_angle
    psi = (target_mean - 1.0) * cycle_angle

    # Kinematics x(psi)
    r = stroke / 2.0
    l = conrod
    # Simple slider crank x(psi)
    sin_p = np.sin(psi)
    cos_p = np.cos(psi)
    root = np.sqrt(l**2 - r**2 * sin_p**2)
    x = r * cos_p + root

    # Velocity dx/dtheta = (dx/dpsi) * (dpsi/dtheta)
    # dpsi/dtheta = target_mean - 1
    dx_dpsi = -r * sin_p - (r**2 * sin_p * cos_p) / root
    v = dx_dpsi * (target_mean - 1.0)

    motion_data = {"theta": cycle_angle.tolist(), "x": x.tolist(), "v": v.tolist()}

    geom = {"stroke": stroke, "conrod": conrod}

    # Run Interpreter
    # We ask it to fit this data. It should recover Mean Ratio approx `target_mean`.
    opts = {
        "mean_ratio": target_mean,
        "n_knots": 50,
        "degree": 3,
        "weights": {"track": 1.0, "smooth": 0.01},  # Low smoothing to allow tracking
    }

    try:
        output = interp.process(motion_data, geom, options=opts)

        result = {
            "status": output["status"],
            "fit_error": output["meta"]["fitting_error"],
            "mean_ratio_result": output["meta"]["mean_ratio"],
            "objective": output["meta"]["fitting_error"],  # Standardize for dashboard
        }
    except Exception as e:
        result = {"status": "Exception", "error": str(e)}

    return result


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "doe_output")

    # 1. Define Axes
    axes = {
        "mean_ratio": np.linspace(1.5, 3.0, 6).tolist(),
        "stroke": np.linspace(0.1, 0.3, 6).tolist(),  # m
        "conrod": np.linspace(0.3, 0.5, 6).tolist(),  # m
    }

    # 2. Init Runner
    runner = DOERunner("phase2_sensitivity", output_dir)

    # 3. Create Design
    # 4 values * 3 values * 3 values = 36 runs. Quick.
    print("Generating Phase 2 Full Factorial Design...")
    runner.create_full_factorial(axes, replicates=1)
    runner.add_randomization(seed=123)

    # 4. Run
    # Use parallel processing
    runner.run(phase2_test, workers=4)


if __name__ == "__main__":
    main()
