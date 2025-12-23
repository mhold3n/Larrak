"""
Generate Phase 2 Golden Matrix.
Tests Interpreter Module across geometric and ratio ranges.
"""

import sys
import os
import numpy as np
import json

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from tests.infra.matrix_runner import run_matrix
from interpreter.interface import Interpreter


def phase2_test(params):
    """
    Execute a single Phase 2 test case.
    Params: degree, mean_ratio, stroke, conrod
    """
    interp = Interpreter()

    # 1. Generate Input Trajectory (Synthetic Phase 1)
    # Based on the requested geometry but perturbed to require fitting
    stroke = params["stroke"]
    conrod = params["conrod"]

    # Create simple slider-crank motion
    # But we want to simulate an input that implies the TARGET ratio
    # If we feed x(theta) from perfect slider crank, ideal ratio is 2.0 (if dphi/dtheta=1).
    # If we want to test mean_ratio != 2.0, we must scale theta?
    # dpsi/dphi = i - 1.
    # If i=3, dpsi/dphi = 2. psi = 2*phi.
    # So if we feed x(psi) where psi = 2*theta_cycle_angle, we imply ratio 3?

    cycle_angle = np.linspace(0, 4 * np.pi, 200)  # 2 cycles

    target_mean = params["mean_ratio"]
    # implied psi = (target_mean - 1) * cycle_angle
    # Note: cycle_angle here acts as Ring Angle * const?
    # No, usually Phase 1 outputs x vs theta, where theta is "Cycle Angle" (1 rev = 360).
    # Does Phase 1 output cycle angle 0..360 deg?
    # Let's assume input 'theta' is strictly mapped to ring angle 'phi' linearly.
    # If we want Ratio R, we need dpsi/dphi = R-1.
    # psi = (R-1) * phi.

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

    output = interp.process(motion_data, geom, options=opts)

    return {
        "fit_error": output["meta"]["fitting_error"],
        "mean_ratio_result": output["meta"]["mean_ratio"],
        "status": output["status"],
    }


def main():
    # Define Matrix Axes
    matrix_name = "ratio_x_geometry"
    output_dir = os.path.join(os.path.dirname(__file__), "matrix_output")

    axes = {
        "mean_ratio": [1.5, 2.0, 2.5, 3.0],
        "stroke": [0.1, 0.2, 0.3],  # m
        "conrod": [0.3, 0.4, 0.5],  # m
    }

    # Run
    run_matrix(matrix_name, axes, phase2_test, output_dir, workers=4)


if __name__ == "__main__":
    main()
