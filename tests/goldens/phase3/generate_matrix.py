"""
Generate Phase 3 Golden Matrix.
Tests Campro Logic (Breathing Gears) across Size and Load ranges.
"""

import sys
import os
import numpy as np
import math

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from tests.infra.matrix_runner import run_matrix
from campro.optimization.driver import solve_cycle


def phase3_test(params):
    """
    Execute a single Phase 3 test case.
    Params: r_max, load_scale, profile_type, rpm
    """
    r_max = params["r_max"]
    load_scale = params["load_scale"]
    profile_type = params.get("profile_type", "sine")
    rpm = params.get("rpm", 2000.0)  # Default to 2000 if not in axis

    # Synthetic Target Ratio Profile
    # Defined over 100 points
    phi = np.linspace(0, 2 * np.pi, 100)
    if profile_type == "sine":
        # Mean 2.0, Amplitude 0.5
        target_ratio = 2.0 + 0.5 * np.sin(2 * phi)
    else:
        # Flat 2.0
        target_ratio = np.full_like(phi, 2.0)

    # Synthetic Load Profile (Gas Force)
    # Peak at TDC (phi=0, 2pi), zero elsewhere ish
    # Gaussian-like pulse
    F_peak = 1000.0 * load_scale
    F_gas = F_peak * (np.exp(-10 * phi**2) + np.exp(-10 * (phi - 2 * np.pi) ** 2))

    # Construct Problem Dict P
    P = {
        "problem_type": "phase3_mech",  # Custom tag
        "load_profile": {"angle": phi.tolist(), "F_gas": F_gas.tolist()},
        "target_ratio_profile": target_ratio.tolist(),
        "bounds": {
            "r_min": 0.05 * r_max,  # 5% of max
            "r_max": r_max,
        },
        "rb_ring": 0.9 * r_max,  # Base circle slightly smaller than max
        "num": {"K": 40, "C": 3, "n_knots": 10},
        "weights": {"track": 10.0, "eff": 1.0, "smooth": 0.1},
        "operating_conditions": {"rpm": rpm, "omega": rpm * 2 * np.pi / 60.0},
        "solver": {
            "ipopt": {
                "max_iter": 3000,
                "tol": 1e-4,
                "print_level": 0,
                "linear_solver": "ma57",
            }
        },
    }

    try:
        # Solve
        sol = solve_cycle(P)

        # Extract results from Solution object
        # Access attributes directly assuming Solution class structure
        sol_data = sol.data
        sol_meta = sol.meta

        # Check status from optimization result in meta
        opt_res = sol_meta.get("optimization", {})
        success = opt_res.get("success", False)
        status = "Optimal" if success else "Failed"

        # Extract Objective
        obj_val = opt_res.get("f_opt", float("nan"))
        iter_count = opt_res.get("iterations", 0)

        # Extract Trajectories
        trajectories = {}
        # Get solution vector
        x_opt = sol_data.get("x")
        if x_opt is None:
            x_opt = opt_res.get("x_opt")  # Fallback

        if x_opt is not None:
            sol_x = np.array(x_opt).flatten()

            # Get variable groups from meta (from CollocationBuilder)
            # Metadata might be nested in sol_meta['meta'] based on driver.py line 1365
            # "meta": {"grid": grid, "meta": meta, ...}
            # So real meta is sol_meta['meta']
            inner_meta = sol_meta.get("meta", {})
            var_groups = inner_meta.get("variable_groups", {})

            # Save relevant trajectories
            # Phase 3 mech usually has: r, v, a, u (control)
            targets = ["r", "v", "a", "u", "T_gas", "T_fric"]

            for name, indices in var_groups.items():
                trajectories[name] = sol_x[indices].tolist()

        output = {
            "status": status,
            "objective": obj_val,
            "iter_count": iter_count,
            "params": params,
            "trajectories": trajectories,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        output = {"status": "Exception", "error": str(e)}

    return output


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "matrix_output")

    # Matrix 1: Size x Load
    axes1 = {
        "r_max": [0.1, 0.3, 0.5],  # 10cm, 30cm, 50cm
        "load_scale": [0.1, 1.0, 5.0],  # Low, Nominal, High
        "profile_type": ["sine"],
        "rpm": [2000.0],
    }
    run_matrix("size_x_load", axes1, phase3_test, output_dir, workers=4)

    # Matrix 2: Size x RPM
    axes2 = {
        "r_max": [0.1, 0.3, 0.5],
        "rpm": [800.0, 2000.0, 6000.0, 8000.0],
        "load_scale": [1.0],
        "profile_type": ["sine"],
    }
    run_matrix("size_x_rpm", axes2, phase3_test, output_dir, workers=4)

    # Matrix 3: Load x RPM
    axes3 = {
        "load_scale": [0.1, 1.0, 5.0],
        "rpm": [800.0, 2000.0, 6000.0, 8000.0],
        "r_max": [0.3],
        "profile_type": ["sine"],
    }
    run_matrix("load_x_rpm", axes3, phase3_test, output_dir, workers=4)


if __name__ == "__main__":
    main()
