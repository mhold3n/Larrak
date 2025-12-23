"""
Generate Phase 3 DOE Golden.
Tests Campro Logic (Breathing Gears) across Size and Load ranges using a unified Design of Experiments.
"""

import os
import sys
from typing import Any

import numpy as np

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

# Suppress CasADi output globally for DOE

from campro.optimization.driver import solve_cycle_orchestrated
from tests.infra.doe_runner import DOERunner


def phase3_test(params: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a single Phase 3 test case.
    Params: r_max, load_scale, profile_type, rpm
    """
    r_max = params["r_max"]
    load_scale = params["load_scale"]
    profile_type = params.get("profile_type", "sine")
    rpm = params.get("rpm", 2000.0)

    # Synthetic Target Ratio Profile
    phi = np.linspace(0, 2 * np.pi, 100)
    if profile_type == "sine":
        target_ratio = 2.0 + 0.5 * np.sin(2 * phi)
    else:
        target_ratio = np.full_like(phi, 2.0)

    # Synthetic Load Profile (Gas Force)
    F_peak = 1000.0 * load_scale
    F_gas = F_peak * (np.exp(-10 * phi**2) + np.exp(-10 * (phi - 2 * np.pi) ** 2))

    # Construct Problem Dict P
    P = {
        "problem_type": "phase3_mech",
        "load_profile": {"angle": phi.tolist(), "F_gas": F_gas.tolist()},
        "target_ratio_profile": target_ratio.tolist(),
        "bounds": {
            "r_min": 0.05 * r_max,
            "r_max": r_max,
        },
        "rb_ring": 0.9 * r_max,
        "num": {"K": 40, "C": 3, "n_knots": 10},
        "weights": {"track": 10.0, "eff": 1.0, "smooth": 0.1},
        "operating_conditions": {"rpm": rpm, "omega": rpm * 2 * np.pi / 60.0},
        "solver": {
            "expand": True,  # Try to enable JIT
            "ipopt": {
                "max_iter": 500,  # Fail fast
                "tol": 1e-3,  # Loose tolerance
                "print_level": 0,
                "linear_solver": "mumps",  # Safe fallback
                "max_cpu_time": 15.0,  # Timeout
            },
        },
    }

    try:
        # Solve using Orchestrator (default budget=500)
        sol = solve_cycle_orchestrated(P, budget=500)

        # Check status
        opt_res = sol.meta.get("optimization", {})
        success = opt_res.get("success", False)
        status = "Optimal" if success else "Failed"
        if not success:
            # Fallback check on IPOPT return status string
            ret_stat = opt_res.get("return_status", "Failed")
            if "Acceptable" in ret_stat:
                status = "Acceptable"

        obj_val = opt_res.get("f_opt", float("nan"))
        iter_count = opt_res.get("iterations", 0)

        # Extract Trajectories for dashboard
        trajectories = {}
        # We need 'a' (acceleration) for peak_acc check if standard dashboard uses it
        # Phase 3 usually has r, v, a
        x_opt = sol.data.get("x")
        if x_opt is not None:
            sol_x = np.array(x_opt).flatten()
            inner_meta = sol.meta.get("meta", {})  # The CollocationBuilder meta
            var_groups = inner_meta.get("variable_groups", {})

            targets = ["r", "v", "a", "u"]
            for name, indices in var_groups.items():
                if name in targets:
                    trajectories[name] = sol_x[indices].tolist()

        output = {
            "status": status,
            "objective": obj_val,
            "iter_count": iter_count,
            "trajectories": trajectories,
            "r_max_actual": r_max,  # Pass thru for easy access if needed
        }

    except Exception as e:
        output = {"status": "Exception", "error": str(e)}

    return output


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "doe_output")

    # Define Axes
    # Replicates the coverage of the 3 matrices but in one factorial
    axes = {
        "r_max": np.linspace(0.1, 0.5, 4).tolist(),
        "load_scale": np.linspace(0.1, 5.0, 4).tolist(),
        "rpm": np.linspace(800.0, 8000.0, 4).tolist(),
        "profile_type": ["sine"],  # Single level
    }
    # 3 * 3 * 4 * 1 = 36 runs.

    # Manual Full Factorial
    import itertools

    import pandas as pd

    keys = list(axes.keys())
    values = list(axes.values())
    doe_list = []

    for combo in itertools.product(*values):
        doe_list.append(dict(zip(keys, combo)))

    runner = DOERunner("phase3_sensitivity", output_dir)
    print(f"Generating Phase 3 Full Factorial Design... ({len(doe_list)} points)")

    # Shuffle for randomization
    runner.design = pd.DataFrame(doe_list).sample(frac=1, random_state=999).reset_index(drop=True)

    # Serial execution for Phase 3 might be safer given CasADi parallel issues with IPOPT instances?
    # Phase 1 used workers=4. Phase 3 solution is heavier. Let's try workers=2.
    runner.run(phase3_test, workers=1)


if __name__ == "__main__":
    main()
