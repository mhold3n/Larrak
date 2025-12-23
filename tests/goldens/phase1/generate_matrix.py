"""
Generate Phase 1 Golden Matrix.
Tests Thermo Module across RPM, Load, and Boost ranges.
"""

import sys
import os
import numpy as np
import casadi as ca
import json

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from tests.infra.matrix_runner import run_matrix
from thermo.nlp import build_thermo_nlp


def phase1_test(params):
    """
    Execute a single Phase 1 test case.
    Params: rpm, q_total, p_boost_bars
    """
    rpm = params["rpm"]
    q_total = params["q_total"]
    p_boost = params["p_boost_bars"] * 1e5

    # RPM to rad/s
    omega = rpm * (2 * np.pi / 60.0)

    # Build NLP
    # Using 40 coll points for speed in goldens
    nlp_tuple = build_thermo_nlp(
        n_coll=40, Q_total=q_total, p_int=p_boost, omega_val=omega, model_type="prechamber"
    )

    # Check return type to handle dict vs tuple ambiguity
    if isinstance(nlp_tuple, tuple):
        nlp_dict = nlp_tuple[0]
        meta_dict = nlp_tuple[1]
    else:
        nlp_dict = nlp_tuple
        meta_dict = {}  # Should not happen based on CollocationBuilder code

    # Build Solver
    nlp_prob = {"x": nlp_dict["x"], "f": nlp_dict["f"], "g": nlp_dict["g"]}  # CasADi Solver
    # Revert to flat dict for safety
    opts = {
        "print_time": True,
        "ipopt.print_level": 3,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-2,
        "ipopt.acceptable_iter": 15,
        "ipopt.max_iter": 10000,
        "ipopt.linear_solver": "ma57",
        "ipopt.nlp_scaling_method": "gradient-based",
    }

    try:
        solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts)

        # Initial Guess from meta
        x0 = meta_dict.get("w0", [])
        lbg = meta_dict.get("lbg", [])
        ubg = meta_dict.get("ubg", [])
        lbx = meta_dict.get("lbw", [])  # Note: CollocationBuilder calls it lbw, CasADi expects lbx
        ubx = meta_dict.get("ubw", [])

        res = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        status = "Optimal" if solver.stats()["success"] else "Failed"
        obj_val = float(res["f"])

        # Extract Trajectories
        trajectories = {}
        sol_x = np.array(res["x"]).flatten()
        var_groups = meta_dict.get("variable_groups", {})

        # We want x, v, acc
        # Also maybe thermodynamic states for debug
        targets = [
            "x",
            "v",
            "acc",
            "m_c",
            "T_c",
            "P_c",
        ]  # P_c not state, but we can compute or skip

        for name, indices in var_groups.items():
            # Only save relevant variables to keep file size small
            if name in ["x", "v", "acc", "m_c", "T_c", "Y_f"]:
                trajectories[name] = sol_x[indices].tolist()

        output = {
            "status": status,
            "objective": obj_val,
            "iter_count": solver.stats()["iter_count"],
            "return_status": solver.stats()["return_status"],
            "trajectories": trajectories,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        output = {"status": "Exception", "error": str(e)}

    return output


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "matrix_output")

    # --- Helper: Dynamic Boost Scaling ---
    # Derive P_boost from Stoic (lambda) and Load (Q_total) to ensure valid A/F
    def stoic_expander(params_in):
        phi = params_in.get("phi", 1.0)
        q = params_in["q_total"]
        # Base: P=2.0 at Q=1000, phi=1.0 => K = 0.002
        p_bar = 2.0 * (q / 1000.0) * (1.0 / phi)
        # Clip params (Open Air Mass -> Minimal constraints, but stay physical)
        p_bar = max(0.2, min(p_bar, 10.0))  # Allow up to 10 bar for high load
        return {"p_boost_bars": p_bar}

    def phase1_wrapper(p):
        # Auto-calculate boost if not manual
        if "p_boost_bars" not in p:
            p.update(stoic_expander(p))
        elif "phi" in p:  # If both present, phi logic overrides or complements?
            # If sweeping phi, we MUST recalculate boost
            p.update(stoic_expander(p))
        return phase1_test(p)

    # --- Matrix 1: RPM x Load (Stoichiometric) ---
    # Vary Load (Q) and RPM. Boost scales automatically to maintain Phi=1.0
    axes1 = {
        "rpm": [800, 2000, 4000, 6000, 8000],
        # Extend to 5000J+ for 150kW potential
        "q_total": [1000.0, 3000.0, 5000.0, 7000.0],
        "phi": [1.0],  # Fixed Stoic
    }
    run_matrix("rpm_x_load", axes1, phase1_wrapper, output_dir, workers=1)

    # --- Matrix 3: Stoic x Load (at fixed RPM) ---
    axes3 = {
        "rpm": [2000],
        "q_total": [1000.0, 3000.0, 5000.0],
        "phi": [0.7, 1.0, 1.3],  # Lean to Rich
    }
    run_matrix("stoic_x_load", axes3, phase1_wrapper, output_dir, workers=1)

    # --- Matrix 4: Stoic x RPM (at fixed Load) ---
    axes4 = {
        "rpm": [800, 2000, 4000, 6000, 8000],
        "q_total": [3000.0],
        "phi": [0.7, 1.0, 1.3],
    }
    run_matrix("stoic_x_rpm", axes4, phase1_wrapper, output_dir, workers=1)


if __name__ == "__main__":
    main()
