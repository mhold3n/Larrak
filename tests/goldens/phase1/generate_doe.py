"""
Generate Phase 1 DOE Golden.
Tests Thermo Module across RPM, Load, and Phi ranges using a unified Design of Experiments.
"""

import os
import sys

# Critical Configuration:
# We use Outer-Loop Parallelism (workers=4 in runner), so we must Disable Inner-Loop Parallelism.
# This prevents overloading the CPU (4 workers * 8 threads = 32 threads -> Thrashing).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from typing import Any

import casadi as ca
import numpy as np
import pandas as pd
import csv
import time
import datetime
import itertools

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from tests.infra.doe_runner import DOERunner
from thermo.nlp import build_thermo_nlp


def phase1_test(params: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a single Phase 1 test case with physical inputs.
    Params: rpm, p_int_bar, fuel_mass_mg
    """
    rpm = params["rpm"]
    p_int_bar = params["p_int_bar"]
    fuel_mass_mg = params["fuel_mass_mg"]

    # 1. Derived Physical Inputs
    p_int = p_int_bar * 1e5
    fuel_mass = fuel_mass_mg * 1e-6

    # Lower Heating Value of generic gasoline/hydrocarbon
    lhv = 44.0e6  # J/kg

    # Total Heat Release (Energy Input)
    # Assuming complete combustion for the energy source definition
    q_total = fuel_mass * lhv

    # 2. Physics / Thermodynamics Reporting
    # We estimate Phi (Equivalence Ratio) for reporting purposes
    # m_air ~ rho * Vd * volumetric_efficiency
    # For reporting, assume VolEff=1.0 and standard conditions
    r_gas = 287.0
    t_int_ref = 300.0  # From NLP default (distinct from input, used for rho calc)
    rho_int = p_int / (r_gas * t_int_ref)

    # Vd for 100mm bore, 200mm stroke
    vd = (np.pi * (0.1**2) / 4.0) * 0.2
    m_air_approx = rho_int * vd

    # Stoichiometric Ratio (Air/Fuel) approx 14.7
    af_stoic = 14.7

    # Actual AF
    af_actual = m_air_approx / fuel_mass if fuel_mass > 0 else 1e9

    # Phi = AF_stoic / AF_actual
    phi_approx = af_stoic / af_actual

    try:
        # Build NLP with physical inputs
        nlp_res = build_thermo_nlp(
            n_coll=40,
            Q_total=q_total,
            p_int=p_int,
            omega_val=rpm * 2 * np.pi / 60.0,
            debug_feasibility=False,
        )

        if isinstance(nlp_res, tuple):
            nlp, meta = nlp_res
            x0 = meta["w0"]
            lbx = meta["lbw"]
            ubx = meta["ubw"]
            lbg = meta["lbg"]
            ubg = meta["ubg"]
        else:
            # Fallback/Legacy if build_thermo_nlp returns single dict
            nlp = nlp_res
            x0 = nlp.get("x0", nlp.get("w0"))
            lbx = nlp.get("lbx", nlp.get("lbw"))
            ubx = nlp.get("ubx", nlp.get("ubw"))
            lbg = nlp.get("lbg")
            ubg = nlp.get("ubg")

        # 3. Create Solver
        opts = {
            "ipopt": {
                "max_iter": 3000,
                "print_level": 0,
                "sb": "yes",
                "tol": 1e-6,
                "linear_solver": "ma57",
            },
            "print_time": 0,
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # 4. Solve
        res = solver(
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
        )  # Analyze Results
        stats = solver.stats()
        status = "Optimal" if stats["success"] else stats["return_status"]
        if "Acceptable" in stats["return_status"]:
            status = "Acceptable"

        obj_val = float(res["f"])

        # Work and Efficiency
        # CRITICAL: Use 'true_work' from diagnostics, NOT objective (which includes penalties)
        abs_work_j = 0.0
        p_max_bar = 0.0

        # Check if diagnostics function is available in meta dict
        # nlp_res is (nlp_dict, meta_dict)
        if isinstance(nlp_res, tuple) and "diagnostics_fn" in nlp_res[1]:
            diag_fn = nlp_res[1]["diagnostics_fn"]
            # diag_fn returns [p_max, work_j]
            # Must pass optimal w (res["x"])
            w_opt = res["x"]
            d_res = diag_fn(w_opt)
            p_max_bar = float(d_res[0]) / 1e5
            abs_work_j = float(d_res[1])
        else:
            # Fallback (likely inaccurate due to penalties)
            abs_work_j = abs(obj_val)

        # Calculate Compression Work Estimate (Adiabatic) to get Net Work
        # W_comp = (P1*V1 / (gamma-1)) * (CR^(gamma-1) - 1)
        # Geometry: OP Symmetric
        B, S_piston, CR_val = 0.1, 0.1, 15.0
        # V_bdc = 2 * (Vc + Vd_piston)
        V_disp_p = (np.pi * B**2 / 4.0) * S_piston
        V_c_p = V_disp_p / (CR_val - 1)
        V_bdc_total = 2.0 * (V_c_p + V_disp_p)

        gamma_air = 1.35
        # p_int is already in Pa
        w_comp_est = (p_int * V_bdc_total / (gamma_air - 1)) * (CR_val ** (gamma_air - 1) - 1)

        abs_work_net_j = abs_work_j - w_comp_est

        # Net Efficiency
        thermal_eff = abs_work_net_j / q_total if q_total > 0 else 0.0

        output = {
            "status": status,
            "doe_status": "Completed",
            "rpm": rpm,
            "p_int_bar": p_int_bar,
            "fuel_mass_mg": fuel_mass_mg,
            "q_total_j": q_total,
            "phi_est": phi_approx,
            "objective": obj_val,
            "abs_work_j": abs_work_j,  # Gross Work
            "abs_work_net_j": abs_work_net_j,  # Net Work
            "w_comp_est_j": w_comp_est,
            "p_max_bar": p_max_bar,
            "thermal_efficiency": thermal_eff,
            "iter_count": stats["iter_count"],
            "solver_status": stats["return_status"],
        }

    except Exception as e:
        output = {
            "status": "Exception",
            "doe_status": "Failed",
            "rpm": rpm,
            "p_int_bar": p_int_bar,
            "fuel_mass_mg": fuel_mass_mg,
            "error": str(e),
        }

    return output


def main():
    # 1. Define Physical Grid (Rectangular Domain)
    # RPM: 1000 to 6000 in 200 rpm steps (26 levels)
    rpm_levels = np.linspace(1000, 6000, 26)

    # Boost/Pressure Sweep: 1.0 bar to 4.0 bar (21 levels)
    p_int_levels = np.linspace(1.0, 4.0, 21)

    # Fuel Mass Sweep: 20mg to 300mg (25 levels)
    # Fuel Type: Gasoline (LHV 44MJ/kg, Density ~0.75 g/cc)
    # Max Flow: 300mg/cycle * 6000rpm = 1.8 kg/min = 2400 cc/min (Meets >2000cc/min req)
    fuel_levels = np.linspace(20, 300, 25)

    # Full Factorial
    doe_list = []
    import itertools

    for r, p, f in itertools.product(rpm_levels, p_int_levels, fuel_levels):
        doe_list.append({"rpm": r, "p_int_bar": p, "fuel_mass_mg": f})

    print(f"Generated {len(doe_list)} DOE points.")

    # 2. Run DOE (Parallel)
    # Define Output Location
    output_dir = "tests/goldens/phase1/doe_output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Runner
    runner = DOERunner("phase1_physical", output_dir)

    # Load Design
    runner.design = pd.DataFrame(doe_list)

    # Run
    # DOERunner handles parallel execution and incremental saving
    print(f"Starting DOE Run...")
    runner.run(test_func=phase1_test, workers=8)

    print(f"DOE Completed.")


if __name__ == "__main__":
    main()
