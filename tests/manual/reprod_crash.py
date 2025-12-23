import numpy as np
import casadi as ca
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from thermo.nlp import build_thermo_nlp


def test_point():
    rpm = 1200.0
    p_int_bar = 1.0
    fuel_mass_mg = 200.0

    print(f"Testing Point: RPM={rpm}, P={p_int_bar} bar, Fuel={fuel_mass_mg} mg")

    # 1. Derived Physical Inputs
    p_int = p_int_bar * 1e5
    fuel_mass = fuel_mass_mg * 1e-6
    lhv = 44.0e6
    q_total = fuel_mass * lhv

    # 2. Build NLP
    print("Building NLP...")
    nlp_res = build_thermo_nlp(
        n_coll=50,
        Q_total=q_total,
        p_int=p_int,
        omega_val=rpm * 2 * np.pi / 60.0,
        model_type="prechamber",
        stroke=0.1,  # 100mm (OP)
        debug_feasibility=True,
    )

    if isinstance(nlp_res, tuple):
        nlp, meta = nlp_res
        x0 = meta["w0"]
        lbx = meta["lbw"]
        ubx = meta["ubw"]
        lbg = meta["lbg"]
        ubg = meta["ubg"]
    else:
        nlp = nlp_res
        x0 = nlp["x0"]
        lbx = nlp["lbx"]
        ubx = nlp["ubx"]
        lbg = nlp["lbg"]
        ubg = nlp["ubg"]

    # 3. Create Solver
    print("Creating Solver...")
    opts = {
        "ipopt": {
            "max_iter": 500,
            "print_level": 5,
            "linear_solver": "ma57",  # Using ma57 as in DOE
        },
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # 4. Solve
    print("Solving...")
    try:
        res = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        print("Solve Finished!")
        print("Status:", solver.stats()["return_status"])
        print("Obj:", res["f"])
    except Exception as e:
        print("Exception during solve:", e)


if __name__ == "__main__":
    test_point()
