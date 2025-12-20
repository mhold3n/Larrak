
import os
import sys
import numpy as np

# Add project root
sys.path.append(os.getcwd())

# PATCH: Explicitly add Conda Library bin and HSL bin to PATH for CasADi
try:
    conda_prefix = sys.prefix
    conda_paths = [
        os.path.join(conda_prefix, "Library", "bin"),
        os.path.join(conda_prefix, "Library", "mingw-w64", "bin"),
        os.path.join(conda_prefix, "Library", "usr", "bin"),
    ]

    # HSL Path (Hardcoded based on detected location)
    hsl_dll = r"c:\Users\maxed\OneDrive\Desktop\Github Projects\Larrak\Libraries\CoinHSL.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin\libcoinhsl.dll"
    hsl_dir = os.path.dirname(hsl_dll)
    conda_paths.insert(0, hsl_dir)

    os.environ["HSLLIB_PATH"] = hsl_dll

    current_path = os.environ.get("PATH", "")
    for p in conda_paths:
        if os.path.exists(p) and p not in current_path:
            os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                try: os.add_dll_directory(p)
                except: pass
except: pass

from campro.optimization.nlp.thermo_nlp import build_thermo_nlp
import casadi as ca

def debug_point():
    print("=== Debugging Single Point Feasibility ===")
    
    # Failing Point from DOE:
    # RPM=1000, P_int=1.0 bar (1e5 Pa), Fuel=20 mg
    rpm = 1000.0
    p_int_pa = 1.0e5
    fuel_mg = 20.0
    fuel_kg = fuel_mg * 1e-6
    lhv = 44.0e6
    q_total = fuel_kg * lhv
    
    T_int = 300.0
    omega = rpm * 2 * np.pi / 60.0
    
    print(f"Inputs: RPM={rpm}, P_int={p_int_pa/1e5} bar, Fuel={fuel_mg} mg")
    print(f"Q_total: {q_total} J")
    
    # Simple Initial Conditions
    gamma = 1.35
    CR = 15.0
    Vd = 0.00157 # approx for 100mm/200mm
    V_bdc = Vd * CR / (CR-1)
    m_trapped = p_int_pa * V_bdc / (287.0 * T_int)
    T_tdc = T_int * CR**(gamma-1)
    
    init_conds = {
        "m_c": m_trapped,
        "T_c": T_tdc
    }
    
    # Build NLP with VERBOSE logging
    # debug_feasibility=True enables soft constraints
    print("Building NLP (debug_feasibility=True)...")
    nlp_res = build_thermo_nlp(
        n_coll=50,
        Q_total=q_total,
        p_int=p_int_pa,
        T_int=T_int,
        omega_val=omega,
        debug_feasibility=True, 
        initial_conditions=init_conds
    )
    
    if isinstance(nlp_res, tuple):
        nlp, meta = nlp_res
        x0 = meta["w0"]
        lbx = meta["lbw"]
        ubx = meta["ubw"]
        lbg = meta["lbg"]
        ubg = meta["ubg"]
    else:
        print("Error: build_thermo_nlp returned dict, expected tuple")
        return

    # Solver Options - HIGH VERBOSITY
    opts = {
        "ipopt": {
            "max_iter": 3000,
            "print_level": 5, # Verbose
            "linear_solver": "ma57", # Use robust solver
        },
        "print_time": 1,
    }
    
    print("Solving...")
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    
    res = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    
    print(f"Status: {solver.stats()['return_status']}")
    print(f"Objective: {res['f']}")
    
    # Check Soft Penalties
    w_opt = res["x"]
    # If debug_feasibility is on, slack variables are part of w
    # We can inspect them if we knew their indices.
    # Instead, let's look at diagnostics.
    
    if "diagnostics_fn" in meta:
        diag = meta["diagnostics_fn"](w_opt)
        print(f"Diagnostics: P_max={float(diag[0])/1e5:.2f} bar, Work={float(diag[1]):.2f} J")

if __name__ == "__main__":
    debug_point()
