
import os
import sys
import casadi as ca
import numpy as np

# PATCH: Explicitly add Conda Library bin and HSL bin to PATH for CasADi
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

current_path = os.environ.get("PATH", "")
for p in conda_paths:
    if os.path.exists(p) and p not in current_path:
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(p)
            except:
                pass

# Explicitly set HSLLIB option availability for Ipopt via Env (optional/fallback)
os.environ["HSLLIB_PATH"] = hsl_dll

# Ensure path
sys.path.append(os.getcwd())

from thermo.nlp import build_thermo_nlp

def debug_diag():
    print("Building NLP...")
    res_tuple = build_thermo_nlp(n_coll=40, debug_feasibility=True)
    
    if not isinstance(res_tuple, tuple):
        print("ERROR: Result is not a tuple")
        return

    nlp = res_tuple[0]
    meta = res_tuple[1]
    
    print(f"Meta Keys: {meta.keys()}")
    
    if "diagnostics_fn" not in meta:
        print("ERROR: diagnostics_fn missing")
        return
        
    diag_fn = meta["diagnostics_fn"]
    print("Diagnostics function found.")
    
    # Run a dummy solve or just eval with w0
    x0 = meta["w0"]
    
    print(f"Evaluating diagnostics with w0 (Length: {x0.shape})...")
    try:
        d_out = diag_fn(x0)
        p_max = float(d_out[0])
        work = float(d_out[1])
        print(f"Result with w0: P_max={p_max}, Work={work}")
    except Exception as e:
        print(f"ERROR calling diagnostics: {e}")

    # Create dummy solver to get a 'valid' x (though x0 should work)
    solver = ca.nlpsol("solver", "ipopt", nlp, {"ipopt": {"max_iter": 1}, "print_time":0})
    sol = solver(x0=x0, lbx=meta["lbw"], ubx=meta["ubw"], lbg=meta["lbg"], ubg=meta["ubg"])
    
    w_opt = sol["x"]
    print(f"Evaluating diagnostics with 1-iter solution...")
    d_out_opt = diag_fn(w_opt)
    p_max_opt = float(d_out_opt[0])
    work_opt = float(d_out_opt[1])
    t_crown_opt = float(d_out_opt[2])
    print(f"Result with w_opt: P_max={p_max_opt}, Work={work_opt}, T_crown={t_crown_opt}")
    
    if t_crown_opt > 0:
        print("PASS: T_crown calculated.")
    else:
        print("FAIL: T_crown is zero or negative.")

if __name__ == "__main__":
    debug_diag()
