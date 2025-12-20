
import os
import sys

# CRITICAL: Fix PATH for CasADi in Conda environment
# When running python directly from envs/larrak/python.exe, Library/bin is NOT in PATH.
# CasADi's C++ core needs this to load plugins (ipopt) via LoadLibrary.
conda_prefix = sys.prefix
conda_lib_bin = os.path.join(conda_prefix, "Library", "bin")
conda_mingw = os.path.join(conda_prefix, "Library", "mingw-w64", "bin")
conda_usr = os.path.join(conda_prefix, "Library", "usr", "bin")

current_path = os.environ.get("PATH", "")
paths_to_prepend = [conda_lib_bin, conda_mingw, conda_usr]

# HSL Path - auto-detected for cross-platform support
from campro.environment.resolve import hsl_path
try:
    hsl_dll = str(hsl_path())
    hsl_dir = os.path.dirname(hsl_dll)
    paths_to_prepend.insert(0, hsl_dir)
except RuntimeError:
    print("Warning: HSL library not detected, solver may fail")
    hsl_dll = None
    hsl_dir = None

for p in paths_to_prepend:
    if os.path.exists(p) and p not in current_path:
        os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
        # Also use add_dll_directory for Python side
        try:
            os.add_dll_directory(p)
        except:
            pass

print(f"Patched PATH. HSL Dir: {hsl_dir}")

import casadi as ca

def check_ma86():
    print("Checking MA86 availability with PATH patched...")
    x = ca.MX.sym("x")
    nlp = {"x": x, "f": x**2, "g": x}
    
    opts = {
        "ipopt.linear_solver": "ma86",
        "ipopt.print_level": 5,
        "ipopt.hsllib": hsl_dll
    }
    
    try:
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        # Try to solve a dummy problem to ensure it actually loads the lib
        res = solver(x0=1.0, lbg=0.0, ubg=10.0)
        print("MA86 is available and working!")
        return True
    except Exception as e:
        print(f"MA86 check failed: {e}")
        return False

if __name__ == "__main__":
    check_ma86()
