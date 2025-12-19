
import os
import sys

print(f"PATH BEFORE: {os.environ.get('PATH')}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX')}")

# Fix Path logic
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    if os.name == 'nt':
        lib_bin = os.path.join(conda_prefix, "Library", "bin")
        if lib_bin not in os.environ["PATH"]:
            print(f"Adding {lib_bin} to PATH")
            os.environ["PATH"] = lib_bin + os.pathsep + os.environ["PATH"]
        else:
            print(f"{lib_bin} is ALREADY in PATH")

import casadi as ca
print(f"CasADi Version: {ca.__version__}")

try:
    # Try creating a solver
    x = ca.SX.sym("x")
    nlp = {"x": x, "f": x**2}
    print("Attempting to create IPOPT solver...")
    solver = ca.nlpsol("solver", "ipopt", nlp)
    print("SUCCESS: IPOPT solver created.")
    
    # Test Solve
    res = solver(x0=2.0)
    print(f"Solve Result: {res['x']}")
except Exception as e:
    print(f"FAILURE: {e}")
