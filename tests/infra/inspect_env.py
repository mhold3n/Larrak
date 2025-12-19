
import casadi as ca
import os
import sys

print(f"Python: {sys.executable}")
print(f"CasADi: {ca.__file__}")
try:
    print(f"CasADi Lib Path: {ca.GlobalOptions.getCasadiPath() if hasattr(ca.GlobalOptions, 'getCasadiPath') else 'N/A'}")
except:
    pass

# Try to load ipopt to see where it comes from
try:
    s = ca.nlpsol('s', 'ipopt', {'x': ca.MX.sym('x'), 'f': ca.MX.sym('x')**2})
    print("Ipopt loaded successfully")
except Exception as e:
    print(f"Ipopt failed: {e}")

# Check PATH
print("PATH:", os.environ.get('PATH'))
