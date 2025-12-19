
import ctypes
import os
import sys

# Path to casadi ipopt plugin in larrak env
dll_path = r"C:\Users\maxed\miniconda3\envs\larrak\Library\bin\casadi_nlpsol_ipopt.dll"

print(f"Testing DLL load: {dll_path}")
print(f"CWD: {os.getcwd()}")
print(f"PATH starting with: {os.environ['PATH'][:100]}...")

if not os.path.exists(dll_path):
    print("DLL file does not exist!")
    sys.exit(1)

# Try loading main DLL
try:
    # Add Library\bin to search path
    lib_bin = os.path.dirname(dll_path)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(lib_bin)
    
    lib = ctypes.CDLL(dll_path)
    print(f"Successfully loaded {os.path.basename(dll_path)}")
except Exception as e:
    print(f"Failed to load {os.path.basename(dll_path)}: {e}")
    # Inspect Windows Error
    import traceback
    traceback.print_exc()
