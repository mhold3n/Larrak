
import ctypes
import os
import sys

dll_path = r"c:\Users\maxed\OneDrive\Desktop\Github Projects\Larrak\Libraries\CoinHSL-archive.v2024.5.15.x86_64-w64-mingw32-libgfortran5\bin\libhsl.dll"
dependencies = [
    "libwinpthread-1.dll",
    "libgcc_s_seh-1.dll",
    "libgfortran-5.dll",
    "libquadmath-0.dll",
    "libopenblas.dll",
    "libmetis.dll",
    "libcoinhsl.dll", # sometimes depend on each other?
]

print(f"Testing DLL load: {dll_path}")
print(f"CWD: {os.getcwd()}")
print(f"PATH: {os.environ['PATH']}")

# Try loading dependencies explicitly
base_dir = os.path.dirname(dll_path)
for dep in dependencies:
    dep_path = os.path.join(base_dir, dep)
    if os.path.exists(dep_path):
        try:
            ctypes.CDLL(dep_path)
            print(f"Successfully loaded dependency: {dep}")
        except Exception as e:
            print(f"Failed to load dependency {dep}: {e}")
    else:
        print(f"Dependency not found: {dep}")

# Try loading main DLL
try:
    # Add dll directory to dll search path (Python 3.8+)
    os.add_dll_directory(base_dir)
    lib = ctypes.CDLL(dll_path)
    print("Successfully loaded libhsl.dll")
except Exception as e:
    print(f"Failed to load libhsl.dll: {e}")

