
import os
import sys

def setup_casadi_path():
    """
    Ensure Conda Library/bin is in PATH on Windows so CasADi can find ipopt.dll.
    This must be called BEFORE importing casadi.
    """
    if os.name != 'nt':
        return

    # Check common locations
    candidates = []
    
    # 1. From sys.executable (Generic Conda)
    conda_prefix = os.path.dirname(sys.executable)
    candidates.append(os.path.join(conda_prefix, "Library", "bin"))
    
    # 2. From Environment Variable
    env_prefix = os.environ.get("CONDA_PREFIX")
    if env_prefix:
        candidates.append(os.path.join(env_prefix, "Library", "bin"))
        
    added = False
    for lib_bin in candidates:
        if os.path.exists(lib_bin) and lib_bin not in os.environ["PATH"]:
            # Prepend to ensure priority
            os.environ["PATH"] = lib_bin + os.pathsep + os.environ["PATH"]
            # print(f"[EnvSetup] Added {lib_bin} to PATH") # Reduce noise
            added = True
            break
    
    # if not added:
    #     print("[EnvSetup] No Conda Library/bin found or already in PATH.")

# Auto-run on import
setup_casadi_path()

# --- PROVENANCE AUTO-HOOK ---
if os.environ.get("LARRAK_RUN_ID"):
    try:
        import provenance.hooks
        provenance.hooks.install()
        # print("[EnvSetup] Provenance hooks installed.")
    except ImportError:
        pass
# ----------------------------
