#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Launcher script for the Cam Motion Law GUI.

This script provides a simple way to launch the GUI application with
environment validation.
"""

import sys
import io
from pathlib import Path
import os

# Fix Windows console encoding for emoji/Unicode characters
if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        # Fallback for older Python versions
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
try:
    # Tk only when needed; avoid early GUI init on headless
    import tkinter as tk  # type: ignore
    from tkinter import messagebox, filedialog  # type: ignore
except Exception:
    tk = None  # type: ignore
    messagebox = None  # type: ignore
    filedialog = None  # type: ignore

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def check_conda_environment():
    """Check if running in the correct conda environment and warn if not."""
    import os
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    
    # Check if we're in a conda environment at all
    in_conda = bool(conda_prefix)
    
    # Check for local environment first
    try:
        from campro.environment.platform_detector import (
            get_local_conda_env_path,
            is_local_conda_env_present,
        )
        project_root = current_dir
        if is_local_conda_env_present(project_root):
            local_env_path = get_local_conda_env_path(project_root)
            if str(Path(conda_prefix).resolve()) == str(local_env_path.resolve()):
                print(f"[DEBUG] Running in local conda environment: {local_env_path}")
                return True
            else:
                print(f"[WARNING] Local environment exists at {local_env_path} but active env is {conda_prefix}")
    except ImportError:
        pass  # If we can't import, skip this check
    
    # Check if in larrak environment (global or local)
    if conda_env == "larrak" or "larrak" in conda_prefix.lower():
        print(f"[DEBUG] Running in conda environment: {conda_env if conda_env else 'larrak'}")
        return True
    
    # Not in larrak environment - check if we can find it
    if not in_conda:
        print("[WARNING] Not running in a conda environment.")
        print("Please activate the larrak environment first:")
        print("  conda activate larrak")
        return False
    
    # In a conda environment but not larrak
    print(f"[WARNING] Running in conda environment '{conda_env}' instead of 'larrak'")
    print("This may cause missing dependencies (numpy, casadi, etc.)")
    print("Please activate the larrak environment:")
    print("  conda activate larrak")
    
    # Try to find larrak environment (simple path-based check without imports)
    import platform
    home = Path.home()
    system = platform.system().lower()
    
    # Check standard conda locations for larrak environment
    possible_paths = []
    if system == "windows":
        # Windows: check common locations
        possible_paths = [
            home / "miniconda3" / "envs" / "larrak",
            home / "anaconda3" / "envs" / "larrak",
            current_dir / "conda_env_windows",
        ]
    elif system == "darwin":
        possible_paths = [
            home / "miniconda3" / "envs" / "larrak",
            home / "anaconda3" / "envs" / "larrak",
            current_dir / "conda_env_macos",
        ]
    else:
        possible_paths = [
            home / "miniconda3" / "envs" / "larrak",
            home / "anaconda3" / "envs" / "larrak",
            current_dir / "conda_env_linux",
        ]
    
    for larrak_path in possible_paths:
        if larrak_path.exists():
            print(f"Found larrak environment at: {larrak_path}")
            print(f"Activate with: conda activate {larrak_path}")
            break
    else:
        print("Could not find larrak environment in standard locations.")
        print("Create it with: python scripts/setup_environment.py")
    
    # Don't fail immediately - let the import errors reveal the problem
    return False


def _prompt_and_resolve_paths() -> bool:
    """Interactive resolution for CasADi and HSL paths via Tk dialogs.

    Returns True if both CasADi import and HSL path are successfully resolved after prompts.
    """
    print("[DEBUG] _prompt_and_resolve_paths: Starting...")
    # Create a transient root for dialogs
    if tk is None or messagebox is None or filedialog is None:
        print("[DEBUG] _prompt_and_resolve_paths: tkinter unavailable")
        print("[ERROR] GUI prompt unavailable (tkinter missing). Cannot resolve paths interactively.")
        return False

    print("[DEBUG] _prompt_and_resolve_paths: Creating Tk root...")
    root = tk.Tk()
    root.withdraw()
    print("[DEBUG] _prompt_and_resolve_paths: Tk root created and hidden")

    # 1) Resolve CasADi
    print("[DEBUG] _prompt_and_resolve_paths: Testing CasADi import...")
    try:
        import casadi  # noqa: F401
        casadi_ok = True
        print("[DEBUG] _prompt_and_resolve_paths: CasADi import successful")
    except Exception as e:
        casadi_ok = False
        print(f"[DEBUG] _prompt_and_resolve_paths: CasADi import failed: {e}")

    if not casadi_ok:
        messagebox.showwarning(
            "CasADi/HSL not resolved",
            "CasADi module not found. Please select the folder containing the 'casadi' Python package.",
        )
        pkg_dir = filedialog.askdirectory(title="Select folder containing 'casadi' package")
        if pkg_dir:
            if pkg_dir not in sys.path:
                sys.path.insert(0, pkg_dir)
        try:
            import importlib

            importlib.invalidate_caches()
            import casadi  # type: ignore  # noqa: F401
            casadi_ok = True
        except Exception as e:
            messagebox.showerror("CasADi import failed", f"Failed to import CasADi after selection: {e}")
            casadi_ok = False

    # 2) Resolve HSL library path
    print("[DEBUG] _prompt_and_resolve_paths: Checking HSL library path...")
    try:
        from campro import constants as _c

        hsl_path = getattr(_c, "HSLLIB_PATH", "")
        print(f"[DEBUG] _prompt_and_resolve_paths: HSLLIB_PATH from constants: {hsl_path}")
    except Exception as e:
        hsl_path = ""
        print(f"[DEBUG] _prompt_and_resolve_paths: Error importing constants: {e}")

    def _valid_hsl(p: str) -> bool:
        result = bool(p) and Path(p).exists()
        print(f"[DEBUG] _valid_hsl: path='{p}', exists={Path(p).exists() if p else False}, valid={result}")
        return result

    if not _valid_hsl(hsl_path):
        print(f"[DEBUG] _prompt_and_resolve_paths: HSL path not valid, prompting user...")
        messagebox.showwarning(
            "CasADi/HSL not resolved",
            "HSL (libcoinhsl) library path not set. Please select the HSL library file (DLL/DYLIB/SO).",
        )
        filetypes = [
            ("Windows DLL", "*.dll"),
            ("macOS dylib", "*.dylib"),
            ("Linux so", "*.so"),
            ("All files", "*.*"),
        ]
        sel = filedialog.askopenfilename(title="Select HSL library (libcoinhsl)", filetypes=filetypes)
        if sel:
            os.environ["HSLLIB_PATH"] = sel
            try:
                from campro import constants as _c2

                _c2.HSLLIB_PATH = sel  # override runtime constant
            except Exception:
                pass
            hsl_path = sel

    root.destroy()
    result = casadi_ok and _valid_hsl(hsl_path)
    print(f"[DEBUG] _prompt_and_resolve_paths: Returning {result} (casadi_ok={casadi_ok}, hsl_valid={_valid_hsl(hsl_path)})")
    return result


def validate_environment_before_launch():
    """Validate environment before launching GUI."""
    try:
        print("[DEBUG] validate_environment_before_launch: Starting...")
        # TEMPORARILY DISABLE VALIDATION TO PREVENT UNEXPECTED LINEAR SOLVER CHANGES
        # The validation process may create solvers that conflict with the MA27/MA57 policy
        # Hard fail if CasADi/HSL unresolved; allow user to resolve interactively
        print("[DEBUG] validate_environment_before_launch: Calling _prompt_and_resolve_paths()...")
        ok = _prompt_and_resolve_paths()
        print(f"[DEBUG] validate_environment_before_launch: _prompt_and_resolve_paths() returned: {ok}")
        if not ok:
            raise RuntimeError("CasADi/HSL not resolved")
        print("Environment OK - proceeding with GUI launch...")
        print("[DEBUG] validate_environment_before_launch: Complete")

        # Original validation code (commented out to avoid unintended solver initialization)
        # from campro.environment.validator import validate_environment
        #
        # print("Validating environment...")
        # results = validate_environment()
        # overall_status = results["summary"]["overall_status"]
        #
        # if overall_status.value == "error":
        #     print("❌ Environment validation failed!")
        #     print("Required dependencies are missing or incompatible.")
        #     print("\nTo fix this issue:")
        #     print("1. Run: python scripts/setup_environment.py")
        #     print("2. Or run: python scripts/check_environment.py")
        #     print("\nExiting...")
        #     sys.exit(1)
        # elif overall_status.value == "warning":
        #     print("⚠️  Environment validation passed with warnings.")
        #     print("Run 'python scripts/check_environment.py' for details.")
        #     print("Continuing with GUI launch...")
        # else:
        #     print("✅ Environment validation passed!")

    except ImportError as e:
        print(f"⚠️  Warning: Could not import environment validator: {e}")
        print("Environment validation skipped.")
    except Exception as e:
        print(f"[WARNING] Error during environment validation: {e}")
        print("Environment validation failed.")


try:
    # Check conda environment first
    print("[DEBUG] Step 0: Checking conda environment...")
    env_ok = check_conda_environment()
    if not env_ok:
        print("[WARNING] Continuing anyway - import errors will reveal missing dependencies...")
    
    # Skip environment validation at launch - it will run when user presses optimize button
    print("[DEBUG] Step 1: Skipping environment validation at launch (will run before optimization)...")

    print("[DEBUG] Step 2: About to import cam_motion_gui...")
    from cam_motion_gui import main
    print("[DEBUG] Step 3: Successfully imported cam_motion_gui.main")

    if __name__ == "__main__":
        print("[DEBUG] Step 4: __name__ == '__main__', calling main()...")
        print("Starting Cam Motion Law GUI...")
        main()
        print("[DEBUG] Step 5: main() returned")

except ImportError as e:
    print(f"[DEBUG] ImportError caught: {e}")
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure all dependencies are installed:")
    print("Run: python scripts/setup_environment.py")
    print("Or: pip install -r requirements.txt")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"[DEBUG] Exception caught: {e}")
    print(f"❌ Error starting GUI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
