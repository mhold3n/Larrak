"""Constants used across the Larrak project."""

# Numerical tolerances
TOLERANCE = 1e-8
COLLOCATION_TOLERANCE = 1e-6

# Default solver settings
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_COLLOCATION_DEGREE = 3

# Motion law types
MOTION_LAW_TYPES = [
    "minimum_time",
    "minimum_energy",
    "minimum_jerk",
    "minimum_snap",
    "custom",
]

# Collocation methods
COLLOCATION_METHODS = [
    "legendre",
    "radau",
    "lobatto",
]

# Geometry and gear synthesis constants (Litvin)
from math import pi as _pi  # noqa: E402

DEG_TO_RAD = _pi / 180.0
RAD_TO_DEG = 180.0 / _pi

DEFAULT_PRESSURE_ANGLE_DEG = 20.0
MIN_PRESSURE_ANGLE_DEG = 12.0
MAX_PRESSURE_ANGLE_DEG = 35.0

GEOM_TOL = 1e-6
PROFILE_CLOSURE_TOL = 1e-2

# OP dynamic gas simulator shared defaults
DEFAULT_DISCHARGE_COEFFICIENT: float = 0.7


# IPOPT / HSL integration constants
# Resolve platform-appropriate HSL library path at import time.
# Falls back to empty string if not found (Ipopt will use non-HSL linear solver).
import os as _os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
import sys as _sys  # noqa: E402

def _detect_hsl_path() -> str:
    """
    Detect HSL library path with priority:
    1. Local conda environment (OS-specific)
    2. Active conda environment
    3. Environment variable HSLLIB_PATH
    4. Project CoinHSL archive (Windows)
    """
    try:
        project_root = _Path(__file__).resolve().parents[1]
        
        # Priority 1: Check local conda environment using env_manager
        try:
            from campro.environment.env_manager import find_hsl_library  # noqa: E402
            
            hsl_path = find_hsl_library()
            if hsl_path and hsl_path.exists():
                return str(hsl_path)
        except ImportError:
            # If env_manager not available, skip this check
            pass
        except Exception:
            # If any error occurs, continue to fallback methods
            pass
        
        # Priority 2: Check environment variable
        hsl_env = _os.environ.get("HSLLIB_PATH", "")
        if hsl_env:
            env_path = _Path(hsl_env)
            if env_path.exists():
                return str(env_path)
        
        # Priority 3: Look for the specific CoinHSL archive folder shipped with the repo
        # Prefer the specific version: CoinHSL-archive.v2024.5.15.x86_64-w64-mingw32-libgfortran5
        if _sys.platform.startswith("win"):
            hsl_folder = project_root / "CoinHSL-archive.v2024.5.15.x86_64-w64-mingw32-libgfortran5"
            if hsl_folder.exists():
                bin_dir = hsl_folder / "bin"
                dll = bin_dir / "libcoinhsl.dll"
                if dll.exists():
                    return str(dll)
            
            # Fallback: search for any CoinHSL-archive.* folder (for compatibility)
            candidates = list(project_root.glob("CoinHSL-archive.*"))
            if candidates:
                bin_dir = candidates[0] / "bin"
                dll = bin_dir / "libcoinhsl.dll"
                if dll.exists():
                    return str(dll)
        
        # Priority 4: Platform-specific common locations (if env var not set)
        if _sys.platform.startswith("linux"):
            # Common Linux locations (as fallback only)
            pass
        elif _sys.platform == "darwin":
            # Common macOS locations (as fallback only)
            pass
    except Exception:
        pass
    return ""

HSLLIB_PATH: str = _detect_hsl_path()

def _detect_ipopt_opt() -> str:
    try:
        project_root = _Path(__file__).resolve().parents[1]
        opt = project_root / "ipopt.opt"
        return str(opt) if opt.exists() else ""
    except Exception:
        return ""

IPOPT_OPT_PATH: str = _detect_ipopt_opt()

# Default directory for Ipopt output files when analysis is enabled
IPOPT_LOG_DIR: str = "logs/ipopt"


# CasADi physics integration constants
# Port of core physics (torque, side loading, Litvin) from Python to CasADi MX
# for automatic differentiation and direct NLP integration with Ipopt
USE_CASADI_PHYSICS: bool = True
# Feature toggle: enable CasADi physics in optimizers (ENABLED BY DEFAULT)
# When True, CrankCenterOptimizer uses symbolic CasADi expressions instead of Python callbacks

CASADI_PHYSICS_EPSILON: float = 1e-12
# Domain guard epsilon for sqrt, division, and other potentially singular operations
# Usage: sqrt(fmax(value, CASADI_PHYSICS_EPSILON))

CASADI_PHYSICS_ASIN_CLAMP: float = 0.999999
# Clamp value for arcsin input to avoid domain errors at ±1
# Usage: arcsin(fmin(fmax(value, -clamp), clamp))

# CasADi physics performance and validation constants
CASADI_PHYSICS_MAX_EVALUATION_TIME_MS: float = 2.0
# Maximum expected evaluation time for unified physics function (ms)
# Used for performance testing and validation
# Based on actual measurements: ~0.7ms with 2x safety margin

CASADI_PHYSICS_MAX_GRADIENT_TIME_MS: float = 3.0
# Maximum expected gradient evaluation time for unified physics function (ms)
# Used for performance testing and validation
# Based on actual measurements: ~1.4ms with 2x safety margin

CASADI_PHYSICS_USE_EFFECTIVE_RADIUS_CORRECTION: bool = False
# Feature flag: enable effective radius offset correction in CasADi kinematics
# When True, accounts for crank center offset effects on effective radius
# Default False until validated against Python baselines

CASADI_PHYSICS_VALIDATION_MODE: bool = True
# Feature flag: enable parallel validation mode for CasADi physics
# When True, runs both Python and CasADi physics and compares results
# Used for confidence building and validation statistics collection

CASADI_PHYSICS_VALIDATION_TOLERANCE: float = 1e-4
# Tolerance for validation mode comparisons between Python and CasADi results
# Used to determine if implementations are within acceptable agreement
