"""Constants used across the Larrak project.

This module provides project-wide constants with engineering metadata.
All physics and numerical tolerance constants use the PhysicalConstant
dataclass for traceability and uncertainty quantification.

For pure configuration constants (solver settings, file paths), raw values
are used. For physics constants, use campro.units instead.
"""

from __future__ import annotations

from math import pi as _pi

from campro.units import PhysicalConstant

# =============================================================================
# Numerical Tolerances (with PhysicalConstant metadata)
# =============================================================================

TOLERANCE = PhysicalConstant(
    value=1e-8,
    unit="dimensionless",
    source="Project standard numerical tolerance",
    notes="General-purpose numerical tolerance for comparisons",
)

COLLOCATION_TOLERANCE = PhysicalConstant(
    value=1e-6,
    unit="dimensionless",
    source="Betts, Practical Methods for Optimal Control, Ch. 4",
    notes="Tolerance for collocation constraint satisfaction",
)

GEOM_TOL = PhysicalConstant(
    value=1e-6,
    unit="m",
    source="Litvin, Gear Geometry and Applied Theory, tolerance recommendations",
    notes="Geometric tolerance for gear profile computations",
)

PROFILE_CLOSURE_TOL = PhysicalConstant(
    value=1e-2,
    unit="m",
    source="Litvin gear synthesis closure tolerance",
    notes="Tolerance for profile closure verification",
)


# =============================================================================
# Geometry and Gear Synthesis Constants
# =============================================================================

DEG_TO_RAD = PhysicalConstant(
    value=_pi / 180.0,
    unit="rad/deg",
    source="Mathematical definition",
    notes="Exact conversion factor",
)

RAD_TO_DEG = PhysicalConstant(
    value=180.0 / _pi,
    unit="deg/rad",
    source="Mathematical definition",
    notes="Exact conversion factor",
)

DEFAULT_PRESSURE_ANGLE_DEG = PhysicalConstant(
    value=20.0,
    unit="deg",
    source="AGMA 2001-D04, standard pressure angle",
    valid_range=(12.0, 35.0),
    notes="Standard pressure angle for involute gears",
)

MIN_PRESSURE_ANGLE_DEG = PhysicalConstant(
    value=12.0,
    unit="deg",
    source="AGMA 2001-D04, minimum for standard gears",
    notes="Below this, undercutting becomes severe",
)

MAX_PRESSURE_ANGLE_DEG = PhysicalConstant(
    value=35.0,
    unit="deg",
    source="AGMA 2001-D04, practical maximum",
    notes="Above this, contact ratio drops too low",
)


# =============================================================================
# Fluid Dynamics Constants
# =============================================================================

DEFAULT_DISCHARGE_COEFFICIENT = PhysicalConstant(
    value=0.7,
    unit="dimensionless",
    source="Heywood, IC Engine Fundamentals, Table 6.1",
    uncertainty=0.05,
    valid_range=(0.6, 0.9),
    notes="Typical poppet valve discharge coefficient",
)


# =============================================================================
# CasADi Physics Integration Constants
# =============================================================================

CASADI_PHYSICS_EPSILON = PhysicalConstant(
    value=1e-9,
    unit="dimensionless",
    source="Numerical analysis best practice for domain guards",
    notes="Small epsilon to prevent division by zero in symbolic physics expressions. "
    "Consolidated from duplicate definitions, standardized to 1e-9 for numerical stability.",
)

CASADI_FLOW_EPSILON = PhysicalConstant(
    value=1e-2,
    unit="dimensionless",
    source="Flow calculation stability requirement",
    notes="Larger epsilon for smooth flow approximation to avoid kinks near zero pressure difference.",
)

CASADI_PHYSICS_ASIN_CLAMP = PhysicalConstant(
    value=0.999999,
    unit="dimensionless",
    source="Numerical analysis - arcsin domain limit",
    notes="Clamp value to avoid arcsin(±1) domain errors",
)

CASADI_PHYSICS_MAX_EVALUATION_TIME_MS = PhysicalConstant(
    value=2.0,
    unit="ms",
    source="Performance measurement with 2x safety margin",
    notes="Max expected unified physics evaluation time",
)

CASADI_PHYSICS_MAX_GRADIENT_TIME_MS = PhysicalConstant(
    value=3.0,
    unit="ms",
    source="Performance measurement with 2x safety margin",
    notes="Max expected gradient evaluation time",
)

CASADI_PHYSICS_VALIDATION_TOLERANCE = PhysicalConstant(
    value=1e-4,
    unit="dimensionless",
    source="Validation mode comparison threshold",
    notes="Max relative error between Python and CasADi implementations",
)


# =============================================================================
# Solver Configuration (non-physics, raw values)
# =============================================================================

DEFAULT_MAX_ITERATIONS: int = 1000
DEFAULT_COLLOCATION_DEGREE: int = 3

MOTION_LAW_TYPES: list[str] = [
    "minimum_time",
    "minimum_energy",
    "minimum_jerk",
    "minimum_snap",
    "custom",
]

COLLOCATION_METHODS: list[str] = [
    "legendre",
    "radau",
    "lobatto",
]


# =============================================================================
# Feature Flags (configuration, raw values)
# =============================================================================

USE_CASADI_PHYSICS: bool = True
CASADI_PHYSICS_USE_EFFECTIVE_RADIUS_CORRECTION: bool = False
CASADI_PHYSICS_VALIDATION_MODE: bool = True


# =============================================================================
# IPOPT / HSL Integration (infrastructure, raw values)
# =============================================================================

import os as _os  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


def _detect_hsl_path() -> str:
    """Detect HSL library path with priority-based search.

    Priority:
    1. Local conda environment (OS-specific)
    2. Active conda environment
    3. Environment variable HSLLIB_PATH
    4. Project CoinHSL archive (OS-specific) - auto-detected

    Returns:
        Path to HSL library

    Raises:
        RuntimeError: If all detection methods fail
    """
    from campro.logging import get_logger

    log = get_logger(__name__)
    detection_attempts: list[str] = []
    last_error: Exception | None = None

    try:
        # Priority 1: Check local conda environment using env_manager
        try:
            from campro.environment.env_manager import find_hsl_library

            hsl_path = find_hsl_library()
            if hsl_path and hsl_path.exists():
                log.debug(f"HSL library found via Priority 1 (conda env): {hsl_path}")
                return str(hsl_path)
            detection_attempts.append("Priority 1: HSL library not found in conda env")
        except ImportError:
            detection_attempts.append("Priority 1: env_manager not available (skipped)")
        except Exception as e:
            detection_attempts.append(f"Priority 1: Error - {e}")
            last_error = e
            log.debug(f"Priority 1 detection failed: {e}")

        # Priority 2: Check environment variable
        hsl_env = _os.environ.get("HSLLIB_PATH", "")
        if hsl_env:
            env_path = _Path(hsl_env)
            if env_path.exists():
                log.debug(f"HSL library found via Priority 2 (env var): {env_path}")
                return str(env_path)
            detection_attempts.append(f"Priority 2: HSLLIB_PATH set but file not found: {hsl_env}")
        else:
            detection_attempts.append("Priority 2: HSLLIB_PATH not set")

        # Priority 3: Auto-detect CoinHSL directory using hsl_detector
        try:
            from campro.environment.hsl_detector import get_hsl_library_path

            hsl_lib_path = get_hsl_library_path()
            if hsl_lib_path and hsl_lib_path.exists():
                log.debug(f"HSL library found via Priority 3 (auto-detect): {hsl_lib_path}")
                return str(hsl_lib_path)
            detection_attempts.append("Priority 3: HSL library not found in project directory")
        except ImportError:
            detection_attempts.append("Priority 3: hsl_detector not available (skipped)")
        except Exception as e:
            detection_attempts.append(f"Priority 3: Error - {e}")
            last_error = e
            log.debug(f"Priority 3 detection failed: {e}")

    except Exception as e:
        detection_attempts.append(f"Outer exception: {e}")
        last_error = e
        log.debug(f"Outer detection exception: {e}")

    # All detection methods failed
    error_msg = "HSL library path detection failed. Attempted methods:\n"
    error_msg += "\n".join(f"  - {attempt}" for attempt in detection_attempts)
    if last_error:
        error_msg += f"\nLast error: {last_error}"

    log.error(error_msg)
    raise RuntimeError(error_msg)


HSLLIB_PATH: str = _detect_hsl_path()


def _detect_ipopt_opt() -> str:
    """Detect ipopt.opt configuration file path."""
    try:
        project_root = _Path(__file__).resolve().parents[1]
        opt = project_root / "ipopt.opt"
        return str(opt) if opt.exists() else ""
    except Exception:
        return ""


IPOPT_OPT_PATH: str = _detect_ipopt_opt()
IPOPT_LOG_DIR: str = "logs/ipopt"


# =============================================================================
# Backward Compatibility - Raw Value Access
# =============================================================================
# For code that expects raw float values, the PhysicalConstant objects
# support float() conversion and arithmetic operations.
# Example: float(TOLERANCE) returns 1e-8

# =============================================================================
# Engine Physics Constants (Friction & Heat Transfer)
# =============================================================================

FRICTION_CHEN_FLYNN_A = PhysicalConstant(
    value=2.0,
    unit="bar",
    source="Chen-Flynn Model calibration (Heywood generic)",
    notes="Constant friction term (FMEP offset)",
)

FRICTION_CHEN_FLYNN_B = PhysicalConstant(
    value=0.005,
    unit="dimensionless",
    source="Chen-Flynn Model calibration",
    notes="Peak pressure scaling term for FMEP",
)

HEAT_TRANSFER_WOSCHNI_C = PhysicalConstant(
    value=12.0,
    unit="dimensionless",
    source="Woschni Correlation calibration",
    notes="Global scaling coefficient for heat transfer (adjusted for strong heat loss)",
)


DEFAULT_WALL_TEMPERATURE = PhysicalConstant(
    value=450.0,
    unit="K",
    source="Typical cylinder liner temperature",
    valid_range=(350.0, 550.0),
    notes="Assumed constant wall temperature for heat loss",
)


# =============================================================================
# Combustion Model Constants (Wiebe & Flame Speed)
# =============================================================================

DEFAULT_WIEBE_M = PhysicalConstant(
    value=2.0,
    unit="dimensionless",
    source="Heywood, ICE Fundamentals, typical SI engine",
    valid_range=(1.5, 3.0),
    notes="Wiebe shape factor (m)",
)

DEFAULT_WIEBE_A = PhysicalConstant(
    value=5.0,
    unit="dimensionless",
    source="Heywood, ICE Fundamentals, 99% burn duration definition",
    notes="Wiebe efficiency factor (a), corresponds to 99% burn complete",
)

DEFAULT_TURBULENCE_FACTOR_K = PhysicalConstant(
    value=0.3,
    unit="dimensionless",
    source="Turbulent flame speed correlation",
    notes="Scaling factor for turbulence intensity (u_turb = k * u_piston)",
)

DEFAULT_BURN_COEF_C = PhysicalConstant(
    value=3.0,
    unit="dimensionless",
    source="Clearance height scaling for burn duration",
    notes="Calibration coefficient for burn duration model",
)

TURBULENCE_EXPONENT = PhysicalConstant(
    value=0.7,
    unit="dimensionless",
    source="Turbulent flame speed power law",
    notes="Exponent for turbulence intensity in flame speed correlation",
)

MIN_FLAME_SPEED = PhysicalConstant(
    value=0.1,
    unit="m/s",
    source="Numerical stability lower bound",
    notes="Minimum allowable flame speed to prevent division by zero",
)

# =============================================================================
# Ignition & Flame Kinetics Constants
# =============================================================================

IGNITION_DELAY_PRE_EXP = PhysicalConstant(
    value=1e-6,
    unit="s",
    source="Simplified Arrhenius correlation",
    notes="Pre-exponential factor for ignition delay",
)

IGNITION_ACTIVATION_ENERGY = PhysicalConstant(
    value=15000.0,
    unit="J/mol",
    source="Simplified single-step kinetics",
    notes="Global activation energy for ignition",
)

FLAME_SPEED_REF_TEMP = PhysicalConstant(
    value=300.0,
    unit="K",
    source="Standard reference temperature",
    notes="Reference temperature for laminar flame speed correlation",
)

FLAME_SPEED_REF_PRESSURE = PhysicalConstant(
    value=100000.0,
    unit="Pa",
    source="Standard reference pressure",
    notes="Reference pressure for laminar flame speed correlation",
)

FLAME_SPEED_TEMP_EXPONENT = PhysicalConstant(
    value=2.0,
    unit="dimensionless",
    source="Metghalchi and Keck correlation (approx)",
    notes="Temperature exponent alpha for laminar flame speed",
)

FLAME_SPEED_PRESSURE_EXPONENT = PhysicalConstant(
    value=-0.5,
    unit="dimensionless",
    source="Metghalchi and Keck correlation (approx)",
    notes="Pressure exponent beta for laminar flame speed",
)
