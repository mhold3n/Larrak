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
from math import pi as _pi

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
# Note: these paths are user-environment specific and centralized here to avoid
# duplication across modules. Update as needed for the active environment.
HSLLIB_PATH: str = "/Users/maxholden/anaconda3/envs/larrak/lib/libcoinhsl.dylib"
IPOPT_OPT_PATH: str = "/Users/maxholden/Documents/GitHub/Larrak/ipopt.opt"

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
