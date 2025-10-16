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





