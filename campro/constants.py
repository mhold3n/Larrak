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





