"""
Utility functions and helpers.

This module provides utility functions for plotting, validation,
conversion, and other common operations.
"""

from .conversion import cam_angle_to_time, convert_units, time_to_cam_angle
from .plotting import apply_smart_scaling, create_smart_scaled_plots, plot_solution
from .validation import validate_constraints, validate_inputs

__all__ = [
    # Plotting utilities
    "plot_solution",
    "create_smart_scaled_plots",
    "apply_smart_scaling",

    # Validation utilities
    "validate_inputs",
    "validate_constraints",

    # Conversion utilities
    "convert_units",
    "time_to_cam_angle",
    "cam_angle_to_time",
]


