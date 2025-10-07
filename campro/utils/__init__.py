"""
Utility functions and helpers.

This module provides utility functions for plotting, validation,
conversion, and other common operations.
"""

from .plotting import plot_solution, create_smart_scaled_plots, apply_smart_scaling
from .validation import validate_inputs, validate_constraints
from .conversion import convert_units, time_to_cam_angle, cam_angle_to_time

__all__ = [
    # Plotting utilities
    'plot_solution',
    'create_smart_scaled_plots',
    'apply_smart_scaling',
    
    # Validation utilities
    'validate_inputs',
    'validate_constraints',
    
    # Conversion utilities
    'convert_units',
    'time_to_cam_angle',
    'cam_angle_to_time',
]


