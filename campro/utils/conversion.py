"""
Unit conversion and transformation utilities.

This module provides functions for converting between different units
and coordinate systems used in motion law problems.
"""

from typing import Union

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


def convert_units(value: Union[float, np.ndarray],
                 from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    Convert values between different units.
    
    Args:
        value: Value(s) to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted value(s)
        
    Raises:
        ValueError: If conversion is not supported
    """
    # Define conversion factors
    conversions = {
        # Length conversions
        ("mm", "m"): 1e-3,
        ("m", "mm"): 1e3,
        ("mm", "in"): 0.0393701,
        ("in", "mm"): 25.4,

        # Time conversions
        ("s", "ms"): 1e3,
        ("ms", "s"): 1e-3,
        ("s", "min"): 1/60,
        ("min", "s"): 60,

        # Velocity conversions
        ("mm/s", "m/s"): 1e-3,
        ("m/s", "mm/s"): 1e3,
        ("mm/s", "in/s"): 0.0393701,
        ("in/s", "mm/s"): 25.4,

        # Acceleration conversions
        ("mm/s²", "m/s²"): 1e-3,
        ("m/s²", "mm/s²"): 1e3,
        ("mm/s²", "g"): 1e-3 / 9.80665,
        ("g", "mm/s²"): 9.80665 * 1e3,

        # Jerk conversions
        ("mm/s³", "m/s³"): 1e-3,
        ("m/s³", "mm/s³"): 1e3,

        # Angle conversions
        ("deg", "rad"): np.pi / 180,
        ("rad", "deg"): 180 / np.pi,
        ("deg", "rev"): 1/360,
        ("rev", "deg"): 360,
    }

    # Check if conversion is supported
    conversion_key = (from_unit, to_unit)
    if conversion_key not in conversions:
        raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")

    # Apply conversion
    factor = conversions[conversion_key]
    return value * factor


def time_to_cam_angle(time_array: np.ndarray, cycle_time: float = 1.0,
                     start_angle: float = 0.0) -> np.ndarray:
    """
    Convert time array to cam angle array.
    
    Args:
        time_array: Time array in seconds
        cycle_time: Total cycle time in seconds
        start_angle: Starting cam angle in degrees
        
    Returns:
        Cam angle array in degrees
    """
    # Normalize time to 0-1 range
    normalized_time = time_array / cycle_time

    # Convert to cam angle (0-360 degrees)
    cam_angle = start_angle + normalized_time * 360.0

    # Ensure angles are in 0-360 range
    cam_angle = cam_angle % 360.0

    return cam_angle


def cam_angle_to_time(cam_angle_array: np.ndarray, cycle_time: float = 1.0,
                     start_angle: float = 0.0) -> np.ndarray:
    """
    Convert cam angle array to time array.
    
    Args:
        cam_angle_array: Cam angle array in degrees
        cycle_time: Total cycle time in seconds
        start_angle: Starting cam angle in degrees
        
    Returns:
        Time array in seconds
    """
    # Normalize cam angle to 0-360 range
    normalized_angle = (cam_angle_array - start_angle) % 360.0

    # Convert to time
    time_array = normalized_angle / 360.0 * cycle_time

    return time_array


def rpm_to_angular_velocity(rpm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert RPM to angular velocity in rad/s.
    
    Args:
        rpm: Revolutions per minute
        
    Returns:
        Angular velocity in rad/s
    """
    return rpm * 2 * np.pi / 60


def angular_velocity_to_rpm(angular_velocity: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert angular velocity in rad/s to RPM.
    
    Args:
        angular_velocity: Angular velocity in rad/s
        
    Returns:
        RPM
    """
    return angular_velocity * 60 / (2 * np.pi)


def frequency_to_rpm(frequency: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert frequency in Hz to RPM.
    
    Args:
        frequency: Frequency in Hz
        
    Returns:
        RPM
    """
    return frequency * 60


def rpm_to_frequency(rpm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert RPM to frequency in Hz.
    
    Args:
        rpm: Revolutions per minute
        
    Returns:
        Frequency in Hz
    """
    return rpm / 60


def pressure_to_stress(pressure: Union[float, np.ndarray],
                      area: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert pressure to stress (force per unit area).
    
    Args:
        pressure: Pressure in Pa
        area: Area in m²
        
    Returns:
        Stress in Pa
    """
    return pressure / area


def stress_to_pressure(stress: Union[float, np.ndarray],
                      area: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert stress to pressure.
    
    Args:
        stress: Stress in Pa
        area: Area in m²
        
    Returns:
        Pressure in Pa
    """
    return stress * area


