"""
Input validation utilities.

This module provides functions for validating inputs, constraints,
and other data structures used throughout the campro library.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


def validate_inputs(inputs: Dict[str, Any], required_keys: List[str],
                   optional_keys: Optional[List[str]] = None) -> bool:
    """
    Validate input dictionary has required keys.
    
    Args:
        inputs: Dictionary to validate
        required_keys: List of required keys
        optional_keys: List of optional keys (for validation)
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(inputs, dict):
        raise ValueError("Inputs must be a dictionary")
    
    # Check required keys
    missing_keys = [key for key in required_keys if key not in inputs]
    if missing_keys:
        raise ValueError(f"Missing required keys: {missing_keys}")
    
    # Check for unexpected keys
    if optional_keys is not None:
        all_valid_keys = set(required_keys) | set(optional_keys)
        unexpected_keys = [key for key in inputs.keys() if key not in all_valid_keys]
        if unexpected_keys:
            log.warning(f"Unexpected keys found: {unexpected_keys}")
    
    return True


def validate_constraints(constraints: Any) -> bool:
    """
    Validate constraint object.
    
    Args:
        constraints: Constraint object to validate
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if constraints is None:
        raise ValueError("Constraints cannot be None")
    
    # Check if constraints have validate method
    if hasattr(constraints, 'validate'):
        return constraints.validate()
    else:
        log.warning("Constraint object does not have validate method")
        return True


def validate_numeric_range(value: Union[int, float], min_val: Optional[float] = None,
                          max_val: Optional[float] = None, name: str = "value") -> bool:
    """
    Validate numeric value is within specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value)}")
    
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
    
    return True


def validate_array_shape(array: np.ndarray, expected_shape: Tuple[int, ...],
                        name: str = "array") -> bool:
    """
    Validate array has expected shape.
    
    Args:
        array: Array to validate
        expected_shape: Expected shape tuple
        name: Name of the array for error messages
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{name} must be numpy array, got {type(array)}")
    
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    
    return True


def validate_positive(value: Union[int, float], name: str = "value") -> bool:
    """
    Validate value is positive.
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    return validate_numeric_range(value, min_val=0.0, name=name)


def validate_percentage(value: Union[int, float], name: str = "percentage") -> bool:
    """
    Validate value is a valid percentage (0-100).
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    return validate_numeric_range(value, min_val=0.0, max_val=100.0, name=name)


