"""
Constraint definitions and validation for cam motion law problems.

This module provides a comprehensive constraint system for defining
motion law constraints, cam-specific constraints, and future physics constraints.
"""

from .base import BaseConstraints, ConstraintType, ConstraintViolation
from .motion import MotionConstraints, MotionConstraintType
from .cam import CamMotionConstraints, CamConstraintType

__all__ = [
    # Base classes
    'BaseConstraints',
    'ConstraintType', 
    'ConstraintViolation',
    
    # Motion constraints
    'MotionConstraints',
    'MotionConstraintType',
    
    # Cam constraints
    'CamMotionConstraints',
    'CamConstraintType',
]


