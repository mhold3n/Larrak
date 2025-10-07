"""
Geometry computation components.

This module provides components for computing geometric properties
of cam systems, including curves, curvature, and transformations.
"""

from .curves import CamCurveComponent
from .curvature import CurvatureComponent
from .transformations import CoordinateTransformComponent

__all__ = [
    'CamCurveComponent',
    'CurvatureComponent', 
    'CoordinateTransformComponent',
]

