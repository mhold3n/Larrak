"""
Geometry computation components.

This module provides components for computing geometric properties
of cam systems, including curves, curvature, and transformations.
"""

from .curvature import CurvatureComponent
from .curves import CamCurveComponent
from .transformations import CoordinateTransformComponent

__all__ = [
    "CamCurveComponent",
    "CoordinateTransformComponent",
    "CurvatureComponent",
]
