"""
Physics simulation modules for combustion and thermodynamics.

This module provides the foundation for physics-based optimization
including combustion simulation, valve timing, cylinder pressure analysis,
and cam-ring-linear follower mapping.
"""

from .base import BaseComponent, BaseSystem, PhysicsResult, PhysicsStatus
from .cam_ring_mapping import CamRingMapper, CamRingParameters

__all__ = [
    # Base classes
    'BaseComponent',
    'BaseSystem',
    'PhysicsResult',
    'PhysicsStatus',
    
    # Cam-ring mapping
    'CamRingMapper',
    'CamRingParameters',
    
    # Future modules will be added here:
    # 'CombustionModel',
    # 'ThermodynamicsModel',
    # 'ValveTimingModel',
]
