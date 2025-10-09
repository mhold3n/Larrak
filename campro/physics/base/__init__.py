"""
Base physics components and interfaces.

This module provides the foundational interfaces and classes for building
modular, adaptable physics systems. All physics components inherit from
these base classes to ensure consistent interfaces and behavior.
"""

from .component import BaseComponent, ComponentResult, ComponentStatus
from .model import BasePhysicsModel
from .result import PhysicsResult, PhysicsStatus
from .system import BaseSystem, SystemResult, SystemStatus

__all__ = [
    # Component interfaces
    "BaseComponent",
    "ComponentResult",
    "ComponentStatus",

    # System interfaces
    "BaseSystem",
    "SystemResult",
    "SystemStatus",

    # Result types
    "PhysicsResult",
    "PhysicsStatus",

    # Physics model base class
    "BasePhysicsModel",
]
