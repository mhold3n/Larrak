"""
Physics simulation modules for combustion and thermodynamics.

This module provides the foundation for physics-based optimization
including combustion simulation, valve timing, cylinder pressure analysis,
and cam-ring-linear follower mapping.
"""

from .base import (
    BaseComponent,
    BasePhysicsModel,
    BaseSystem,
    PhysicsResult,
    PhysicsStatus,
)
from .cam_ring_mapping import CamRingMapper, CamRingParameters
from .geometry.litvin import LitvinGearGeometry, LitvinSynthesis, LitvinSynthesisResult
from .kinematics.crank_kinematics import CrankKinematics, CrankKinematicsResult
from .mechanics.side_loading import SideLoadAnalyzer, SideLoadResult
from .mechanics.torque_analysis import PistonTorqueCalculator, TorqueAnalysisResult

__all__ = [
    # Base classes
    "BaseComponent",
    "BaseSystem",
    "BasePhysicsModel",
    "PhysicsResult",
    "PhysicsStatus",
    # Cam-ring mapping
    "CamRingMapper",
    "CamRingParameters",
    "LitvinSynthesis",
    "LitvinSynthesisResult",
    "LitvinGearGeometry",
    # Mechanics modules
    "PistonTorqueCalculator",
    "TorqueAnalysisResult",
    "SideLoadAnalyzer",
    "SideLoadResult",
    # Kinematics modules
    "CrankKinematics",
    "CrankKinematicsResult",
    # Future modules will be added here:
    # 'CombustionModel',
    # 'ThermodynamicsModel',
    # 'ValveTimingModel',
]
