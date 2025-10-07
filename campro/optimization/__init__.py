"""
Optimization routines and solvers for motion law problems.

This module provides a comprehensive optimization framework for solving
motion law problems using various methods including collocation and
future physics-based optimization.
"""

from .base import BaseOptimizer, OptimizationResult, OptimizationStatus
from .collocation import CollocationSettings, CollocationMethod, CollocationOptimizer
from .motion import MotionOptimizer, MotionObjectiveType
from .secondary import SecondaryOptimizer
from .tertiary import TertiaryOptimizer, LinkageParameters
from .cam_ring_optimizer import CamRingOptimizer, CamRingOptimizationConstraints, CamRingOptimizationTargets
from .sun_gear_optimizer import SunGearOptimizer, SunGearParameters, SunGearOptimizationConstraints, SunGearOptimizationTargets
from .unified_framework import (
    UnifiedOptimizationFramework, 
    UnifiedOptimizationSettings, 
    UnifiedOptimizationConstraints, 
    UnifiedOptimizationTargets,
    UnifiedOptimizationData,
    OptimizationMethod,
    OptimizationLayer
)
from .motion_law import (
    MotionLawConstraints,
    MotionLawResult,
    MotionLawValidator,
    MotionType
)
from .motion_law_optimizer import MotionLawOptimizer
from .cam_ring_processing import (
    process_linear_to_ring_follower,
    process_ring_optimization,
    process_multi_objective_ring_design,
    create_constant_ring_design,
    create_optimized_ring_design
)

__all__ = [
    # Base classes
    'BaseOptimizer',
    'OptimizationResult',
    'OptimizationStatus',
    
    # Collocation
    'CollocationSettings',
    'CollocationMethod',
    'CollocationOptimizer',
    
    # Motion optimization
    'MotionOptimizer',
    'MotionObjectiveType',
    
    # Secondary optimization
    'SecondaryOptimizer',
    
    # Tertiary optimization
    'TertiaryOptimizer',
    'LinkageParameters',
    
    # Cam-ring optimization
    'CamRingOptimizer',
    'CamRingOptimizationConstraints',
    'CamRingOptimizationTargets',
    
    # Sun gear optimization
    'SunGearOptimizer',
    'SunGearParameters',
    'SunGearOptimizationConstraints',
    'SunGearOptimizationTargets',
    
    # Unified optimization framework
    'UnifiedOptimizationFramework',
    'UnifiedOptimizationSettings',
    'UnifiedOptimizationConstraints',
    'UnifiedOptimizationTargets',
    'UnifiedOptimizationData',
    'OptimizationMethod',
    'OptimizationLayer',
    
    # Motion law optimization
    'MotionLawConstraints',
    'MotionLawResult',
    'MotionLawValidator',
    'MotionType',
    'MotionLawOptimizer',
    
    # Cam-ring processing
    'process_linear_to_ring_follower',
    'process_ring_optimization',
    'process_multi_objective_ring_design',
    'create_constant_ring_design',
    'create_optimized_ring_design',
]
