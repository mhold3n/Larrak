"""
Optimization strategies for different types of problems.

This module provides a framework for implementing different optimization
strategies, enabling easy extension and modification of optimization approaches.
"""

from .base_strategy import BaseOptimizationStrategy, OptimizationStrategyResult
from .motion_strategy import MotionOptimizationStrategy
from .geometry_strategy import GeometryOptimizationStrategy
from .multi_objective import MultiObjectiveStrategy

__all__ = [
    # Base strategy
    'BaseOptimizationStrategy',
    'OptimizationStrategyResult',
    
    # Specific strategies
    'MotionOptimizationStrategy',
    'GeometryOptimizationStrategy', 
    'MultiObjectiveStrategy',
]

