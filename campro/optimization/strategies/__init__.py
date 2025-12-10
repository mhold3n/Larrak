"""
Optimization strategies for different types of problems.

This module provides a framework for implementing different optimization
strategies.
"""

from .base_strategy import BaseOptimizationStrategy, OptimizationStrategyResult

# from .gear_synthesis import optimize_geometry  # Can export if needed, but safe to import directly

__all__ = [
    "BaseOptimizationStrategy",
    "OptimizationStrategyResult",
]
