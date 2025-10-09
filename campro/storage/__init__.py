"""
Result storage and sharing system.

This module provides a centralized storage system for optimization results,
enabling different optimizers to share and access results from previous
optimization runs.
"""

from .base import BaseStorage, StorageResult, StorageStatus
from .memory import MemoryStorage
from .registry import OptimizationRegistry

__all__ = [
    # Base classes
    "BaseStorage",
    "StorageResult",
    "StorageStatus",

    # Storage implementations
    "MemoryStorage",

    # Registry system
    "OptimizationRegistry",
]


