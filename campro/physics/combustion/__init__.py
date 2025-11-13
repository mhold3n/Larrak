"""
Combustion physics utilities.

This package exposes the integrated combustion model used by the optimization
stack to derive burn profiles, mass-fraction-burned markers, and heat-release
rates from ignition decisions and cylinder state.
"""

from .model import CombustionModel, CombustionOutputs, CombustionConfig

__all__ = ["CombustionModel", "CombustionOutputs", "CombustionConfig"]
