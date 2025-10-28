"""
Mechanics modules for torque analysis and side-loading computation.

This package provides physics models for crank center optimization including
torque computation from motion law and gear geometry, side-loading analysis,
and kinematic analysis for crank center offset effects.
"""

from .side_loading import SideLoadAnalyzer, SideLoadResult
from .torque_analysis import PistonTorqueCalculator, TorqueAnalysisResult

__all__ = [
    # Torque analysis
    "PistonTorqueCalculator",
    "TorqueAnalysisResult",
    # Side-loading analysis
    "SideLoadAnalyzer",
    "SideLoadResult",
]
