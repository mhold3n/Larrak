"""Litvin planetary synthesis package."""

# CLI functionality
from campro.optimization.strategies.gear_synthesis import OptimResult, optimize_geometry

from .cli import main as cli_main
from .config import GeometrySearchConfig, OptimizationOrder, PlanetSynthesisConfig
from .metrics import Order0Metrics, evaluate_order0_metrics
from .motion import RadialSlotMotion
from .planetary_synthesis import PlanetToothProfile, synthesize_planet_from_motion

__all__ = [
    "GeometrySearchConfig",
    "OptimResult",
    "OptimizationOrder",
    "Order0Metrics",
    "PlanetSynthesisConfig",
    "PlanetToothProfile",
    "RadialSlotMotion",
    "cli_main",
    "evaluate_order0_metrics",
    "optimize_geometry",
    "synthesize_planet_from_motion",
]
