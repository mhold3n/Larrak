"""
Motion Law Optimization Library

This package provides a comprehensive, reusable library for OP engine motion law
optimization. It encapsulates the complex optimization pipeline into a clean,
configurable API that can be used as part of larger optimization routines.

Key Features:
- Clean separation of concerns with modular architecture
- Multiple solver backends (IPOPT, robust IPOPT, adaptive)
- Flexible problem configuration with presets and scenarios
- Comprehensive result handling and validation
- Built-in post-processing and metrics computation
- Extensible architecture for custom objectives and constraints
- Extensive test coverage and examples

Main Components:
- MotionLawOptimizer: Main optimization class
- OptimizationConfig: Configuration management
- ConfigFactory: Factory for creating configurations
- Solver backends: IPOPT, robust, and adaptive solvers
- Result processing: Validation and metrics computation

Usage Examples:

Basic optimization:
    from campro.freepiston.opt import MotionLawOptimizer, ConfigFactory

    config = ConfigFactory.create_default_config()
    optimizer = MotionLawOptimizer(config)
    result = optimizer.optimize()

Quick optimization:
    from campro.freepiston.opt import quick_optimize, get_preset_config

    config = get_preset_config("high_performance")
    result = quick_optimize(config, backend="robust")

Custom configuration:
    from campro.freepiston.opt import create_engine_config, create_optimization_scenario

    config = create_engine_config("opposed_piston")
    config = create_optimization_scenario("efficiency", **config.__dict__)
    result = quick_optimize(config)
"""

from __future__ import annotations

# Configuration factory and presets
from .config_factory import (
    ConfigFactory,
    create_engine_config,
    create_optimization_scenario,
    get_preset_config,
)
from .cons import comprehensive_path_constraints

# Legacy imports for backward compatibility
from .driver import solve_cycle, solve_cycle_adaptive, solve_cycle_robust
from .ipopt_solver import IPOPTOptions, IPOPTResult, IPOPTSolver
from .nlp import build_collocation_nlp, build_collocation_nlp_with_1d_coupling
from .obj import get_objective_function

# Core optimization classes
from .optimization_lib import (
    AdaptiveBackend,
    IPOPTBackend,
    MotionLawOptimizer,
    OptimizationConfig,
    OptimizationResult,
    ProblemBuilder,
    ResultProcessor,
    RobustIPOPTBackend,
    create_adaptive_optimizer,
    create_robust_optimizer,
    create_standard_optimizer,
    quick_optimize,
)
from .solution import Solution

__all__ = [
    # Core optimization classes
    "OptimizationConfig",
    "OptimizationResult",
    "MotionLawOptimizer",
    "IPOPTBackend",
    "RobustIPOPTBackend",
    "AdaptiveBackend",
    "ProblemBuilder",
    "ResultProcessor",
    # Convenience functions
    "create_standard_optimizer",
    "create_robust_optimizer",
    "create_adaptive_optimizer",
    "quick_optimize",
    # Configuration factory
    "ConfigFactory",
    "get_preset_config",
    "create_engine_config",
    "create_optimization_scenario",
    # Legacy imports for backward compatibility
    "solve_cycle",
    "solve_cycle_robust",
    "solve_cycle_adaptive",
    "build_collocation_nlp",
    "build_collocation_nlp_with_1d_coupling",
    "get_objective_function",
    "comprehensive_path_constraints",
    "Solution",
    "IPOPTSolver",
    "IPOPTOptions",
    "IPOPTResult",
]

# Version information
__version__ = "1.0.0"
__author__ = "OP Engine Optimization Team"
__description__ = "Motion Law Optimization Library for OP Engines"
