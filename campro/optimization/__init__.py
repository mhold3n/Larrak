"""
Motion Law Optimization Library
"""

from __future__ import annotations

# Active components
from .driver import solve_cycle, solve_cycle_adaptive, solve_cycle_robust
from .nlp import build_collocation_nlp

# Solvers are still valid
from .solvers.ipopt_solver import IPOPTOptions, IPOPTResult, IPOPTSolver

__all__ = [
    "IPOPTOptions",
    "IPOPTResult",
    "IPOPTSolver",
    "build_collocation_nlp",
    "solve_cycle",
    "solve_cycle_adaptive",
    "solve_cycle_robust",
]

# Version information
__version__ = "1.0.0"
__author__ = "OP Engine Optimization Team"
__description__ = "Motion Law Optimization Library for OP Engines"
