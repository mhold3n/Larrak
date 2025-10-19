"""
Centralized IPOPT solver factory to prevent linear solver clobbering warnings.

This module ensures that the linear solver is only set once across all IPOPT
solver creations, preventing the "clobbering" warnings that occur when
multiple solvers try to set the same option.
"""

from __future__ import annotations

import casadi as ca
from typing import Any, Dict, Optional

from campro.constants import HSLLIB_PATH
from campro.logging import get_logger

log = get_logger(__name__)

# Global flag to track if linear solver has been set
_LINEAR_SOLVER_INITIALIZED = False


def create_ipopt_solver(
    name: str,
    nlp: Any,
    options: Optional[Dict[str, Any]] = None,
    force_linear_solver: bool = True
) -> Any:
    """
    Create an IPOPT solver with proper linear solver configuration.
    
    Args:
        name: Name for the solver instance
        nlp: NLP problem definition
        options: Additional IPOPT options
        force_linear_solver: Whether to force MA27 linear solver (default: True)
        
    Returns:
        CasADi IPOPT solver instance
    """
    global _LINEAR_SOLVER_INITIALIZED
    
    # Start with default options
    opts = options.copy() if options else {}
    
    # Set linear solver and HSL library only if not already initialized
    if force_linear_solver and not _LINEAR_SOLVER_INITIALIZED:
        opts["ipopt.linear_solver"] = "ma27"
        opts["ipopt.hsllib"] = HSLLIB_PATH
        _LINEAR_SOLVER_INITIALIZED = True
        log.debug(f"Setting linear solver to MA27 for solver '{name}'")
    elif force_linear_solver and _LINEAR_SOLVER_INITIALIZED:
        log.debug(f"Linear solver already initialized, skipping for solver '{name}'")
        # Don't set linear_solver again to avoid clobbering
    elif not force_linear_solver:
        log.debug(f"Not forcing linear solver for solver '{name}'")
    
    # Create the solver
    solver = ca.nlpsol(name, "ipopt", nlp, opts)
    
    return solver


def reset_linear_solver_flag() -> None:
    """Reset the linear solver initialization flag (for testing)."""
    global _LINEAR_SOLVER_INITIALIZED
    _LINEAR_SOLVER_INITIALIZED = False
    log.debug("Linear solver initialization flag reset")


def is_linear_solver_initialized() -> bool:
    """Check if the linear solver has been initialized."""
    return _LINEAR_SOLVER_INITIALIZED
