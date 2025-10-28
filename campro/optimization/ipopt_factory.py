"""
Centralized IPOPT solver factory to prevent linear solver clobbering warnings.

This module ensures that the linear solver is only set once across all IPOPT
solver creations, preventing the "clobbering" warnings that occur when
multiple solvers try to set the same option.
"""

from __future__ import annotations

from typing import Any

import casadi as ca

from campro.constants import HSLLIB_PATH
from campro.diagnostics.ipopt_logger import ensure_runs_dir
from campro.diagnostics.run_metadata import RUN_ID
from campro.logging import get_logger

log = get_logger(__name__)

# Default linear solver for this machine
DEFAULT_LINEAR_SOLVER = "ma27"


def create_ipopt_solver(
    name: str,
    nlp: Any,
    options: dict[str, Any] | None = None,
    linear_solver: str | None = None,
) -> Any:
    """
    Create an IPOPT solver with explicit linear solver configuration.

    Args:
        name: Name for the solver instance
        nlp: NLP problem definition
        options: Additional IPOPT options
        linear_solver: Linear solver to use (default: ma27 for this machine)

    Returns:
        CasADi IPOPT solver instance
    """
    # Start with default options
    opts = options.copy() if options else {}

    # Set linear solver explicitly
    solver_to_use = linear_solver or DEFAULT_LINEAR_SOLVER
    opts["ipopt.linear_solver"] = solver_to_use

    # Set HSL library path if using MA27/MA57
    if solver_to_use in ["ma27", "ma57"]:
        opts["ipopt.hsllib"] = HSLLIB_PATH

    # Default Ipopt file sink per run (unless caller provided one)
    try:
        ensure_runs_dir("runs")
        opts.setdefault("ipopt.output_file", f"runs/{RUN_ID}-ipopt.log")
        opts.setdefault("ipopt.print_level", 5)
        opts.setdefault("ipopt.file_print_level", 5)
    except Exception:
        # If directory cannot be created, continue without file sink
        pass

    log.debug(f"Creating solver '{name}' with linear solver: {solver_to_use}")

    # Create the solver
    solver = ca.nlpsol(name, "ipopt", nlp, opts)

    return solver


def get_default_linear_solver() -> str:
    """Get the default linear solver for this machine."""
    return DEFAULT_LINEAR_SOLVER


def set_default_linear_solver(solver: str) -> None:
    """Set the default linear solver for this machine."""
    global DEFAULT_LINEAR_SOLVER
    DEFAULT_LINEAR_SOLVER = solver
    log.info(f"Default linear solver set to: {solver}")
