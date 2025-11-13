"""
Centralized IPOPT solver factory to prevent linear solver clobbering warnings.

This module ensures that the linear solver is only set once across all IPOPT
solver creations, preventing the "clobbering" warnings that occur when
multiple solvers try to set the same option.
"""

from __future__ import annotations

from typing import Any

import casadi as ca

from pathlib import Path

from campro.constants import HSLLIB_PATH, IPOPT_OPT_PATH
from campro.diagnostics.ipopt_logger import ensure_runs_dir
from campro.diagnostics.run_metadata import RUN_ID
from campro.logging import get_logger

log = get_logger(__name__)

# Default linear solver for this machine (MA27 only)
DEFAULT_LINEAR_SOLVER = "ma27"


def _option_file_has_linear_solver() -> bool:
    """Check if ipopt.opt file contains linear_solver setting."""
    if not IPOPT_OPT_PATH:
        return False
    try:
        opt_path = Path(IPOPT_OPT_PATH)
        if not opt_path.exists():
            return False
        with open(opt_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
            # Check for linear_solver setting (with or without ipopt. prefix)
            return "linear_solver" in content
    except Exception:  # pragma: no cover - best effort only
        return False


def _resolve_hsl_path() -> str | None:
    """Return the configured HSL library path if available."""
    if HSLLIB_PATH:
        return HSLLIB_PATH
    try:
        from campro.environment.env_manager import find_hsl_library

        detected = find_hsl_library()
        if detected:
            return str(detected)
    except Exception:  # pragma: no cover - best effort only
        pass
    return None


def build_ipopt_solver_options(
    options: dict[str, Any] | None = None,
    linear_solver: str | None = None,
    *,
    enable_log_sink: bool = True,
) -> dict[str, Any]:
    """Return IPOPT options configured for MA27."""
    # Always set linear_solver programmatically for consistent behavior
    # The option file no longer contains linear_solver to avoid conflicts
    opts = options.copy() if options else {}
    
    # Remove any existing linear_solver from opts to ensure we set it consistently
    opts.pop("ipopt.linear_solver", None)
    
    # Always set linear_solver programmatically
    requested = linear_solver or opts.get("ipopt.linear_solver")
    if requested and requested.lower() != DEFAULT_LINEAR_SOLVER:
        log.info(
            "Forcing IPOPT linear solver to %s (requested '%s')",
            DEFAULT_LINEAR_SOLVER,
            requested,
        )
    opts["ipopt.linear_solver"] = DEFAULT_LINEAR_SOLVER

    hsl_path = _resolve_hsl_path()
    if hsl_path:
        opts["ipopt.hsllib"] = hsl_path
    else:
        log.warning(
            "HSL library path not configured; IPOPT will rely on builtin defaults."
        )

    if enable_log_sink:
        try:
            ensure_runs_dir("runs")
            opts.setdefault("ipopt.output_file", f"runs/{RUN_ID}-ipopt.log")
            opts.setdefault("ipopt.print_level", 5)
            opts.setdefault("ipopt.file_print_level", 5)
        except Exception:  # pragma: no cover - logging only
            pass

    if IPOPT_OPT_PATH:
        opts.setdefault("ipopt.option_file_name", IPOPT_OPT_PATH)

    return opts

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
    opts = build_ipopt_solver_options(options, linear_solver)

    log.debug(f"Creating solver '{name}' with linear solver: {opts.get('ipopt.linear_solver')}")

    # Create the solver
    solver = ca.nlpsol(name, "ipopt", nlp, opts)

    return solver


def get_default_linear_solver() -> str:
    """Get the default linear solver for this machine."""
    return DEFAULT_LINEAR_SOLVER


def set_default_linear_solver(solver: str) -> None:
    """Set the default linear solver for this machine."""
    global DEFAULT_LINEAR_SOLVER
    if solver.lower() != DEFAULT_LINEAR_SOLVER:
        log.warning(
            "Only MA27 is supported; ignoring request to set linear solver to '%s'.",
            solver,
        )
        DEFAULT_LINEAR_SOLVER = "ma27"
    else:
        DEFAULT_LINEAR_SOLVER = "ma27"
    log.info("Default linear solver set to: ma27")
