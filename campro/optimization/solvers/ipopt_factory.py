"""
Centralized IPOPT solver factory to prevent linear solver clobbering warnings.

This module ensures that the linear solver is only set once across all IPOPT
solver creations, preventing the "clobbering" warnings that occur when
multiple solvers try to set the same option.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import casadi as ca

from campro.constants import HSLLIB_PATH, IPOPT_OPT_PATH
from campro.diagnostics.ipopt_logger import ensure_runs_dir
from campro.diagnostics.run_metadata import RUN_ID
from campro.logging import get_logger

log = get_logger(__name__)

# Default linear solver - will be set based on available solvers
_DEFAULT_LINEAR_SOLVER: str | None = None
# Track whether hsllib has already been configured (via ipopt.opt or env flag)
_HSLLIB_LOCKED_EXTERNALLY: bool | None = None


def _get_default_linear_solver() -> str:
    """Get default linear solver, detecting available solvers if needed."""
    global _DEFAULT_LINEAR_SOLVER

    if _DEFAULT_LINEAR_SOLVER is not None:
        return _DEFAULT_LINEAR_SOLVER

    # Detect available solvers and prefer MA57 over MA27 when available
    try:
        from campro.environment.hsl_detector import detect_available_solvers

        available = detect_available_solvers(test_runtime=False)
        if available:
            # Prefer MA57 for better performance, fall back to MA27, then first available
            if "ma57" in available:
                _DEFAULT_LINEAR_SOLVER = "ma57"
                log.info(f"Default linear solver set to MA57 (available: {', '.join(available)})")
            elif "ma27" in available:
                _DEFAULT_LINEAR_SOLVER = "ma27"
                log.info(
                    f"Default linear solver set to MA27 (MA57 not available, found: {', '.join(available)})"
                )
            else:
                _DEFAULT_LINEAR_SOLVER = available[0]
                log.info(
                    f"Default linear solver set to {available[0]} "
                    f"(MA57/MA27 not available, found: {', '.join(available)})"
                )
        else:
            _DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
            log.warning("No HSL solvers detected; defaulting to MA27 (fallback)")
    except ImportError as e:
        _DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
        log.warning(f"hsl_detector not available ({e}); defaulting to MA27 (fallback)")
    except Exception as e:
        _DEFAULT_LINEAR_SOLVER = "ma27"  # Fallback
        log.warning(f"Error detecting available solvers ({e}); defaulting to MA27 (fallback)")

    return _DEFAULT_LINEAR_SOLVER


def _option_file_has_linear_solver() -> bool:
    """Check if ipopt.opt file contains linear_solver setting."""
    if not IPOPT_OPT_PATH:
        return False
    try:
        opt_path = Path(IPOPT_OPT_PATH)
        if not opt_path.exists():
            return False
        with open(opt_path, encoding="utf-8") as f:
            content = f.read().lower()
            # Check for linear_solver setting (with or without ipopt. prefix)
            return "linear_solver" in content
    except Exception:  # pragma: no cover - best effort only
        return False


def _option_file_has_hsllib() -> bool:
    """Check if ipopt.opt file contains hsllib setting."""
    if not IPOPT_OPT_PATH:
        return False
    try:
        opt_path = Path(IPOPT_OPT_PATH)
        if not opt_path.exists():
            return False
        with open(opt_path, encoding="utf-8") as f:
            content = f.read().lower()
            # Check for hsllib setting (IPOPT option file uses bare "hsllib" key)
            return "hsllib" in content
    except Exception:  # pragma: no cover - best effort only
        return False


def _hsllib_locked_externally() -> bool:
    """
    Determine if IPOPT already supplies hsllib (option file or env flag).

    IPOPT disallows resetting hsllib once configured (disallow_clobbering),
    so we skip programmatic overrides when ipopt.opt already defines it or
    when the user explicitly tells us it's managed upstream.
    """
    global _HSLLIB_LOCKED_EXTERNALLY
    if _HSLLIB_LOCKED_EXTERNALLY is not None:
        return _HSLLIB_LOCKED_EXTERNALLY

    explicit_skip = os.getenv("CAMPRO_ASSUME_HSLLIB_LOCKED", "").strip().lower()
    if explicit_skip in {"1", "true", "yes"}:
        _HSLLIB_LOCKED_EXTERNALLY = True
        log.debug(
            "Skipping ipopt.hsllib override (CAMPRO_ASSUME_HSLLIB_LOCKED=%s)",
            explicit_skip,
        )
        return True

    if _option_file_has_hsllib():
        _HSLLIB_LOCKED_EXTERNALLY = True
        log.debug(
            "ipopt.opt already defines hsllib; skipping programmatic override to avoid clobber warnings",
        )
        return True

    _HSLLIB_LOCKED_EXTERNALLY = False
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
    """Return IPOPT options configured with appropriate linear solver."""
    # Always set linear_solver programmatically for consistent behavior
    # The option file no longer contains linear_solver to avoid conflicts
    opts = options.copy() if options else {}

    # Remove any existing linear_solver from opts to ensure we set it consistently
    opts.pop("ipopt.linear_solver", None)

    # Determine which solver to use
    requested = linear_solver or opts.get("ipopt.linear_solver")
    default_solver = _get_default_linear_solver()

    # Use requested solver if provided, otherwise use default
    solver_to_use = (requested or default_solver).lower()

    # Validate solver is available (best effort)
    try:
        from campro.environment.hsl_detector import detect_available_solvers

        available = detect_available_solvers(test_runtime=False)
        if available and solver_to_use not in available:
            log.warning(
                f"Requested solver '{solver_to_use}' may not be available. "
                f"Available solvers: {', '.join(available)}. Using '{solver_to_use}' anyway."
            )
    except Exception:
        # If detection fails, proceed with requested/default solver
        pass

    opts["ipopt.linear_solver"] = solver_to_use

    if requested and requested.lower() != default_solver:
        log.debug(
            f"Using requested linear solver '{solver_to_use}' (default would be '{default_solver}')"
        )

    # Always set hsllib programmatically (no option file dependency)
    if not _hsllib_locked_externally():
        hsl_path = _resolve_hsl_path()
        if hsl_path:
            opts["ipopt.hsllib"] = hsl_path
            # Verbose diagnostics only when explicitly requested (for troubleshooting)
            import sys

            show_verbose_diagnostics = os.getenv("CAMPRO_DEBUG_HSL", "").lower() in (
                "1",
                "true",
                "yes",
            )

            if show_verbose_diagnostics:
                print(
                    f"[DEBUG ipopt_factory] Setting ipopt.hsllib programmatically: {hsl_path}",
                    file=sys.stderr,
                    flush=True,
                )
                log.debug(f"Setting ipopt.hsllib programmatically: {hsl_path}")
        else:
            log.warning("HSL library path not configured; IPOPT will rely on builtin defaults.")
    else:
        log.debug("ipopt.hsllib already configured upstream; leaving existing value in place")

    if enable_log_sink:
        try:
            ensure_runs_dir("/tmp/runs")
            opts.setdefault("ipopt.output_file", f"/tmp/runs/{RUN_ID}-ipopt.log")
            opts.setdefault("ipopt.print_level", 5)
            opts.setdefault("ipopt.file_print_level", 5)
        except Exception:  # pragma: no cover - logging only
            pass

    # Do not use option file - all configuration is programmatic

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
    return _get_default_linear_solver()


def set_default_linear_solver(solver: str) -> None:
    """Set the default linear solver for this machine."""
    global _DEFAULT_LINEAR_SOLVER

    solver_lower = solver.lower()

    # Validate solver is available
    try:
        from campro.environment.hsl_detector import detect_available_solvers

        available = detect_available_solvers(test_runtime=False)
        if available and solver_lower not in available:
            log.warning(
                f"Solver '{solver_lower}' may not be available. "
                f"Available solvers: {', '.join(available)}. "
                f"Setting default anyway."
            )
    except Exception:
        # If detection fails, proceed anyway
        pass

    _DEFAULT_LINEAR_SOLVER = solver_lower
    log.info(f"Default linear solver set to: {solver_lower}")
