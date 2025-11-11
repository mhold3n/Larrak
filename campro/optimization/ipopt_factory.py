"""
Centralized IPOPT solver factory to prevent linear solver clobbering warnings.

This module ensures that the linear solver is only set once across all IPOPT
solver creations, preventing the "clobbering" warnings that occur when
multiple solvers try to set the same option.
"""

from __future__ import annotations

import platform
from typing import Any, Dict

import casadi as ca

from campro.constants import HSLLIB_PATH, IPOPT_OPT_PATH
from campro.diagnostics.ipopt_logger import ensure_runs_dir
from campro.diagnostics.run_metadata import RUN_ID
from campro.logging import get_logger

log = get_logger(__name__)

# Detect macOS platform
IS_MACOS = platform.system().lower() == "darwin"

# Default linear solver for this machine
DEFAULT_LINEAR_SOLVER = "ma27"

_MA57_SUPPORT_CACHE: Dict[str, bool] = {}


def _supports_ma57(hsl_path: str) -> bool:
    """Return True if the provided HSL library exposes MA57 symbols and works at runtime.
    
    Performs both symbol existence check and runtime verification by attempting
    to create and use an Ipopt solver with MA57.
    """
    cached = _MA57_SUPPORT_CACHE.get(hsl_path)
    if cached is not None:
        return cached
    try:
        import ctypes

        # First check: symbol existence
        lib = ctypes.CDLL(hsl_path)
        for symbol in ("ma57ad_", "ma57cd_", "ma57id_"):
            getattr(lib, symbol)
        
        # Second check: runtime verification via CasADi/Ipopt
        try:
            import casadi as ca
            x = ca.SX.sym("x")
            nlp = {"x": x, "f": x * x}
            # Try to create solver with MA57
            solver = ca.nlpsol(
                "_ma57_runtime_test",
                "ipopt",
                nlp,
                {
                    "ipopt.linear_solver": "ma57",
                    "ipopt.hsllib": hsl_path,
                    "ipopt.print_level": 0,
                    "ipopt.sb": "yes",
                },
            )
            # Try to solve a simple problem
            result = solver(x0=0.0, lbx=-10.0, ubx=10.0)
            # If we get here without exception, MA57 works
            _MA57_SUPPORT_CACHE[hsl_path] = True
            return True
        except Exception as runtime_exc:
            log.debug(f"MA57 runtime verification failed for {hsl_path}: {runtime_exc}")
            _MA57_SUPPORT_CACHE[hsl_path] = False
            return False
    except (OSError, AttributeError) as symbol_exc:
        log.debug(f"MA57 symbol check failed for {hsl_path}: {symbol_exc}")
        _MA57_SUPPORT_CACHE[hsl_path] = False
        return False


def build_ipopt_solver_options(
    options: dict[str, Any] | None = None,
    linear_solver: str | None = None,
    *,
    enable_log_sink: bool = True,
) -> dict[str, Any]:
    """Return IPOPT options with centralized linear-solver wiring."""
    opts = options.copy() if options else {}

    solver_to_use = linear_solver or opts.get("ipopt.linear_solver") or DEFAULT_LINEAR_SOLVER
    solver_lower = solver_to_use.lower()

    # Guard against MA97 on macOS – fall back to MA57 as in create_ipopt_solver
    if IS_MACOS and solver_lower == "ma97":
        log.warning(
            "MA97 solver is not supported on macOS due to known crash bug. "
            "Falling back to MA57 for large problems.",
        )
        solver_to_use = "ma57"
        solver_lower = "ma57"

    has_option_file = bool(IPOPT_OPT_PATH)

    # Inject hsllib path for HSL solvers (ma27/ma57/ma77/ma86)
    if solver_lower in {"ma27", "ma57", "ma77", "ma86"}:
        hsl_path = HSLLIB_PATH
        if not hsl_path:
            try:
                from campro.environment.env_manager import find_hsl_library

                detected = find_hsl_library()
                if detected:
                    hsl_path = str(detected)
            except Exception:
                hsl_path = None

        if solver_lower == "ma57":
            if not hsl_path:
                log.warning(
                    "MA57 requested but no HSL library found. Falling back to MA27.",
                )
                solver_to_use = "ma27"
                solver_lower = "ma27"
            elif not _supports_ma57(hsl_path):
                log.warning(
                    "HSL library %s does not expose MA57 symbols. Falling back to MA27.",
                    hsl_path,
                )
                solver_to_use = "ma27"
                solver_lower = "ma27"

        if solver_lower in {"ma27", "ma77", "ma86"} and not hsl_path:
            if has_option_file:
                log.debug(
                    "No HSLLIB_PATH detected for %s; deferring to ipopt.opt: %s",
                    solver_to_use,
                    IPOPT_OPT_PATH,
                )
            else:
                log.warning(
                    "Requested HSL solver '%s' but libcoinhsl was not found. "
                    "Falling back to MUMPS.",
                    solver_to_use,
                )
                solver_to_use = "mumps"
                solver_lower = "mumps"

        if hsl_path and solver_lower in {"ma27", "ma57", "ma77", "ma86"}:
            opts["ipopt.hsllib"] = hsl_path

    opts["ipopt.linear_solver"] = solver_to_use

    # Keep log sink / output files consistent with IPOPT factory defaults
    if enable_log_sink:
        try:
            ensure_runs_dir("runs")
            opts.setdefault("ipopt.output_file", f"runs/{RUN_ID}-ipopt.log")
            opts.setdefault("ipopt.print_level", 5)
            opts.setdefault("ipopt.file_print_level", 5)
        except Exception:
            pass

    if has_option_file:
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
    DEFAULT_LINEAR_SOLVER = solver
    log.info(f"Default linear solver set to: {solver}")
