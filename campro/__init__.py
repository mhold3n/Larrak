"""Larrak: Optimal motion law problems using CasADi and Ipopt collocation."""
from __future__ import annotations

import os

from campro.diagnostics.run_metadata import set_global_seeds
from campro.logging import get_logger

__version__ = "0.1.0"

log = get_logger(__name__)

# Global flag to track ipopt availability
_IPOPT_AVAILABLE: bool | None = None


def _check_ipopt_availability() -> bool:
    """Check if ipopt is available in CasADi (fast, import-time friendly)."""
    global _IPOPT_AVAILABLE

    if _IPOPT_AVAILABLE is not None:
        return _IPOPT_AVAILABLE

    # Skip validation if explicitly disabled
    if os.getenv("CAMPRO_SKIP_VALIDATION") == "1":
        _IPOPT_AVAILABLE = True
        return _IPOPT_AVAILABLE

    try:
        import casadi as ca

        _IPOPT_AVAILABLE = False
        plugins = None
        if hasattr(ca, "nlpsol_plugins"):
            try:
                plugins = ca.nlpsol_plugins()
                if "ipopt" in plugins:
                    _IPOPT_AVAILABLE = True
            except Exception:
                plugins = None

        if not _IPOPT_AVAILABLE:
            # Try a direct instantiation as a secondary check
            try:
                x = ca.SX.sym("x")
                f = x**2
                # Use the centralized factory with explicit linear solver
                from campro.optimization.ipopt_factory import create_ipopt_solver

                create_ipopt_solver(
                    "ipopt_probe", {"x": x, "f": f}, linear_solver="ma27",
                )
                _IPOPT_AVAILABLE = True
                log.info("IPOPT availability check completed with MA27 linear solver")
            except Exception as e:
                log.error(f"IPOPT availability check failed: {e}")
                _IPOPT_AVAILABLE = False

        if not _IPOPT_AVAILABLE:
            log.warning(
                "IPOPT solver is not available in CasADi. Run 'python scripts/check_environment.py' "
                "for details or 'python scripts/setup_environment.py' to install.",
            )
    except Exception as exc:
        _IPOPT_AVAILABLE = False
        log.warning(
            "IPOPT availability check failed: %s. Run 'python scripts/check_environment.py' for details.",
            exc,
        )

    return _IPOPT_AVAILABLE


def is_ipopt_available() -> bool:
    """
    Check if ipopt solver is available.

    Returns:
        True if ipopt is available, False otherwise
    """
    return _check_ipopt_availability()


# Perform lightweight validation on import
_check_ipopt_availability()

# Ensure deterministic RNG state for reproducibility at import time.
try:
    set_global_seeds()
except Exception:
    # Seeding should never fail import; ignore in edge environments.
    pass
