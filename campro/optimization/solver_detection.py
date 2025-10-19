from __future__ import annotations

"""Runtime detection utilities for linear solvers.

This module currently provides a cached probe for MA57 availability in the
installed CasADi + Ipopt environment. The probe creates a tiny dummy NLP and
attempts to instantiate an Ipopt solver with ``ipopt.linear_solver = 'ma57'``.

The result is cached to avoid repeated expensive checks.  Clients should call
:func:`is_ma57_available` exactly *once* at application start-up and reuse the
returned boolean for the lifetime of the process.

The implementation is intentionally **side-effect free**: it never raises on
failure and logs the outcome via the project logger.
"""

from typing import Final

import casadi as ca

from campro.logging import get_logger

__all__: Final = ["is_ma57_available"]

log = get_logger(__name__)

# Cache sentinel: None → not probed yet, otherwise bool result
_ma57_available: bool | None = None


def is_ma57_available() -> bool:  # noqa: D401 – simple function, no docstring needed
    """Return *True* if MA57 linear solver is usable, else *False*.

    The function performs a single probe on first invocation and subsequently
    returns the cached result.
    """

    global _ma57_available  # pylint: disable=global-statement

    if _ma57_available is not None:
        return _ma57_available

    try:
        # Minimal NLP: f(x) = x**2 with single SX variable.
        x = ca.SX.sym("x")
        nlp = {"x": x, "f": x * x}

        # Attempt to create an Ipopt solver with MA57 using factory
        from campro.optimization.ipopt_factory import create_ipopt_solver
        create_ipopt_solver("_ma57_probe", nlp, {"ipopt.linear_solver": "ma57"}, force_linear_solver=False)

        _ma57_available = True
    except Exception as exc:  # pylint: disable=broad-except
        log.debug("MA57 detection failed: %s", exc)
        _ma57_available = False

    log.info("MA57 available: %s", _ma57_available)
    return _ma57_available
