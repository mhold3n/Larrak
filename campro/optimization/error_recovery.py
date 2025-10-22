from __future__ import annotations

"""Staged error-recovery orchestration for Ipopt solves.

The public entry point :func:`safe_solve` wraps an *optimizer callable* and
attempts a sequence of increasingly conservative retries when the first solve
raises an exception or returns a "non-converged" status.

It is deliberately agnostic to the specific NLP implementation – the caller
supplies a *factory* that, given override dictionaries, instantiates and
executes the solver returning an object with ``success`` and ``status``
attributes (compatible with SciPy / CasADi wrappers) or raises on failure.

Strategy order (configurable):
1. Relax line-search & tolerances.
2. Switch Hessian approximation mode.
3. Adjust barrier parameters.
4. Alternate HSL solver selection if available.
5. Final attempt with reduced print and CPU time.

Each stage logs parameters, elapsed time and outcome via project logger.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Sequence

from campro.logging import get_logger

log = get_logger(__name__)

__all__ = ["safe_solve"]


class SolveResultProtocol:  # pragma: no cover – used only for typing
    success: bool
    status: str
    info: Dict[str, Any]


# Type alias for a function that performs a solve given option overrides
SolveFn = Callable[[Dict[str, Any]], "SolveResultProtocol"]


class RetryStrategy:
    """Single retry strategy descriptor."""

    def __init__(self, name: str, overrides: Dict[str, Any] | None = None):
        self.name = name
        self.overrides = overrides or {}

    def __repr__(self) -> str:  # noqa: D401
        return f"<RetryStrategy {self.name}>"


_DEFAULT_STRATEGIES: Sequence[RetryStrategy] = [
    RetryStrategy(
        "relax_tol_line_search",
        {
            "ipopt.line_search_method": "cg-penalty-equality",
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.tol": 1e-6,
        },
    ),
    RetryStrategy(
        "switch_hessian",
        {
            "ipopt.hessian_approximation": "limited-memory",
        },
    ),
    RetryStrategy(
        "adjust_barrier",
        {
            "ipopt.mu_strategy": "monotone",
            "ipopt.mu_init": 0.01,
        },
    ),
    RetryStrategy(
        "alt_linear_solver",
        {},  # Note: linear_solver is set by the IPOPT factory
    ),
    RetryStrategy(
        "final_low_print",
        {
            "ipopt.print_level": 0,
            "ipopt.max_cpu_time": 60,
        },
    ),
]


class MaxRetriesExceeded(Exception):
    """Raised when all retry strategies fail."""


def safe_solve(
    solve_fn: SolveFn,
    *,
    base_options: Dict[str, Any],
    strategies: Sequence[RetryStrategy] | None = None,
) -> "SolveResultProtocol":
    """Attempt NLP solve with staged recovery.

    Parameters
    ----------
    solve_fn
        Callable receiving *options* dict and returning SolveResultProtocol or
        raising on catastrophic failure.
    base_options
        Initial options dict used for the first attempt and as the baseline for
        subsequent strategy overrides.
    strategies
        Custom retry strategies; defaults to built-in sequence.
    """

    strategies = list(strategies or _DEFAULT_STRATEGIES)

    attempt_opts = deepcopy(base_options)

    for idx, strat in enumerate([None, *strategies]):
        label = "initial" if strat is None else strat.name
        if strat is not None:
            attempt_opts.update(strat.overrides)

        log.info("Solve attempt %d (%s) with %d option overrides", idx, label, len(attempt_opts))
        t0 = time.perf_counter()
        try:
            result = solve_fn(attempt_opts)
        except Exception as exc:  # pylint: disable=broad-except
            elapsed = time.perf_counter() - t0
            log.warning("Solve attempt %s raised %s after %.3fs", label, exc.__class__.__name__, elapsed)
            continue  # proceed to next strategy

        elapsed = time.perf_counter() - t0
        if getattr(result, "success", False):
            log.info("Solve succeeded on attempt %s in %.3fs", label, elapsed)
            return result

        log.warning(
            "Solve attempt %s finished without convergence (status=%s) in %.3fs",
            label,
            getattr(result, "status", "unknown"),
            elapsed,
        )

    raise MaxRetriesExceeded("All retry strategies exhausted without success")
