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

import time  # noqa: E402
from collections.abc import (
    Callable,
    Sequence,
)
from copy import deepcopy  # noqa: E402
from typing import Any

from campro.logging import get_logger  # noqa: E402

log = get_logger(__name__)

__all__ = ["MaxRetriesExceeded", "RetryStrategy", "safe_solve"]


class SolveResultProtocol:  # pragma: no cover – used only for typing
    success: bool
    status: str
    info: dict[str, Any]


# Type alias for a function that performs a solve given option overrides
SolveFn = Callable[[dict[str, Any]], "SolveResultProtocol"]


class RetryStrategy:
    """Single retry strategy descriptor."""

    def __init__(self, name: str, overrides: dict[str, Any] | None = None):
        self.name = name
        self.overrides = overrides or {}

    def __repr__(self) -> str:
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


class MaxRetriesExceededError(Exception):
    """Raised when all retry strategies fail."""


MaxRetriesExceeded = MaxRetriesExceededError


def safe_solve(
    solve_fn: SolveFn,
    *,
    base_options: dict[str, Any],
    strategies: Sequence[RetryStrategy] | None = None,
    use_default_strategies: bool = False,
) -> SolveResultProtocol:
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
        Custom retry strategies; defaults to none (single attempt).
    use_default_strategies
        When True and *strategies* is None, enable the built-in retry sequence.
    """

    if strategies is None:
        strategy_list = list(_DEFAULT_STRATEGIES) if use_default_strategies else []
    else:
        strategy_list = list(strategies)

    attempt_opts = deepcopy(base_options)

    for idx, strat in enumerate([None, *strategy_list]):
        label = "initial" if strat is None else strat.name
        if strat is not None:
            attempt_opts.update(strat.overrides)

        log.info(
            "Solve attempt %d (%s) with %d option overrides",
            idx,
            label,
            len(attempt_opts),
        )
        t0 = time.perf_counter()
        try:
            result = solve_fn(attempt_opts)
        except Exception as exc:  # pylint: disable=broad-except
            elapsed = time.perf_counter() - t0
            log.warning(
                "Solve attempt %s raised %s after %.3fs",
                label,
                exc.__class__.__name__,
                elapsed,
            )
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

    raise MaxRetriesExceededError("All retry strategies exhausted without success")
