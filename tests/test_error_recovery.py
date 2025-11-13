from types import SimpleNamespace

import pytest

from campro.optimization.error_recovery import (
    MaxRetriesExceeded,
    RetryStrategy,
    safe_solve,
)


class _CounterSolve:
    """Mock solve function succeeding after *n_fail* attempts."""

    def __init__(self, n_fail: int):
        self.n_fail = n_fail
        self.calls = 0

    def __call__(self, _opts):
        self.calls += 1
        if self.calls <= self.n_fail:
            raise RuntimeError("Synthetic failure")
        return SimpleNamespace(success=True, status="optimal", info={})


def test_safe_solve_succeeds_after_retries():
    solve = _CounterSolve(n_fail=2)
    result = safe_solve(solve, base_options={}, use_default_strategies=True)
    assert result.success is True
    assert solve.calls == 3  # initial + 2 retries


def test_safe_solve_exhausts_strategies():
    solve = _CounterSolve(n_fail=10)
    with pytest.raises(MaxRetriesExceeded):
        safe_solve(solve, base_options={}, strategies=[RetryStrategy("dummy")])


def test_safe_solve_without_default_retries_fails_fast():
    solve = _CounterSolve(n_fail=1)
    with pytest.raises(MaxRetriesExceeded):
        safe_solve(solve, base_options={})
