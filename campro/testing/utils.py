import multiprocessing
import platform
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np


@dataclass
class SolverResult:
    """Standardized result object for solver tests."""

    success: bool
    status: str
    f: float
    x: np.ndarray
    max_constr_violation: float
    kkt_error: float
    metadata: dict[str, Any]


def check_gradients(
    build_nlp: Callable[[Any], dict[str, Any]],
    x0: np.ndarray,
    p: np.ndarray | None = None,
    eps: float = 1e-6,
    tol: float = 1e-4,
) -> None:
    """
    Verify CasADi analytical gradients against finite differences.

    Args:
        build_nlp: Function calling the NLP builder, returning dict with 'x', 'f', 'g'
        x0: Initial guess / linearization point
        p: Parameters (optional)
        eps: Finite difference step size
        tol: Tolerance for comparison
    """
    nlp = build_nlp(x0, p) if p is not None else build_nlp(x0)

    if hasattr(nlp, "f") and hasattr(nlp, "x"):
        f_sym = nlp.f
        x_sym = nlp.x
    else:
        f_sym = nlp["f"]
        x_sym = nlp["x"]

    grad_sym = ca.gradient(f_sym, x_sym)
    grad_f = ca.Function("grad_f", [x_sym], [grad_sym])

    g_auto = np.array(grad_f(x0)).ravel()

    # Finite Differences
    f_func = ca.Function("f_func", [x_sym], [f_sym])
    g_fd = []

    for i in range(len(x0)):
        e = np.zeros_like(x0)
        e[i] = eps
        f_plus = float(f_func(x0 + e))
        f_minus = float(f_func(x0 - e))
        g_fd.append((f_plus - f_minus) / (2 * eps))

    g_fd = np.array(g_fd)

    diff = np.abs(g_auto - g_fd)
    max_diff = np.max(diff)

    if max_diff > tol:
        bad_indices = np.where(diff > tol)[0]
        msg = f"Gradient mismatch! Max error: {max_diff:.2e} > tol {tol}\n"
        raise AssertionError(msg)


def assert_solver_success(result: SolverResult, tol: float = 1e-5) -> None:
    """Standard assertions for a successful optimization run."""
    assert result.success, f"Solver failed with status: {result.status}"
    assert result.max_constr_violation < tol, (
        f"Constraint violation {result.max_constr_violation} > {tol}"
    )


def _isolated_wrapper(temp_filename: str, target_func: Callable, args: tuple, kwargs: dict) -> None:
    """Internal wrapper for process isolation using file IPC."""
    import os
    import pickle

    try:
        res = target_func(*args, **kwargs)
        with open(temp_filename, "wb") as f:
            pickle.dump((True, res), f)
        # Flush and sync to ensure data is on disk
        # f.flush() handled by context manager exit, but sync is good
        # os.fsync(f.fileno()) # Can't do this after close.

    except Exception as e:
        # Try to report error
        try:
            with open(temp_filename, "wb") as f:
                pickle.dump((False, e), f)
        except Exception:
            pass  # We failed to report failure. Parent will see empty file or crash.

    # Force exit to avoid CasADi teardown crash
    # Using explicit exit to bypass Python's cleanup of C++ extensions
    os._exit(0)


def run_isolated(target_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Run a function in a separate process to avoid CasADi/IPOPT segfaults on macOS.
    Returns the result of the function.
    Uses file-based IPC to avoid Queue threading issues with os._exit().
    """
    if platform.system() != "Darwin":
        return target_func(*args, **kwargs)

    import os
    import pickle
    import tempfile

    # Create a temp file for the result
    fd, temp_path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)  # We just need the path

    ctx = multiprocessing.get_context("spawn")

    try:
        p = ctx.Process(target=_isolated_wrapper, args=(temp_path, target_func, args, kwargs))
        p.start()
        p.join()

        # Check if process crashed with segfault (code -11)
        # We don't raise immediately, we check if we got a result first.

        success = False
        value = None
        loaded = False

        try:
            if os.path.getsize(temp_path) > 0:
                with open(temp_path, "rb") as f:
                    success, value = pickle.load(f)
                loaded = True
        except (EOFError, pickle.UnpicklingError):
            pass

        if loaded:
            if success:
                return value
            else:
                raise value

        # If we didn't get a result, check exit code
        if p.exitcode is not None and p.exitcode != 0:
            raise RuntimeError(
                f"Isolated process crashed (Exit code: {p.exitcode}) and no result was written."
            )

        raise RuntimeError("Isolated process finished without data.")

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
