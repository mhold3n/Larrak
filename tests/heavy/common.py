"""Shared helpers for heavy IPOPT integration tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"
HEAVY_ENV_FLAG = "CAMPRO_RUN_HEAVY_TESTS"


def heavy_env_enabled() -> bool:
    """Return True when the caller explicitly opted in to IPOPT-heavy tests."""
    value = os.environ.get(HEAVY_ENV_FLAG, "").lower()
    return value in {"1", "true", "yes", "on"}


def require_golden_initialized(payload: dict[str, Any], golden_name: str) -> None:
    """Raise a descriptive error when the stored golden payload is uninitialized."""
    if not payload.get("initialized"):
        raise RuntimeError(
            f"Golden reference '{golden_name}' is not initialized.",
        )


def load_golden_json(relative_name: str) -> dict[str, Any]:
    """Load a golden JSON payload from tests/golden."""
    path = GOLDEN_DIR / relative_name
    if not path.exists():
        raise FileNotFoundError(relative_name)
    return json.loads(path.read_text())


def save_golden_json(relative_name: str, payload: dict[str, Any]) -> None:
    """Write a golden JSON payload to tests/golden (with trailing newline)."""
    path = GOLDEN_DIR / relative_name
    path.write_text(json.dumps(payload, indent=2) + "\n")


def arrays_close(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    """Assert that two numeric arrays match within tolerances."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Array '{name}' shape mismatch: {actual.shape} != {expected.shape}",
        )
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = np.max(np.abs(actual - expected))
        raise AssertionError(
            f"Array '{name}' mismatch (max abs diff={diff:.3e}, rtol={rtol}, atol={atol})",
        )


def to_array(seq: Iterable[float]) -> np.ndarray:
    """Convert an iterable into a float64 NumPy array."""
    return np.asarray(list(seq), dtype=float)


def serialize_array(values: np.ndarray | Iterable[float]) -> list[float]:
    """Convert an array-like object to a plain list for JSON output."""
    return np.asarray(values, dtype=float).tolist()
