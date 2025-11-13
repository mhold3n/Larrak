from __future__ import annotations

"""Simple warm-start persistence for Ipopt via CasADi.

Persists the last primal/dual to disk under `runs/` and loads them on
subsequent solves if dimensions match. Designed to be a lightweight utility
that callers can opt into.
"""

from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402

from campro.logging import get_logger  # noqa: E402

log = get_logger(__name__)


@dataclass(slots=True)
class WarmStart:
    x0: np.ndarray
    lam_g0: np.ndarray | None = None
    lam_x0: np.ndarray | None = None
    zl0: np.ndarray | None = None
    zu0: np.ndarray | None = None


def save_warmstart(
    x: np.ndarray,
    *,
    lam_g: np.ndarray | None = None,
    lam_x: np.ndarray | None = None,
    zl: np.ndarray | None = None,
    zu: np.ndarray | None = None,
    tag: str = "",
) -> tuple[str, str]:
    """Warm-start persistence disabled; keep signature for legacy callers."""
    log.debug("Skipping warm-start save; persistence disabled")
    return "", ""


def load_warmstart(
    n_x: int,
    n_lam_g: int | None = None,
    *,
    tag: str = "",
) -> dict[str, np.ndarray]:
    """Warm-start persistence disabled; return empty kwargs."""
    log.debug(
        "Skipping warm-start load for %s variables (tag=%s); persistence disabled",
        n_x,
        tag,
    )
    return {}
