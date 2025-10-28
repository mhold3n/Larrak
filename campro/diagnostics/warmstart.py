from __future__ import annotations

"""Simple warm-start persistence for Ipopt via CasADi.

Persists the last primal/dual to disk under `runs/` and loads them on
subsequent solves if dimensions match. Designed to be a lightweight utility
that callers can opt into.
"""

from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .run_metadata import RUN_ID  # noqa: E402


@dataclass(slots=True)
class WarmStart:
    x0: np.ndarray
    lam_g0: np.ndarray | None = None
    lam_x0: np.ndarray | None = None
    zl0: np.ndarray | None = None
    zu0: np.ndarray | None = None


def _runs_dir() -> Path:
    p = Path("runs")
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_warmstart(
    x: np.ndarray,
    *,
    lam_g: np.ndarray | None = None,
    lam_x: np.ndarray | None = None,
    zl: np.ndarray | None = None,
    zu: np.ndarray | None = None,
    tag: str = "",
) -> tuple[str, str]:
    """Persist warm-start arrays to `runs/{RUN_ID}-warmstart.npz` and `latest-warmstart.npz`.

    Returns `(by_run_path, latest_path)`.
    """
    runs = _runs_dir()
    by_run = runs / f"{RUN_ID}-warmstart{('-' + tag) if tag else ''}.npz"
    latest = runs / f"latest-warmstart{('-' + tag) if tag else ''}.npz"

    np.savez_compressed(
        by_run,
        x0=np.asarray(x),
        lam_g0=None if lam_g is None else np.asarray(lam_g),
        lam_x0=None if lam_x is None else np.asarray(lam_x),
        zl0=None if zl is None else np.asarray(zl),
        zu0=None if zu is None else np.asarray(zu),
        n_x=int(np.asarray(x).size),
        n_lam_g=0 if lam_g is None else int(np.asarray(lam_g).size),
    )
    # Overwrite 'latest'
    np.savez_compressed(
        latest,
        x0=np.asarray(x),
        lam_g0=None if lam_g is None else np.asarray(lam_g),
        lam_x0=None if lam_x is None else np.asarray(lam_x),
        zl0=None if zl is None else np.asarray(zl),
        zu0=None if zu is None else np.asarray(zu),
        n_x=int(np.asarray(x).size),
        n_lam_g=0 if lam_g is None else int(np.asarray(lam_g).size),
    )
    return str(by_run), str(latest)


def load_warmstart(
    n_x: int,
    n_lam_g: int | None = None,
    *,
    tag: str = "",
) -> dict[str, np.ndarray]:
    """Load compatible warm-start arrays if shapes match current problem.

    Returns a dict suitable for passing to CasADi `nlpsol`: possibly including
    `x0`, `lam_g0`, `lam_x0`, `zl0`, `zu0` depending on availability.
    Returns empty dict if nothing compatible is found.
    """
    runs = _runs_dir()
    for name in (
        f"latest-warmstart{('-' + tag) if tag else ''}.npz",
        f"{RUN_ID}-warmstart{('-' + tag) if tag else ''}.npz",
    ):
        path = runs / name
        if not path.exists():
            continue
        try:
            data = np.load(path, allow_pickle=True)
        except Exception as e:
            log.debug(f"Skipping warmstart file {path} due to error: {e}")
            continue
        try:
            if int(data.get("n_x", -1)) != int(n_x):
                continue
            if n_lam_g is not None and int(data.get("n_lam_g", -2)) not in (
                int(n_lam_g),
                0,
            ):
                continue
            out: dict[str, np.ndarray] = {}
            for key in ("x0", "lam_g0", "lam_x0", "zl0", "zu0"):
                arr = data.get(key, None)
                if arr is not None and arr is not np.array(None):
                    out[key] = arr
            if "x0" in out:
                return out
        except Exception as e:
            log.debug(f"Skipping warmstart file {path} due to error: {e}")
            continue
    return {}
