from __future__ import annotations

"""Parse Ipopt log files into structured metrics.

The parser is *best-effort*: missing patterns are ignored instead of raising.
Only a subset of metrics required for MA57 readiness analysis is extracted.
Extend the regular expressions as new insights become necessary.

All times are returned in seconds.  Counts are integers.  Ratios are floats in
[0, 1] where available.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from re import Pattern
from typing import Any, Dict, Final

from campro.logging import get_logger

log = get_logger(__name__)

__all__: Final = [
    "IpoptLogStats",
    "parse_ipopt_log_file",
    "parse_ipopt_log_text",
]


@dataclass(slots=True)
class IpoptLogStats:
    """Structured subset of Ipopt run statistics."""

    status: str | None = None
    n_iterations: int | None = None
    cpu_time: float | None = None
    ls_time: float | None = None
    ls_time_ratio: float | None = None
    restoration_count: int = 0
    refactorizations: int = 0
    small_pivot_warnings: int = 0
    primal_inf: float | None = None
    dual_inf: float | None = None
    compl_inf: float | None = None

    def as_dict(self) -> Dict[str, Any]:
        """Return stats as a JSON-serialisable dict."""
        return {
            "status": self.status,
            "iterations": self.n_iterations,
            "cpu_time": self.cpu_time,
            "ls_time": self.ls_time,
            "ls_time_ratio": self.ls_time_ratio,
            "restoration_count": self.restoration_count,
            "refactorizations": self.refactorizations,
            "small_pivot_warnings": self.small_pivot_warnings,
            "primal_inf": self.primal_inf,
            "dual_inf": self.dual_inf,
            "compl_inf": self.compl_inf,
        }


# -- Regular expressions ----------------------------------------------------

_RE_STATUS: Final[Pattern[str]] = re.compile(r"^\s*EXIT:\s*(.+?)\s*$", re.MULTILINE)
_RE_ITER: Final[Pattern[str]] = re.compile(r"Number of Iterations\s*:\s*(\d+)")
_RE_CPU: Final[Pattern[str]] = re.compile(r"Total CPU secs in IPOPT.*?=\s*([0-9.]+)")
_RE_LS_TIME: Final[Pattern[str]] = re.compile(
    r"Time\s+for\s+linear\s+solve\s+=\s+([0-9.]+)",
)
_RE_LS_RATIO: Final[Pattern[str]] = re.compile(
    r"Linear\s+solve\s+time\s+ratio\s+=\s+([0-9.]+)",
)
_RE_PIVOT: Final[Pattern[str]] = re.compile(r"Warning:\s+Small\s+pivot")
_RE_RESTOR: Final[Pattern[str]] = re.compile(r"Restoration phase start", re.IGNORECASE)
_RE_REFACT: Final[Pattern[str]] = re.compile(
    r"Number of Iterative Refinements:\s*(\d+)",
)
_RE_INFEAS: Final[Pattern[str]] = re.compile(
    r"Primal infeasibility\s*=\s*([0-9.eE+-]+).*?Dual infeasibility\s*=\s*([0-9.eE+-]+).*?Complementarity\s*=\s*([0-9.eE+-]+)",
    re.DOTALL,
)


def _extract_float(match: re.Match[str] | None) -> float | None:
    return float(match.group(1)) if match else None


def _extract_int(match: re.Match[str] | None) -> int | None:
    return int(match.group(1)) if match else None


def parse_ipopt_log_text(text: str) -> IpoptLogStats:
    """Return :class:`IpoptLogStats` from raw Ipopt output string."""

    stats = IpoptLogStats()

    stats.status = (
        _RE_STATUS.search(text).group(1).strip() if _RE_STATUS.search(text) else None
    )
    stats.n_iterations = _extract_int(_RE_ITER.search(text))
    stats.cpu_time = _extract_float(_RE_CPU.search(text))
    stats.ls_time = _extract_float(_RE_LS_TIME.search(text))
    stats.ls_time_ratio = _extract_float(_RE_LS_RATIO.search(text))

    stats.restoration_count = len(_RE_RESTOR.findall(text))
    stats.small_pivot_warnings = len(_RE_PIVOT.findall(text))

    ref_m = _RE_REFACT.search(text)
    if ref_m:
        stats.refactorizations = int(ref_m.group(1))

    infeas_m = _RE_INFEAS.search(text)
    if infeas_m:
        stats.primal_inf = float(infeas_m.group(1))
        stats.dual_inf = float(infeas_m.group(2))
        stats.compl_inf = float(infeas_m.group(3))

    # Derive ls_time_ratio if missing and both components present
    if stats.ls_time_ratio is None and stats.cpu_time and stats.ls_time:
        try:
            stats.ls_time_ratio = (
                stats.ls_time / stats.cpu_time if stats.cpu_time > 0 else None
            )
        except ZeroDivisionError:
            stats.ls_time_ratio = None

    return stats


def parse_ipopt_log_file(path: str | Path) -> IpoptLogStats:
    """Read *path* and parse Ipopt statistics."""

    p = Path(path)
    try:
        text = p.read_text(errors="ignore")
    except FileNotFoundError:
        log.error("Ipopt log file not found: %s", p)
        raise

    return parse_ipopt_log_text(text)
