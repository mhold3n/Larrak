"""Ipopt logging helpers.

Provides a stable file sink for Ipopt output per run and helpers to parse
basic statistics from the log file when available.
"""
from __future__ import annotations

from pathlib import Path

from campro.optimization.solvers.ipopt_log_parser import parse_ipopt_log_file

from .run_metadata import RUN_ID


def ensure_runs_dir(path: str = "runs") -> str:
    """Ensure the runs directory exists and return its path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def ipopt_output_path() -> str:
    """Return the default Ipopt output log file for this run."""
    ensure_runs_dir()
    return f"runs/{RUN_ID}-ipopt.log"


def inject_ipopt_file_sink(opts: dict[str, object]) -> dict[str, object]:
    """Inject file sink options into an Ipopt options dict if absent.

    Does not override existing settings.
    """
    opts = dict(opts) if opts is not None else {}
    opts.setdefault("ipopt.output_file", ipopt_output_path())
    opts.setdefault("ipopt.print_level", 5)
    opts.setdefault("ipopt.file_print_level", 5)
    return opts


def get_ipopt_log_stats(path: str | None = None) -> dict[str, object]:
    """Parse Ipopt log file and return a stats dict.

    If ``path`` is None, uses the current run's default log path.
    Returns an empty dict if parsing fails.
    """
    try:
        p = path or ipopt_output_path()
        stats = parse_ipopt_log_file(p)
        return stats.as_dict()
    except Exception:
        return {}
