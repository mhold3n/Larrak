from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SolveReport:
    """Structured result of a motion law optimization run."""

    run_id: str
    status: str  # "Solve_Success", "Infeasible", "Diverged", ...
    kkt: dict[str, float] = field(default_factory=dict)
    n_iter: int = 0
    scaling_stats: dict[str, Any] = field(default_factory=dict)
    residuals: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)  # paths to CSV/PNG/logs
