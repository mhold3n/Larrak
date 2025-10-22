from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class SolveReport:
    """Structured result of a motion law optimization run."""

    run_id: str
    status: str  # "Solve_Success", "Infeasible", "Diverged", ...
    kkt: Dict[str, float] = field(default_factory=dict)
    n_iter: int = 0
    scaling_stats: Dict[str, Any] = field(default_factory=dict)
    residuals: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)  # paths to CSV/PNG/logs

