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
    # Extended fields for Phase-1 free-piston integration
    motion: dict[str, Any] = field(default_factory=dict)     # e.g., {time_s, theta, position_mm, velocity, acceleration}
    pressure: dict[str, Any] = field(default_factory=dict)   # e.g., {"vs_time": {...}, "vs_position": {...}, "vs_theta": {...}}
    thermo: dict[str, Any] = field(default_factory=dict)     # e.g., {"eta_th": ..., "imep": ..., "p_max": ...}
