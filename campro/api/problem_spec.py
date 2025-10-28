from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProblemSpec:
    """High-level problem specification for motion law optimization.

    Fields intentionally minimal to decouple UI and solver internals.
    """

    stroke: float
    cycle_time: float
    phases: Dict[str, float]
    bounds: Dict[str, float]
    objective: str  # "min_jerk" | "custom_thermo" | ...
    gear_mode: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
