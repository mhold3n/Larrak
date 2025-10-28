from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ProblemSpec:
    """High-level problem specification for motion law optimization.

    Fields intentionally minimal to decouple UI and solver internals.
    """

    stroke: float
    cycle_time: float
    phases: dict[str, float]
    bounds: dict[str, float]
    objective: str  # "min_jerk" | "custom_thermo" | ...
    gear_mode: str | None = None
    extra: dict[str, Any] | None = None
