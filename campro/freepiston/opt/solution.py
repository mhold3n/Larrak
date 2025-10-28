from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Solution:
    meta: dict[str, Any]
    data: dict[str, Any]

    @property
    def success(self) -> bool:
        """Extract success from optimization metadata."""
        return self.meta.get("optimization", {}).get("success", False)

    @property
    def iterations(self) -> int:
        """Extract iterations from optimization metadata."""
        return self.meta.get("optimization", {}).get("iterations", 0)

    @property
    def states(self) -> dict[str, Any]:
        """Extract states from data if available."""
        return self.data.get("states", {})

    @property
    def performance_metrics(self) -> dict[str, Any]:
        """Extract performance metrics from metadata."""
        return self.meta.get("performance_metrics", {})

    @property
    def objective_value(self) -> float:
        """Extract objective value from optimization metadata."""
        return self.meta.get("optimization", {}).get("f_opt", float("inf"))

    @property
    def cpu_time(self) -> float:
        """Extract CPU time from optimization metadata."""
        return self.meta.get("optimization", {}).get("cpu_time", 0.0)

    @property
    def message(self) -> str:
        """Extract solver message from optimization metadata."""
        return self.meta.get("optimization", {}).get("message", "")

    @property
    def status(self) -> int:
        """Extract solver status from optimization metadata."""
        return self.meta.get("optimization", {}).get("status", -1)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        # Minimal save: write meta and sizes
        with (p / "meta.txt").open("w", encoding="utf-8") as f:
            for k, v in self.meta.items():
                f.write(f"{k}: {v}\n")
