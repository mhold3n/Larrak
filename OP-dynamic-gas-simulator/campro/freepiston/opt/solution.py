from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class Solution:
    meta: Dict[str, Any]
    data: Dict[str, Any]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        # Minimal save: write meta and sizes
        with (p / "meta.txt").open("w", encoding="utf-8") as f:
            for k, v in self.meta.items():
                f.write(f"{k}: {v}\n")
