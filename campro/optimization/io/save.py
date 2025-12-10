from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from campro.logging import get_logger

log = get_logger(__name__)


def save_json(
    obj: dict[str, Any], path: str | Path, filename: str = "state.json",
) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    with (p / filename).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
