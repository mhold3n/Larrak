from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from campro.logging import get_logger

log = get_logger(__name__)


def load_cfg(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping")
    return cfg
