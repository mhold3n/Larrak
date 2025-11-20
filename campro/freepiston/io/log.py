from __future__ import annotations

from datetime import datetime
from pathlib import Path

from campro.logging import get_logger

log = get_logger(__name__)


def append_run_log(path: str | Path, message: str) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().isoformat()
    with (p / "run.log").open("a", encoding="utf-8") as f:
        f.write(f"{ts} {message}\n")
