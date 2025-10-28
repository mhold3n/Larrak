import json
import random
import time
import uuid
from pathlib import Path

import numpy as np

# A unique identifier for this Python process/run. Used to name artifacts.
RUN_ID: str = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


def set_global_seeds(seed: int = 1337) -> None:
    """Set global RNG seeds for reproducibility.

    Applies to Python's ``random`` and NumPy's PRNG.
    """
    random.seed(seed)
    np.random.seed(seed)


def log_run_metadata(meta: dict, folder: str = "runs") -> str:
    """Write run metadata JSON next to solver logs.

    Returns the path to the JSON file for convenience.
    """
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{RUN_ID}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return str(out_path)
