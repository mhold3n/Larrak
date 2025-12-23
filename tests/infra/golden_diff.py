"""
Golden Diff Utility.
Compares two JSON golden files or directories.
"""

import json
import os
import glob
import numpy as np
from typing import Dict, Any, Tuple


def compare_json(expected: Dict, actual: Dict, tolerance: float = 1e-6) -> Tuple[bool, str]:
    """
    Compare two separate JSON dictionaries (expected vs actual).
    Recursive comparison for floats/lists with tolerance.
    """

    def _compare_val(k, v1, v2, path):
        if type(v1) != type(v2):
            return False, f"Type mismatch at {path}: {type(v1)} vs {type(v2)}"

        if isinstance(v1, dict):
            # Check keys match
            if set(v1.keys()) != set(v2.keys()):
                return False, f"Key mismatch at {path}: {set(v1.keys()) ^ set(v2.keys())}"
            for sub_k in v1:
                ok, msg = _compare_val(sub_k, v1[sub_k], v2[sub_k], f"{path}.{sub_k}")
                if not ok:
                    return False, msg
            return True, ""

        elif isinstance(v1, list):
            if len(v1) != len(v2):
                return False, f"Length mismatch at {path}: {len(v1)} vs {len(v2)}"
            for i, (x, y) in enumerate(zip(v1, v2)):
                ok, msg = _compare_val(str(i), x, y, f"{path}[{i}]")
                if not ok:
                    return False, msg
            return True, ""

        elif isinstance(v1, (int, float)):
            # Numeric comparison
            diff = abs(v1 - v2)
            denom = max(abs(v1), abs(v2))
            if denom < 1e-9:  # Both near zero
                if diff > tolerance:
                    return False, f"Value mismatch at {path}: {v1} vs {v2}"
            else:
                rel_diff = diff / denom
                if rel_diff > tolerance:
                    return False, f"Value mismatch at {path} ({rel_diff:.2%}): {v1} vs {v2}"
            return True, ""

        else:
            # String/Bool/Null
            if v1 != v2:
                return False, f"Value mismatch at {path}: {v1} vs {v2}"
            return True, ""

    return _compare_val("root", expected, actual, "root")


def check_goldens(new_dir: str, baseline_dir: str) -> Dict[str, Any]:
    """
    Check new goldens against baseline.
    """
    # Implementation simpler: just structure for now
    pass
