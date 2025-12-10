from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class IpoptAnalysisReport:
    grade: str  # "low" | "medium" | "high"
    reasons: list[str]
    suggested_action: str
    stats: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "grade": self.grade,
                "reasons": self.reasons,
                "suggested_action": self.suggested_action,
                "stats": self.stats,
            },
            indent=2,
        )


def _safe_get(d: dict[str, Any], key: str, default: Any) -> Any:
    try:
        return d.get(key, default)
    except Exception:
        return default


def analyze_ipopt_run(
    stats: dict[str, Any], ipopt_output_file: str | None,
) -> IpoptAnalysisReport:
    """
    Analyze an Ipopt run (stats + optional output file) and estimate whether MA57
    would likely yield better robustness/performance than MA27.
    """
    reasons: list[str] = []
    indicators: list[str] = []

    success = bool(_safe_get(stats, "success", False))
    return_status = str(_safe_get(stats, "return_status", ""))
    iter_count = int(_safe_get(stats, "iter_count", _safe_get(stats, "iterations", 0)))
    primal_inf = float(_safe_get(stats, "primal_inf", 0.0))
    dual_inf = float(_safe_get(stats, "dual_inf", 0.0))

    # Parse output file if present
    ls_time_ratio: float | None = None
    refactorizations: int = 0
    restoration_occurrences: int = 0
    inertia_corrections: int = 0
    small_pivot_warnings: int = 0

    if ipopt_output_file and Path(ipopt_output_file).exists():
        try:
            text = Path(ipopt_output_file).read_text(errors="ignore")
            # Heuristic parsing: look for common lines
            if (
                "Total CPU secs in linear solver" in text
                and "Total CPU secs in IPOPT" in text
            ):
                try:
                    # Rough extraction
                    import re

                    ls_match = re.search(
                        r"Total CPU secs in linear solver\s*=\s*([0-9.]+)", text,
                    )
                    total_match = re.search(
                        r"Total CPU secs in IPOPT\s*=\s*([0-9.]+)", text,
                    )
                    if ls_match and total_match:
                        ls = float(ls_match.group(1))
                        total = float(total_match.group(1))
                        if total > 0:
                            ls_time_ratio = ls / total
                except Exception:
                    pass
            refactorizations = text.count("Factorization CPU time") + text.count(
                "KKT: Cholesky",
            )
            restoration_occurrences = text.count("Restoration phase activated")
            inertia_corrections = text.count("inertia")
            small_pivot_warnings = text.count("small pivot") + text.count(
                "singular KKT",
            )
        except Exception as exc:
            log.warning(f"Failed to parse Ipopt output file {ipopt_output_file}: {exc}")

    # Heuristics
    if not success:
        reasons.append("Ipopt did not succeed")
    if "Restoration" in return_status or restoration_occurrences > 0:
        reasons.append("Restoration phase encountered")
    if ls_time_ratio is not None and ls_time_ratio > 0.5:
        reasons.append(f"Linear solver dominates runtime (ratio ~{ls_time_ratio:.2f})")
    if refactorizations >= 10:
        reasons.append(f"Many refactorizations detected ({refactorizations})")
    if inertia_corrections > 0:
        reasons.append("Inertia corrections/regularization occurrences detected")
    if small_pivot_warnings > 0:
        reasons.append("Small pivot / near-singularity warnings detected")
    if iter_count > 3000:
        reasons.append(f"High iteration count ({iter_count})")
    if max(primal_inf, dual_inf) > 1e-3:
        reasons.append("Significant primal/dual infeasibility remained")

    # Grade
    if any(
        [
            "Ipopt did not succeed" in r
            or "Restoration" in r
            or "near-singularity" in r
            for r in reasons
        ],
    ):
        grade = "high"
    elif any(
        [
            "dominates runtime" in r or "refactorizations" in r or "inertia" in r
            for r in reasons
        ],
    ):
        grade = "medium"
    else:
        grade = "low"

    suggested_action = {
        "low": "MA27 performing well; maintain current configuration.",
        "medium": "Investigate scaling and warm-start strategies to assist MA27.",
        "high": "Significant issues detected; revisit model conditioning, scaling, and bounds before re-running MA27.",
    }[grade]

    report = IpoptAnalysisReport(
        grade=grade,
        reasons=reasons or ["No adverse indicators detected"],
        suggested_action=suggested_action,
        stats={
            "success": success,
            "return_status": return_status,
            "iter_count": iter_count,
            "primal_inf": primal_inf,
            "dual_inf": dual_inf,
            "ls_time_ratio": ls_time_ratio,
            "refactorizations": refactorizations,
            "restoration_occurrences": restoration_occurrences,
            "inertia_corrections": inertia_corrections,
            "small_pivot_warnings": small_pivot_warnings,
            "ipopt_output_file": ipopt_output_file,
        },
    )
    return report
