#!/usr/bin/env python3
"""Larrak command-line interface.

Usage:
  python -m scripts.larrak_cli solve --spec specs/op_phaseA.yml --diagnose --export-gear
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from campro.api import ProblemSpec, solve_motion
from campro.diagnostics.run_metadata import RUN_ID
from campro.optimization.solver_analysis import analyze_ipopt_run


def _load_spec(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required to parse YAML specs. Install with 'pip install pyyaml'.") from exc
        return yaml.safe_load(text) or {}
    # Default to JSON
    return json.loads(text)


def _to_problem_spec(d: Dict[str, Any]) -> ProblemSpec:
    stroke = float(d.get("stroke", 20.0))
    cycle_time = float(d.get("cycle_time", 1.0))
    phases = d.get("phases", {}) or {}
    bounds = d.get("bounds", {}) or {}
    objective = str(d.get("objective", "minimum_jerk"))
    gear_mode = d.get("gear_mode")
    extra = d.get("extra")
    return ProblemSpec(
        stroke=stroke,
        cycle_time=cycle_time,
        phases=phases,
        bounds=bounds,
        objective=objective,
        gear_mode=gear_mode,
        extra=extra,
    )


def cmd_solve(args: argparse.Namespace) -> int:
    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"Spec file not found: {spec_path}", file=sys.stderr)
        return 2

    try:
        raw = _load_spec(spec_path)
        spec = _to_problem_spec(raw)
    except Exception as exc:
        print(f"Failed to load spec: {exc}", file=sys.stderr)
        return 2

    report = solve_motion(spec)

    # Save report JSON
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"{RUN_ID}-report.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, indent=2)

    print(f"SolveReport written: {out_path}")

    if args.diagnose:
        ipopt_log = None
        try:
            ipopt_log = report.artifacts.get("ipopt_log") if hasattr(report, "artifacts") else None
        except Exception:
            ipopt_log = None
        try:
            readiness = analyze_ipopt_run({}, ipopt_log)
            print("Diagnosis:")
            print(f"  MA57 Readiness: {readiness.grade.upper()}")
            if readiness.reasons:
                print("  Reasons:")
                for r in readiness.reasons:
                    print(f"    - {r}")
            print(f"  Suggested: {readiness.suggested_action}")
        except Exception as exc:
            print(f"Diagnosis failed: {exc}")

    if args.export_gear:
        print("--export-gear requested, but gear synthesis/export is not implemented yet.")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="larrak", description="Larrak CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_solve = sub.add_parser("solve", help="Solve motion law from spec file")
    p_solve.add_argument("--spec", required=True, help="Path to YAML/JSON ProblemSpec file")
    p_solve.add_argument("--diagnose", action="store_true", help="Run Ipopt log diagnostics")
    p_solve.add_argument("--export-gear", action="store_true", help="Stub: export gear DXF")
    p_solve.set_defaults(func=cmd_solve)

    ns = parser.parse_args(argv)
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())

