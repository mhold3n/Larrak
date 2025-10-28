from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from campro.constants import HSLLIB_PATH, IPOPT_OPT_PATH  # noqa: E402
from campro.logging import get_logger  # noqa: E402
from campro.optimization.solver_analysis import analyze_ipopt_run  # noqa: E402

log = get_logger(__name__)


def mini_nlp_and_run(enable_analysis: bool = True):
    import casadi as ca  # type: ignore

    x = ca.SX.sym("x")
    nlp = {"x": x, "f": (x - 1) ** 2}

    opts = {
        "ipopt.linear_solver": "ma27",
        "ipopt.hsllib": HSLLIB_PATH,
        "ipopt.option_file_name": IPOPT_OPT_PATH,
        "ipopt.print_level": 3,
    }
    ipopt_output_file = None
    if enable_analysis:
        out_dir = Path("logs/ipopt")
        out_dir.mkdir(parents=True, exist_ok=True)
        ipopt_output_file = str(out_dir / "probe.log")
        opts["ipopt.output_file"] = ipopt_output_file
        opts["ipopt.print_timing_statistics"] = "yes"

    s = ca.nlpsol("s", "ipopt", nlp, opts)
    r = s(x0=0)
    stats = dict(s.stats())

    report = analyze_ipopt_run(stats, ipopt_output_file)
    print(report.to_json())
    return 0 if report.grade in ("low", "medium") else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MA27 run and grade MA57 readiness",
    )
    parser.add_argument(
        "--no-analysis", action="store_true", help="Disable Ipopt log capture",
    )
    args = parser.parse_args()

    sys.exit(mini_nlp_and_run(enable_analysis=not args.no_analysis))


if __name__ == "__main__":
    main()
