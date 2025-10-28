from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Final

# Ensure repository root is on sys.path so 'campro' package can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from campro.logging import get_logger

log = get_logger(__name__)


IPOPT_OPT_PATH: Final[str] = "/Users/maxholden/Documents/GitHub/Larrak/ipopt.opt"
HSL_LIB_PATH: Final[str] = "/Users/maxholden/anaconda3/envs/larrak/lib/libcoinhsl.dylib"


def verify_ipopt_with_solver(solver_name: str) -> int:
    """
    Create a trivial NLP, run Ipopt once, and verify the requested linear solver is active.

    Returns
    -------
    int
        0 on success, non-zero on failure.
    """
    try:
        import casadi as ca  # type: ignore
    except Exception as exc:  # pragma: no cover - CLI diagnostic
        log.error(f"Failed to import CasADi: {exc}")
        return 2

    # Diagnostics: confirm files exist (do not hard fail if missing; Ipopt may still run)
    ipopt_opt_exists = Path(IPOPT_OPT_PATH).exists()
    hsl_exists = Path(HSL_LIB_PATH).exists()
    if not ipopt_opt_exists:
        log.warning(f"Option file not found at {IPOPT_OPT_PATH}")
    if not hsl_exists:
        log.warning(f"HSL library not found at {HSL_LIB_PATH}")

    # Simple scalar quadratic: f(x) = (x - 1)^2
    x = ca.SX.sym("x")
    nlp: Dict[str, Any] = {"x": x, "f": (x - 1) ** 2}

    # Provide creation-time options so Ipopt initializes MA57 & HSL before reading ipopt.opt
    opts: Dict[str, Any] = {
        "ipopt.linear_solver": solver_name,
        "ipopt.hsllib": HSL_LIB_PATH,
        "ipopt.option_file_name": IPOPT_OPT_PATH,
    }

    try:
        solver = ca.nlpsol("verify", "ipopt", nlp, opts)
    except Exception as exc:  # pragma: no cover - CLI diagnostic
        log.error(f"Failed to create Ipopt solver: {exc}")
        return 3

    # Solve once to populate stats
    try:
        _ = solver(x0=0)
    except Exception as exc:  # pragma: no cover - CLI diagnostic
        log.error(f"Ipopt solve failed: {exc}")
        return 4

    stats: Dict[str, Any] = dict(solver.stats())
    linear_solver = stats.get("linear_solver")
    success = bool(stats.get("success", False))
    return_status = stats.get("return_status")

    log.info(f"Ipopt created: {bool(solver)}")
    log.info(f"Ipopt success: {success} | status: {return_status}")
    log.info(f"Ipopt linear_solver (reported): {linear_solver}")
    if ipopt_opt_exists:
        log.info(f"Option file detected at: {IPOPT_OPT_PATH}")
    if hsl_exists:
        log.info(f"HSL library detected at: {HSL_LIB_PATH}")

    if linear_solver is None:
        # Not all builds expose 'linear_solver' in stats; emit keys for debugging
        log.warning(
            f"'linear_solver' not present in stats. Keys: {sorted(stats.keys())}",
        )

    # Verification: we expect the requested solver to be active
    if str(linear_solver).lower() == solver_name.lower():
        log.info(
            f"Verification OK: Ipopt reports {solver_name.upper()} as the active linear solver.",
        )
        return 0

    log.error(
        f"Verification FAILED: Ipopt did not report {solver_name.upper()} as the linear solver.",
    )
    return 5


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Ipopt linear solver initialization",
    )
    parser.add_argument(
        "--solver", default="ma57", help="Linear solver to verify (e.g., ma57, ma27)",
    )
    args = parser.parse_args()

    exit_code = verify_ipopt_with_solver(args.solver)
    # Avoid bare prints (CI rule); rely on process exit code for scripting
    import sys

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
