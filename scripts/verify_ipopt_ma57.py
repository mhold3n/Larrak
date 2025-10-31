from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path so 'campro' package can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse  # noqa: E402

from campro.environment.env_manager import find_hsl_library  # noqa: E402
from campro.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def get_ipopt_opt_path() -> str:
    """Get the path to ipopt.opt file in the project root."""
    opt_file = PROJECT_ROOT / "ipopt.opt"
    return str(opt_file) if opt_file.exists() else ""


def get_hsl_lib_path() -> str:
    """Get the path to HSL library, checking local conda env first."""
    # First, try to find in active conda environment
    hsl_path = find_hsl_library()
    if hsl_path:
        return str(hsl_path)
    
    # Fallback: check environment variable
    import os
    hsl_env = os.environ.get("HSLLIB_PATH", "")
    if hsl_env and Path(hsl_env).exists():
        return hsl_env
    
    # Fallback: try project CoinHSL archive (Windows)
    if sys.platform.startswith("win"):
        # Prefer the specific CoinHSL archive folder
        hsl_folder = PROJECT_ROOT / "CoinHSL-archive.v2024.5.15.x86_64-w64-mingw32-libgfortran5"
        if hsl_folder.exists():
            bin_dir = hsl_folder / "bin"
            dll = bin_dir / "libcoinhsl.dll"
            if dll.exists():
                return str(dll)
        
        # Fallback: search for any CoinHSL-archive.* folder
        candidates = list(PROJECT_ROOT.glob("CoinHSL-archive.*"))
        if candidates:
            bin_dir = candidates[0] / "bin"
            dll = bin_dir / "libcoinhsl.dll"
            if dll.exists():
                return str(dll)
    
    return ""


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

    # Get paths dynamically
    ipopt_opt_path = get_ipopt_opt_path()
    hsl_lib_path = get_hsl_lib_path()
    
    # Diagnostics: confirm files exist (do not hard fail if missing; Ipopt may still run)
    ipopt_opt_exists = bool(ipopt_opt_path) and Path(ipopt_opt_path).exists()
    hsl_exists = bool(hsl_lib_path) and Path(hsl_lib_path).exists()
    
    if not ipopt_opt_exists:
        log.warning(f"Option file not found at {ipopt_opt_path or '(not set)'}")
    if not hsl_exists:
        log.warning(f"HSL library not found at {hsl_lib_path or '(not set)'}")

    # Simple scalar quadratic: f(x) = (x - 1)^2
    x = ca.SX.sym("x")
    nlp: dict[str, Any] = {"x": x, "f": (x - 1) ** 2}

    # Provide creation-time options so Ipopt initializes MA57 & HSL before reading ipopt.opt
    opts: dict[str, Any] = {
        "ipopt.linear_solver": solver_name,
    }
    
    # Only add HSL library path if found
    if hsl_exists:
        opts["ipopt.hsllib"] = hsl_lib_path
    
    # Only add option file path if found
    if ipopt_opt_exists:
        opts["ipopt.option_file_name"] = ipopt_opt_path

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

    stats: dict[str, Any] = dict(solver.stats())
    linear_solver = stats.get("linear_solver")
    success = bool(stats.get("success", False))
    return_status = stats.get("return_status")

    log.info(f"Ipopt created: {bool(solver)}")
    log.info(f"Ipopt success: {success} | status: {return_status}")
    log.info(f"Ipopt linear_solver (reported): {linear_solver}")
    if ipopt_opt_exists:
        log.info(f"Option file detected at: {ipopt_opt_path}")
    if hsl_exists:
        log.info(f"HSL library detected at: {hsl_lib_path}")

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
