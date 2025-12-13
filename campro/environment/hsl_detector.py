"""
HSL library detection and solver availability testing.

This module provides automatic detection of CoinHSL libraries based on platform
(Windows/macOS) and testing of available HSL solvers (MA27, MA57, MA77, MA86, MA97).
"""

from __future__ import annotations

import re
from pathlib import Path

from campro.environment.platform_detector import IS_MACOS, IS_WINDOWS
from campro.logging import get_logger

log = get_logger(__name__)

# Cache for detected solvers to avoid repeated testing
_AVAILABLE_SOLVERS_CACHE: list[str] | None = None
_COINHSL_DIRECTORY_CACHE: Path | None = None


def _get_project_root() -> Path:
    """Get the project root directory."""
    try:
        import campro

        return Path(campro.__file__).parent.parent
    except ImportError:
        # Fallback: assume current working directory
        return Path.cwd()


def _extract_version_from_dirname(dirname: str) -> tuple[int, ...]:
    """
    Extract version tuple from CoinHSL directory name for sorting.

    Example: "CoinHSL.v2024.5.15.x86_64-apple-darwin-libgfortran5" -> (2024, 5, 15)
    """
    # Match pattern: CoinHSL.vYYYY.M.D.*
    match = re.search(r"\.v(\d+)\.(\d+)\.(\d+)", dirname)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return (0, 0, 0)


def find_coinhsl_directory(project_root: Path | None = None) -> Path | None:
    """
    Auto-detect CoinHSL directory based on platform.

    Searches for CoinHSL.v* directories matching the current platform:
    - Windows: *w64-mingw32* or *mingw*
    - macOS: *darwin* or *apple*

    Returns the most recent version if multiple directories are found.

    Parameters
    ----------
    project_root : Path, optional
        Root directory to search. If None, uses detected project root.

    Returns
    -------
    Optional[Path]
        Path to CoinHSL directory, or None if not found
    """
    global _COINHSL_DIRECTORY_CACHE

    if _COINHSL_DIRECTORY_CACHE is not None:
        return _COINHSL_DIRECTORY_CACHE

    if project_root is None:
        project_root = _get_project_root()

    # Search for CoinHSL.v* directories in libraries/
    libraries_dir = project_root / "libraries"
    if not libraries_dir.exists():
        # Fallback to root for backward compatibility
        candidates = list(project_root.glob("CoinHSL.v*"))
    else:
        candidates = list(libraries_dir.glob("CoinHSL.v*"))

    if not candidates:
        log.debug("No CoinHSL.v* directories found in project root")
        return None

    # Filter by platform
    platform_matches = []
    if IS_WINDOWS:
        # Windows: look for w64-mingw32 or mingw in name
        for candidate in candidates:
            name_lower = candidate.name.lower()
            if "w64" in name_lower or "mingw" in name_lower:
                platform_matches.append(candidate)
    elif IS_MACOS:
        # macOS: look for darwin or apple in name
        for candidate in candidates:
            name_lower = candidate.name.lower()
            if "darwin" in name_lower or "apple" in name_lower:
                platform_matches.append(candidate)
    else:
        # Linux: accept any CoinHSL directory
        platform_matches = candidates

    if not platform_matches:
        log.debug(
            f"No CoinHSL directories found matching platform (Windows={IS_WINDOWS}, macOS={IS_MACOS})"
        )
        return None

    # Sort by version (most recent first)
    platform_matches.sort(key=lambda p: _extract_version_from_dirname(p.name), reverse=True)

    selected = platform_matches[0]
    log.info(f"Found CoinHSL directory: {selected.name}")

    _COINHSL_DIRECTORY_CACHE = selected
    return selected


def get_hsl_library_path(coinhsl_dir: Path | None = None) -> Path | None:
    """
    Get the path to the HSL library file (libcoinhsl.dll/dylib/so).

    Parameters
    ----------
    coinhsl_dir : Path, optional
        CoinHSL directory. If None, attempts to auto-detect.

    Returns
    -------
    Optional[Path]
        Path to HSL library file, or None if not found
    """
    if coinhsl_dir is None:
        coinhsl_dir = find_coinhsl_directory()

    if coinhsl_dir is None:
        return None

    # Determine library file name and location based on platform
    if IS_WINDOWS:
        lib_path = coinhsl_dir / "bin" / "libcoinhsl.dll"
    elif IS_MACOS:
        # Try lib directory first (standard macOS location)
        lib_path = coinhsl_dir / "lib" / "libcoinhsl.dylib"
        if not lib_path.exists():
            # Fallback to bin directory
            lib_path = coinhsl_dir / "bin" / "libcoinhsl.dylib"
    else:  # Linux
        lib_path = coinhsl_dir / "lib" / "libcoinhsl.so"

    if lib_path.exists():
        log.debug(f"Found HSL library at: {lib_path}")
        return lib_path

    log.debug(f"HSL library not found at expected location: {lib_path}")
    return None


def _read_coinhsl_config(coinhsl_dir: Path) -> dict[str, bool]:
    """
    Read CoinHslConfig.h to determine compile-time solver availability.

    Parameters
    ----------
    coinhsl_dir : Path
        CoinHSL directory containing include/CoinHslConfig.h

    Returns
    -------
    dict[str, bool]
        Mapping of solver names to availability (e.g., {"ma27": True, "ma57": True})
    """
    config_path = coinhsl_dir / "include" / "CoinHslConfig.h"

    if not config_path.exists():
        log.debug(f"CoinHslConfig.h not found at {config_path}")
        return {}

    solver_availability = {}
    solver_names = ["ma27", "ma57", "ma77", "ma86", "ma97"]

    try:
        with open(config_path, encoding="utf-8") as f:
            content = f.read()
            for solver in solver_names:
                # Look for #define COINHSL_HAS_MA27 1 pattern
                pattern = f"#define COINHSL_HAS_{solver.upper()} 1"
                solver_availability[solver] = pattern in content
    except Exception as e:
        log.warning(f"Error reading CoinHslConfig.h: {e}")
        return {}

    return solver_availability


def _check_solver_symbols_in_library(lib_path: Path, solver_name: str) -> bool:
    """
    Check if a solver's symbols exist in the HSL library.

    Args:
        lib_path: Path to the HSL library (.dylib or .so)
        solver_name: Solver name (e.g., 'ma57', 'ma27')

    Returns:
        True if symbols are found, False otherwise
    """
    if not lib_path.exists():
        return False

    try:
        import subprocess

        # Use nm to check for symbols (works on macOS and Linux)
        # MA57 symbols are prefixed with ma57a_, ma57b_, etc.
        # MA27 symbols are prefixed with ma27a_, ma27b_, etc.
        result = subprocess.run(
            ["nm", "-gU", str(lib_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False

        # Check for solver-specific symbols
        # Each solver has multiple entry points (e.g., ma57a_, ma57ad_, ma57b_, etc.)
        solver_prefix = solver_name.lower()
        # Look for the main analysis symbol (e.g., ma57a_ or ma27a_)
        main_symbol = f"_{solver_prefix}a_"
        return main_symbol in result.stdout.lower()
    except Exception:
        # If symbol checking fails, assume available (conservative fallback)
        return True


def _get_hsl_library_from_ipopt_opt() -> Path | None:
    """Get HSL library path from ipopt.opt file if it exists."""
    try:
        # Check current directory first, then project root
        opt_file = Path("ipopt.opt")
        if not opt_file.exists():
            # Try project root
            project_root = _get_project_root()
            opt_file = project_root / "ipopt.opt"
        if not opt_file.exists():
            return None

        content = opt_file.read_text()
        # Look for "hsllib /path/to/library" pattern
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("hsllib"):
                # Extract path (everything after "hsllib")
                parts = line.split(None, 1)
                if len(parts) > 1:
                    lib_path = Path(parts[1].strip())
                    if lib_path.exists():
                        return lib_path
    except Exception:
        pass
    return None


def detect_available_solvers(
    coinhsl_dir: Path | None = None,
    test_runtime: bool = True,
) -> list[str]:
    """
    Detect which HSL solvers are available.

    First checks compile-time availability from CoinHslConfig.h,
    then optionally tests runtime availability by attempting to create CasADi solvers.

    Parameters
    ----------
    coinhsl_dir : Path, optional
        CoinHSL directory. If None, attempts to auto-detect.
    test_runtime : bool, default=True
        If True, test runtime availability by creating test solvers.
        If False, only check compile-time availability.

    Returns
    -------
    list[str]
        List of available solver names (e.g., ["ma27", "ma57"])
    """
    global _AVAILABLE_SOLVERS_CACHE

    if _AVAILABLE_SOLVERS_CACHE is not None:
        return _AVAILABLE_SOLVERS_CACHE

    if coinhsl_dir is None:
        coinhsl_dir = find_coinhsl_directory()

    if coinhsl_dir is None:
        log.warning("CoinHSL directory not found; cannot detect solvers")
        _AVAILABLE_SOLVERS_CACHE = []
        return []

    # Read compile-time availability
    compile_time_available = _read_coinhsl_config(coinhsl_dir)

    # All possible solvers
    all_solvers = ["ma27", "ma57", "ma77", "ma86", "ma97"]

    # Filter by compile-time availability
    potentially_available = [
        solver for solver in all_solvers if compile_time_available.get(solver, False)
    ]

    # Skip MA97 on macOS due to known segmentation fault bug
    if IS_MACOS and "ma97" in potentially_available:
        potentially_available.remove("ma97")
        log.debug("Skipping MA97 on macOS due to known crash bug")

    if not test_runtime:
        _AVAILABLE_SOLVERS_CACHE = potentially_available
        return potentially_available

    # Test runtime availability
    available_solvers = []

    try:
        import casadi as ca

        # Create a minimal test problem
        x = ca.SX.sym("x")
        f = x**2
        g = x - 1
        nlp = {"x": x, "f": f, "g": g}

        # Get HSL library path for solver configuration
        # Only set hsllib if not already configured via ipopt.opt
        solver_opts: dict[str, Any] = {}
        try:
            # Check if ipopt.opt exists and has hsllib setting
            opt_file = Path("ipopt.opt")
            if opt_file.exists():
                content = opt_file.read_text().lower()
                if "hsllib" in content:
                    # ipopt.opt already configures hsllib; don't override
                    log.debug("ipopt.opt already defines hsllib; skipping programmatic override")
                else:
                    # No hsllib in ipopt.opt; set it programmatically
                    hsl_lib_path = get_hsl_library_path(coinhsl_dir)
                    if hsl_lib_path:
                        solver_opts["ipopt.hsllib"] = str(hsl_lib_path)
            else:
                # No ipopt.opt file; set hsllib programmatically
                hsl_lib_path = get_hsl_library_path(coinhsl_dir)
                if hsl_lib_path:
                    solver_opts["ipopt.hsllib"] = str(hsl_lib_path)
        except Exception as e:
            log.debug(f"Could not check ipopt.opt: {e}; proceeding without hsllib override")

        # Check which library is actually being used
        hsl_lib_in_use = _get_hsl_library_from_ipopt_opt()
        if hsl_lib_in_use:
            log.debug(f"HSL library from ipopt.opt: {hsl_lib_in_use}")

        for solver_name in potentially_available:
            # If we know which library is being used, verify symbols exist
            if hsl_lib_in_use:
                if not _check_solver_symbols_in_library(hsl_lib_in_use, solver_name):
                    log.debug(
                        f"Solver {solver_name.upper()} symbols not found in library {hsl_lib_in_use}"
                    )
                    continue
            try:
                # Attempt to create solver with this linear solver
                solver_opts["ipopt.linear_solver"] = solver_name
                solver_opts["ipopt.print_level"] = 0
                solver_opts["ipopt.sb"] = "yes"

                # Just create the solver - don't solve to avoid segfaults
                # If creation succeeds, the solver symbols are available
                solver = ca.nlpsol(f"test_{solver_name}", "ipopt", nlp, solver_opts)
                available_solvers.append(solver_name)
                log.debug(f"Solver {solver_name.upper()} is available (created successfully)")
            except Exception as e:
                error_msg = str(e)
                # Check if this is a symbol loading failure during solver creation
                if "symbol not found" in error_msg or "DYNAMIC_LIBRARY_FAILURE" in error_msg:
                    log.debug(f"Solver {solver_name.upper()} symbols not found: {error_msg[:100]}")
                else:
                    log.debug(f"Solver {solver_name.upper()} not available: {e}")

    except ImportError:
        log.warning("CasADi not available; cannot test runtime solver availability")
        # Return compile-time availability as best guess
        available_solvers = potentially_available
    except Exception as e:
        log.warning(f"Error testing solver availability: {e}")
        # Return compile-time availability as best guess
        available_solvers = potentially_available

    if not available_solvers:
        log.warning("No HSL solvers detected as available")
    else:
        log.info(
            f"Detected {len(available_solvers)} available HSL solvers: {', '.join(available_solvers)}"
        )

    _AVAILABLE_SOLVERS_CACHE = available_solvers
    return available_solvers


def clear_cache() -> None:
    """Clear cached detection results (useful for testing or re-detection)."""
    global _AVAILABLE_SOLVERS_CACHE, _COINHSL_DIRECTORY_CACHE
    _AVAILABLE_SOLVERS_CACHE = None
    _COINHSL_DIRECTORY_CACHE = None
    log.debug("Cleared HSL detector cache")


__all__ = [
    "clear_cache",
    "detect_available_solvers",
    "find_coinhsl_directory",
    "get_hsl_library_path",
]
