"""
Quick-access functions for common platform-specific resources.

This module provides simple functions for scripts that don't need
the full PlatformContext machinery.

Usage in scripts:
    from campro.environment.resolve import hsl_path, exit_safely
    
    hsl = hsl_path()  # Returns Path, raises if not found
    exit_safely(0)    # Platform-appropriate exit
"""

from __future__ import annotations

from pathlib import Path

from campro.environment.context import ctx


def hsl_path() -> Path:
    """
    Get HSL library path or raise helpful error.

    Returns
    -------
    Path
        Absolute path to the HSL library (libcoinhsl.dll/.dylib/.so)

    Raises
    ------
    RuntimeError
        If CoinHSL library cannot be found for the current platform
    """
    path = ctx.resources.hsl_library
    if path is None:
        raise RuntimeError(
            f"CoinHSL library not found for platform '{ctx.platform}'. "
            f"Ensure Libraries/ contains a CoinHSL build matching your platform. "
            f"Expected patterns: Windows=*mingw*, macOS=*darwin* or *apple*"
        )
    return path


def project_root() -> Path:
    """
    Get project root directory.

    Returns
    -------
    Path
        Absolute path to the project root (parent of campro/)
    """
    return ctx.paths.project_root


def python_exe() -> Path:
    """
    Get Python executable path or raise.

    Returns
    -------
    Path
        Path to the Python executable in the active conda environment

    Raises
    ------
    RuntimeError
        If Python executable cannot be found
    """
    path = ctx.resources.python_executable
    if path is None:
        raise RuntimeError(
            "Python executable not found in conda environment. "
            "Ensure a conda environment is active."
        )
    return path


def exit_safely(code: int = 0) -> None:
    """
    Exit with platform-appropriate method.

    On macOS, uses os._exit() to avoid CasADi/IPOPT teardown segfaults.
    On Windows/Linux, uses standard sys.exit().

    Parameters
    ----------
    code : int, default=0
        Exit code to return
    """
    ctx.workflows.exit_safely(code)


def requires_isolation() -> bool:
    """
    Check if CasADi operations require process isolation.

    Returns
    -------
    bool
        True on macOS (where teardown segfaults occur), False otherwise
    """
    return ctx.workflows.requires_process_isolation()


def libraries_dir() -> Path:
    """
    Get the Libraries directory containing CoinHSL and other dependencies.

    Returns
    -------
    Path
        Absolute path to Libraries/ directory
    """
    return ctx.paths.libraries_dir()


__all__ = [
    "exit_safely",
    "hsl_path",
    "libraries_dir",
    "project_root",
    "python_exe",
    "requires_isolation",
]
