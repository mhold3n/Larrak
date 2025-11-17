"""
Environment management utilities for local conda environments.

Provides functions to detect, validate, and work with local conda environments
stored in the repository, with OS-aware path resolution.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional

from campro.environment.platform_detector import (
    IS_MACOS,
    IS_WINDOWS,
    get_local_conda_env_path,
    is_local_conda_env_present,
)
from campro.logging import get_logger

log = get_logger(__name__)


def get_active_conda_env_path(project_root: Path | None = None) -> Optional[Path]:
    """
    Get the path to the currently active conda environment, or the local env if available.
    
    This function checks:
    1. If a local conda environment exists for the current OS, return that path
    2. If CONDA_PREFIX is set (active conda env), return that path
    3. If CONDA_DEFAULT_ENV is 'larrak', try to find it in standard conda locations
    4. Otherwise, return None
    
    Parameters
    ----------
    project_root : Path, optional
        Root directory of the project. If None, attempts to detect it.
    
    Returns
    -------
    Optional[Path]
        Path to the conda environment, or None if not found
    """
    # First priority: check for local conda environment
    if project_root is None:
        # Try to detect project root
        try:
            # If campro is importable, we can detect the root
            import campro
            project_root = Path(campro.__file__).parent.parent
        except ImportError:
            # Fallback: assume current working directory
            project_root = Path.cwd()
    
    if is_local_conda_env_present(project_root):
        local_env_path = get_local_conda_env_path(project_root)
        log.info(f"Using local conda environment: {local_env_path}")
        return local_env_path
    
    # Second priority: check for active conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        env_path = Path(conda_prefix)
        if env_path.exists():
            log.info(f"Using active conda environment: {env_path}")
            return env_path
    
    # Third priority: check for 'larrak' environment in standard conda locations
    # Always check, not just when CONDA_DEFAULT_ENV is 'larrak'
    
    # Method 1: Use CONDA_BASE or extract from CONDA_EXE
    conda_base = os.environ.get("CONDA_BASE")
    if not conda_base:
        # Try to extract from CONDA_EXE (works on both Unix and Windows)
        conda_exe = os.environ.get("CONDA_EXE", "")
        if conda_exe:
            # On Windows: C:\path\to\conda\Scripts\conda.exe -> C:\path\to\conda
            # On Unix: /path/to/conda/bin/conda -> /path/to/conda
            if platform.system().lower() == "windows":
                conda_base = str(Path(conda_exe).parent.parent)
            else:
                conda_base = conda_exe.replace("/bin/conda", "")
    
    if conda_base:
        env_path = Path(conda_base) / "envs" / "larrak"
        if env_path.exists():
            log.info(f"Found larrak environment at: {env_path}")
            return env_path
    
    # Method 2: Try home directory locations (common installations)
    home = Path.home()
    for base_name in ["anaconda3", "miniconda3", "Miniforge3", "miniforge3"]:
        base_path = home / base_name / "envs" / "larrak"
        if base_path.exists():
            log.info(f"Found larrak environment at: {base_path}")
            return base_path
    
    # Method 3: Try .conda directory
    conda_home = home / ".conda" / "envs" / "larrak"
    if conda_home.exists():
        log.info(f"Found larrak environment at: {conda_home}")
        return conda_home
    
    log.warning("No conda environment found (local or global)")
    return None


def get_python_executable_path(project_root: Path | None = None) -> Optional[Path]:
    """
    Get the path to the Python executable in the active conda environment.
    
    Parameters
    ----------
    project_root : Path, optional
        Root directory of the project. If None, attempts to detect it.
    
    Returns
    -------
    Optional[Path]
        Path to the Python executable, or None if not found
    """
    env_path = get_active_conda_env_path(project_root)
    if env_path is None:
        return None
    
    system = platform.system().lower()
    if system == "windows":
        python_exe = env_path / "python.exe"
        if not python_exe.exists():
            python_exe = env_path / "Scripts" / "python.exe"
    else:
        python_exe = env_path / "bin" / "python"
    
    if python_exe.exists():
        return python_exe
    
    return None


def get_lib_path(env_path: Path | None = None) -> Optional[Path]:
    """
    Get the library path for the given conda environment.
    
    Parameters
    ----------
    env_path : Path, optional
        Path to conda environment. If None, uses get_active_conda_env_path()
    
    Returns
    -------
    Optional[Path]
        Path to the lib directory, or None if not found
    """
    if env_path is None:
        env_path = get_active_conda_env_path()
    
    if env_path is None:
        return None
    
    system = platform.system().lower()
    if system == "windows":
        lib_path = env_path / "Library" / "lib"
    else:
        lib_path = env_path / "lib"
    
    if lib_path.exists():
        return lib_path
    
    return None


def find_hsl_library(env_path: Path | None = None) -> Optional[Path]:
    """
    Find the HSL library (libcoinhsl) in the given conda environment.
    
    Parameters
    ----------
    env_path : Path, optional
        Path to conda environment. If None, uses get_active_conda_env_path()
    
    Returns
    -------
    Optional[Path]
        Path to the HSL library file, or None if not found
    """
    if env_path is None:
        env_path = get_active_conda_env_path()
    
    if env_path is None:
        return None
    
    system = platform.system().lower()
    
    # Determine library file extension
    if system == "windows":
        lib_name = "libcoinhsl.dll"
        search_paths = [
            env_path / "Library" / "bin" / lib_name,
            env_path / "Library" / "lib" / lib_name,
            env_path / "bin" / lib_name,
            env_path / "Scripts" / lib_name,
        ]
    elif system == "darwin":
        lib_name = "libcoinhsl.dylib"
        search_paths = [
            env_path / "lib" / lib_name,
            env_path / "lib" / "libcoinhsl.dylib",
        ]
    else:  # Linux
        lib_name = "libcoinhsl.so"
        search_paths = [
            env_path / "lib" / lib_name,
            env_path / "lib" / "libcoinhsl.so",
        ]
    
    # Search for the library
    for search_path in search_paths:
        if search_path.exists():
            log.info(f"Found HSL library at: {search_path}")
            return search_path
    
    # Fallback: Check project CoinHSL directory using hsl_detector
    try:
        from campro.environment.hsl_detector import get_hsl_library_path
        
        hsl_lib_path = get_hsl_library_path()
        if hsl_lib_path and hsl_lib_path.exists():
            log.info(f"Found HSL library in project CoinHSL directory: {hsl_lib_path}")
            return hsl_lib_path
    except ImportError:
        # If hsl_detector not available, skip this check
        log.debug("hsl_detector not available; skipping project directory check")
        pass
    except Exception as e:
        # Log error but continue (this is a fallback path)
        log.warning(f"Error checking project CoinHSL directory: {e}")
        # Continue to return None (not found)
        pass
    
    log.debug("HSL library not found in conda environment or project folder")
    return None


def ensure_local_env_activated(
    project_root: Path | None = None,
    strict: bool = False,
) -> bool:
    """
    Validate that a local conda environment is activated or available.
    
    Parameters
    ----------
    project_root : Path, optional
        Root directory of the project. If None, attempts to detect it.
    strict : bool, default False
        If True, requires local env to be present. If False, allows fallback to global env.
    
    Returns
    -------
    bool
        True if a valid conda environment is available (local or global), False otherwise
    """
    if project_root is None:
        try:
            import campro
            project_root = Path(campro.__file__).parent.parent
        except ImportError:
            project_root = Path.cwd()
    
    # Check for local environment
    if is_local_conda_env_present(project_root):
        log.info("Local conda environment is available")
        return True
    
    if strict:
        log.error(
            "Local conda environment not found and strict mode is enabled. "
            "Please create a local environment using the install scripts."
        )
        return False
    
    # Fallback to global environment check
    env_path = get_active_conda_env_path(project_root)
    if env_path is not None:
        log.info(f"Using global conda environment: {env_path}")
        return True
    
    log.warning("No conda environment found (local or global)")
    return False


__all__ = [
    "get_active_conda_env_path",
    "get_python_executable_path",
    "get_lib_path",
    "find_hsl_library",
    "ensure_local_env_activated",
]
