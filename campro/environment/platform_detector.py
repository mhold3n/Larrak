"""
Platform detection utilities.

Provides functions to detect operating system, CPU architecture,
and availability of package managers (conda, mamba, brew, apt, choco).
Also provides utilities for managing local conda environments.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from campro.logging import get_logger

log = get_logger(__name__)

# OS detection constants for cross-platform compatibility
IS_WINDOWS = platform.system().lower() == "windows"
IS_MACOS = platform.system().lower() == "darwin"
IS_LINUX = platform.system().lower() == "linux"


@dataclass
class PlatformInfo:
    os_name: str
    arch: str
    has_conda: bool
    has_mamba: bool
    has_brew: bool
    has_apt: bool
    has_choco: bool


def _has_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def detect_platform() -> PlatformInfo:
    """Detect platform and common package managers/architecture."""
    system = platform.system().lower()  # 'darwin', 'linux', 'windows'
    arch = platform.machine().lower()

    # Normalize architecture
    if arch in {"x86_64", "amd64"}:
        arch = "x86_64"
    elif arch in {"arm64", "aarch64"}:
        arch = "arm64"

    info = PlatformInfo(
        os_name=system,
        arch=arch,
        has_conda=_has_cmd("conda"),
        has_mamba=_has_cmd("mamba"),
        has_brew=_has_cmd("brew"),
        has_apt=_has_cmd("apt") or _has_cmd("apt-get"),
        has_choco=_has_cmd("choco"),
    )

    log.info(
        "Platform detected: os=%s arch=%s conda=%s mamba=%s brew=%s apt=%s choco=%s",
        info.os_name,
        info.arch,
        info.has_conda,
        info.has_mamba,
        info.has_brew,
        info.has_apt,
        info.has_choco,
    )

    return info


def get_local_conda_env_name() -> str:
    """
    Get the local conda environment folder name based on the current OS.
    
    Returns
    -------
    str
        Environment folder name: 'conda_env_windows', 'conda_env_macos', or 'conda_env_linux'
    """
    system = platform.system().lower()
    if system == "windows":
        return "conda_env_windows"
    elif system == "darwin":
        return "conda_env_macos"
    elif system == "linux":
        return "conda_env_linux"
    else:
        # Fallback for unknown systems
        return f"conda_env_{system}"


def get_local_conda_env_path(project_root: Path | None = None) -> Path:
    """
    Get the path to the local conda environment for the current OS.
    
    Parameters
    ----------
    project_root : Path, optional
        Root directory of the project. If None, attempts to detect it.
    
    Returns
    -------
    Path
        Path to the local conda environment directory
    """
    if project_root is None:
        # Try to detect project root by looking for campro package
        current_file = Path(__file__).resolve()
        # Go up from campro/environment/platform_detector.py to project root
        project_root = current_file.parent.parent.parent
    
    env_name = get_local_conda_env_name()
    return project_root / env_name


def get_local_conda_env_activate_command() -> str:
    """
    Get the command to activate the local conda environment for the current OS.
    
    Returns
    -------
    str
        Activation command appropriate for the current OS and shell
    """
    system = platform.system().lower()
    env_name = get_local_conda_env_name()
    
    if system == "windows":
        # PowerShell activation
        return f"conda activate {env_name}"
    else:
        # Bash/Zsh activation (macOS/Linux)
        return f"conda activate {env_name}"


def is_local_conda_env_present(project_root: Path | None = None) -> bool:
    """
    Check if the local conda environment exists for the current OS.
    
    Parameters
    ----------
    project_root : Path, optional
        Root directory of the project. If None, attempts to detect it.
    
    Returns
    -------
    bool
        True if the local conda environment directory exists and appears valid
    """
    env_path = get_local_conda_env_path(project_root)
    if not env_path.exists():
        return False
    
    # Check for typical conda environment markers
    markers = ["conda-meta", "bin", "Scripts", "lib", "Library"]
    has_markers = any((env_path / marker).exists() for marker in markers)
    
    return has_markers


__all__ = [
    "IS_LINUX",
    "IS_MACOS",
    "IS_WINDOWS",
    "PlatformInfo",
    "detect_platform",
    "get_local_conda_env_activate_command",
    "get_local_conda_env_name",
    "get_local_conda_env_path",
    "is_local_conda_env_present",
]
