"""
Platform detection utilities.

Provides functions to detect operating system, CPU architecture,
and availability of package managers (conda, mamba, brew, apt, choco).
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass

from campro.logging import get_logger

log = get_logger(__name__)


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


__all__ = ["PlatformInfo", "detect_platform"]
