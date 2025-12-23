"""
Unified platform context for cross-platform resource and workflow management.

This module provides a singleton `PlatformContext` that centralizes all
platform-specific logic, replacing scattered `if sys.platform == ...` checks.

Usage:
    from campro.environment.context import ctx
    
    # Check platform
    if ctx.platform == "macos":
        ...
    
    # Get resources
    hsl_path = ctx.resources.hsl_library
    python_exe = ctx.resources.python_executable
    
    # Platform-safe workflows
    ctx.workflows.exit_safely(0)
    
    # Paths
    root = ctx.paths.project_root
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal

from campro.environment.platform_detector import (
    IS_MACOS,
    IS_WINDOWS,
    PlatformInfo,
    detect_platform,
)
from campro.logging import get_logger

log = get_logger(__name__)

PlatformName = Literal["windows", "macos", "linux"]


@dataclass
class ResourceResolver:
    """Resolves platform-specific resources (libraries, paths)."""

    _platform: PlatformName
    _project_root: Path

    @cached_property
    def hsl_library(self) -> Path | None:
        """Get HSL library path for current platform."""
        from campro.environment.hsl_detector import get_hsl_library_path

        return get_hsl_library_path()

    @cached_property
    def python_executable(self) -> Path | None:
        """Get Python executable for current platform."""
        from campro.environment.env_manager import get_python_executable_path

        return get_python_executable_path(self._project_root)

    @cached_property
    def conda_env_path(self) -> Path | None:
        """Get active conda environment path."""
        from campro.environment.env_manager import get_active_conda_env_path

        return get_active_conda_env_path(self._project_root)

    @cached_property
    def library_dir(self) -> Path | None:
        """Get platform-specific library directory."""
        from campro.environment.env_manager import get_lib_path

        return get_lib_path()


@dataclass
class WorkflowHandler:
    """Platform-specific workflow strategies."""

    _platform: PlatformName

    def exit_safely(self, code: int = 0) -> None:
        """
        Exit the process safely, handling platform-specific teardown issues.

        On macOS, uses os._exit() to bypass CasADi/IPOPT segfault on teardown.
        On Windows/Linux, uses standard sys.exit().
        """
        import os

        if self._platform == "macos":
            log.debug(f"Using os._exit({code}) for macOS to avoid teardown segfault")
            os._exit(code)
        else:
            sys.exit(code)

    def requires_process_isolation(self) -> bool:
        """
        Check if CasADi calls require subprocess isolation.

        Returns True on macOS where CasADi/IPOPT teardown can segfault.
        """
        return self._platform == "macos"

    def run_isolated(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Run function in isolation if required by platform.

        On macOS, spawns a subprocess to avoid segfaults.
        On other platforms, runs directly.
        """
        from campro.testing.utils import run_isolated

        return run_isolated(func, *args, **kwargs)

    @property
    def shell_extension(self) -> str:
        """Get shell script extension for platform."""
        return ".ps1" if self._platform == "windows" else ".sh"

    @property
    def executable_extension(self) -> str:
        """Get executable extension for platform."""
        return ".exe" if self._platform == "windows" else ""


@dataclass
class PathResolver:
    """Cross-platform path resolution."""

    _platform: PlatformName
    _project_root: Path

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self._project_root

    def output_dir(self, name: str) -> Path:
        """Get output directory, creating if needed."""
        path = self._project_root / "out" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def logs_dir(self) -> Path:
        """Get logs directory."""
        path = self._project_root / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def libraries_dir(self) -> Path:
        """Get Libraries directory containing CoinHSL and other deps."""
        return self._project_root / "Libraries"


@dataclass
class PlatformContext:
    """
    Singleton context providing unified access to platform-specific resources.

    Access via module-level `ctx` instance:
        from campro.environment.context import ctx

        print(ctx.platform)  # 'windows', 'macos', or 'linux'
        hsl = ctx.resources.hsl_library
        ctx.workflows.exit_safely(0)
    """

    platform: PlatformName
    info: PlatformInfo
    resources: ResourceResolver = field(init=False)
    workflows: WorkflowHandler = field(init=False)
    paths: PathResolver = field(init=False)

    def __post_init__(self) -> None:
        project_root = self._detect_project_root()
        self.resources = ResourceResolver(self.platform, project_root)
        self.workflows = WorkflowHandler(self.platform)
        self.paths = PathResolver(self.platform, project_root)
        log.debug(
            f"PlatformContext initialized: platform={self.platform}, "
            f"project_root={project_root}"
        )

    @staticmethod
    def _detect_project_root() -> Path:
        """Detect project root from campro package location."""
        try:
            import campro

            return Path(campro.__file__).parent.parent
        except ImportError:
            return Path.cwd()

    @classmethod
    def initialize(cls) -> "PlatformContext":
        """Create and return the platform context."""
        if IS_WINDOWS:
            platform_name: PlatformName = "windows"
        elif IS_MACOS:
            platform_name = "macos"
        else:
            platform_name = "linux"

        info = detect_platform()
        return cls(platform=platform_name, info=info)


# Module-level singleton - initialized on first import
ctx = PlatformContext.initialize()

__all__ = ["PlatformContext", "ctx", "PlatformName"]
