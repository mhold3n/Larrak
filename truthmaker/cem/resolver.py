"""
CEM Runtime Resolver: Cross-platform detection and management of the CEM service.

Follows the same patterns as campro/environment/context.py for platform interoperability.

Usage:
    from truthmaker.cem.resolver import cem_runtime
    
    # Get CEM executable path
    exe = cem_runtime.executable_path
    
    # Check if CEM is available
    if cem_runtime.is_available:
        with cem_runtime.start_service() as port:
            # Use gRPC client on this port
            ...
    
    # Or auto-connect (starts service if needed)
    client = cem_runtime.get_client()
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Iterator
from contextlib import contextmanager

# Reuse platform detection from campro
try:
    from campro.environment.context import ctx as platform_ctx
    PLATFORM = platform_ctx.platform
    PROJECT_ROOT = platform_ctx.paths.project_root
except ImportError:
    # Fallback if campro not available
    if sys.platform == "win32":
        PLATFORM = "windows"
    elif sys.platform == "darwin":
        PLATFORM = "macos"
    else:
        PLATFORM = "linux"
    PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class CEMRuntimeConfig:
    """Configuration for CEM runtime discovery and startup."""
    
    # Default gRPC port
    default_port: int = 50051
    
    # Timeout for CEM startup (seconds)
    startup_timeout: float = 30.0
    
    # Health check interval (seconds)
    health_check_interval: float = 1.0
    
    # CEM build configuration
    build_config: str = "Release"  # or "Debug"
    
    # .NET runtime requirements
    dotnet_version_min: str = "8.0"


@dataclass
class CEMRuntime:
    """
    Cross-platform CEM runtime resolver.
    
    Handles:
    - Detecting installed .NET SDK
    - Building CEM if needed
    - Starting/stopping the CEM gRPC service
    - Platform-specific executable paths
    """
    
    config: CEMRuntimeConfig = field(default_factory=CEMRuntimeConfig)
    
    @cached_property
    def cem_project_dir(self) -> Path:
        """Path to Larrak.CEM C# project."""
        return PROJECT_ROOT / "Larrak.CEM"
    
    @cached_property
    def dotnet_executable(self) -> Optional[Path]:
        """Find dotnet CLI executable."""
        dotnet = shutil.which("dotnet")
        return Path(dotnet) if dotnet else None
    
    @cached_property
    def dotnet_version(self) -> Optional[str]:
        """Get installed .NET SDK version."""
        if not self.dotnet_executable:
            return None
        
        try:
            result = subprocess.run(
                [str(self.dotnet_executable), "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    @property
    def is_dotnet_available(self) -> bool:
        """Check if .NET SDK is installed with minimum version."""
        version = self.dotnet_version
        if not version:
            return False
        
        try:
            major = int(version.split(".")[0])
            required = int(self.config.dotnet_version_min.split(".")[0])
            return major >= required
        except ValueError:
            return False
    
    @cached_property
    def executable_path(self) -> Optional[Path]:
        """
        Get path to CEM executable for current platform.
        
        On Windows: Larrak.CEM/src/Larrak.CEM.API/bin/Release/net8.0/Larrak.CEM.API.exe
        On macOS/Linux: Larrak.CEM/src/Larrak.CEM.API/bin/Release/net8.0/Larrak.CEM.API
        """
        api_dir = self.cem_project_dir / "src" / "Larrak.CEM.API"
        bin_dir = api_dir / "bin" / self.config.build_config / "net8.0"
        
        if PLATFORM == "windows":
            exe = bin_dir / "Larrak.CEM.API.exe"
        else:
            exe = bin_dir / "Larrak.CEM.API"
        
        return exe if exe.exists() else None
    
    @property
    def is_built(self) -> bool:
        """Check if CEM has been built."""
        return self.executable_path is not None
    
    @property
    def is_available(self) -> bool:
        """Check if CEM is available (built or can be built)."""
        return self.is_built or self.is_dotnet_available
    
    def build(self, force: bool = False) -> bool:
        """
        Build the CEM project.
        
        Args:
            force: Rebuild even if already built
            
        Returns:
            True if build succeeded
        """
        if not self.is_dotnet_available:
            print("[CEM] .NET SDK not found. Install from: https://dotnet.microsoft.com/download/dotnet/8.0")
            return False
        
        if self.is_built and not force:
            print("[CEM] Already built. Use force=True to rebuild.")
            return True
        
        print(f"[CEM] Building {self.config.build_config} configuration...")
        
        try:
            result = subprocess.run(
                [
                    str(self.dotnet_executable),
                    "build",
                    "-c", self.config.build_config,
                    str(self.cem_project_dir / "Larrak.CEM.sln")
                ],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                print("[CEM] Build succeeded")
                # Clear cached property
                if "executable_path" in self.__dict__:
                    del self.__dict__["executable_path"]
                return True
            else:
                print(f"[CEM] Build failed:\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("[CEM] Build timed out")
            return False
        except Exception as e:
            print(f"[CEM] Build error: {e}")
            return False
    
    @contextmanager
    def start_service(self, port: int = None) -> Iterator[int]:
        """
        Start the CEM gRPC service.
        
        Args:
            port: Port to listen on (default: config.default_port)
            
        Yields:
            The port the service is listening on
        """
        port = port or self.config.default_port
        
        if not self.is_built:
            if not self.build():
                raise RuntimeError("CEM build failed")
        
        exe = self.executable_path
        if not exe:
            raise RuntimeError("CEM executable not found after build")
        
        print(f"[CEM] Starting service on port {port}...")
        
        # Start the service
        env = os.environ.copy()
        env["CEM_PORT"] = str(port)
        
        process = subprocess.Popen(
            [str(exe)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(exe.parent)
        )
        
        try:
            # Wait for service to be ready
            if not self._wait_for_ready(port):
                process.terminate()
                raise RuntimeError("CEM service failed to start")
            
            yield port
            
        finally:
            print("[CEM] Stopping service...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
    def _wait_for_ready(self, port: int) -> bool:
        """Wait for CEM service to be ready."""
        import time
        
        start = time.time()
        while time.time() - start < self.config.startup_timeout:
            if self._check_health(port):
                return True
            time.sleep(self.config.health_check_interval)
        
        return False
    
    def _check_health(self, port: int) -> bool:
        """Check if CEM service is responding."""
        import socket
        
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                return True
        except (socket.error, socket.timeout):
            return False
    
    def get_client(self, auto_start: bool = True):
        """
        Get a CEM client, optionally starting the service.
        
        Args:
            auto_start: If True and service not running, start it
            
        Returns:
            CEMClient configured for local service
        """
        from truthmaker.cem.client import CEMClient
        
        port = self.config.default_port
        
        # Check if service is running
        if self._check_health(port):
            return CEMClient(port=port, mock=False)
        
        # Service not running
        if not auto_start:
            print("[CEM] Service not running, falling back to mock mode")
            return CEMClient(mock=True)
        
        if not self.is_available:
            print("[CEM] Not available, using mock mode")
            return CEMClient(mock=True)
        
        # TODO: Start service in background
        # For now, fall back to mock
        print("[CEM] Auto-start not yet implemented, using mock mode")
        return CEMClient(mock=True)


# Module-level singleton
cem_runtime = CEMRuntime()

# Convenience functions
def get_cem_runtime() -> CEMRuntime:
    """Get the CEM runtime singleton."""
    return cem_runtime

def is_cem_available() -> bool:
    """Check if CEM is available."""
    return cem_runtime.is_available

def build_cem(force: bool = False) -> bool:
    """Build the CEM project."""
    return cem_runtime.build(force)


__all__ = [
    "CEMRuntime",
    "CEMRuntimeConfig", 
    "cem_runtime",
    "get_cem_runtime",
    "is_cem_available",
    "build_cem",
]
