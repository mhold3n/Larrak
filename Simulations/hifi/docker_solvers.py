"""Docker-based OpenFOAM solver wrapper.

Executes OpenFOAM solvers via Docker container instead of local install.
Avoids macOS dyld library conflicts and ensures reproducibility.

Usage:
    from Simulations.hifi.docker_openfoam import DockerOpenFOAM

    foam = DockerOpenFOAM()
    foam.run_solver("simpleFoam", case_dir="/path/to/case")
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from campro.logging import get_logger

log = get_logger(__name__)


class DockerOpenFOAM:
    """
    Docker wrapper for OpenFOAM solvers.

    Executes OpenFOAM commands inside the Docker container,
    mounting case directories as needed.
    """

    # Official OpenFOAM Foundation Docker image (v11 with ParaView)
    DEFAULT_IMAGE = "openfoam/openfoam11-paraview510"

    # OpenFOAM environment setup inside container
    BASHRC_PATH = "/opt/openfoam11/etc/bashrc"

    def __init__(
        self,
        image: str | None = None,
        container_name: str | None = None,
    ):
        """
        Initialize Docker OpenFOAM wrapper.

        Args:
            image: Docker image to use (default: openfoam2406-default)
            container_name: Named container from docker-compose, or None for fresh
        """
        self.image = image or self.DEFAULT_IMAGE
        self.container_name = container_name  # e.g., "larrak-openfoam-1"

    def run_solver(
        self,
        solver: str,
        case_dir: str | Path,
        args: list[str] | None = None,
        timeout: int = 3600,
        log_file: str | None = None,
    ) -> tuple[int, str, str]:
        """
        Run an OpenFOAM solver in Docker.

        Args:
            solver: Solver name (e.g., "simpleFoam", "chtMultiRegionFoam")
            case_dir: Path to case directory (will be mounted)
            args: Additional solver arguments
            timeout: Timeout in seconds
            log_file: Optional file to save solver log

        Returns:
            (return_code, stdout, stderr)
        """
        case_path = Path(case_dir).resolve()

        # Build the OpenFOAM command with environment setup
        foam_cmd = f"source {self.BASHRC_PATH} && cd /case && {solver}"
        if args:
            foam_cmd += " " + " ".join(args)

        # Docker command
        if self.container_name:
            # Use existing running container
            cmd = [
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-c",
                foam_cmd,
            ]
        else:
            # Run fresh container with case mounted
            # Use --entrypoint to bypass the interactive startup script
            cmd = [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "--entrypoint",
                "/bin/bash",
                "-v",
                f"{case_path}:/case:rw",
                "-w",
                "/case",
                self.image,
                "-c",
                foam_cmd,
            ]

        log.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if log_file:
                with open(log_file, "w") as f:
                    f.write(f"=== Command ===\n{' '.join(cmd)}\n\n")
                    f.write(f"=== stdout ===\n{result.stdout}\n\n")
                    f.write(f"=== stderr ===\n{result.stderr}\n")

            if result.returncode != 0:
                log.warning(f"OpenFOAM {solver} failed: exit {result.returncode}")
            else:
                log.info(f"OpenFOAM {solver} completed successfully")

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            log.error(f"OpenFOAM {solver} timed out after {timeout}s")
            return -1, "", "Timeout expired"

        except FileNotFoundError:
            log.error("Docker not found. Is Docker installed and running?")
            return -2, "", "Docker not found"

    def run_utility(
        self,
        utility: str,
        case_dir: str | Path,
        args: list[str] | None = None,
        timeout: int = 300,
    ) -> tuple[int, str, str]:
        """
        Run an OpenFOAM utility (blockMesh, decomposePar, etc).

        Args:
            utility: Utility name
            case_dir: Case directory
            args: Additional arguments
            timeout: Timeout in seconds

        Returns:
            (return_code, stdout, stderr)
        """
        return self.run_solver(utility, case_dir, args, timeout)

    def check_availability(self) -> bool:
        """Check if Docker OpenFOAM is available."""
        try:
            cmd = [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                self.image,
                "bash",
                "-c",
                f"source {self.BASHRC_PATH} && echo ready",
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.returncode == 0 and "ready" in result.stdout
        except Exception as e:
            log.warning(f"Docker OpenFOAM not available: {e}")
            return False

    def pull_image(self) -> bool:
        """Pull the OpenFOAM Docker image."""
        log.info(f"Pulling OpenFOAM image: {self.image}")
        try:
            result = subprocess.run(
                ["docker", "pull", self.image],
                capture_output=True,
                text=True,
                timeout=600,
            )
            return result.returncode == 0
        except Exception as e:
            log.error(f"Failed to pull image: {e}")
            return False


class DockerCalculiX:
    """
    Docker wrapper for CalculiX FEA solver.

    Similar to DockerOpenFOAM but for CalculiX.
    """

    DEFAULT_IMAGE = "calculix/ccx:latest"

    def __init__(self, image: str | None = None):
        self.image = image or self.DEFAULT_IMAGE

    def run_solver(
        self,
        input_file: str,
        case_dir: str | Path,
        timeout: int = 3600,
    ) -> tuple[int, str, str]:
        """
        Run CalculiX solver in Docker.

        Args:
            input_file: Input file name (without .inp extension)
            case_dir: Case directory
            timeout: Timeout in seconds

        Returns:
            (return_code, stdout, stderr)
        """
        case_path = Path(case_dir).resolve()

        cmd = [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "-v",
            f"{case_path}:/work:rw",
            "-w",
            "/work",
            self.image,
            "ccx",
            "-i",
            input_file,
        ]

        log.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return -1, "", "Timeout expired"
        except FileNotFoundError:
            return -2, "", "Docker not found"


__all__ = ["DockerOpenFOAM", "DockerCalculiX"]
