#!/usr/bin/env python3
"""
Environment setup script for Larrak.

This script automates the creation and setup of the conda environment
required for Larrak to run properly.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Avoid import-time validation before env exists
os.environ.setdefault("CAMPRO_SKIP_VALIDATION", "1")

from campro.environment.platform_detector import (  # noqa: E402
    get_local_conda_env_path,
    is_local_conda_env_present,
)
from campro.environment.validator import validate_environment  # noqa: E402
from campro.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def run_command(
    cmd: list, check: bool = True, stream: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a command and return the result.

    Args:
        cmd: Command to run as list
        check: Whether to raise exception on non-zero exit

    Returns:
        Completed process result
    """
    log.info(f"Running command: {' '.join(cmd)}")
    try:
        if stream:
            # Inherit stdout/stderr so progress is visible to the user
            result = subprocess.run(
                cmd,
                check=check,
                cwd=project_root,
            )
            return result
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.stdout:
            log.info(f"Command output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed: {e}")
        if e.stderr:
            log.error(f"Error output: {e.stderr}")
        raise


def check_conda_available() -> bool:
    """Check if conda is available in the system."""
    try:
        result = run_command(["conda", "--version"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_mamba_available() -> bool:
    """Check if mamba is available in the system."""
    try:
        result = run_command(["mamba", "--version"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_conda_environment(
    env_file: str,
    use_mamba: bool = False,
    use_local: bool = True,
) -> bool:
    """
    Create conda environment from environment file.

    Args:
        env_file: Path to environment.yml file
        use_mamba: Whether to use mamba instead of conda
        use_local: Whether to create environment locally (OS-specific) or globally

    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(env_file):
        log.error(f"Environment file not found: {env_file}")
        return False
    # Validate non-empty env file to avoid SpecNotFound empty error
    try:
        with open(env_file, encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            log.error(f"Environment file is empty: {env_file}")
            return False
    except Exception as read_exc:
        log.error(f"Error reading environment file: {read_exc}")
        return False

    package_manager = "mamba" if use_mamba else "conda"

    try:
        if use_local:
            # Create local OS-specific conda environment
            local_env_path = get_local_conda_env_path(project_root)
            env_path_str = str(local_env_path)

            if is_local_conda_env_present(project_root):
                log.info(
                    f"Local environment already exists at '{env_path_str}'. Updating...",
                )
                cmd = [
                    package_manager,
                    "env",
                    "update",
                    "-f",
                    env_file,
                    "--prefix",
                    env_path_str,
                ]
            else:
                log.info(f"Creating new local environment at '{env_path_str}'...")
                cmd = [
                    package_manager,
                    "env",
                    "create",
                    "-f",
                    env_file,
                    "--prefix",
                    env_path_str,
                ]
        else:
            # Fallback to global named environment (legacy behavior)
            env_name = "larrak"
            result = run_command([package_manager, "env", "list"], check=False)

            if env_name in result.stdout:
                log.info(f"Environment '{env_name}' already exists. Updating...")
                cmd = [
                    package_manager,
                    "env",
                    "update",
                    "-f",
                    env_file,
                    "--name",
                    env_name,
                ]
            else:
                log.info(f"Creating new environment '{env_name}'...")
                cmd = [package_manager, "env", "create", "-f", env_file]

        # Prefer libmamba solver for speed on conda
        if package_manager == "conda":
            try:
                run_command(
                    ["conda", "config", "--set", "solver", "libmamba"], check=False,
                )
                run_command(
                    ["conda", "config", "--set", "channel_priority", "strict"],
                    check=False,
                )
            except Exception:
                pass

        print(
            "⏳ Creating/updating conda environment... (this may take several minutes)",
        )
        # Stream conda/mamba output so user sees native progress bars
        run_command(cmd, stream=True)
        log.info(f"Environment setup completed successfully using {package_manager}")
        return True

    except subprocess.CalledProcessError as e:
        log.error(f"Failed to create/update environment: {e}")
        return False


def validate_setup() -> bool:
    """
    Validate that the environment setup was successful.

    Returns:
        True if validation passes, False otherwise
    """
    log.info("Validating environment setup...")

    try:
        results = validate_environment()
        overall_status = results["summary"]["overall_status"]

        if overall_status.value == "pass":
            log.info("✓ Environment validation passed")
            return True
        if overall_status.value == "warning":
            log.warning("⚠ Environment validation passed with warnings")
            return True
        log.error("✗ Environment validation failed")
        return False

    except Exception as e:
        log.error(f"Error during validation: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup Larrak environment using conda/mamba",
    )
    parser.add_argument(
        "--env-file",
        default="environment.yml",
        help="Environment file to use (default: environment.yml)",
    )
    parser.add_argument(
        "--use-mamba",
        action="store_true",
        help="Use mamba instead of conda (faster)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation after setup",
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Create global named environment instead of local OS-specific environment",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    print("Larrak Environment Setup")
    print("=" * 50)

    # Check for conda/mamba
    if args.use_mamba and check_mamba_available():
        package_manager = "mamba"
        log.info("Using mamba for environment setup")
    elif check_mamba_available():
        package_manager = "mamba"
        log.info("Using mamba for environment setup (auto-detected)")
    elif check_conda_available():
        package_manager = "conda"
        log.info("Using conda for environment setup")
    else:
        print("❌ Neither conda nor mamba is available!")
        print("\nPlease install Miniconda or Miniforge:")
        print("  - Miniconda: https://docs.conda.io/en/latest/miniconda.html")
        print("  - Miniforge: https://github.com/conda-forge/miniforge")
        print("\nAfter installation, restart your terminal and run this script again.")
        sys.exit(1)

    # Create environment
    env_file = args.env_file
    if not os.path.isabs(env_file):
        env_file = os.path.join(project_root, env_file)

    print(f"📦 Creating environment from {env_file}...")
    use_local = not args.no_local
    if use_local:
        local_env_path = get_local_conda_env_path(project_root)
        print(f"📍 Using local environment path: {local_env_path}")
    
    success = create_conda_environment(
        env_file,
        use_mamba=(package_manager == "mamba"),
        use_local=use_local,
    )

    if not success:
        print("❌ Failed to create environment")
        # Retry with mamba if available and not already used
        if package_manager == "conda" and check_mamba_available():
            print("🔄 Retrying with mamba (faster solver)...")
            if create_conda_environment(
                env_file,
                use_mamba=True,
                use_local=use_local,
            ):
                success = True
        if not success:
            sys.exit(1)

    # Validate setup
    if not args.skip_validation:
        print("🔍 Validating environment...")
        if validate_setup():
            print("✅ Environment setup completed successfully!")
        else:
            print("⚠️  Environment setup completed with issues")
            print("Run 'python scripts/check_environment.py' for details")
    else:
        print("✅ Environment setup completed (validation skipped)")

    # Print next steps
    print("\nNext steps:")
    print("1. Activate the environment:")
    if use_local:
        local_env_path = get_local_conda_env_path(project_root)
        print(f"   conda activate {local_env_path}")
    else:
        print("   conda activate larrak")
    print("\n2. Verify the installation:")
    print("   python scripts/check_environment.py")
    print("\n3. Run the GUI:")
    print("   python cam_motion_gui.py")
    print("\n4. Or run examples:")
    print("   python scripts/example_usage.py")


if __name__ == "__main__":
    main()
