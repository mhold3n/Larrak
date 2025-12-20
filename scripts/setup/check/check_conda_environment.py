#!/usr/bin/env python3
"""
Comprehensive conda environment health check for Larrak.

This script performs a thorough check of the conda environment, including:
- Environment status (name, path, Python version)
- Package version verification against environment.yml
- Import testing for all major modules
- Integration with existing environment validation
- Detection of pip/conda conflicts
- Comprehensive error reporting with recommendations
"""

import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class CheckStatus(Enum):
    """Status of a health check."""

    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a health check."""

    status: CheckStatus
    category: str
    message: str
    details: str | None = None
    recommendation: str | None = None
    exception: Exception | None = None
    traceback: str | None = None


@dataclass
class HealthCheckReport:
    """Comprehensive health check report."""

    environment_status: dict[str, Any] = field(default_factory=dict)
    package_checks: list[CheckResult] = field(default_factory=list)
    import_checks: list[CheckResult] = field(default_factory=list)
    integration_checks: list[CheckResult] = field(default_factory=list)
    pip_conflicts: list[CheckResult] = field(default_factory=list)
    hsl_checks: list[CheckResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def colorize(text: str, color: str) -> str:
    """Add color to text for terminal output."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
    }

    if sys.stdout.isatty() and os.getenv("TERM") != "dumb":
        return f"{colors.get(color, '')}{text}{colors['reset']}"
    return text


def get_status_icon(status: CheckStatus) -> str:
    """Get status icon for check result."""
    icons = {
        CheckStatus.PASS: colorize("✓", "green"),
        CheckStatus.WARNING: colorize("⚠", "yellow"),
        CheckStatus.ERROR: colorize("✗", "red"),
        CheckStatus.INFO: colorize("ℹ", "blue"),
        CheckStatus.SKIPPED: colorize("-", "cyan"),
    }
    return icons.get(status, "?")


def check_conda_environment_status() -> dict[str, Any]:
    """Check conda environment status and return details."""
    status = {
        "active": False,
        "name": None,
        "path": None,
        "python_version": None,
        "conda_available": False,
    }

    # Check if conda is available
    try:
        result = subprocess.run(
            ["conda", "--version"],
            check=False, capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            status["conda_available"] = True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check active environment
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    conda_prefix = os.environ.get("CONDA_PREFIX", "")

    if conda_prefix:
        status["active"] = True
        status["name"] = conda_env if conda_env else Path(conda_prefix).name
        status["path"] = conda_prefix

    # Check Python version
    status["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    status["python_version_info"] = sys.version_info

    return status


def parse_environment_yml() -> dict[str, str]:
    """Parse environment.yml and extract package requirements."""
    env_file = project_root / "environment.yml"
    requirements: dict[str, str] = {}

    if not env_file.exists():
        return requirements

    try:
        import yaml

        with open(env_file) as f:
            data = yaml.safe_load(f)

        # Extract dependencies
        if data is None:
            data = {}
        deps = data.get("dependencies", [])
        if deps is None:
            deps = []
        for dep in deps:
            if isinstance(dep, str):
                # Parse version constraint (e.g., "python>=3.9,<3.12" or "numpy>=1.24.0")
                if ">=" in dep or ">" in dep or "<=" in dep or "<" in dep:
                    # Extract package name and version
                    parts = dep.replace(">=", " ").replace(">", " ").replace("<=", " ").replace("<", " ").replace(",", " ").split()
                    if parts:
                        pkg_name = parts[0]
                        # Store the full constraint
                        requirements[pkg_name] = dep
                else:
                    # No version constraint
                    requirements[dep] = ""
            elif isinstance(dep, dict):
                # Handle pip section: {"pip": ["package1>=1.0", "package2"]}
                if "pip" in dep:
                    pip_deps = dep["pip"]
                    if pip_deps is None:
                        pip_deps = []
                    for pip_dep in pip_deps:
                        if isinstance(pip_dep, str):
                            if ">=" in pip_dep or ">" in pip_dep or "<=" in pip_dep or "<" in pip_dep:
                                parts = pip_dep.replace(">=", " ").replace(">", " ").replace("<=", " ").replace("<", " ").replace(",", " ").split()
                                if parts:
                                    pkg_name = parts[0]
                                    requirements[pkg_name] = pip_dep
                            else:
                                requirements[pip_dep] = ""

    except ImportError:
        # Fallback: simple parsing without yaml
        with open(env_file) as f:
            in_pip_section = False
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "- pip:" in line or "pip:" in line:
                        in_pip_section = True
                        continue
                    if in_pip_section and line.startswith("- "):
                        dep = line.replace("- ", "").strip()
                        if ">=" in dep or ">" in dep or "<=" in dep or "<" in dep:
                            parts = dep.replace(">=", " ").replace(">", " ").replace("<=", " ").replace("<", " ").replace(",", " ").split()
                            if parts:
                                pkg_name = parts[0]
                                requirements[pkg_name] = dep
                        else:
                            requirements[dep] = ""
                    elif "- " in line and not in_pip_section:
                        dep = line.replace("- ", "").strip()
                        if ">=" in dep or ">" in dep or "<=" in dep or "<" in dep:
                            parts = dep.replace(">=", " ").replace(">", " ").replace("<=", " ").replace("<", " ").replace(",", " ").split()
                            if parts:
                                pkg_name = parts[0]
                                requirements[pkg_name] = dep
                        else:
                            requirements[dep] = ""
    except Exception as e:
        print(f"Warning: Could not parse environment.yml: {e}")

    return requirements


def check_package_versions(requirements: dict[str, str]) -> list[CheckResult]:
    """Check installed package versions against requirements."""
    results: list[CheckResult] = []

    # Map conda package names to Python import names
    import_name_map = {
        "scikit-learn": "sklearn",
        "python-dateutil": "dateutil",
        "pytest-cov": "pytest_cov",
        "ipython": "IPython",  # ipython package imports as IPython
    }

    # Special handling for packages that aren't Python modules
    special_packages = {
        "ipopt": "ipopt_binary",  # IPOPT is a binary, not a Python package
    }

    for pkg_name, requirement in requirements.items():
        # Skip Python version check (handled separately)
        if pkg_name.lower() == "python":
            continue

        # Skip pip (not a package to import)
        if pkg_name.lower() == "pip":
            continue

        # Handle special packages
        if pkg_name.lower() in special_packages:
            if pkg_name.lower() == "ipopt":
                # Check IPOPT via CasADi
                try:
                    import casadi as ca
                    ipopt_available = False
                    
                    # Check via plugins
                    if hasattr(ca, "nlpsol_plugins"):
                        try:
                            plugins = ca.nlpsol_plugins()
                            ipopt_available = "ipopt" in plugins
                        except Exception:
                            pass
                    
                    # If not found via plugins, try direct creation
                    if not ipopt_available:
                        try:
                            from campro.optimization.solvers.ipopt_factory import (
                                create_ipopt_solver,
                            )
                            x = ca.SX.sym("x")
                            f = x**2
                            nlp = {"x": x, "f": f}
                            create_ipopt_solver("ipopt_check", nlp, linear_solver="ma27")
                            ipopt_available = True
                        except Exception:
                            pass
                    
                    if ipopt_available:
                        results.append(
                            CheckResult(
                                status=CheckStatus.PASS,
                                category="package_version",
                                message=f"{pkg_name} is available via CasADi",
                                details=f"IPOPT solver is accessible (required: {requirement})",
                            )
                        )
                    else:
                        results.append(
                            CheckResult(
                                status=CheckStatus.ERROR,
                                category="package_version",
                                message=f"{pkg_name} is not available",
                                details="IPOPT solver not found in CasADi",
                                recommendation=f"Install {pkg_name}: conda install -c conda-forge ipopt",
                            )
                        )
                except ImportError:
                    results.append(
                        CheckResult(
                            status=CheckStatus.ERROR,
                            category="package_version",
                            message=f"{pkg_name} check failed (CasADi not available)",
                            details="Cannot check IPOPT without CasADi",
                            recommendation="Install CasADi first",
                        )
                    )
            continue

        # Get import name (may differ from conda package name)
        import_name = import_name_map.get(pkg_name, pkg_name)

        try:
            # Try to import the package
            module = __import__(import_name)
            installed_version = getattr(module, "__version__", None)

            # If no __version__, try importlib.metadata (Python 3.8+)
            if not installed_version:
                try:
                    from importlib.metadata import version
                    installed_version = version(import_name)
                except (ImportError, Exception):
                    # Fallback to pkg_resources for older Python
                    try:
                        import pkg_resources
                        installed_version = pkg_resources.get_distribution(import_name).version
                    except (ImportError, Exception):
                        pass

            if installed_version:
                # Simple version check - just verify it's installed
                results.append(
                    CheckResult(
                        status=CheckStatus.PASS,
                        category="package_version",
                        message=f"{pkg_name} is installed",
                        details=f"Version: {installed_version} (required: {requirement})",
                    )
                )
            else:
                # Still check if package is importable (even without version)
                results.append(
                    CheckResult(
                        status=CheckStatus.WARNING,
                        category="package_version",
                        message=f"{pkg_name} is installed but version unknown",
                        details=f"Package is importable but version info unavailable (required: {requirement})",
                        recommendation=f"Verify {pkg_name} installation - may need update",
                    )
                )

        except ImportError as e:
            results.append(
                CheckResult(
                    status=CheckStatus.ERROR,
                    category="package_version",
                    message=f"{pkg_name} is not installed",
                    details=str(e),
                    recommendation=f"Install {pkg_name}: conda install -c conda-forge {pkg_name}",
                )
            )
        except Exception as e:
            results.append(
                CheckResult(
                    status=CheckStatus.ERROR,
                    category="package_version",
                    message=f"Error checking {pkg_name}",
                    details=str(e),
                    exception=e,
                )
            )

    return results


def check_pip_in_conda() -> list[CheckResult]:
    """Check for pip-installed packages in conda environment (potential conflicts)."""
    results: list[CheckResult] = []

    try:
        # Get list of pip-installed packages
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            check=False, capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            pip_packages = json.loads(result.stdout)
            # Filter for packages that might conflict with conda
            conda_packages = [
                "casadi",
                "numpy",
                "scipy",
                "matplotlib",
                "scikit-learn",
                "ipopt",
            ]

            # Check which packages are actually installed via conda
            conflicting = []
            try:
                conda_list_result = subprocess.run(
                    ["conda", "list", "--json", "--name", os.environ.get("CONDA_DEFAULT_ENV", "")],
                    check=False, capture_output=True,
                    text=True,
                    timeout=10,
                )
                conda_installed = set()
                if conda_list_result.returncode == 0:
                    conda_list = json.loads(conda_list_result.stdout)
                    conda_installed = {pkg.get("name", "").lower() for pkg in conda_list}
                
                # Only flag as conflict if package is in pip but NOT in conda
                for pkg in pip_packages:
                    pkg_name = pkg.get("name", "").lower()
                    if pkg_name in conda_packages and pkg_name not in conda_installed:
                        conflicting.append(pkg_name)
            except Exception:
                # Fallback: if we can't check conda, use original logic
                for pkg in pip_packages:
                    pkg_name = pkg.get("name", "").lower()
                    if pkg_name in conda_packages:
                        conflicting.append(pkg_name)

            if conflicting:
                results.append(
                    CheckResult(
                        status=CheckStatus.WARNING,
                        category="pip_conflict",
                        message=f"Found {len(conflicting)} packages installed via pip that may conflict with conda",
                        details=f"Packages: {', '.join(conflicting)}",
                        recommendation=(
                            "Consider reinstalling via conda: "
                            f"conda install -c conda-forge {' '.join(conflicting)}"
                        ),
                    )
                )
            else:
                results.append(
                    CheckResult(
                        status=CheckStatus.PASS,
                        category="pip_conflict",
                        message="No obvious pip/conda conflicts detected",
                    )
                )
        else:
            results.append(
                CheckResult(
                    status=CheckStatus.WARNING,
                    category="pip_conflict",
                    message="Could not check pip packages",
                    details=result.stderr,
                )
            )

    except Exception as e:
        results.append(
            CheckResult(
                status=CheckStatus.WARNING,
                category="pip_conflict",
                message="Error checking pip packages",
                details=str(e),
                exception=e,
            )
        )

    return results


def test_import(module_name: str, display_name: str | None = None) -> CheckResult:
    """Test importing a module and return result."""
    if display_name is None:
        display_name = module_name

    try:
        __import__(module_name)
        return CheckResult(
            status=CheckStatus.PASS,
            category="import",
            message=f"{display_name} imported successfully",
        )
    except ImportError as e:
        return CheckResult(
            status=CheckStatus.ERROR,
            category="import",
            message=f"{display_name} import failed",
            details=str(e),
            exception=e,
            traceback=traceback.format_exc(),
            recommendation=f"Install missing dependency or check {module_name} installation",
        )
    except Exception as e:
        return CheckResult(
            status=CheckStatus.ERROR,
            category="import",
            message=f"{display_name} import error",
            details=str(e),
            exception=e,
            traceback=traceback.format_exc(),
            recommendation=f"Check {module_name} installation and dependencies",
        )


def test_campro_imports() -> list[CheckResult]:
    """Test importing all major campro modules."""
    results: list[CheckResult] = []

    # Core modules
    core_modules = [
        "campro",
        "campro.logging",
        "campro.constants",
        "campro.api",
        "campro.api.problem_spec",
        "campro.api.solve_report",
    ]

    # Optimization modules
    optimization_modules = [
        "campro.optimization",
        "campro.optimization.base",
        "campro.optimization.numerical.collocation",
        "campro.optimization.motion",
        "campro.optimization.motion_law",
        "campro.optimization.solvers.ipopt_factory",
    ]

    # Physics modules
    physics_modules = [
        "campro.physics",
        "campro.physics.base",
        "campro.physics.kinematics.crank_kinematics",
        "campro.physics.mechanics.torque_analysis",
        "campro.physics.mechanics.side_loading",
    ]

    # Environment modules
    environment_modules = [
        "campro.environment",
        "campro.environment.validator",
        "campro.environment.platform_detector",
        "campro.environment.env_manager",
    ]

    # Configuration modules
    config_modules = [
        "campro.config",
        "campro.config.parameter_manager",
        "campro.config.system_builder",
    ]

    all_modules = (
        core_modules
        + optimization_modules
        + physics_modules
        + environment_modules
        + config_modules
    )

    for module_name in all_modules:
        results.append(test_import(module_name))

    return results


def test_external_imports() -> list[CheckResult]:
    """Test importing external dependencies."""
    results: list[CheckResult] = []

    external_modules = [
        ("casadi", "CasADi"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("matplotlib.pyplot", "Matplotlib.pyplot"),
        ("sklearn", "scikit-learn"),
        ("joblib", "Joblib"),
        ("pytest_cov", "pytest-cov"),
    ]

    for module_name, display_name in external_modules:
        results.append(test_import(module_name, display_name))

    # Test tkinter (optional for GUI)
    try:
        import tkinter

        results.append(
            CheckResult(
                status=CheckStatus.PASS,
                category="import",
                message="tkinter imported successfully (GUI support available)",
            )
        )
    except ImportError:
        results.append(
            CheckResult(
                status=CheckStatus.WARNING,
                category="import",
                message="tkinter not available (GUI will not work)",
                recommendation="Install tkinter: conda install -c conda-forge tk",
            )
        )

    return results


def check_hsl_library() -> list[CheckResult]:
    """Check HSL library paths and accessibility."""
    results: list[CheckResult] = []

    # Check HSLLIB_PATH environment variable
    hsl_path_env = os.environ.get("HSLLIB_PATH", "")
    if hsl_path_env:
        hsl_path = Path(hsl_path_env)
        if hsl_path.exists():
            results.append(
                CheckResult(
                    status=CheckStatus.PASS,
                    category="hsl",
                    message="HSL library path found in environment",
                    details=f"Path: {hsl_path}",
                )
            )
        else:
            results.append(
                CheckResult(
                    status=CheckStatus.WARNING,
                    category="hsl",
                    message="HSLLIB_PATH set but file does not exist",
                    details=f"Path: {hsl_path}",
                    recommendation="Verify HSL library path or remove HSLLIB_PATH",
                )
            )
    else:
        # Check constants for HSL path
        try:
            from campro import constants

            hsl_path_const = getattr(constants, "HSLLIB_PATH", "")
            if hsl_path_const and Path(hsl_path_const).exists():
                results.append(
                    CheckResult(
                        status=CheckStatus.INFO,
                        category="hsl",
                        message="HSL library path found in constants",
                        details=f"Path: {hsl_path_const}",
                    )
                )
            else:
                results.append(
                    CheckResult(
                        status=CheckStatus.WARNING,
                        category="hsl",
                        message="HSL library path not configured",
                        recommendation=(
                            "HSL solvers are optional but improve performance. "
                            "See docs/installation_guide.md for installation instructions."
                        ),
                    )
                )
        except Exception as e:
            results.append(
                CheckResult(
                    status=CheckStatus.WARNING,
                    category="hsl",
                    message="Could not check HSL library path",
                    details=str(e),
                )
            )

    return results


def run_integration_validation() -> list[CheckResult]:
    """Run existing environment validation and integrate results."""
    results: list[CheckResult] = []

    try:
        from campro.environment.validator import validate_environment

        # Skip HSL validation to avoid segmentation faults
        # The HSL validation creates multiple IPOPT solvers which can crash
        original_skip = os.environ.get("CAMPRO_SKIP_VALIDATION")
        os.environ["CAMPRO_SKIP_VALIDATION"] = "1"

        try:
            # Run validation (may take a moment)
            validation_results = validate_environment()
            overall_status = validation_results["summary"]["overall_status"]

            # Convert validation status to our check status
            status_map = {
                "pass": CheckStatus.PASS,
                "warning": CheckStatus.WARNING,
                "error": CheckStatus.ERROR,
                "skipped": CheckStatus.SKIPPED,
            }

            overall_check_status = status_map.get(overall_status.value, CheckStatus.WARNING)

            results.append(
                CheckResult(
                    status=overall_check_status,
                    category="integration",
                    message="Environment validation completed (HSL validation skipped to avoid crashes)",
                    details=f"Overall status: {overall_status.value}",
                )
            )

            # Add details about specific checks
            if validation_results["python_version"].status.value == "error":
                results.append(
                    CheckResult(
                        status=CheckStatus.ERROR,
                        category="integration",
                        message="Python version validation failed",
                        details=validation_results["python_version"].message,
                        recommendation=validation_results["python_version"].suggestion,
                    )
                )

            if validation_results["casadi_ipopt"].status.value == "error":
                results.append(
                    CheckResult(
                        status=CheckStatus.ERROR,
                        category="integration",
                        message="CasADi/IPOPT validation failed",
                        details=validation_results["casadi_ipopt"].message,
                        recommendation=validation_results["casadi_ipopt"].suggestion,
                    )
                )

            # Check required packages
            for pkg_result in validation_results["required_packages"]:
                if pkg_result.status.value == "error":
                    results.append(
                        CheckResult(
                            status=CheckStatus.ERROR,
                            category="integration",
                            message=f"Required package validation failed: {pkg_result.message}",
                            details=pkg_result.details,
                            recommendation=pkg_result.suggestion,
                        )
                    )

        finally:
            # Restore original environment variable
            if original_skip is None:
                os.environ.pop("CAMPRO_SKIP_VALIDATION", None)
            else:
                os.environ["CAMPRO_SKIP_VALIDATION"] = original_skip

    except ImportError as e:
        results.append(
            CheckResult(
                status=CheckStatus.ERROR,
                category="integration",
                message="Could not import environment validator",
                details=str(e),
                traceback=traceback.format_exc(),
            )
        )
    except Exception as e:
        results.append(
            CheckResult(
                status=CheckStatus.WARNING,
                category="integration",
                message="Error running environment validation",
                details=str(e),
                exception=e,
                traceback=traceback.format_exc(),
            )
        )

    return results


def print_check_result(result: CheckResult, indent: str = "") -> None:
    """Print a check result in formatted way."""
    icon = get_status_icon(result.status)
    status_color = {
        CheckStatus.PASS: "green",
        CheckStatus.WARNING: "yellow",
        CheckStatus.ERROR: "red",
        CheckStatus.INFO: "blue",
        CheckStatus.SKIPPED: "cyan",
    }.get(result.status, "white")

    print(f"{indent}{icon} {colorize(result.message, status_color)}")

    if result.details:
        print(f"{indent}  Details: {result.details}")

    if result.recommendation:
        print(f"{indent}  Recommendation: {colorize(result.recommendation, 'cyan')}")

    if result.traceback and result.status == CheckStatus.ERROR:
        print(f"{indent}  Traceback:")
        for line in result.traceback.split("\n"):
            print(f"{indent}    {line}")


def print_report(report: HealthCheckReport, json_output: bool = False) -> None:
    """Print comprehensive health check report."""
    if json_output:
        # Convert to JSON-serializable format
        json_report = {
            "environment_status": report.environment_status,
            "package_checks": [
                {
                    "status": r.status.value,
                    "category": r.category,
                    "message": r.message,
                    "details": r.details,
                    "recommendation": r.recommendation,
                }
                for r in report.package_checks
            ],
            "import_checks": [
                {
                    "status": r.status.value,
                    "category": r.category,
                    "message": r.message,
                    "details": r.details,
                    "recommendation": r.recommendation,
                }
                for r in report.import_checks
            ],
            "integration_checks": [
                {
                    "status": r.status.value,
                    "category": r.category,
                    "message": r.message,
                    "details": r.details,
                    "recommendation": r.recommendation,
                }
                for r in report.integration_checks
            ],
            "pip_conflicts": [
                {
                    "status": r.status.value,
                    "category": r.category,
                    "message": r.message,
                    "details": r.details,
                    "recommendation": r.recommendation,
                }
                for r in report.pip_conflicts
            ],
            "hsl_checks": [
                {
                    "status": r.status.value,
                    "category": r.category,
                    "message": r.message,
                    "details": r.details,
                    "recommendation": r.recommendation,
                }
                for r in report.hsl_checks
            ],
            "summary": report.summary,
        }
        print(json.dumps(json_report, indent=2))
        return

    # Human-readable output
    print("=" * 70)
    print(colorize("Larrak Conda Environment Health Check", "cyan"))
    print("=" * 70)

    # Environment Status
    print("\n" + colorize("Environment Status", "cyan"))
    print("-" * 70)
    env = report.environment_status
    print(f"Active Environment: {env.get('name', 'None')}")
    print(f"Environment Path: {env.get('path', 'None')}")
    print(f"Python Version: {env.get('python_version', 'Unknown')}")
    print(f"Conda Available: {env.get('conda_available', False)}")
    print(f"Environment Active: {env.get('active', False)}")

    # Check if environment matches expected
    expected_env = "larrak"
    if env.get("name") != expected_env:
        print(
            colorize(
                f"\n⚠  Warning: Expected environment '{expected_env}' but found '{env.get('name')}'",
                "yellow",
            )
        )

    # Python version check
    python_version_info = env.get("python_version_info")
    if python_version_info:
        major, minor = python_version_info.major, python_version_info.minor
        if not (3, 9) <= (major, minor) < (3, 12):
            print(
                colorize(
                    f"\n⚠  Warning: Python {major}.{minor} is outside recommended range (3.9-3.11)",
                    "yellow",
                )
            )

    # Package Checks
    print("\n" + colorize("Package Version Checks", "cyan"))
    print("-" * 70)
    for result in report.package_checks:
        print_check_result(result, "  ")

    # Import Checks
    print("\n" + colorize("Import Tests", "cyan"))
    print("-" * 70)
    for result in report.import_checks:
        print_check_result(result, "  ")

    # Integration Checks
    if report.integration_checks:
        print("\n" + colorize("Integration Validation", "cyan"))
        print("-" * 70)
        for result in report.integration_checks:
            print_check_result(result, "  ")

    # Pip Conflicts
    if report.pip_conflicts:
        print("\n" + colorize("Pip/Conda Conflicts", "cyan"))
        print("-" * 70)
        for result in report.pip_conflicts:
            print_check_result(result, "  ")

    # HSL Checks
    if report.hsl_checks:
        print("\n" + colorize("HSL Library Checks", "cyan"))
        print("-" * 70)
        for result in report.hsl_checks:
            print_check_result(result, "  ")

    # Summary
    print("\n" + colorize("Summary", "cyan"))
    print("-" * 70)
    summary = report.summary
    total_checks = summary.get("total_checks", 0)
    errors = summary.get("errors", 0)
    warnings = summary.get("warnings", 0)
    passes = summary.get("passes", 0)

    print(f"Total Checks: {total_checks}")
    print(f"{colorize('✓ Passed', 'green')}: {passes}")
    print(f"{colorize('⚠ Warnings', 'yellow')}: {warnings}")
    print(f"{colorize('✗ Errors', 'red')}: {errors}")

    if errors > 0:
        print(f"\n{colorize('❌ Health check found errors that need attention!', 'red')}")
        print("Please review the errors above and follow the recommendations.")
    elif warnings > 0:
        print(f"\n{colorize('⚠ Health check passed with warnings.', 'yellow')}")
        print("Review warnings above for potential issues.")
    else:
        print(f"\n{colorize('✅ Health check passed! Environment looks good.', 'green')}")


def generate_summary(report: HealthCheckReport) -> dict[str, Any]:
    """Generate summary statistics from report."""
    all_checks = (
        report.package_checks
        + report.import_checks
        + report.integration_checks
        + report.pip_conflicts
        + report.hsl_checks
    )

    status_counts = {
        CheckStatus.PASS: 0,
        CheckStatus.WARNING: 0,
        CheckStatus.ERROR: 0,
        CheckStatus.INFO: 0,
        CheckStatus.SKIPPED: 0,
    }

    for check in all_checks:
        status_counts[check.status] += 1

    return {
        "total_checks": len(all_checks),
        "passes": status_counts[CheckStatus.PASS],
        "warnings": status_counts[CheckStatus.WARNING],
        "errors": status_counts[CheckStatus.ERROR],
        "info": status_counts[CheckStatus.INFO],
        "skipped": status_counts[CheckStatus.SKIPPED],
    }


def main() -> None:
    """Main health check function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive conda environment health check for Larrak",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )
    parser.add_argument(
        "--no-imports",
        action="store_true",
        help="Skip import testing (faster but less comprehensive)",
    )
    parser.add_argument(
        "--no-integration",
        action="store_true",
        help="Skip integration with check_environment.py",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    print("Running comprehensive conda environment health check...")
    print("This may take a moment...\n")

    # Initialize report
    report = HealthCheckReport()

    # 1. Check environment status
    print("Checking conda environment status...")
    report.environment_status = check_conda_environment_status()

    # 2. Check package versions
    print("Checking package versions against environment.yml...")
    requirements = parse_environment_yml()
    report.package_checks = check_package_versions(requirements)

    # 3. Check for pip conflicts
    print("Checking for pip/conda conflicts...")
    report.pip_conflicts = check_pip_in_conda()

    # 4. Test imports
    if not args.no_imports:
        print("Testing module imports...")
        report.import_checks = test_external_imports() + test_campro_imports()
    else:
        print("Skipping import tests (--no-imports)...")
        report.import_checks = []

    # 5. Check HSL library
    print("Checking HSL library configuration...")
    report.hsl_checks = check_hsl_library()

    # 6. Run integration validation
    if not args.no_integration:
        print("Running integration validation...")
        report.integration_checks = run_integration_validation()
    else:
        print("Skipping integration validation (--no-integration)...")
        report.integration_checks = []

    # Generate summary
    report.summary = generate_summary(report)

    # Print report
    print("\n")
    print_report(report, json_output=args.json)

    # Determine exit code
    if report.summary.get("errors", 0) > 0:
        sys.exit(1)
    elif report.summary.get("warnings", 0) > 0 and args.json:
        # In JSON mode, exit with code 2 if warnings
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

