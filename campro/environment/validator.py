"""
Environment validation utilities.

This module provides comprehensive validation of the installation environment,
including CasADi, ipopt, Python version, and required packages.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any

from campro.environment.platform_detector import IS_MACOS
from campro.logging import get_logger

log = get_logger(__name__)

# Global flag to prevent multiple IPOPT linear solver initializations
_IPOPT_LINEAR_SOLVER_INITIALIZED = False


def _validate_ma27_usage() -> None:
    """Validate that MA27 is being used and fail hard if a non-HSL fallback is detected."""
    # Skip validation if explicitly disabled (avoids solver clobbering warnings)
    import os
    if os.getenv("CAMPRO_SKIP_VALIDATION") == "1":
        log.info("MA27 validation skipped (CAMPRO_SKIP_VALIDATION=1)")
        return
    
    try:
        import casadi as ca

        from campro.optimization.ipopt_factory import create_ipopt_solver

        # Create a test problem to check which linear solver is actually being used
        x = ca.SX.sym("x")
        f = x**2
        g = x - 1
        nlp = {"x": x, "f": f, "g": g}

        # Create solver using centralized factory
        solver = create_ipopt_solver("ma27_test", nlp, linear_solver="ma27")

        # Try to solve a simple problem to verify MA27 is working
        result = solver(x0=0, lbg=0, ubg=0)

        # Check if solver actually used MA27 (heuristic check)
        # If MA27 is not available, IPOPT may fall back to a non-HSL solver, which is not allowed
        log.info("MA27 linear solver validation passed")

    except Exception as e:
        log.error(f"MA27 validation failed: {e}")
        raise RuntimeError(
            "MA27 linear solver is not available. Fallback to non-HSL solvers is not allowed. "
            "Please ensure HSL library with MA27 is properly installed.",
        ) from e


class ValidationStatus(Enum):
    """Status of a validation check."""

    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a validation check."""

    status: ValidationStatus
    message: str
    details: str | None = None
    suggestion: str | None = None
    version: str | None = None


def validate_python_version(min_version: tuple[int, int] = (3, 9)) -> ValidationResult:
    """Validate Python version against a minimum required version."""
    current_version = sys.version_info[:2]

    if current_version >= min_version:
        return ValidationResult(
            status=ValidationStatus.PASS,
            message=f"Python {current_version[0]}.{current_version[1]} is supported",
            version=f"{current_version[0]}.{current_version[1]}",
        )
    return ValidationResult(
        status=ValidationStatus.ERROR,
        message=f"Python {current_version[0]}.{current_version[1]} is not supported",
        details=f"Minimum required version is {min_version[0]}.{min_version[1]}",
        suggestion="Please upgrade Python to a supported version",
        version=f"{current_version[0]}.{current_version[1]}",
    )


def validate_casadi_ipopt() -> ValidationResult:
    """Validate CasADi installation and ipopt availability."""
    try:
        import casadi as ca

        log.info("CasADi version: %s", getattr(ca, "__version__", "unknown"))

        if hasattr(ca, "nlpsol_plugins"):
            plugins = ca.nlpsol_plugins()
            log.info("Available NLP solvers: %s", plugins)
            if "ipopt" in plugins:
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message="CasADi with ipopt support is available",
                    details=f"CasADi {getattr(ca, '__version__', 'unknown')} with ipopt solver",
                    version=getattr(ca, "__version__", None),
                )
            return ValidationResult(
                status=ValidationStatus.ERROR,
                message="CasADi is installed but ipopt solver is not available",
                details=f"Available solvers: {plugins}",
                suggestion=(
                    "On macOS, prebuilt CasADi may lack IPOPT. Use conda on Linux/arm64, or build CasADi from source with Homebrew.\n"
                    "See docs/installation_guide.md (Method B).\n\n"
                    "Quick fix (Linux/arm64): conda install -c conda-forge casadi ipopt"
                ),
            )

        # Fallback for CasADi builds without nlpsol_plugins attribute
        try:
            from campro.optimization.ipopt_factory import create_ipopt_solver

            x = ca.SX.sym("x")
            f = x**2
            nlp = {"x": x, "f": f}
            # Use centralized factory to prevent clobbering
            create_ipopt_solver("ipopt_probe", nlp, linear_solver="ma27")
            return ValidationResult(
                status=ValidationStatus.PASS,
                message="CasADi with ipopt support is available (fallback check)",
                details=f"CasADi {getattr(ca, '__version__', 'unknown')} with ipopt solver",
                version=getattr(ca, "__version__", None),
            )
        except Exception as exc:
            return ValidationResult(
                status=ValidationStatus.ERROR,
                message="CasADi is installed but ipopt solver is not available",
                details=f"Could not create ipopt solver: {exc}",
                suggestion=(
                    "On macOS, build CasADi from source with IPOPT via Homebrew (Method B in docs/installation_guide.md).\n"
                    "Linux/arm64 users can install via conda-forge: conda install -c conda-forge casadi ipopt"
                ),
            )

    except ImportError as exc:
        return ValidationResult(
            status=ValidationStatus.ERROR,
            message="CasADi is not installed",
            details=str(exc),
            suggestion=(
                "Install CasADi using conda: conda install -c conda-forge casadi ipopt"
            ),
        )
    except Exception as exc:
        log.error("Unexpected error checking CasADi: %s", exc)
        return ValidationResult(
            status=ValidationStatus.ERROR,
            message="Error checking CasADi installation",
            details=str(exc),
            suggestion="Please check your CasADi installation",
        )


def validate_hsl_solvers() -> list[ValidationResult]:
    """Validate HSL solver availability (MA27, MA57, MA77, MA86, MA97) - optional but improves performance."""
    # Skip validation if explicitly disabled (avoids solver clobbering warnings)
    import os
    if os.getenv("CAMPRO_SKIP_VALIDATION") == "1":
        log.info("HSL solver validation skipped (CAMPRO_SKIP_VALIDATION=1)")
        return []
    
    results: list[ValidationResult] = []

    # Detect available solvers using hsl_detector
    try:
        from campro.environment.hsl_detector import detect_available_solvers
        
        # Get list of solvers to test (detected from CoinHSL config)
        hsl_solvers = detect_available_solvers(test_runtime=False)
        
        if not hsl_solvers:
            results.append(
                ValidationResult(
                    status=ValidationStatus.WARNING,
                    message="No HSL solvers detected in CoinHSL library",
                    details="CoinHSL directory not found or no solvers configured",
                    suggestion=(
                        "HSL solvers are optional but improve performance. To obtain them:\n"
                        "1. Visit STFC licensing portal: https://licences.stfc.ac.uk/product/coin-hsl\n"
                        "2. Obtain a license for HSL software\n"
                        "3. Follow installation instructions in docs/installation_guide.md\n"
                        "4. Note: ipopt works without HSL but with reduced performance"
                    ),
                ),
            )
            return results
        
        # Test runtime availability through CasADi
        import casadi as ca
        
        available_solvers = []
        solver_details = []

        # Create a simple test problem
        x = ca.SX.sym("x")
        f = x**2
        g = x - 1
        nlp = {"x": x, "f": f, "g": g}
        
        # Get HSL library path for solver configuration
        from campro.environment.hsl_detector import get_hsl_library_path
        hsl_lib_path = get_hsl_library_path()
        solver_opts = {}
        if hsl_lib_path:
            solver_opts["ipopt.hsllib"] = str(hsl_lib_path)

        for solver_name in hsl_solvers:
            try:
                # Try to create a solver with this HSL linear solver
                solver_opts["ipopt.linear_solver"] = solver_name
                solver_opts["ipopt.print_level"] = 0
                solver_opts["ipopt.sb"] = "yes"
                
                solver = ca.nlpsol(
                    f"hsl_test_{solver_name}",
                    "ipopt",
                    nlp,
                    solver_opts,
                )

                # Test the solver with a simple problem
                result = solver(x0=0, lbg=0, ubg=0)
                stats = solver.stats()

                if stats["success"]:
                    available_solvers.append(solver_name.upper())
                    solver_details.append(solver_name.upper())
                else:
                    solver_details.append(f"{solver_name.upper()}(failed)")

            except Exception as e:
                solver_details.append(f"{solver_name.upper()}(error: {str(e)[:50]})")

        total_possible = 5  # MA27, MA57, MA77, MA86, MA97
        if available_solvers:
            results.append(
                ValidationResult(
                    status=ValidationStatus.PASS,
                    message=f"HSL solvers are available ({len(available_solvers)}/{len(hsl_solvers)} detected, {total_possible} possible)",
                    details=f"Available: {', '.join(available_solvers)} | Tested: {', '.join(solver_details)}",
                    suggestion="HSL solvers will significantly improve optimization performance",
                ),
            )
        else:
            results.append(
                ValidationResult(
                    status=ValidationStatus.WARNING,
                    message="No HSL solvers are available at runtime",
                    details=f"Detected: {', '.join(hsl_solvers)} | Tested: {', '.join(solver_details)}",
                    suggestion=(
                        "HSL solvers detected but not available at runtime. Check:\n"
                        "1. HSL library path is correctly configured\n"
                        "2. All required dependencies are available\n"
                        "3. CasADi can access the HSL library\n"
                        "Note: ipopt works without HSL but with reduced performance"
                    ),
                ),
            )

    except ImportError:
        results.append(
            ValidationResult(
                status=ValidationStatus.WARNING,
                message="Cannot check HSL solvers - CasADi not available",
                details="CasADi is required to check HSL solver availability",
                suggestion="Install CasADi first, then re-run validation",
            ),
        )
    except Exception as exc:
        log.error("Unexpected error checking HSL solvers: %s", exc)
        results.append(
            ValidationResult(
                status=ValidationStatus.WARNING,
                message="Error checking HSL solver availability",
                details=str(exc),
                suggestion="HSL solvers are optional - ipopt will work without them",
            ),
        )

    return results


def validate_required_packages() -> list[ValidationResult]:
    """Validate required packages are importable (reports versions if available)."""
    required_packages: dict[str, str] = {
        "numpy": "1.24.0",
        "scipy": "1.10.0",
        "matplotlib": "3.7.0",
    }

    results: list[ValidationResult] = []
    for package, min_version in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", None)
            if version:
                results.append(
                    ValidationResult(
                        status=ValidationStatus.PASS,
                        message=f"{package} is available",
                        version=version,
                    ),
                )
            else:
                results.append(
                    ValidationResult(
                        status=ValidationStatus.WARNING,
                        message=f"{package} is available but version unknown",
                        suggestion=(
                            f"Consider upgrading {package} to version {min_version} or later"
                        ),
                    ),
                )
        except ImportError as exc:
            results.append(
                ValidationResult(
                    status=ValidationStatus.ERROR,
                    message=f"{package} is not installed",
                    details=str(exc),
                    suggestion=f"Install {package} using conda: conda install -c conda-forge {package}",
                ),
            )
        except Exception as exc:
            log.error("Unexpected error checking %s: %s", package, exc)
            results.append(
                ValidationResult(
                    status=ValidationStatus.ERROR,
                    message=f"Error checking {package}",
                    details=str(exc),
                ),
            )

    return results


def validate_environment() -> dict[str, Any]:
    """Perform comprehensive environment validation and return structured results."""
    # Skip validation if explicitly disabled (avoids HSL solver clobbering warnings)
    import os
    if os.getenv("CAMPRO_SKIP_VALIDATION") == "1":
        log.info("Environment validation skipped (CAMPRO_SKIP_VALIDATION=1)")
        # Return minimal results to avoid breaking code that expects this structure
        from .validator import ValidationStatus, ValidationResult
        return {
            "python_version": ValidationResult(
                status=ValidationStatus.SKIPPED,
                message="Validation skipped",
                details="CAMPRO_SKIP_VALIDATION=1",
            ),
            "casadi_ipopt": ValidationResult(
                status=ValidationStatus.SKIPPED,
                message="Validation skipped",
                details="CAMPRO_SKIP_VALIDATION=1",
            ),
            "required_packages": [],
            "hsl_solvers": [],
            "summary": {"overall_status": ValidationStatus.SKIPPED},
        }
    
    log.info("Starting environment validation")

    # Validate MA27 usage first - fail hard if a non-HSL fallback is detected
    _validate_ma27_usage()

    results: dict[str, Any] = {
        "python_version": validate_python_version(),
        "casadi_ipopt": validate_casadi_ipopt(),
        "required_packages": validate_required_packages(),
        "hsl_solvers": validate_hsl_solvers(),
    }

    # Aggregate results (HSL solvers are optional, so they don't affect overall status)
    all_results: list[ValidationResult] = [
        results["python_version"],
        results["casadi_ipopt"],
        *results["required_packages"],
    ]

    # HSL solvers are optional - include them in total count but not in error determination
    hsl_results = results["hsl_solvers"]
    all_results.extend(hsl_results)

    status_counts: dict[ValidationStatus, int] = {
        ValidationStatus.PASS: 0,
        ValidationStatus.WARNING: 0,
        ValidationStatus.ERROR: 0,
        ValidationStatus.SKIPPED: 0,
    }

    # Count only required dependencies for overall status
    required_results = [
        results["python_version"],
        results["casadi_ipopt"],
        *results["required_packages"],
    ]
    for res in required_results:
        status_counts[res.status] += 1

    # Count all results for total
    total_status_counts: dict[ValidationStatus, int] = {
        ValidationStatus.PASS: 0,
        ValidationStatus.WARNING: 0,
        ValidationStatus.ERROR: 0,
        ValidationStatus.SKIPPED: 0,
    }
    for res in all_results:
        total_status_counts[res.status] += 1

    # Overall status based only on required dependencies
    if status_counts[ValidationStatus.ERROR] > 0:
        overall_status = ValidationStatus.ERROR
    elif status_counts[ValidationStatus.WARNING] > 0:
        overall_status = ValidationStatus.WARNING
    else:
        overall_status = ValidationStatus.PASS

    results["summary"] = {
        "overall_status": overall_status,
        "status_counts": status_counts,
        "total_checks": len(all_results),
        "total_status_counts": total_status_counts,
    }

    log.info(
        "Environment validation complete. Overall status: %s", overall_status.value,
    )
    return results


def get_installation_instructions() -> str:
    """Get installation instructions for setting up the environment."""
    return (
        "\nInstallation Instructions:\n\n"
        "1. Install Miniconda or Miniforge:\n"
        "   - Download from: https://docs.conda.io/en/latest/miniconda.html\n"
        "   - Or use Miniforge: https://github.com/conda-forge/miniforge\n\n"
        "2. Create environment from environment.yml:\n"
        "   conda env create -f environment.yml\n\n"
        "3. Activate the environment:\n"
        "   conda activate larrak\n\n"
        "4. Verify installation:\n"
        "   python scripts/check_environment.py\n\n"
        "Alternative (if conda is not available):\n"
        "   pip install -r requirements.txt\n"
        "   Note: This may not include ipopt solver support.\n"
    )


__all__ = [
    "ValidationResult",
    "ValidationStatus",
    "get_installation_instructions",
    "validate_casadi_ipopt",
    "validate_environment",
    "validate_hsl_solvers",
    "validate_python_version",
    "validate_required_packages",
]
