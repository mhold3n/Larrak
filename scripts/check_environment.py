#!/usr/bin/env python3
"""
Environment validation script for Larrak.

This script checks if the environment is properly set up with all required
dependencies, including CasADi with ipopt support.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from campro.environment.validator import (  # noqa: E402
    ValidationResult,
    ValidationStatus,
    get_installation_instructions,
    validate_environment,
)
from campro.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def colorize(text: str, color: str) -> str:
    """
    Add color to text for terminal output.

    Args:
        text: Text to colorize
        color: Color name (red, green, yellow, blue)

    Returns:
        Colorized text (or original if colors not supported)
    """
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }

    # Check if colors are supported
    if sys.stdout.isatty() and os.getenv("TERM") != "dumb":
        return f"{colors.get(color, '')}{text}{colors['reset']}"
    return text


def print_status_icon(status: ValidationStatus) -> str:
    """Get status icon for validation result."""
    icons = {
        ValidationStatus.PASS: colorize("✓", "green"),
        ValidationStatus.WARNING: colorize("⚠", "yellow"),
        ValidationStatus.ERROR: colorize("✗", "red"),
        ValidationStatus.SKIPPED: colorize("-", "blue"),
    }
    return icons.get(status, "?")


def print_validation_result(result: ValidationResult, indent: str = "") -> None:
    """
    Print a validation result in a formatted way.

    Args:
        result: Validation result to print
        indent: Indentation string
    """
    icon = print_status_icon(result.status)
    status_color = {
        ValidationStatus.PASS: "green",
        ValidationStatus.WARNING: "yellow",
        ValidationStatus.ERROR: "red",
        ValidationStatus.SKIPPED: "blue",
    }.get(result.status, "white")

    print(f"{indent}{icon} {colorize(result.message, status_color)}")

    if result.version:
        print(f"{indent}  Version: {result.version}")

    if result.details:
        print(f"{indent}  Details: {result.details}")

    if result.suggestion:
        print(f"{indent}  Suggestion: {colorize(result.suggestion, 'blue')}")


def print_environment_report(
    results: dict[str, Any], json_output: bool = False,
) -> None:
    """
    Print comprehensive environment validation report.

    Args:
        results: Validation results dictionary
        json_output: Whether to output in JSON format
    """
    if json_output:
        # Convert to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if key == "summary":
                json_results[key] = {
                    "overall_status": value["overall_status"].value,
                    "status_counts": {
                        k.value: v for k, v in value["status_counts"].items()
                    },
                    "total_checks": value["total_checks"],
                }
            elif isinstance(value, list):
                json_results[key] = [
                    {
                        "status": item.status.value,
                        "message": item.message,
                        "details": item.details,
                        "suggestion": item.suggestion,
                        "version": item.version,
                    }
                    for item in value
                ]
            else:
                json_results[key] = {
                    "status": value.status.value,
                    "message": value.message,
                    "details": value.details,
                    "suggestion": value.suggestion,
                    "version": value.version,
                }

        print(json.dumps(json_results, indent=2))
        return

    # Human-readable output
    print("Larrak Environment Validation Report")
    print("=" * 50)

    # Overall status
    overall_status = results["summary"]["overall_status"]
    status_color = {
        ValidationStatus.PASS: "green",
        ValidationStatus.WARNING: "yellow",
        ValidationStatus.ERROR: "red",
    }.get(overall_status, "white")

    print(f"\nOverall Status: {colorize(overall_status.value.upper(), status_color)}")

    # Status summary
    counts = results["summary"]["status_counts"]
    print(
        f"Checks: {counts[ValidationStatus.PASS]} passed, "
        f"{counts[ValidationStatus.WARNING]} warnings, "
        f"{counts[ValidationStatus.ERROR]} errors",
    )

    # Detailed results
    print("\nDetailed Results:")
    print("-" * 30)

    # Python version
    print("\nPython Version:")
    print_validation_result(results["python_version"], "  ")

    # CasADi and ipopt
    print("\nCasADi and ipopt:")
    print_validation_result(results["casadi_ipopt"], "  ")

    # Required packages
    print("\nRequired Packages:")
    for result in results["required_packages"]:
        print_validation_result(result, "  ")

    # Optional HSL solvers
    print("\nOptional HSL Solvers (improves performance):")
    for result in results["hsl_solvers"]:
        print_validation_result(result, "  ")

    # Installation instructions if needed
    if overall_status == ValidationStatus.ERROR:
        print(f"\n{colorize('Installation Instructions:', 'yellow')}")
        print(get_installation_instructions())


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate Larrak environment setup",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
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

    try:
        # Run validation
        results = validate_environment()
        overall_status = results["summary"]["overall_status"]

        # Print report
        print_environment_report(results, json_output=args.json)

        # Determine exit code
        if overall_status == ValidationStatus.ERROR:
            exit_code = 1
        elif overall_status == ValidationStatus.WARNING and args.strict:
            exit_code = 2
        else:
            exit_code = 0

        # Print summary for non-JSON output
        if not args.json:
            if exit_code == 0:
                print(f"\n{colorize('Environment validation passed!', 'green')}")
            elif exit_code == 1:
                print(f"\n{colorize('Environment validation failed!', 'red')}")
                print("Please fix the errors above and run this script again.")
            else:
                print(
                    f"\n{colorize('Environment validation passed with warnings.', 'yellow')}",
                )
                print("Run without --strict to treat warnings as non-fatal.")

        sys.exit(exit_code)

    except Exception as e:
        log.error(f"Error during validation: {e}")
        if not args.json:
            print(f"\n{colorize('Error during validation:', 'red')} {e}")
        sys.exit(1)


if __name__ == "__main__":
    import os

    main()
