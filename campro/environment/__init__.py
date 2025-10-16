"""
Environment validation and setup utilities.

This module provides tools for validating the installation environment,
checking dependencies, and ensuring proper setup of CasADi with ipopt support.
"""

from .validator import (
    ValidationResult,
    ValidationStatus,
    validate_environment,
    validate_casadi_ipopt,
    validate_python_version,
    validate_required_packages,
)

__all__ = [
    "ValidationResult",
    "ValidationStatus", 
    "validate_environment",
    "validate_casadi_ipopt",
    "validate_python_version",
    "validate_required_packages",
]
