"""
Environment validation and setup utilities.

This module provides tools for validating the installation environment,
checking dependencies, and ensuring proper setup of CasADi with ipopt support.

Cross-Platform Support:
    The `ctx` singleton provides unified access to platform-specific resources:
    
        from campro.environment import ctx
        
        hsl = ctx.resources.hsl_library
        ctx.workflows.exit_safely(0)
    
    Convenience functions are also available:
    
        from campro.environment.resolve import hsl_path, exit_safely
"""

from .context import PlatformContext, ctx
from .resolve import (
    exit_safely,
    hsl_path,
    libraries_dir,
    project_root,
    python_exe,
    requires_isolation,
)
from .validator import (
    ValidationResult,
    ValidationStatus,
    validate_casadi_ipopt,
    validate_environment,
    validate_python_version,
    validate_required_packages,
)

__all__ = [
    # Platform context
    "PlatformContext",
    "ctx",
    # Resolve convenience functions
    "exit_safely",
    "hsl_path",
    "libraries_dir",
    "project_root",
    "python_exe",
    "requires_isolation",
    # Validation
    "ValidationResult",
    "ValidationStatus",
    "validate_casadi_ipopt",
    "validate_environment",
    "validate_python_version",
    "validate_required_packages",
]
