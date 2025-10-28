"""
Parameter management utilities.

This module provides tools for managing and validating parameters
across the modular system.
"""

from typing import Any, Dict, List, Optional

from campro.logging import get_logger

log = get_logger(__name__)


class ParameterValidator:
    """Validator for system parameters."""

    @staticmethod
    def validate_positive_float(value: Any, name: str) -> bool:
        """Validate that a value is a positive float."""
        try:
            float_val = float(value)
            if float_val <= 0:
                log.error(f"{name} must be positive, got {float_val}")
                return False
            return True
        except (ValueError, TypeError):
            log.error(f"{name} must be a number, got {type(value)}")
            return False

    @staticmethod
    def validate_float_range(
        value: Any, name: str, min_val: float, max_val: float,
    ) -> bool:
        """Validate that a value is within a float range."""
        try:
            float_val = float(value)
            if not (min_val <= float_val <= max_val):
                log.error(
                    f"{name} must be between {min_val} and {max_val}, got {float_val}",
                )
                return False
            return True
        except (ValueError, TypeError):
            log.error(f"{name} must be a number, got {type(value)}")
            return False

    @staticmethod
    def validate_choice(value: Any, name: str, choices: List[str]) -> bool:
        """Validate that a value is one of the allowed choices."""
        if value not in choices:
            log.error(f"{name} must be one of {choices}, got {value}")
            return False
        return True


class ParameterManager:
    """Manager for system parameters."""

    def __init__(self):
        """Initialize parameter manager."""
        self.parameters = {}
        self.validators = {}
        self.log = get_logger(__name__)

    def add_parameter(
        self, name: str, value: Any, validator: Optional[callable] = None,
    ) -> None:
        """Add a parameter with optional validation."""
        if validator and not validator(value):
            raise ValueError(f"Parameter {name} failed validation")

        self.parameters[name] = value
        if validator:
            self.validators[name] = validator

        log.debug(f"Added parameter {name} = {value}")

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.parameters.get(name, default)

    def update_parameter(self, name: str, value: Any) -> None:
        """Update a parameter value."""
        if name in self.validators and not self.validators[name](value):
            raise ValueError(f"Parameter {name} failed validation")

        self.parameters[name] = value
        log.debug(f"Updated parameter {name} = {value}")

    def validate_all(self) -> bool:
        """Validate all parameters."""
        for name, validator in self.validators.items():
            if name in self.parameters:
                if not validator(self.parameters[name]):
                    return False
        return True

    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all parameters."""
        return self.parameters.copy()
