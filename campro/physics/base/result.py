"""
Standardized result types for physics computations.

This module defines common result types used across the physics system
for consistent data handling and error reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


class PhysicsStatus(Enum):
    """Status of physics computations."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"
    CANCELLED = "cancelled"


@dataclass
class PhysicsResult:
    """
    Standardized result container for physics computations.

    This class provides a consistent interface for returning results
    from physics computations, including data, metadata, and status information.
    """

    status: PhysicsStatus
    data: dict[str, np.ndarray]
    metadata: dict[str, Any]
    error_message: str | None = None
    warnings: list | None = None

    def __post_init__(self):
        """Initialize warnings list if not provided."""
        if self.warnings is None:
            self.warnings = []

    @property
    def is_successful(self) -> bool:
        """Check if computation was successful."""
        return self.status == PhysicsStatus.COMPLETED

    @property
    def has_error(self) -> bool:
        """Check if computation had an error."""
        return self.status == PhysicsStatus.FAILED

    @property
    def has_warnings(self) -> bool:
        """Check if computation had warnings."""
        return len(self.warnings) > 0

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        log.warning(f"Added warning: {warning}")

    def get_data_keys(self) -> list:
        """Get list of data keys."""
        return list(self.data.keys())

    def get_data_shape(self, key: str) -> tuple:
        """Get shape of data array for given key."""
        if key not in self.data:
            raise KeyError(f"Data key '{key}' not found")
        return self.data[key].shape

    def get_data_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all data arrays."""
        info = {}
        for key, array in self.data.items():
            info[key] = {
                "shape": array.shape,
                "dtype": array.dtype,
                "min": float(np.min(array)) if array.size > 0 else None,
                "max": float(np.max(array)) if array.size > 0 else None,
                "mean": float(np.mean(array)) if array.size > 0 else None,
            }
        return info

    def validate_data(self) -> bool:
        """Validate that all data arrays are valid."""
        for key, array in self.data.items():
            if not isinstance(array, np.ndarray):
                log.error(f"Data '{key}' is not a numpy array")
                return False
            if not np.all(np.isfinite(array)):
                log.error(f"Data '{key}' contains non-finite values")
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary (excluding numpy arrays)."""
        return {
            "status": self.status.value,
            "metadata": self.metadata,
            "error_message": self.error_message,
            "warnings": self.warnings,
            "data_keys": list(self.data.keys()),
            "data_info": self.get_data_info(),
        }

    @classmethod
    def success(
        cls, data: dict[str, np.ndarray], metadata: dict[str, Any] | None = None,
    ) -> PhysicsResult:
        """Create a successful result."""
        return cls(
            status=PhysicsStatus.COMPLETED,
            data=data,
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls, error_message: str, metadata: dict[str, Any] | None = None,
    ) -> PhysicsResult:
        """Create a failed result."""
        return cls(
            status=PhysicsStatus.FAILED,
            data={},
            metadata=metadata or {},
            error_message=error_message,
        )

    def __repr__(self) -> str:
        """String representation of result."""
        status_str = self.status.value
        data_keys = list(self.data.keys())
        return f"PhysicsResult(status={status_str}, data_keys={data_keys})"
