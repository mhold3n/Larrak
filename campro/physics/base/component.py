"""
Base component interface for physics components.

This module defines the fundamental interface that all physics components
must implement, ensuring consistency and enabling modular system design.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


class ComponentStatus(Enum):
    """Status of a component computation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"


@dataclass
class ComponentResult:
    """Result from a component computation."""

    status: ComponentStatus
    outputs: dict[str, np.ndarray]
    metadata: dict[str, Any]
    error_message: str | None = None

    @property
    def is_successful(self) -> bool:
        """Check if computation was successful."""
        return self.status == ComponentStatus.COMPLETED

    @property
    def has_error(self) -> bool:
        """Check if computation had an error."""
        return self.status == ComponentStatus.FAILED


class BaseComponent(ABC):
    """
    Base interface for all physics components.

    This class defines the contract that all physics components must follow,
    enabling modular system design and easy testing.
    """

    def __init__(self, parameters: dict[str, Any], name: str | None = None):
        """
        Initialize the component.

        Parameters
        ----------
        parameters : Dict[str, Any]
            Component parameters
        name : str, optional
            Component name for identification
        """
        self.parameters = parameters.copy()
        self.name = name or self.__class__.__name__
        self._validate_parameters()
        log.debug(
            f"Initialized component {self.name} with parameters: {self.parameters}",
        )

    @abstractmethod
    def _validate_parameters(self) -> None:
        """
        Validate component parameters.

        Raises
        ------
        ValueError
            If parameters are invalid
        """

    @abstractmethod
    def compute(self, inputs: dict[str, np.ndarray]) -> ComponentResult:
        """
        Compute component outputs from inputs.

        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            Input data arrays

        Returns
        -------
        ComponentResult
            Computation result with outputs and metadata
        """

    def get_parameters(self) -> dict[str, Any]:
        """
        Get component parameters.

        Returns
        -------
        Dict[str, Any]
            Copy of component parameters
        """
        return self.parameters.copy()

    def update_parameters(self, parameters: dict[str, Any]) -> None:
        """
        Update component parameters.

        Parameters
        ----------
        parameters : Dict[str, Any]
            New parameters to update
        """
        self.parameters.update(parameters)
        self._validate_parameters()
        log.debug(f"Updated parameters for component {self.name}")

    def get_required_inputs(self) -> list[str]:
        """
        Get list of required input names.

        Returns
        -------
        List[str]
            List of required input parameter names
        """
        return []

    def get_optional_inputs(self) -> list[str]:
        """
        Get list of optional input names.

        Returns
        -------
        List[str]
            List of optional input parameter names
        """
        return []

    def get_outputs(self) -> list[str]:
        """
        Get list of output names.

        Returns
        -------
        List[str]
            List of output parameter names
        """
        return []

    def validate_inputs(self, inputs: dict[str, np.ndarray]) -> bool:
        """
        Validate input data.

        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            Input data to validate

        Returns
        -------
        bool
            True if inputs are valid
        """
        required = self.get_required_inputs()
        for input_name in required:
            if input_name not in inputs:
                log.error(f"Missing required input: {input_name}")
                return False
            if not isinstance(inputs[input_name], np.ndarray):
                log.error(f"Input {input_name} must be numpy array")
                return False
        return True

    def __repr__(self) -> str:
        """String representation of component."""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"
