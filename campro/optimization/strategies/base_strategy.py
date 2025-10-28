"""
Base optimization strategy interface.

This module defines the fundamental interface for optimization strategies,
enabling pluggable optimization approaches for different problem types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from campro.logging import get_logger

log = get_logger(__name__)


class OptimizationStatus(Enum):
    """Status of optimization computation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"
    CANCELLED = "cancelled"


@dataclass
class OptimizationStrategyResult:
    """Result from an optimization strategy."""

    status: OptimizationStatus
    solution: Dict[str, Any]
    objective_value: float
    iterations: int
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None

    def __post_init__(self):
        """Initialize warnings list if not provided."""
        if self.warnings is None:
            self.warnings = []

    @property
    def is_successful(self) -> bool:
        """Check if optimization was successful."""
        return self.status == OptimizationStatus.COMPLETED

    @property
    def has_error(self) -> bool:
        """Check if optimization had an error."""
        return self.status == OptimizationStatus.FAILED


class BaseOptimizationStrategy(ABC):
    """
    Base interface for optimization strategies.

    This class defines the contract that all optimization strategies must follow,
    enabling modular optimization approaches and easy testing.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the optimization strategy.

        Parameters
        ----------
        name : str, optional
            Strategy name for identification
        """
        self.name = name or self.__class__.__name__
        self._validate_strategy()
        log.debug(f"Initialized optimization strategy: {self.name}")

    @abstractmethod
    def _validate_strategy(self) -> None:
        """
        Validate strategy configuration.

        Raises
        ------
        ValueError
            If strategy configuration is invalid
        """

    @abstractmethod
    def optimize(
        self,
        objective_function: Callable,
        initial_guess: Dict[str, Any],
        constraints: Optional[List[Callable]] = None,
        bounds: Optional[Dict[str, tuple]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> OptimizationStrategyResult:
        """
        Perform optimization.

        Parameters
        ----------
        objective_function : Callable
            Function to minimize
        initial_guess : Dict[str, Any]
            Initial parameter values
        constraints : List[Callable], optional
            Constraint functions
        bounds : Dict[str, tuple], optional
            Parameter bounds
        options : Dict[str, Any], optional
            Optimization options

        Returns
        -------
        OptimizationStrategyResult
            Optimization result
        """

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information.

        Returns
        -------
        Dict[str, Any]
            Strategy information
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "description": self.__doc__ or "No description available",
        }

    def validate_inputs(
        self, objective_function: Callable, initial_guess: Dict[str, Any],
    ) -> bool:
        """
        Validate optimization inputs.

        Parameters
        ----------
        objective_function : Callable
            Objective function to validate
        initial_guess : Dict[str, Any]
            Initial guess to validate

        Returns
        -------
        bool
            True if inputs are valid
        """
        if not callable(objective_function):
            log.error("Objective function must be callable")
            return False

        if not isinstance(initial_guess, dict):
            log.error("Initial guess must be a dictionary")
            return False

        return True

    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.__class__.__name__}(name='{self.name}')"
