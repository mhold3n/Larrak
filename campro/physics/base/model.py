"""
Base physics model class.

This module defines the fundamental physics simulation framework that will
be extended for various physics models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from campro.logging import get_logger

from .result import PhysicsResult, PhysicsStatus

log = get_logger(__name__)


class BasePhysicsModel(ABC):
    """
    Base class for all physics simulation models.

    Provides a common interface for physics simulation across different
    domains (combustion, thermodynamics, fluid dynamics).
    """

    def __init__(self, name: str = "BasePhysicsModel"):
        self.name = name
        self._is_configured = False
        self._current_result: Optional[PhysicsResult] = None
        self._simulation_history: List[PhysicsResult] = []

    @abstractmethod
    def configure(self, **kwargs) -> None:
        """
        Configure the physics model with problem-specific parameters.

        Args:
            **kwargs: Configuration parameters
        """

    @abstractmethod
    def simulate(self, inputs: Dict[str, Any], **kwargs) -> PhysicsResult:
        """
        Run a physics simulation.

        Args:
            inputs: Input parameters for simulation
            **kwargs: Additional simulation parameters

        Returns:
            PhysicsResult object
        """

    def is_configured(self) -> bool:
        """Check if physics model is configured."""
        return self._is_configured

    def get_current_result(self) -> Optional[PhysicsResult]:
        """Get the most recent simulation result."""
        return self._current_result

    def get_simulation_history(self) -> List[PhysicsResult]:
        """Get all simulation results."""
        return self._simulation_history.copy()

    def clear_history(self) -> None:
        """Clear simulation history."""
        self._simulation_history.clear()

    def _start_simulation(self) -> PhysicsResult:
        """Start a new simulation process."""
        result = PhysicsResult(
            status=PhysicsStatus.RUNNING,
            data={},
            metadata={},
        )
        self._current_result = result
        return result

    def _finish_simulation(
        self,
        result: PhysicsResult,
        data: Dict[str, Any],
        convergence_info: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> PhysicsResult:
        """Finish a simulation process."""
        # Update result data
        result.data = data
        result.metadata.update(convergence_info or {})
        result.error_message = error_message

        # Determine final status
        if error_message:
            result.status = PhysicsStatus.FAILED
        elif len(data) > 0:
            result.status = PhysicsStatus.COMPLETED
        else:
            result.status = PhysicsStatus.FAILED

        # Store in history
        self._simulation_history.append(result)

        log.info(f"Physics simulation {result.status.value}: {self.name}")

        return result

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Validate simulation inputs."""
        if not isinstance(inputs, dict):
            raise ValueError("Inputs must be a dictionary")

        if not self._is_configured:
            raise RuntimeError(
                f"Physics model {self.name} is not configured. Call configure() first.",
            )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all simulations."""
        if not self._simulation_history:
            return {}

        successful_results = [r for r in self._simulation_history if r.is_successful()]

        summary = {
            "total_simulations": len(self._simulation_history),
            "successful_simulations": len(successful_results),
            "success_rate": len(successful_results) / len(self._simulation_history),
        }

        if successful_results:
            simulation_times = [
                r.simulation_time
                for r in successful_results
                if r.simulation_time is not None
            ]
            if simulation_times:
                import numpy as np

                summary["avg_simulation_time"] = np.mean(simulation_times)
                summary["min_simulation_time"] = np.min(simulation_times)
                summary["max_simulation_time"] = np.max(simulation_times)

        return summary
