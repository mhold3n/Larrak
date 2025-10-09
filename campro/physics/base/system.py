"""
Base system interface for physics systems.

This module defines the interface for complete physics systems that coordinate
multiple components to solve complex problems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from campro.logging import get_logger

from .component import BaseComponent, ComponentResult

log = get_logger(__name__)


class SystemStatus(Enum):
    """Status of a system computation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALID = "invalid"


@dataclass
class SystemResult:
    """Result from a system computation."""
    status: SystemStatus
    outputs: Dict[str, np.ndarray]
    component_results: Dict[str, ComponentResult]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        """Check if computation was successful."""
        return self.status == SystemStatus.COMPLETED

    @property
    def has_error(self) -> bool:
        """Check if computation had an error."""
        return self.status == SystemStatus.FAILED


class BaseSystem(ABC):
    """
    Base interface for complete physics systems.
    
    Systems coordinate multiple components to solve complex problems.
    They manage data flow between components and handle system-level validation.
    """

    def __init__(self, components: Dict[str, BaseComponent], name: Optional[str] = None):
        """
        Initialize the system.
        
        Parameters
        ----------
        components : Dict[str, BaseComponent]
            Dictionary of components in the system
        name : str, optional
            System name for identification
        """
        self.components = components.copy()
        self.name = name or self.__class__.__name__
        self._validate_system()
        log.info(f"Initialized system {self.name} with {len(self.components)} components")

    @abstractmethod
    def _validate_system(self) -> None:
        """
        Validate system configuration.
        
        Raises
        ------
        ValueError
            If system configuration is invalid
        """

    @abstractmethod
    def solve(self, inputs: Dict[str, np.ndarray]) -> SystemResult:
        """
        Solve the complete system.
        
        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            System input data
            
        Returns
        -------
        SystemResult
            System computation result
        """

    def add_component(self, name: str, component: BaseComponent) -> None:
        """
        Add a component to the system.
        
        Parameters
        ----------
        name : str
            Component name
        component : BaseComponent
            Component instance
        """
        self.components[name] = component
        self._validate_system()
        log.debug(f"Added component {name} to system {self.name}")

    def remove_component(self, name: str) -> None:
        """
        Remove a component from the system.
        
        Parameters
        ----------
        name : str
            Component name to remove
        """
        if name in self.components:
            del self.components[name]
            self._validate_system()
            log.debug(f"Removed component {name} from system {self.name}")
        else:
            log.warning(f"Component {name} not found in system {self.name}")

    def get_component(self, name: str) -> Optional[BaseComponent]:
        """
        Get a component by name.
        
        Parameters
        ----------
        name : str
            Component name
            
        Returns
        -------
        BaseComponent or None
            Component instance if found
        """
        return self.components.get(name)

    def list_components(self) -> List[str]:
        """
        Get list of component names.
        
        Returns
        -------
        List[str]
            List of component names
        """
        return list(self.components.keys())

    def validate_inputs(self, inputs: Dict[str, np.ndarray]) -> bool:
        """
        Validate system inputs.
        
        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            Input data to validate
            
        Returns
        -------
        bool
            True if inputs are valid
        """
        # Override in subclasses for system-specific validation
        return True

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns
        -------
        Dict[str, Any]
            System information including components and parameters
        """
        component_info = {}
        for name, component in self.components.items():
            component_info[name] = {
                "type": component.__class__.__name__,
                "parameters": component.get_parameters(),
                "required_inputs": component.get_required_inputs(),
                "outputs": component.get_outputs(),
            }

        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "components": component_info,
            "num_components": len(self.components),
        }

    def __repr__(self) -> str:
        """String representation of system."""
        return f"{self.__class__.__name__}(name='{self.name}', components={list(self.components.keys())})"

