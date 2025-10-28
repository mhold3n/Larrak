"""
System builder for creating configurable physics systems.

This module provides a builder pattern for creating complex systems
from modular components with flexible configuration.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from campro.logging import get_logger

from ..physics.base import BaseComponent, BaseSystem
from ..physics.geometry import CamCurveComponent, CurvatureComponent
from ..physics.kinematics import MeshingLawComponent

log = get_logger(__name__)


@dataclass
class SystemConfiguration:
    """Configuration for a complete system."""

    name: str
    components: Dict[str, Dict[str, Any]]
    connections: List[Dict[str, str]]
    parameters: Dict[str, Any]


class SystemBuilder:
    """
    Builder for creating configurable physics systems.

    This class enables the creation of complex systems by combining
    modular components with flexible configuration options.
    """

    def __init__(self, name: str = "CustomSystem"):
        """
        Initialize the system builder.

        Parameters
        ----------
        name : str
            Name of the system to build
        """
        self.name = name
        self.components = {}
        self.connections = []
        self.parameters = {}
        self._component_registry = self._initialize_component_registry()
        log.debug(f"Initialized system builder for: {name}")

    def _initialize_component_registry(self) -> Dict[str, Type[BaseComponent]]:
        """Initialize registry of available components."""
        return {
            "cam_curve": CamCurveComponent,
            "curvature": CurvatureComponent,
            "meshing_law": MeshingLawComponent,
            # Add more components as they are created
        }

    def add_component(
        self,
        name: str,
        component_type: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> "SystemBuilder":
        """
        Add a component to the system.

        Parameters
        ----------
        name : str
            Component name
        component_type : str
            Type of component to add
        parameters : Dict[str, Any], optional
            Component parameters

        Returns
        -------
        SystemBuilder
            Self for method chaining
        """
        if component_type not in self._component_registry:
            raise ValueError(f"Unknown component type: {component_type}")

        component_class = self._component_registry[component_type]
        component_params = parameters or {}

        self.components[name] = {
            "type": component_type,
            "class": component_class,
            "parameters": component_params,
        }

        log.debug(f"Added component {name} of type {component_type}")
        return self

    def connect_components(
        self, from_component: str, to_component: str, connection_type: str = "data_flow",
    ) -> "SystemBuilder":
        """
        Connect components in the system.

        Parameters
        ----------
        from_component : str
            Source component name
        to_component : str
            Target component name
        connection_type : str
            Type of connection

        Returns
        -------
        SystemBuilder
            Self for method chaining
        """
        if from_component not in self.components:
            raise ValueError(f"Source component {from_component} not found")
        if to_component not in self.components:
            raise ValueError(f"Target component {to_component} not found")

        self.connections.append(
            {
                "from": from_component,
                "to": to_component,
                "type": connection_type,
            },
        )

        log.debug(f"Connected {from_component} -> {to_component} ({connection_type})")
        return self

    def set_parameters(self, parameters: Dict[str, Any]) -> "SystemBuilder":
        """
        Set system-level parameters.

        Parameters
        ----------
        parameters : Dict[str, Any]
            System parameters

        Returns
        -------
        SystemBuilder
            Self for method chaining
        """
        self.parameters.update(parameters)
        log.debug(f"Set system parameters: {list(parameters.keys())}")
        return self

    def build_cam_ring_system(
        self, base_radius: float = 10.0, connecting_rod_length: float = 25.0,
    ) -> "SystemBuilder":
        """
        Build a standard cam-ring system configuration.

        Parameters
        ----------
        base_radius : float
            Cam base radius
        connecting_rod_length : float
            Connecting rod length

        Returns
        -------
        SystemBuilder
            Self for method chaining
        """
        # Add cam curve component
        self.add_component(
            "cam_curves",
            "cam_curve",
            {"base_radius": base_radius},
        )

        # Add curvature component
        self.add_component(
            "curvature",
            "curvature",
            {},
        )

        # Add meshing law component
        self.add_component(
            "meshing_law",
            "meshing_law",
            {},
        )

        # Connect components
        self.connect_components("cam_curves", "curvature")
        self.connect_components("curvature", "meshing_law")

        # Set system parameters
        self.set_parameters(
            {
                "base_radius": base_radius,
                "connecting_rod_length": connecting_rod_length,
                "system_type": "cam_ring",
            },
        )

        log.info(f"Built cam-ring system with base_radius={base_radius}")
        return self

    def get_configuration(self) -> SystemConfiguration:
        """
        Get the current system configuration.

        Returns
        -------
        SystemConfiguration
            Current configuration
        """
        return SystemConfiguration(
            name=self.name,
            components=self.components,
            connections=self.connections,
            parameters=self.parameters,
        )

    def create_system(self) -> BaseSystem:
        """
        Create the configured system.

        Returns
        -------
        BaseSystem
            Configured system instance
        """
        # Create component instances
        component_instances = {}
        for name, config in self.components.items():
            component_class = config["class"]
            parameters = config["parameters"]
            component_instances[name] = component_class(parameters, name=name)

        # Create system (this would be implemented in a specific system class)
        # For now, return a placeholder
        log.info(
            f"Created system {self.name} with {len(component_instances)} components",
        )

        # This would be replaced with actual system creation
        from ..physics.base import BaseSystem

        return BaseSystem(component_instances, name=self.name)

    def validate_configuration(self) -> bool:
        """
        Validate the current configuration.

        Returns
        -------
        bool
            True if configuration is valid
        """
        try:
            # Check that all components exist
            for connection in self.connections:
                if connection["from"] not in self.components:
                    log.error(f"Connection source {connection['from']} not found")
                    return False
                if connection["to"] not in self.components:
                    log.error(f"Connection target {connection['to']} not found")
                    return False

            # Check that all components can be instantiated
            for name, config in self.components.items():
                component_class = config["class"]
                parameters = config["parameters"]
                try:
                    component_class(parameters, name=name)
                except Exception as e:
                    log.error(f"Component {name} validation failed: {e}")
                    return False

            log.info("Configuration validation passed")
            return True

        except Exception as e:
            log.error(f"Configuration validation failed: {e}")
            return False

    def __repr__(self) -> str:
        """String representation of builder."""
        return f"SystemBuilder(name='{self.name}', components={len(self.components)})"
