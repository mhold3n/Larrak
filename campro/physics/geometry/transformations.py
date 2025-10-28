"""
Coordinate transformation component.

This module provides a modular component for coordinate transformations
between different reference frames.
"""

from typing import Dict, List

import numpy as np

from campro.logging import get_logger

from ..base import BaseComponent, ComponentResult, ComponentStatus

log = get_logger(__name__)


class CoordinateTransformComponent(BaseComponent):
    """
    Component for coordinate transformations.

    This component handles transformations between different coordinate systems
    used in cam-ring systems.
    """

    def _validate_parameters(self) -> None:
        """Validate component parameters."""
        # No specific parameters required for basic transformations

    def compute(self, inputs: Dict[str, np.ndarray]) -> ComponentResult:
        """
        Perform coordinate transformations.

        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            Input data containing:
            - 'theta': Angles (radians)
            - 'r': Radius values
            - 'transform_type': Type of transformation to perform

        Returns
        -------
        ComponentResult
            Result containing transformed coordinates
        """
        try:
            # Validate inputs
            if not self.validate_inputs(inputs):
                return ComponentResult(
                    status=ComponentStatus.FAILED,
                    outputs={},
                    metadata={},
                    error_message="Invalid inputs",
                )

            theta = inputs["theta"]
            r = inputs["r"]
            transform_type = inputs.get("transform_type", "polar_to_cartesian")

            log.info(
                f"Performing {transform_type} transformation for {len(theta)} points",
            )

            if transform_type == "polar_to_cartesian":
                outputs = self._polar_to_cartesian(theta, r)
            elif transform_type == "cartesian_to_polar":
                outputs = self._cartesian_to_polar(theta, r)
            else:
                raise ValueError(f"Unknown transformation type: {transform_type}")

            # Prepare metadata
            metadata = {
                "num_points": len(theta),
                "transform_type": transform_type,
            }

            log.info("Transformation completed successfully")

            return ComponentResult(
                status=ComponentStatus.COMPLETED,
                outputs=outputs,
                metadata=metadata,
            )

        except Exception as e:
            log.error(f"Error in coordinate transformation: {e}")
            return ComponentResult(
                status=ComponentStatus.FAILED,
                outputs={},
                metadata={},
                error_message=str(e),
            )

    def _polar_to_cartesian(
        self, theta: np.ndarray, r: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Convert polar to cartesian coordinates."""
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return {
            "x": x,
            "y": y,
            "theta": theta,
            "r": r,
        }

    def _cartesian_to_polar(
        self, x: np.ndarray, y: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Convert cartesian to polar coordinates."""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        return {
            "x": x,
            "y": y,
            "theta": theta,
            "r": r,
        }

    def get_required_inputs(self) -> List[str]:
        """Get list of required input names."""
        return ["theta", "r"]

    def get_optional_inputs(self) -> List[str]:
        """Get list of optional input names."""
        return ["transform_type"]

    def get_outputs(self) -> List[str]:
        """Get list of output names."""
        return ["x", "y", "theta", "r"]
