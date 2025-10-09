"""
Cam curve computation component.

This module provides a modular component for computing cam curves
from linear follower motion laws.
"""

from typing import Dict, List

import numpy as np

from campro.logging import get_logger

from ..base import BaseComponent, ComponentResult, ComponentStatus

log = get_logger(__name__)


class CamCurveComponent(BaseComponent):
    """
    Component for computing cam curves from linear follower motion law.
    
    This component computes the cam pitch curve, profile curve, and contact curve
    based on the linear follower displacement and system parameters.
    """

    def _validate_parameters(self) -> None:
        """Validate component parameters."""
        required_params = ["base_radius"]
        for param in required_params:
            if param not in self.parameters:
                raise ValueError(f"Missing required parameter: {param}")

        if self.parameters["base_radius"] <= 0:
            raise ValueError("Base radius must be positive")

    def compute(self, inputs: Dict[str, np.ndarray]) -> ComponentResult:
        """
        Compute cam curves from linear follower motion law.
        
        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            Input data containing:
            - 'theta': Cam angles (radians)
            - 'x_theta': Linear follower displacement vs cam angle
            
        Returns
        -------
        ComponentResult
            Result containing cam curves
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
            x_theta = inputs["x_theta"]

            log.info(f"Computing cam curves for {len(theta)} points")

            # Compute cam curves
            base_radius = self.parameters["base_radius"]

            # Pitch curve (connecting rod connection point)
            pitch_radius = base_radius + x_theta

            # Profile curve (actual cam surface)
            # For direct contact with ring follower, profile equals pitch
            profile_radius = pitch_radius.copy()

            # Contact curve (for ring contact)
            # Since cam directly contacts ring follower, use profile radius
            contact_radius = profile_radius.copy()

            # Prepare outputs
            outputs = {
                "pitch_radius": pitch_radius,
                "profile_radius": profile_radius,
                "contact_radius": contact_radius,
            }

            # Prepare metadata
            metadata = {
                "num_points": len(theta),
                "base_radius": base_radius,
                "min_radius": float(np.min(profile_radius)),
                "max_radius": float(np.max(profile_radius)),
                "radius_range": float(np.max(profile_radius) - np.min(profile_radius)),
            }

            log.info(f"Cam curves computed successfully: radius range {metadata['radius_range']:.3f}")

            return ComponentResult(
                status=ComponentStatus.COMPLETED,
                outputs=outputs,
                metadata=metadata,
            )

        except Exception as e:
            log.error(f"Error computing cam curves: {e}")
            return ComponentResult(
                status=ComponentStatus.FAILED,
                outputs={},
                metadata={},
                error_message=str(e),
            )

    def get_required_inputs(self) -> List[str]:
        """Get list of required input names."""
        return ["theta", "x_theta"]

    def get_outputs(self) -> List[str]:
        """Get list of output names."""
        return ["pitch_radius", "profile_radius", "contact_radius"]

