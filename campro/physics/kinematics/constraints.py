"""
Kinematic constraints component.

This module provides a modular component for kinematic constraint analysis.
"""

import numpy as np

from campro.logging import get_logger

from ..base import BaseComponent, ComponentResult, ComponentStatus

log = get_logger(__name__)


class KinematicConstraintsComponent(BaseComponent):
    """
    Component for kinematic constraint analysis.

    This component checks and enforces kinematic constraints
    for cam-ring systems.
    """

    def _validate_parameters(self) -> None:
        """Validate component parameters."""

    def compute(self, inputs: dict[str, np.ndarray]) -> ComponentResult:
        """
        Compute kinematic constraints.

        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            Input data

        Returns
        -------
        ComponentResult
            Result containing constraint analysis
        """
        try:
            violations = []

            # 1. Velocity Check
            velocity = inputs.get("velocity")
            if velocity is not None:
                max_v_abs = np.max(np.abs(velocity))
                # Limit: 25 m/s (25000 mm/s) - typical high performance limit
                limit_v = 25000.0
                if max_v_abs > limit_v:
                    violations.append(
                        f"Max velocity {max_v_abs:.1f} mm/s exceeds limit {limit_v:.1f}"
                    )

            # 2. Acceleration Check
            acceleration = inputs.get("acceleration")
            if acceleration is not None:
                max_a_abs = np.max(np.abs(acceleration))
                # Limit: 5000 m/s^2 (5e6 mm/s^2) - very high, but depends on mass
                limit_a = 5.0e6
                if max_a_abs > limit_a:
                    violations.append(
                        f"Max acceleration {max_a_abs:.1e} mm/s^2 exceeds limit {limit_a:.1e}"
                    )

            # 3. Curvature / Undercut Check
            radius_of_curvature = inputs.get("radius_of_curvature")
            if radius_of_curvature is not None:
                min_rho = np.min(radius_of_curvature)
                # Limit: Must be positive or > specific radius (depending on follower)
                # For convex cam, rho > -R_base roughly.
                # Assuming roller radius needs to be checked against, but simple check:
                limit_rho = -10.0  # Example negative limit allowed for concatenation
                if min_rho < limit_rho:
                    violations.append(
                        f"Min curvature radius {min_rho:.1f} mm suggests potential undercut"
                    )

            status = ComponentStatus.COMPLETED
            if violations:
                # We log violations but maybe don't fail the component itself, just report properties
                # Or set status to WARNING if we had one.
                pass

            return ComponentResult(
                status=status,
                outputs={
                    "is_feasible": np.array([len(violations) == 0]),
                    "violation_count": np.array([len(violations)]),
                },
                metadata={"violations": violations, "checked_fields": list(inputs.keys())},
            )

        except Exception as e:
            log.error(f"Error in kinematic constraints: {e}")
            return ComponentResult(
                status=ComponentStatus.FAILED,
                outputs={},
                metadata={},
                error_message=str(e),
            )

    def get_required_inputs(self) -> list[str]:
        """Get list of required input names."""
        return []

    def get_outputs(self) -> list[str]:
        """Get list of output names."""
        return []
