"""
Crank kinematics analysis for piston-crank systems with offset effects.

This module provides kinematic analysis capabilities for crank center optimization,
computing connecting rod angles, velocities, and accelerations while accounting
for crank center offset effects on piston motion.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from campro.logging import get_logger
from campro.physics.base import BasePhysicsModel, PhysicsResult

log = get_logger(__name__)


@dataclass
class CrankKinematicsResult:
    """Result of crank kinematics analysis computation."""

    # Kinematic data
    rod_angles: np.ndarray  # Connecting rod angles (rad)
    rod_angular_velocities: np.ndarray  # Rod angular velocities (rad/s)
    rod_angular_accelerations: np.ndarray  # Rod angular accelerations (rad/s²)

    # Piston motion data (corrected for offset)
    piston_displacements: np.ndarray  # Piston displacements (mm)
    piston_velocities: np.ndarray  # Piston velocities (mm/s)
    piston_accelerations: np.ndarray  # Piston accelerations (mm/s²)

    # Crank angle data
    crank_angles: np.ndarray  # Crank angles (rad)

    # Analysis parameters
    crank_center_offset: tuple[float, float]  # (x, y) offset from gear center (mm)
    crank_radius: float  # Crank radius (mm)
    rod_length: float  # Connecting rod length (mm)

    # Performance metrics
    max_rod_angle: float  # Maximum rod angle (rad)
    max_piston_velocity: float  # Maximum piston velocity (mm/s)
    max_piston_acceleration: float  # Maximum piston acceleration (mm/s²)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class CrankKinematics(BasePhysicsModel):
    """
    Analyzes crank kinematics with crank center offset effects.

    This class computes connecting rod kinematics and corrected piston motion
    accounting for crank center offset effects, providing the foundation for
    accurate torque and side-loading analysis.
    """

    def __init__(self, name: str = "CrankKinematics"):
        super().__init__(name)
        self._crank_radius: float | None = None
        self._rod_length: float | None = None
        self._motion_law_data: dict[str, np.ndarray] | None = None

    def configure(self, crank_radius: float, rod_length: float, **kwargs) -> None:
        """
        Configure the crank kinematics analyzer with system parameters.

        Args:
            crank_radius: Crank radius in mm
            rod_length: Connecting rod length in mm
            **kwargs: Additional configuration parameters
        """
        if crank_radius <= 0:
            raise ValueError("Crank radius must be positive")
        if rod_length <= 0:
            raise ValueError("Rod length must be positive")

        self._crank_radius = crank_radius
        self._rod_length = rod_length
        self._is_configured = True

        log.info(
            f"Configured {self.name}: crank_radius={crank_radius}mm, rod_length={rod_length}mm",
        )

    def simulate(self, inputs: dict[str, Any], **kwargs) -> PhysicsResult:
        """
        Compute crank kinematics analysis for given inputs.

        Args:
            inputs: Dictionary containing:
                - motion_law_data: Dict with 'theta', 'displacement', 'velocity', 'acceleration'
                - crank_center_offset: Tuple (x, y) offset from gear center (mm)
                - angular_velocity: Optional constant angular velocity (rad/s)
            **kwargs: Additional simulation parameters

        Returns:
            PhysicsResult with crank kinematics analysis data
        """
        self._validate_inputs(inputs)

        result = self._start_simulation()
        result.simulation_time = time.time()

        try:
            # Extract inputs
            motion_law_data = inputs["motion_law_data"]
            crank_center_offset = inputs["crank_center_offset"]
            angular_velocity = inputs.get(
                "angular_velocity", 100.0,
            )  # Default 100 rad/s

            # Validate input data
            self._validate_motion_law_data(motion_law_data)

            # Compute crank kinematics analysis
            kinematics_result = self._compute_kinematics_analysis(
                motion_law_data,
                crank_center_offset,
                angular_velocity,
            )

            # Prepare result data
            data = {
                "rod_angles": kinematics_result.rod_angles,
                "rod_angular_velocities": kinematics_result.rod_angular_velocities,
                "rod_angular_accelerations": kinematics_result.rod_angular_accelerations,
                "piston_displacements": kinematics_result.piston_displacements,
                "piston_velocities": kinematics_result.piston_velocities,
                "piston_accelerations": kinematics_result.piston_accelerations,
                "crank_angles": kinematics_result.crank_angles,
                "crank_center_offset": np.array(crank_center_offset),
                "max_rod_angle": np.array([kinematics_result.max_rod_angle]),
                "max_piston_velocity": np.array(
                    [kinematics_result.max_piston_velocity],
                ),
                "max_piston_acceleration": np.array(
                    [kinematics_result.max_piston_acceleration],
                ),
            }

            # Store kinematics result in metadata for easy access
            convergence_info = {
                "kinematics_result": kinematics_result,
                "crank_radius": kinematics_result.crank_radius,
                "rod_length": kinematics_result.rod_length,
                "angular_velocity": angular_velocity,
            }

            return self._finish_simulation(result, data, convergence_info)

        except Exception as e:
            error_msg = f"Crank kinematics analysis failed: {e!s}"
            log.error(error_msg)
            return self._finish_simulation(result, {}, error_message=error_msg)

    def compute_rod_angle(
        self, crank_angle: float, crank_center_offset: tuple[float, float],
    ) -> float:
        """
        Compute connecting rod angle for given crank angle and center offset.

        Args:
            crank_angle: Crank angle (rad)
            crank_center_offset: (x, y) offset from gear center (mm)

        Returns:
            Connecting rod angle (rad)
        """
        if not self._is_configured:
            raise RuntimeError("Kinematics analyzer must be configured before use")

        # Basic rod angle computation
        # sin(φ) = (r * sin(θ)) / L
        # where r is crank radius, L is rod length, θ is crank angle

        # Account for crank center offset
        # Offset affects the effective crank position
        effective_crank_angle = crank_angle + np.arctan2(
            crank_center_offset[1], crank_center_offset[0],
        )

        rod_angle = np.arcsin(
            (self._crank_radius * np.sin(effective_crank_angle)) / self._rod_length,
        )

        return rod_angle

    def compute_rod_angular_velocity(
        self,
        crank_angle: float,
        crank_angular_velocity: float,
        crank_center_offset: tuple[float, float],
    ) -> float:
        """
        Compute connecting rod angular velocity.

        Args:
            crank_angle: Crank angle (rad)
            crank_angular_velocity: Crank angular velocity (rad/s)
            crank_center_offset: (x, y) offset from gear center (mm)

        Returns:
            Rod angular velocity (rad/s)
        """
        if not self._is_configured:
            raise RuntimeError("Kinematics analyzer must be configured before use")

        # Compute rod angle
        rod_angle = self.compute_rod_angle(crank_angle, crank_center_offset)

        # Compute rod angular velocity using chain rule
        # dφ/dt = (dφ/dθ) * (dθ/dt)
        # where dφ/dθ is the derivative of rod angle w.r.t. crank angle

        # Derivative of arcsin((r*sin(θ))/L) w.r.t. θ
        sin_theta = np.sin(crank_angle)
        cos_theta = np.cos(crank_angle)

        # dφ/dθ = (r*cos(θ)) / (L * sqrt(1 - (r*sin(θ)/L)²))
        denominator = np.sqrt(
            1 - (self._crank_radius * sin_theta / self._rod_length) ** 2,
        )
        if denominator == 0:
            # Handle singularity case
            rod_angular_velocity = 0.0
        else:
            dphi_dtheta = (self._crank_radius * cos_theta) / (
                self._rod_length * denominator
            )
            rod_angular_velocity = dphi_dtheta * crank_angular_velocity

        return rod_angular_velocity

    def compute_corrected_piston_motion(
        self,
        motion_law_data: dict[str, np.ndarray],
        crank_center_offset: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        """
        Compute piston motion corrected for crank center offset effects.

        Args:
            motion_law_data: Original motion law data
            crank_center_offset: (x, y) offset from gear center (mm)

        Returns:
            Dictionary with corrected piston motion data
        """
        if not self._is_configured:
            raise RuntimeError("Kinematics analyzer must be configured before use")

        crank_angles = motion_law_data["theta"]
        n_points = len(crank_angles)

        # Initialize arrays
        corrected_displacements = np.zeros(n_points)
        corrected_velocities = np.zeros(n_points)
        corrected_accelerations = np.zeros(n_points)

        for i, crank_angle in enumerate(crank_angles):
            # Compute rod angle
            rod_angle = self.compute_rod_angle(crank_angle, crank_center_offset)

            # Compute corrected piston displacement
            # Original displacement plus offset effects
            original_displacement = motion_law_data["displacement"][i]

            # Offset correction: account for crank center offset
            offset_correction = crank_center_offset[0] * np.cos(
                crank_angle,
            ) + crank_center_offset[1] * np.sin(crank_angle)

            corrected_displacements[i] = original_displacement + offset_correction

            # Compute corrected velocity (numerical derivative)
            if i > 0:
                dt = crank_angle - crank_angles[i - 1]
                corrected_velocities[i] = (
                    corrected_displacements[i] - corrected_displacements[i - 1]
                ) / dt
            else:
                corrected_velocities[i] = motion_law_data["velocity"][i]

            # Compute corrected acceleration (numerical derivative)
            if i > 1:
                dt = crank_angle - crank_angles[i - 1]
                corrected_accelerations[i] = (
                    corrected_velocities[i] - corrected_velocities[i - 1]
                ) / dt
            else:
                corrected_accelerations[i] = motion_law_data["acceleration"][i]

        return {
            "displacement": corrected_displacements,
            "velocity": corrected_velocities,
            "acceleration": corrected_accelerations,
        }

    def _compute_kinematics_analysis(
        self,
        motion_law_data: dict[str, np.ndarray],
        crank_center_offset: tuple[float, float],
        angular_velocity: float,
    ) -> CrankKinematicsResult:
        """Compute complete crank kinematics analysis."""

        crank_angles = motion_law_data["theta"]
        n_points = len(crank_angles)

        # Initialize arrays
        rod_angles = np.zeros(n_points)
        rod_angular_velocities = np.zeros(n_points)
        rod_angular_accelerations = np.zeros(n_points)

        # Compute rod angles and velocities
        for i, crank_angle in enumerate(crank_angles):
            rod_angles[i] = self.compute_rod_angle(crank_angle, crank_center_offset)
            rod_angular_velocities[i] = self.compute_rod_angular_velocity(
                crank_angle,
                angular_velocity,
                crank_center_offset,
            )

        # Compute rod angular accelerations (numerical derivative)
        for i in range(1, n_points):
            dt = crank_angles[i] - crank_angles[i - 1]
            rod_angular_accelerations[i] = (
                rod_angular_velocities[i] - rod_angular_velocities[i - 1]
            ) / dt

        # Compute corrected piston motion
        corrected_motion = self.compute_corrected_piston_motion(
            motion_law_data, crank_center_offset,
        )

        # Compute performance metrics
        max_rod_angle = np.max(np.abs(rod_angles))
        max_piston_velocity = np.max(np.abs(corrected_motion["velocity"]))
        max_piston_acceleration = np.max(np.abs(corrected_motion["acceleration"]))

        return CrankKinematicsResult(
            rod_angles=rod_angles,
            rod_angular_velocities=rod_angular_velocities,
            rod_angular_accelerations=rod_angular_accelerations,
            piston_displacements=corrected_motion["displacement"],
            piston_velocities=corrected_motion["velocity"],
            piston_accelerations=corrected_motion["acceleration"],
            crank_angles=crank_angles,
            crank_center_offset=crank_center_offset,
            crank_radius=self._crank_radius,
            rod_length=self._rod_length,
            max_rod_angle=max_rod_angle,
            max_piston_velocity=max_piston_velocity,
            max_piston_acceleration=max_piston_acceleration,
            metadata={
                "angular_velocity": angular_velocity,
                "n_points": n_points,
            },
        )

    def _validate_motion_law_data(self, motion_law_data: dict[str, np.ndarray]) -> None:
        """Validate motion law data structure."""

        required_keys = ["theta", "displacement", "velocity", "acceleration"]
        for key in required_keys:
            if key not in motion_law_data:
                raise ValueError(f"Motion law data missing required key: {key}")

            if not isinstance(motion_law_data[key], np.ndarray):
                raise TypeError(f"Motion law data[{key}] must be numpy array")

        # Check that all arrays have same length
        lengths = [len(motion_law_data[key]) for key in required_keys]
        if len(set(lengths)) > 1:
            raise ValueError("All motion law arrays must have same length")
