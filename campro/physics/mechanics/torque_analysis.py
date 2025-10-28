"""
Torque analysis for piston-crank systems with Litvin gear geometry.

This module provides torque computation capabilities for crank center optimization,
integrating motion law data, Litvin gear geometry, and crank kinematics to compute
instantaneous and cycle-averaged torque outputs.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from campro.logging import get_logger
from campro.physics.base import BasePhysicsModel, PhysicsResult
from campro.physics.geometry.litvin import LitvinGearGeometry

log = get_logger(__name__)


@dataclass
class TorqueAnalysisResult:
    """Result of torque analysis computation."""

    # Torque data
    instantaneous_torque: np.ndarray  # Torque at each crank angle (N⋅m)
    cycle_average_torque: float  # Average torque over full cycle (N⋅m)
    max_torque: float  # Maximum instantaneous torque (N⋅m)
    min_torque: float  # Minimum instantaneous torque (N⋅m)

    # Crank angle data
    crank_angles: np.ndarray  # Crank angles (rad)

    # Analysis parameters
    crank_center_offset: Tuple[float, float]  # (x, y) offset from gear center (mm)
    crank_radius: float  # Crank radius (mm)
    rod_length: float  # Connecting rod length (mm)

    # Performance metrics
    torque_ripple: float  # Torque variation coefficient
    power_output: float  # Average power output (W)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class PistonTorqueCalculator(BasePhysicsModel):
    """
    Computes piston torque from motion law, gear geometry, and crank kinematics.

    This class integrates motion law data with Litvin gear geometry to compute
    instantaneous torque at the crank, accounting for crank center offset effects
    and connecting rod kinematics.
    """

    def __init__(self, name: str = "PistonTorqueCalculator"):
        super().__init__(name)
        self._crank_radius: Optional[float] = None
        self._rod_length: Optional[float] = None
        self._gear_geometry: Optional[LitvinGearGeometry] = None
        self._motion_law_data: Optional[Dict[str, np.ndarray]] = None
        self._load_profile: Optional[np.ndarray] = None

    def configure(
        self,
        crank_radius: float,
        rod_length: float,
        gear_geometry: LitvinGearGeometry,
        **kwargs,
    ) -> None:
        """
        Configure the torque calculator with system parameters.

        Args:
            crank_radius: Crank radius in mm
            rod_length: Connecting rod length in mm
            gear_geometry: Litvin gear geometry from secondary optimization
            **kwargs: Additional configuration parameters
        """
        if crank_radius <= 0:
            raise ValueError("Crank radius must be positive")
        if rod_length <= 0:
            raise ValueError("Rod length must be positive")
        if not isinstance(gear_geometry, LitvinGearGeometry):
            raise ValueError("gear_geometry must be a LitvinGearGeometry instance")

        self._crank_radius = crank_radius
        self._rod_length = rod_length
        self._gear_geometry = gear_geometry
        self._is_configured = True

        log.info(
            f"Configured {self.name}: crank_radius={crank_radius}mm, rod_length={rod_length}mm",
        )

    def simulate(self, inputs: Dict[str, Any], **kwargs) -> PhysicsResult:
        """
        Compute torque analysis for given inputs.

        Args:
            inputs: Dictionary containing:
                - motion_law_data: Dict with 'theta', 'displacement', 'velocity', 'acceleration'
                - load_profile: Array of piston forces (N)
                - crank_center_offset: Tuple (x, y) offset from gear center (mm)
            **kwargs: Additional simulation parameters

        Returns:
            PhysicsResult with torque analysis data
        """
        self._validate_inputs(inputs)

        result = self._start_simulation()
        result.simulation_time = time.time()

        try:
            # Extract inputs
            motion_law_data = inputs["motion_law_data"]
            load_profile = inputs["load_profile"]
            crank_center_offset = inputs["crank_center_offset"]

            # Validate input data
            self._validate_motion_law_data(motion_law_data)
            self._validate_load_profile(load_profile, motion_law_data)

            # Compute torque analysis
            torque_result = self._compute_torque_analysis(
                motion_law_data,
                load_profile,
                crank_center_offset,
            )

            # Prepare result data
            data = {
                "instantaneous_torque": torque_result.instantaneous_torque,
                "cycle_average_torque": np.array([torque_result.cycle_average_torque]),
                "crank_angles": torque_result.crank_angles,
                "crank_center_offset": np.array(crank_center_offset),
                "torque_ripple": np.array([torque_result.torque_ripple]),
                "power_output": np.array([torque_result.power_output]),
            }

            # Store torque result in metadata for easy access
            convergence_info = {
                "torque_result": torque_result,
                "max_torque": torque_result.max_torque,
                "min_torque": torque_result.min_torque,
                "crank_radius": torque_result.crank_radius,
                "rod_length": torque_result.rod_length,
            }

            return self._finish_simulation(result, data, convergence_info)

        except Exception as e:
            error_msg = f"Torque analysis failed: {e!s}"
            log.error(error_msg)
            return self._finish_simulation(result, {}, error_message=error_msg)

    def compute_instantaneous_torque(
        self,
        piston_force: float,
        crank_angle: float,
        crank_center_offset: Tuple[float, float],
        pressure_angle: float,
    ) -> float:
        """
        Compute instantaneous torque for given conditions.

        Args:
            piston_force: Piston force (N)
            crank_angle: Crank angle (rad)
            crank_center_offset: (x, y) offset from gear center (mm)
            pressure_angle: Gear pressure angle (rad)

        Returns:
            Instantaneous torque (N⋅m)
        """
        if not self._is_configured:
            raise RuntimeError("Calculator must be configured before use")

        # Compute connecting rod angle
        rod_angle = self._compute_rod_angle(crank_angle, crank_center_offset)

        # Compute effective crank radius (accounting for offset)
        effective_crank_radius = self._compute_effective_crank_radius(
            crank_angle,
            crank_center_offset,
        )

        # Compute torque component from piston force
        # T = F_piston * r_effective * sin(θ + φ) * cos(α)
        # where φ is rod angle and α is pressure angle
        torque_component = (
            piston_force
            * effective_crank_radius
            * np.sin(crank_angle + rod_angle)
            * np.cos(pressure_angle)
        )

        return torque_component

    def compute_cycle_average_torque(
        self,
        motion_law_data: Dict[str, np.ndarray],
        load_profile: np.ndarray,
        crank_center_offset: Tuple[float, float],
    ) -> float:
        """
        Compute cycle-averaged torque for complete motion cycle.

        Args:
            motion_law_data: Motion law data with theta, displacement, velocity, acceleration
            load_profile: Piston force profile (N)
            crank_center_offset: (x, y) offset from gear center (mm)

        Returns:
            Cycle-averaged torque (N⋅m)
        """
        if not self._is_configured:
            raise RuntimeError("Calculator must be configured before use")

        # Get pressure angle from gear geometry
        pressure_angle = self._get_pressure_angle()

        # Compute instantaneous torques
        crank_angles = motion_law_data["theta"]
        instantaneous_torques = np.zeros_like(crank_angles)

        for i, (crank_angle, piston_force) in enumerate(
            zip(crank_angles, load_profile),
        ):
            instantaneous_torques[i] = self.compute_instantaneous_torque(
                piston_force,
                crank_angle,
                crank_center_offset,
                pressure_angle,
            )

        # Compute cycle average
        cycle_average = np.mean(instantaneous_torques)

        return cycle_average

    def _compute_torque_analysis(
        self,
        motion_law_data: Dict[str, np.ndarray],
        load_profile: np.ndarray,
        crank_center_offset: Tuple[float, float],
    ) -> TorqueAnalysisResult:
        """Compute complete torque analysis."""

        crank_angles = motion_law_data["theta"]
        pressure_angle = self._get_pressure_angle()

        # Compute instantaneous torques
        instantaneous_torques = np.zeros_like(crank_angles)

        for i, (crank_angle, piston_force) in enumerate(
            zip(crank_angles, load_profile),
        ):
            instantaneous_torques[i] = self.compute_instantaneous_torque(
                piston_force,
                crank_angle,
                crank_center_offset,
                pressure_angle,
            )

        # Compute cycle metrics
        cycle_average_torque = np.mean(instantaneous_torques)
        max_torque = np.max(instantaneous_torques)
        min_torque = np.min(instantaneous_torques)

        # Compute torque ripple (coefficient of variation)
        torque_std = np.std(instantaneous_torques)
        torque_ripple = (
            torque_std / abs(cycle_average_torque) if cycle_average_torque != 0 else 0
        )

        # Compute power output (assuming constant angular velocity)
        # For now, use a nominal angular velocity - this could be made configurable
        nominal_angular_velocity = 100.0  # rad/s
        power_output = cycle_average_torque * nominal_angular_velocity

        return TorqueAnalysisResult(
            instantaneous_torque=instantaneous_torques,
            cycle_average_torque=cycle_average_torque,
            max_torque=max_torque,
            min_torque=min_torque,
            crank_angles=crank_angles,
            crank_center_offset=crank_center_offset,
            crank_radius=self._crank_radius,
            rod_length=self._rod_length,
            torque_ripple=torque_ripple,
            power_output=power_output,
            metadata={
                "pressure_angle": pressure_angle,
                "nominal_angular_velocity": nominal_angular_velocity,
            },
        )

    def _compute_rod_angle(
        self, crank_angle: float, crank_center_offset: Tuple[float, float],
    ) -> float:
        """Compute connecting rod angle accounting for crank center offset."""

        # For now, use simplified rod angle computation
        # This could be enhanced to account for crank center offset effects
        # sin(φ) = (r * sin(θ)) / L
        # where r is crank radius, L is rod length, θ is crank angle

        rod_angle = np.arcsin(
            (self._crank_radius * np.sin(crank_angle)) / self._rod_length,
        )

        return rod_angle

    def _compute_effective_crank_radius(
        self, crank_angle: float, crank_center_offset: Tuple[float, float],
    ) -> float:
        """Compute effective crank radius accounting for center offset."""

        # For now, use nominal crank radius
        # This could be enhanced to account for offset effects on effective radius
        effective_radius = self._crank_radius

        return effective_radius

    def _get_pressure_angle(self) -> float:
        """Get pressure angle from gear geometry."""

        if self._gear_geometry is None:
            # Use default pressure angle if no gear geometry available
            return np.radians(20.0)  # 20 degrees default

        # Extract pressure angle from gear geometry
        # This is a simplified implementation - could be enhanced based on actual gear geometry structure
        if hasattr(self._gear_geometry, "pressure_angle"):
            return self._gear_geometry.pressure_angle
        # Use default if not available
        return np.radians(20.0)

    def _validate_motion_law_data(self, motion_law_data: Dict[str, np.ndarray]) -> None:
        """Validate motion law data structure."""

        required_keys = ["theta", "displacement", "velocity", "acceleration"]
        for key in required_keys:
            if key not in motion_law_data:
                raise ValueError(f"Motion law data missing required key: {key}")

            if not isinstance(motion_law_data[key], np.ndarray):
                raise ValueError(f"Motion law data[{key}] must be numpy array")

        # Check that all arrays have same length
        lengths = [len(motion_law_data[key]) for key in required_keys]
        if len(set(lengths)) > 1:
            raise ValueError("All motion law arrays must have same length")

    def _validate_load_profile(
        self, load_profile: np.ndarray, motion_law_data: Dict[str, np.ndarray],
    ) -> None:
        """Validate load profile data."""

        if not isinstance(load_profile, np.ndarray):
            raise ValueError("Load profile must be numpy array")

        if len(load_profile) != len(motion_law_data["theta"]):
            raise ValueError("Load profile length must match motion law data length")

        if np.any(np.isnan(load_profile)) or np.any(np.isinf(load_profile)):
            raise ValueError("Load profile contains invalid values (NaN or Inf)")
