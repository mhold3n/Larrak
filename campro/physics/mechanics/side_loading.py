"""
Side-loading analysis for piston-crank systems.

This module provides side-loading computation capabilities for crank center optimization,
integrating with existing piston dynamics to compute lateral forces and side-loading
penalties during compression and combustion phases.
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
class SideLoadResult:
    """Result of side-loading analysis computation."""

    # Side-loading data
    side_load_profile: np.ndarray  # Lateral force at each crank angle (N)
    max_side_load: float  # Maximum lateral force (N)
    avg_side_load: float  # Average lateral force (N)

    # Phase-specific analysis
    compression_side_load: np.ndarray  # Side-loading during compression phases (N)
    combustion_side_load: np.ndarray  # Side-loading during combustion phases (N)
    compression_penalty: float  # Side-loading penalty during compression
    combustion_penalty: float  # Side-loading penalty during combustion

    # Crank angle data
    crank_angles: np.ndarray  # Crank angles (rad)

    # Analysis parameters
    crank_center_offset: tuple[float, float]  # (x, y) offset from gear center (mm)
    piston_geometry: dict[str, float]  # Piston geometry parameters

    # Performance metrics
    side_load_ripple: float  # Side-loading variation coefficient
    total_penalty: float  # Combined side-loading penalty

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class SideLoadAnalyzer(BasePhysicsModel):
    """
    Analyzes side-loading effects in piston-crank systems.

    This class computes lateral forces on the piston due to connecting rod angle
    and crank center offset effects, with special attention to compression and
    combustion phases where side-loading is most critical.
    """

    def __init__(self, name: str = "SideLoadAnalyzer"):
        super().__init__(name)
        self._piston_geometry: dict[str, float] | None = None
        self._motion_law_data: dict[str, np.ndarray] | None = None
        self._load_profile: np.ndarray | None = None

    def configure(self, piston_geometry: dict[str, float], **kwargs) -> None:
        """
        Configure the side-load analyzer with piston geometry parameters.

        Args:
            piston_geometry: Dictionary containing:
                - bore_diameter: Cylinder bore diameter (mm)
                - piston_clearance: Piston-to-bore clearance (mm)
                - rod_length: Connecting rod length (mm)
                - crank_radius: Crank radius (mm)
            **kwargs: Additional configuration parameters
        """
        required_keys = [
            "bore_diameter",
            "piston_clearance",
            "rod_length",
            "crank_radius",
        ]
        for key in required_keys:
            if key not in piston_geometry:
                raise ValueError(f"piston_geometry missing required key: {key}")
            if piston_geometry[key] <= 0:
                raise ValueError(f"piston_geometry[{key}] must be positive")

        self._piston_geometry = piston_geometry.copy()
        self._is_configured = True

        log.info(
            f"Configured {self.name}: bore={piston_geometry['bore_diameter']}mm, "
            f"clearance={piston_geometry['piston_clearance']}mm",
        )

    def simulate(self, inputs: dict[str, Any], **kwargs) -> PhysicsResult:
        """
        Compute side-loading analysis for given inputs.

        Args:
            inputs: Dictionary containing:
                - motion_law_data: Dict with 'theta', 'displacement', 'velocity', 'acceleration'
                - load_profile: Array of piston forces (N)
                - crank_center_offset: Tuple (x, y) offset from gear center (mm)
                - compression_phases: Array of boolean flags for compression phases
                - combustion_phases: Array of boolean flags for combustion phases
            **kwargs: Additional simulation parameters

        Returns:
            PhysicsResult with side-loading analysis data
        """
        self._validate_inputs(inputs)

        result = self._start_simulation()
        result.simulation_time = time.time()

        try:
            # Extract inputs
            motion_law_data = inputs["motion_law_data"]
            load_profile = inputs["load_profile"]
            crank_center_offset = inputs["crank_center_offset"]
            compression_phases = inputs.get("compression_phases")
            combustion_phases = inputs.get("combustion_phases")

            # Validate input data
            self._validate_motion_law_data(motion_law_data)
            self._validate_load_profile(load_profile, motion_law_data)

            # Compute side-loading analysis
            side_load_result = self._compute_side_load_analysis(
                motion_law_data,
                load_profile,
                crank_center_offset,
                compression_phases,
                combustion_phases,
            )

            # Prepare result data
            data = {
                "side_load_profile": side_load_result.side_load_profile,
                "max_side_load": np.array([side_load_result.max_side_load]),
                "avg_side_load": np.array([side_load_result.avg_side_load]),
                "crank_angles": side_load_result.crank_angles,
                "crank_center_offset": np.array(crank_center_offset),
                "side_load_ripple": np.array([side_load_result.side_load_ripple]),
                "total_penalty": np.array([side_load_result.total_penalty]),
            }

            # Store side-load result in metadata for easy access
            convergence_info = {
                "side_load_result": side_load_result,
                "compression_penalty": side_load_result.compression_penalty,
                "combustion_penalty": side_load_result.combustion_penalty,
                "piston_geometry": side_load_result.piston_geometry,
            }

            return self._finish_simulation(result, data, convergence_info)

        except Exception as e:
            error_msg = f"Side-loading analysis failed: {e!s}"
            log.error(error_msg)
            return self._finish_simulation(result, {}, error_message=error_msg)

    def compute_side_load_profile(
        self,
        motion_law_data: dict[str, np.ndarray],
        crank_center_offset: tuple[float, float],
    ) -> np.ndarray:
        """
        Compute side-loading profile for complete motion cycle.

        Args:
            motion_law_data: Motion law data with theta, displacement, velocity, acceleration
            crank_center_offset: (x, y) offset from gear center (mm)

        Returns:
            Side-loading profile (N)
        """
        if not self._is_configured:
            raise RuntimeError("Analyzer must be configured before use")

        crank_angles = motion_law_data["theta"]
        side_loads = np.zeros_like(crank_angles)

        for i, crank_angle in enumerate(crank_angles):
            side_loads[i] = self._compute_instantaneous_side_load(
                crank_angle,
                crank_center_offset,
            )

        return side_loads

    def compute_side_load_penalty(
        self,
        side_load_profile: np.ndarray,
        compression_phases: np.ndarray | None = None,
        combustion_phases: np.ndarray | None = None,
    ) -> float:
        """
        Compute side-loading penalty considering phase-specific effects.

        Args:
            side_load_profile: Side-loading profile (N)
            compression_phases: Boolean array for compression phases
            combustion_phases: Boolean array for combustion phases

        Returns:
            Total side-loading penalty
        """
        if not self._is_configured:
            raise RuntimeError("Analyzer must be configured before use")

        # Default penalty weights
        compression_weight = 1.2
        combustion_weight = 1.5
        general_weight = 1.0

        total_penalty = 0.0

        # General side-loading penalty (RMS)
        general_penalty = np.sqrt(np.mean(side_load_profile**2))
        total_penalty += general_weight * general_penalty

        # Compression phase penalty
        if compression_phases is not None:
            compression_loads = side_load_profile[compression_phases]
            if len(compression_loads) > 0:
                compression_penalty = np.sqrt(np.mean(compression_loads**2))
                total_penalty += compression_weight * compression_penalty

        # Combustion phase penalty
        if combustion_phases is not None:
            combustion_loads = side_load_profile[combustion_phases]
            if len(combustion_loads) > 0:
                combustion_penalty = np.sqrt(np.mean(combustion_loads**2))
                total_penalty += combustion_weight * combustion_penalty

        return total_penalty

    def _compute_side_load_analysis(
        self,
        motion_law_data: dict[str, np.ndarray],
        load_profile: np.ndarray,
        crank_center_offset: tuple[float, float],
        compression_phases: np.ndarray | None = None,
        combustion_phases: np.ndarray | None = None,
    ) -> SideLoadResult:
        """Compute complete side-loading analysis."""

        crank_angles = motion_law_data["theta"]

        # Compute side-loading profile
        side_load_profile = self.compute_side_load_profile(
            motion_law_data, crank_center_offset,
        )

        # Compute basic metrics
        max_side_load = np.max(np.abs(side_load_profile))
        avg_side_load = np.mean(np.abs(side_load_profile))

        # Compute side-loading ripple
        side_load_std = np.std(side_load_profile)
        side_load_ripple = side_load_std / avg_side_load if avg_side_load != 0 else 0

        # Phase-specific analysis
        compression_side_load = np.array([])
        combustion_side_load = np.array([])
        compression_penalty = 0.0
        combustion_penalty = 0.0

        if compression_phases is not None:
            compression_side_load = side_load_profile[compression_phases]
            if len(compression_side_load) > 0:
                compression_penalty = np.sqrt(np.mean(compression_side_load**2))

        if combustion_phases is not None:
            combustion_side_load = side_load_profile[combustion_phases]
            if len(combustion_side_load) > 0:
                combustion_penalty = np.sqrt(np.mean(combustion_side_load**2))

        # Compute total penalty
        total_penalty = self.compute_side_load_penalty(
            side_load_profile,
            compression_phases,
            combustion_phases,
        )

        return SideLoadResult(
            side_load_profile=side_load_profile,
            max_side_load=max_side_load,
            avg_side_load=avg_side_load,
            compression_side_load=compression_side_load,
            combustion_side_load=combustion_side_load,
            compression_penalty=compression_penalty,
            combustion_penalty=combustion_penalty,
            crank_angles=crank_angles,
            crank_center_offset=crank_center_offset,
            piston_geometry=self._piston_geometry.copy(),
            side_load_ripple=side_load_ripple,
            total_penalty=total_penalty,
            metadata={
                "compression_phases_provided": compression_phases is not None,
                "combustion_phases_provided": combustion_phases is not None,
            },
        )

    def _compute_instantaneous_side_load(
        self, crank_angle: float, crank_center_offset: tuple[float, float],
    ) -> float:
        """Compute instantaneous side-loading force."""

        # Get geometry parameters
        rod_length = self._piston_geometry["rod_length"]
        crank_radius = self._piston_geometry["crank_radius"]
        bore_diameter = self._piston_geometry["bore_diameter"]
        piston_clearance = self._piston_geometry["piston_clearance"]

        # Compute connecting rod angle
        # sin(φ) = (r * sin(θ)) / L
        # where r is crank radius, L is rod length, θ is crank angle
        rod_angle = np.arcsin((crank_radius * np.sin(crank_angle)) / rod_length)

        # Compute side-loading force
        # F_side = F_piston * tan(φ)
        # This is a simplified model - could be enhanced with more sophisticated dynamics

        # For now, use a nominal piston force based on crank angle
        # In practice, this would come from the actual piston force profile
        nominal_piston_force = 1000.0  # N - this should be passed as parameter

        side_load = nominal_piston_force * np.tan(rod_angle)

        # Account for crank center offset effects
        # Offset in x-direction affects side-loading magnitude
        offset_factor = 1.0 + 0.1 * abs(crank_center_offset[0]) / crank_radius
        side_load *= offset_factor

        # Account for piston clearance effects
        # Larger clearance allows more side-loading
        clearance_factor = 1.0 + piston_clearance / (bore_diameter * 0.01)
        side_load *= clearance_factor

        return side_load

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

    def _validate_load_profile(
        self, load_profile: np.ndarray, motion_law_data: dict[str, np.ndarray],
    ) -> None:
        """Validate load profile data."""

        if not isinstance(load_profile, np.ndarray):
            raise TypeError("Load profile must be numpy array")

        if len(load_profile) != len(motion_law_data["theta"]):
            raise ValueError("Load profile length must match motion law data length")

        if np.any(np.isnan(load_profile)) or np.any(np.isinf(load_profile)):
            raise ValueError("Load profile contains invalid values (NaN or Inf)")
