"""
Configuration parameters for Cam-Ring system.
"""

from dataclasses import dataclass


@dataclass
class CamRingParameters:
    """Parameters for cam-ring system design."""

    # Cam parameters
    base_radius: float = 10.0  # r_b: base radius of cam

    # Connecting rod parameters
    connecting_rod_length: float = 25.0  # Length of connecting rod

    # Ring parameters
    ring_center_x: float = 0.0
    ring_center_y: float = 0.0

    # Contact parameters
    contact_type: str = "external"  # "external" or "internal"

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "base_radius": self.base_radius,
            "connecting_rod_length": self.connecting_rod_length,
            "ring_center_x": self.ring_center_x,
            "ring_center_y": self.ring_center_y,
            "contact_type": self.contact_type,
        }
