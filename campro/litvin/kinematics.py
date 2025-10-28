from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from campro.logging import get_logger

from .motion import RadialSlotMotion

log = get_logger(__name__)


@dataclass(frozen=True)
class PlanetKinematics:
    R0: float
    motion: RadialSlotMotion

    def center_distance(self, theta_r: float) -> float:
        return self.R0 + self.motion.center_offset_fn(theta_r)

    def planet_angle(self, theta_r: float) -> float:
        return self.motion.planet_angle_fn(theta_r)

    def transform_ring_to_planet(
        self, theta_r: float,
    ) -> tuple[tuple[tuple[float, float], tuple[float, float]], tuple[float, float]]:
        d = self.center_distance(theta_r)
        theta_p = self.planet_angle(theta_r)
        cos_p = cos(theta_p)
        sin_p = sin(theta_p)
        # 2D transform: rotate then translate (planet frame origin at planet center)
        rot = ((cos_p, -sin_p), (sin_p, cos_p))
        trans = (d, 0.0)
        return rot, trans
