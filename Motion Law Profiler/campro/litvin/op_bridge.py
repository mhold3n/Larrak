from __future__ import annotations

from typing import Callable, Dict, Protocol

from .motion import RadialSlotMotion


class MotionProvider(Protocol):
    def center_offset(self, theta_r: float) -> float: ...
    def planet_angle(self, theta_r: float) -> float: ...


def from_provider(provider: MotionProvider) -> RadialSlotMotion:
    return RadialSlotMotion(
        center_offset_fn=provider.center_offset,
        planet_angle_fn=provider.planet_angle,
        d_center_offset_fn=None,
        d2_center_offset_fn=None,
    )


def from_dict(funcs: Dict[str, Callable[[float], float]]) -> RadialSlotMotion:
    return RadialSlotMotion(
        center_offset_fn=funcs["center_offset"],
        planet_angle_fn=funcs["planet_angle"],
        d_center_offset_fn=funcs.get("d_center_offset"),
        d2_center_offset_fn=funcs.get("d2_center_offset"),
    )
