from __future__ import annotations

from dataclasses import dataclass

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class MechState:
    """Left/right piston kinematics.

    Parameters
    ----------
    x_L : float
        Left piston position [m].
    v_L : float
        Left piston velocity [m/s].
    x_R : float
        Right piston position [m].
    v_R : float
        Right piston velocity [m/s].
    """

    x_L: float
    v_L: float
    x_R: float
    v_R: float

    def pack(self) -> tuple[float, float, float, float]:
        return (self.x_L, self.v_L, self.x_R, self.v_R)


@dataclass
class GasState:
    """Conservative variables for a 0D/1D gas model (placeholder).

    Stores a flattened array of state variables; detailed layout is defined by
    the selected fidelity backend. Use typed façades at higher layers.
    """

    data: tuple[float, ...]

    def dof(self) -> int:
        return len(self.data)
