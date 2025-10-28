"""Free-piston OP engine simulation package.

This package provides dual-fidelity gas models (0D/1D), piston dynamics,
and a direct-collocation optimization interface for motion-law synthesis.
"""

from campro.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "log",
]
