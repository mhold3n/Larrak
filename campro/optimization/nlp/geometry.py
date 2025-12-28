"""Geometry lookup tables for 2D precomputed data.

This module defines the interface and standard implementations for accessing
precomputed 2D geometry data (volume, area, etc.) as scalar functions of theta.

Note: This module intentionally uses physics/engineering notation (Volume, A_head, etc.)
instead of snake_case. See NAMING_CONVENTIONS.md for rationale.
"""
# pylint: disable=invalid-name,missing-function-docstring
# pyright: reportInvalidName=false

from __future__ import annotations

from typing import Protocol

import casadi as ca
import numpy as np


class GeometryInterface(Protocol):
    """Interface for geometry lookup.

    Required attributes for engine geometry:
        B: Bore diameter [m]
        S: Stroke length [m]
        bore: Alias for B (bore diameter) [m]
        stroke: Alias for S (stroke length) [m]
    """

    # Geometry dimensions - required for physics calculations
    B: float  # Bore diameter [m]
    S: float  # Stroke length [m]

    @property
    def bore(self) -> float:
        """Bore diameter [m] (alias for B)."""
        ...  # type: ignore[empty-body]

    @property
    def stroke(self) -> float:
        """Stroke length [m] (alias for S)."""
        ...  # type: ignore[empty-body]

    def Volume(self, theta: ca.SX) -> ca.SX:
        """Cylinder volume [m^3] at angle theta."""
        ...  # type: ignore[empty-body]

    def dV_dtheta(self, theta: ca.SX) -> ca.SX:
        """Derivative of cylinder volume wrt theta [m^3/rad]."""
        ...  # type: ignore[empty-body]

    def Area_wall(self, theta: ca.SX) -> ca.SX:
        """Wetted wall area [m^2] at angle theta."""
        ...  # type: ignore[empty-body]

    def Area_intake(
        self,
        theta: ca.SX,
        open_rad: ca.SX | float | None = None,
        duration_rad: ca.SX | float | None = None,
    ) -> ca.SX:
        """Effective intake valve area [m^2] at angle theta."""
        ...  # type: ignore[empty-body]

    def Area_exhaust(
        self,
        theta: ca.SX,
        open_rad: ca.SX | float | None = None,
        duration_rad: ca.SX | float | None = None,
    ) -> ca.SX:
        """Effective exhaust valve area [m^2] at angle theta."""
        ...  # type: ignore[empty-body]


class StandardSliderCrankGeometry:
    """Standard slider-crank geometry implementation (analytical)."""

    def __init__(
        self,
        bore: float,
        stroke: float,
        conrod: float,
        compression_ratio: float,
        intake_open: float = -30.0,  # Degrees BBDC
        intake_duration: float = 60.0,
        exhaust_open: float = -50.0,  # Degrees BBDC
        exhaust_duration: float = 100.0,
    ):
        self.B = bore
        self.S = stroke
        self.L = conrod
        self.CR = compression_ratio
        self.a = stroke / 2.0  # Crank radius
        self.R = conrod / self.a  # Ratio of rod to crank radius

        # Clearance volume
        self.V_disp = (np.pi * self.B**2 / 4) * self.S
        self.V_c = self.V_disp / (self.CR - 1.0)

        # Valve timing (simple cosine bumps for now)
        self.intake_open_rad = np.radians(180.0 - intake_open)
        self.intake_close_rad = np.radians(180.0 - intake_open + intake_duration)
        self.exhaust_open_rad = np.radians(180.0 - exhaust_open)
        self.exhaust_close_rad = np.radians(180.0 - exhaust_open + exhaust_duration)

    @property
    def bore(self) -> float:
        """Bore diameter [m] (alias for B)."""
        return self.B

    @property
    def stroke(self) -> float:
        """Stroke length [m] (alias for S)."""
        return self.S

    def _piston_position(self, theta: ca.SX | ca.MX) -> ca.SX | ca.MX:
        """Distance from TDC [m]."""
        # x = a * (1 - cos(theta) + (R - sqrt(R^2 - sin(theta)^2)))
        # Simplified: x approx a * (1 - cos(theta) + lambda/2 * sin(theta)^2)
        # Using exact formula:
        term1 = 1 - ca.cos(theta)
        term2 = self.R - ca.sqrt(self.R**2 - ca.sin(theta) ** 2)
        return self.a * (term1 + term2)

    def Volume(self, theta: ca.SX) -> ca.SX | ca.MX:
        x = self._piston_position(theta)
        return self.V_c + (np.pi * self.B**2 / 4) * x

    def dV_dtheta(self, theta: ca.SX) -> ca.SX | ca.MX:
        # Analytical derivative of Volume
        # dV/dtheta = A_piston * dx/dtheta
        # dx/dtheta = a * (sin(theta) + sin(theta)cos(theta)/sqrt(R^2 - sin^2(theta)))
        dx_dth = (
            self.a * ca.sin(theta) * (1 + ca.cos(theta) / ca.sqrt(self.R**2 - ca.sin(theta) ** 2))
        )
        return (np.pi * self.B**2 / 4) * dx_dth

    def Area_wall(self, theta: ca.SX) -> ca.SX | ca.MX:
        # A_wall = A_head + A_piston + A_liner
        A_head = np.pi * self.B**2 / 4
        A_piston = np.pi * self.B**2 / 4
        x = self._piston_position(theta)
        A_liner = np.pi * self.B * x
        return A_head + A_piston + A_liner

    def _valve_area(
        self, _theta: ca.SX, _open_rad: float, _close_rad: float, _max_area: float
    ) -> ca.SX | ca.MX:
        """Placeholder valve area calculation (currently returns zero)."""
        # Simple "lift" function: 0 if outside, sine wave inside
        # Note: Handling periodicity and wrap-around is tricky in pure symbolic
        # For Phase 1, we might treat theta as 0..720 or 0..360.
        # Assuming 0..360 for 2-stroke.

        # TODO: Implement smooth bump function when needed
        # Could use: max_area * sin(pi * (theta - open) / dur)^2
        # For now returning zero as placeholder
        return ca.SX(0.0)  # Placeholder for now, typically 2D lookup is better.

    def Area_intake(
        self,
        theta: ca.SX,
        open_rad: ca.SX | float | None = None,
        duration_rad: ca.SX | float | None = None,
    ) -> ca.SX | ca.MX:
        # Defaults to instance values if not provided
        open_rad_val: ca.SX | float = self.intake_open_rad if open_rad is None else open_rad
        duration_rad_val: ca.SX | float = (
            self.intake_close_rad - self.intake_open_rad if duration_rad is None else duration_rad
        )

        # Implementing a simple differentiable bump centered at the window
        # Center = open + duration/2
        # Width (sigma) approx duration/4
        center = open_rad_val + duration_rad_val / 2.0
        sigma = duration_rad_val / 4.0

        # Max Area parameter could also be passed, fixed for now
        # Use periodic distance for 0-2pi wrapping?
        # For 2-stroke centered at PI, simple diff is fine if range is [0, 2pi]
        return 0.005 * ca.exp(-((theta - center) ** 2) / (2 * sigma**2))

    def Area_exhaust(
        self,
        theta: ca.SX,
        open_rad: ca.SX | float | None = None,
        duration_rad: ca.SX | float | None = None,
    ) -> ca.SX | ca.MX:
        open_rad_val: ca.SX | float = self.exhaust_open_rad if open_rad is None else open_rad
        duration_rad_val: ca.SX | float = (
            self.exhaust_close_rad - self.exhaust_open_rad if duration_rad is None else duration_rad
        )

        center = open_rad_val + duration_rad_val / 2.0
        sigma = duration_rad_val / 4.0
        return 0.006 * ca.exp(-((theta - center) ** 2) / (2 * sigma**2))


class InterpolatedGeometry:
    """
    Geometry defined by lookup tables (interpolated).
    Use this for complex port shapes or non-analytical kinematics.
    """

    def __init__(
        self, theta_arr: np.ndarray, V_arr: np.ndarray, A_wall_arr: np.ndarray | None = None
    ):
        """
        Initialize with data arrays.
        theta_arr: expected in [0, 2*pi] or similar cycle domain.
        """
        self.theta_data = theta_arr
        self.V_data = V_arr

        # Create CasADi interpolants
        # 'bspline' gives smooth derivatives
        # Create volume interpolant (bspline for smooth derivatives)
        self._V_interp = ca.interpolant("V_interp", "bspline", [theta_arr], V_arr)  # type: ignore[attr-defined]

        if A_wall_arr is not None:
            self._A_wall_interp = ca.interpolant("A_wall_interp", "linear", [theta_arr], A_wall_arr)  # type: ignore[attr-defined]
        else:
            # Placeholder sphere approx
            self._A_wall_interp = None

    def Volume(self, theta: ca.SX) -> ca.SX | ca.MX:
        return self._V_interp(theta)  # type: ignore[no-any-return]

    def dV_dtheta(self, theta: ca.SX) -> ca.SX | ca.MX:
        # CasADi interpolants are differentiable
        return ca.jacobian(self._V_interp(theta), theta)  # type: ignore[no-any-return]

    def Area_wall(self, theta: ca.SX) -> ca.SX | ca.MX:
        if self._A_wall_interp:
            return self._A_wall_interp(theta)  # type: ignore[no-any-return]

        # Fallback: Spherical approx V = 4/3 pi r^3 => r = (3V/4pi)^(1/3) => A = 4 pi r^2
        V = self.Volume(theta)
        r = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
        return 4.0 * np.pi * r**2

    def Area_intake(self, theta: ca.SX) -> ca.SX | ca.MX:
        # Placeholder
        return ca.fmax(0.0, ca.sin(theta)) * 1e-4  # type: ignore[no-any-return]

    def Area_exhaust(self, theta: ca.SX) -> ca.SX | ca.MX:
        # Placeholder
        return ca.fmax(0.0, ca.sin(theta - np.pi)) * 1e-4  # type: ignore[no-any-return]
