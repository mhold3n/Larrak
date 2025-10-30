from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class CycleGeometry:
    area_mm2: float  # piston crown area [mm^2]
    Vc_mm3: float  # clearance volume [mm^3]


@dataclass
class CycleThermo:
    gamma_bounce: float  # polytropic exponent for bounce chamber
    p_atm_kpa: float  # ambient/reference pressure [kPa]


@dataclass
class WiebeParams:
    a: float
    m: float
    start_deg: float
    duration_deg: float


class SimpleCycleAdapter:
    """Lightweight 0-D cycle adapter for Phase-1 scoring.

    This adapter computes p(θ), normalized dp/dθ shape, and iMEP for a given
    motion law (x(θ), v(θ)), with fuel→base-pressure scheduling and an
    optional electrical load damping parameter carried through for diagnostics.
    All computations are vectorized NumPy and differentiable for smooth
    objective evaluation under SciPy-based optimizers.
    """

    def __init__(
        self,
        wiebe: WiebeParams,
        alpha_fuel_to_base: float,
        beta_base: float,
    ) -> None:
        self.wiebe = wiebe
        self.alpha = float(alpha_fuel_to_base)
        self.beta = float(beta_base)

    def evaluate(
        self,
        theta: np.ndarray,
        x_mm: np.ndarray,
        v_mm_per_theta: np.ndarray,
        fuel_multiplier: float,
        c_load: float,
        geom: CycleGeometry,
        thermo: CycleThermo,
    ) -> Dict[str, np.ndarray | float]:
        """Compute pressure trace, normalized slope shape, and iMEP.

        Parameters
        ----------
        theta : np.ndarray
            Phase angle array [rad], length N, monotonically increasing, periodic 0..2π
        x_mm : np.ndarray
            Position vs θ [mm]
        v_mm_per_theta : np.ndarray
            d(x)/dθ [mm/rad]
        fuel_multiplier : float
            Relative fueling multiplier ∈ R+
        c_load : float
            Electrical damping (N·s/m equivalent); carried through for diagnostics
        geom : CycleGeometry
            Geometry parameters
        thermo : CycleThermo
            Thermo parameters

        Returns
        -------
        dict with keys: 'p' (kPa), 'slope' (unitless normalized dp/dθ), 'imep' (kPa)
        """
        theta = np.asarray(theta, dtype=float)
        x = np.asarray(x_mm, dtype=float)
        v = np.asarray(v_mm_per_theta, dtype=float)

        # Volumes [mm^3]
        V = geom.Vc_mm3 + geom.area_mm2 * x

        # Combustion-side pressure via Wiebe burn shaping (toy model for shape scoring)
        xb = self._wiebe_xb(theta)
        p_comb = thermo.p_atm_kpa * (1.0 + 0.3 * float(fuel_multiplier) * xb)

        # Bounce/base pressure scheduling (Toyota-style linear mapping)
        p0_bounce = self.alpha * float(fuel_multiplier) + self.beta
        # Polytropic bounce chamber (use first-volume reference for scaling)
        V0 = V[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            p_bounce = p0_bounce * (V0 / np.clip(V, 1e-9, None)) ** float(thermo.gamma_bounce)

        # Net indicated pressure at crown (kPa)
        p = p_comb - p_bounce

        # Light symmetric FIR smoothing on p(θ) to stabilize numerical derivative
        p_smooth = self._fir_smooth_periodic(p)

        # dp/dθ using periodic central differences
        dp_dth = self._circ_derivative(theta, p_smooth)

        # Normalize slope shape: zero-mean, unit L2 over θ
        s = self._normalize(dp_dth)

        # iMEP ≈ ∮ p dV, using trapezoidal over θ; dV = A dx = A * x'(θ) dθ
        dtheta = np.diff(theta, prepend=theta[0])
        dx = v * dtheta
        dV = geom.area_mm2 * dx
        # Trapezoidal integration over periodic domain
        imep = float(np.trapz(p * np.gradient(V, theta), theta)) / max(1.0, np.ptp(V))

        return {"p": p_smooth, "slope": s, "imep": imep, "p_raw": p, "p_comb": p_comb, "p_bounce": p_bounce}

    def _wiebe_xb(self, theta: np.ndarray) -> np.ndarray:
        th_deg = np.rad2deg(theta)
        s = self.wiebe
        z = np.clip((th_deg - float(s.start_deg)) / max(float(s.duration_deg), 1e-9), 0.0, 1.0)
        return 1.0 - np.exp(-float(s.a) * z ** (float(s.m) + 1.0))

    @staticmethod
    def _fir_smooth_periodic(y: np.ndarray) -> np.ndarray:
        # 5-tap binomial kernel [1,4,6,4,1]/16 applied circularly
        k = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float) / 16.0
        n = len(y)
        yp = np.concatenate([y[-2:], y, y[:2]])
        conv = np.convolve(yp, k, mode="same")
        return conv[2 : 2 + n]

    @staticmethod
    def _circ_derivative(theta: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Periodic central difference on nonuniform grid
        n = len(y)
        y_plus = np.roll(y, -1)
        y_minus = np.roll(y, 1)
        th_plus = np.roll(theta, -1)
        th_minus = np.roll(theta, 1)
        dth = th_plus - th_minus
        # Fix wrap at endpoints
        dth[0] = (theta[1] - theta[-1] + 2 * np.pi)
        dth[-1] = (theta[0] - theta[-2] + 2 * np.pi)
        return (y_plus - y_minus) / np.where(np.abs(dth) > 1e-12, dth, 1e-12)

    @staticmethod
    def _normalize(s: np.ndarray) -> np.ndarray:
        s0 = s - float(np.mean(s))
        nrm = float(np.sqrt(np.sum(s0 * s0)) + 1e-12)
        return s0 / nrm

    @staticmethod
    def phase_align(s: np.ndarray, sref: np.ndarray, window: Tuple[int, int] | None = None) -> np.ndarray:
        """Align s to sref by circular shift maximizing correlation.

        If window is provided as (center, half_width), only consider shifts within
        that window around center to reduce cost.
        """
        n = len(s)
        if window is None:
            shifts = range(n)
        else:
            c, hw = window
            shifts = [(c + k) % n for k in range(-hw, hw + 1)]
        best_shift = 0
        best_val = -1e300
        for k in shifts:
            sh = np.roll(s, -k)
            val = float(np.dot(sh, sref))
            if val > best_val:
                best_val = val
                best_shift = k
        return np.roll(s, -best_shift)


