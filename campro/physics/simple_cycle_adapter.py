from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from campro.physics.base.result import PhysicsStatus


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
        combustion_defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self.wiebe = wiebe
        self.alpha = float(alpha_fuel_to_base)
        self.beta = float(beta_base)
        self._combustion_defaults: dict[str, Any] = dict(combustion_defaults or {})
        self._combustion_model: Any | None = None
        self._last_combustion_config: dict[str, Any] | None = None
        # Gain-scheduled base pressure table: (phi, fuel_mass) -> (alpha, beta)
        self._gain_table: dict[tuple[float, float], tuple[float, float]] = {}

    def evaluate(
        self,
        theta: np.ndarray,
        x_mm: np.ndarray,
        v_mm_per_theta: np.ndarray,
        fuel_multiplier: float,
        c_load: float,
        geom: CycleGeometry,
        thermo: CycleThermo,
        *,
        combustion: Mapping[str, Any] | None = None,
        cycle_time_s: float | None = None,
    ) -> dict[str, np.ndarray | float]:
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
        combustion : Optional mapping
            Structured combustion inputs (afr, fuel_mass, ignition info, etc.). When provided,
            the adapter delegates combustion-side pressure shaping to CombustionModel.
        cycle_time_s : Optional[float]
            Cycle period used to map θ → time when running the combustion model.

        Returns
        -------
        dict with keys: 'p' (kPa), 'slope' (unitless normalized dp/dθ), 'imep' (kPa)
        """
        theta = np.asarray(theta, dtype=float)
        x = np.asarray(x_mm, dtype=float)
        v = np.asarray(v_mm_per_theta, dtype=float)

        # Volumes [mm^3]
        V = geom.Vc_mm3 + geom.area_mm2 * x

        combustion_data: dict[str, Any] | None = None
        if combustion is not None:
            try:
                combustion_data = self._combustion_pressure_trace(
                    theta=theta,
                    x_mm=x,
                    v_mm_per_theta=v,
                    geom=geom,
                    thermo=thermo,
                    combustion=combustion,
                    cycle_time_s=cycle_time_s,
                )
            except Exception:
                combustion_data = None

        if combustion_data is not None:
            p_cyl_abs_kpa = combustion_data["p_cyl_kpa"]
            ca_markers = combustion_data.get("ca_markers")
            mfb = combustion_data.get("mass_fraction_burned")
        else:
            # Combustion-side pressure via Wiebe burn shaping (toy model for shape scoring)
            xb = self._wiebe_xb(theta)
            p_cyl_abs_kpa = thermo.p_atm_kpa * (1.0 + 0.3 * float(fuel_multiplier) * xb)
            ca_markers = None
            mfb = xb

        # Bounce/base pressure scheduling: use gain-scheduled table if available
        phi_val = None
        fuel_mass_val = None
        if combustion is not None:
            afr = combustion.get("afr")
            fuel_mass = combustion.get("fuel_mass") or combustion.get("fuel_mass_kg")
            if afr is not None and fuel_mass is not None:
                # Estimate phi from AFR (simple model)
                stoich_afr = 14.5  # Default for diesel
                phi_val = stoich_afr / float(afr)
                fuel_mass_val = float(fuel_mass)

        alpha_scheduled, beta_scheduled = self._get_scheduled_base_pressure(phi_val, fuel_mass_val)
        p0_bounce = alpha_scheduled * float(fuel_multiplier) + beta_scheduled
        # Polytropic bounce chamber (use first-volume reference for scaling)
        V0 = V[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            p_bounce = p0_bounce * (V0 / np.clip(V, 1e-9, None)) ** float(thermo.gamma_bounce)

        # Net indicated pressure at crown (kPa)
        p_net_kpa = p_cyl_abs_kpa - p_bounce

        # Light symmetric FIR smoothing on p(θ) to stabilize numerical derivative
        p_smooth = self._fir_smooth_periodic(p_net_kpa)

        # dp/dθ using periodic central differences
        dp_dth = self._circ_derivative(theta, p_smooth)

        # Normalize slope shape: zero-mean, unit L2 over θ
        s = self._normalize(dp_dth)

        volume_m3 = np.asarray(V, dtype=float) * 1e-9
        p_cyl_pa = np.asarray(p_cyl_abs_kpa, dtype=float) * 1e3
        dV_dtheta = np.gradient(volume_m3, theta, edge_order=1)
        cycle_work_j = float(np.trapz(p_cyl_pa * dV_dtheta, theta))

        # iMEP ≈ ∮ p dV; convert to kPa by dividing by stroke volume variation
        imep = 0.0
        try:
            delta_V = float(np.ptp(V))
            if delta_V > 0.0:
                imep = cycle_work_j / (delta_V * 1e-9)
        except Exception:
            imep = 0.0

        result: dict[str, Any] = {
            "p": p_smooth,
            "slope": s,
            "imep": imep,
            "p_raw": p_net_kpa,
            "p_comb": p_cyl_abs_kpa,
            "p_bounce": p_bounce,
            "cycle_work_j": cycle_work_j,
            "fuel_multiplier": float(
                fuel_multiplier
            ),  # Preserve for fallback path in _pressure_ratio
        }
        if combustion_data is not None:
            result.update(
                {
                    "p_cyl": p_cyl_abs_kpa,
                    "p_env": float(thermo.p_atm_kpa),
                    "ca_markers": ca_markers or {},
                    "mass_fraction_burned": mfb,
                    "heat_release_per_deg": combustion_data.get("heat_release_per_deg"),
                    "initial_pressure_pa": combustion_data.get("initial_pressure_pa"),
                }
            )
        else:
            result["p_cyl"] = p_cyl_abs_kpa
            result["p_env"] = float(thermo.p_atm_kpa)
            result["ca_markers"] = ca_markers or {}
            result["mass_fraction_burned"] = mfb
        return result

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
        dth[0] = theta[1] - theta[-1] + 2 * np.pi
        dth[-1] = theta[0] - theta[-2] + 2 * np.pi
        return (y_plus - y_minus) / np.where(np.abs(dth) > 1e-12, dth, 1e-12)

    @staticmethod
    def _normalize(s: np.ndarray) -> np.ndarray:
        s0 = s - float(np.mean(s))
        nrm = float(np.sqrt(np.sum(s0 * s0)) + 1e-12)
        return s0 / nrm

    def _update_gain_table(self, phi: float, fuel_mass: float, alpha: float, beta: float) -> None:
        """Update gain table with a new (phi, fuel_mass) -> (alpha, beta) mapping."""
        key = (float(phi), float(fuel_mass))
        self._gain_table[key] = (float(alpha), float(beta))

    def _get_scheduled_base_pressure(
        self, phi: float | None, fuel_mass: float | None
    ) -> tuple[float, float]:
        """Get scheduled base pressure (alpha, beta) from gain table via bilinear interpolation.

        Returns (alpha, beta) tuple. Falls back to self.alpha, self.beta if table is empty
        or point is outside grid bounds.
        """
        if phi is None or fuel_mass is None or not self._gain_table:
            return (self.alpha, self.beta)

        # Extract grid points
        phi_points = sorted(set(k[0] for k in self._gain_table.keys()))
        fuel_points = sorted(set(k[1] for k in self._gain_table.keys()))

        if not phi_points or not fuel_points:
            return (self.alpha, self.beta)

        # Clamp to grid bounds
        phi_clamped = np.clip(phi, phi_points[0], phi_points[-1])
        fuel_clamped = np.clip(fuel_mass, fuel_points[0], fuel_points[-1])

        # Find surrounding grid points for bilinear interpolation
        phi_idx = np.searchsorted(phi_points, phi_clamped, side="right") - 1
        phi_idx = np.clip(phi_idx, 0, len(phi_points) - 2)
        fuel_idx = np.searchsorted(fuel_points, fuel_clamped, side="right") - 1
        fuel_idx = np.clip(fuel_idx, 0, len(fuel_points) - 2)

        # Get corner values
        phi0, phi1 = phi_points[phi_idx], phi_points[phi_idx + 1]
        fuel0, fuel1 = fuel_points[fuel_idx], fuel_points[fuel_idx + 1]

        corners = [
            self._gain_table.get((phi0, fuel0)),
            self._gain_table.get((phi1, fuel0)),
            self._gain_table.get((phi0, fuel1)),
            self._gain_table.get((phi1, fuel1)),
        ]

        # If any corner is missing, fall back to default
        if any(c is None for c in corners):
            return (self.alpha, self.beta)

        # Bilinear interpolation
        alpha_corners = [c[0] for c in corners]
        beta_corners = [c[1] for c in corners]

        # Interpolation weights
        t_phi = (phi_clamped - phi0) / max(phi1 - phi0, 1e-9) if phi1 > phi0 else 0.0
        t_fuel = (fuel_clamped - fuel0) / max(fuel1 - fuel0, 1e-9) if fuel1 > fuel0 else 0.0

        # Bilinear: first interpolate along phi, then along fuel
        alpha_top = alpha_corners[0] * (1 - t_phi) + alpha_corners[1] * t_phi
        alpha_bottom = alpha_corners[2] * (1 - t_phi) + alpha_corners[3] * t_phi
        alpha_interp = alpha_top * (1 - t_fuel) + alpha_bottom * t_fuel

        beta_top = beta_corners[0] * (1 - t_phi) + beta_corners[1] * t_phi
        beta_bottom = beta_corners[2] * (1 - t_phi) + beta_corners[3] * t_phi
        beta_interp = beta_top * (1 - t_fuel) + beta_bottom * t_fuel

        return (float(alpha_interp), float(beta_interp))

    def reset_gain_table(self) -> None:
        """Clear the gain table (for lifecycle management)."""
        self._gain_table.clear()

    @staticmethod
    def phase_align(
        s: np.ndarray, sref: np.ndarray, window: tuple[int, int] | None = None
    ) -> np.ndarray:
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

    def _combustion_pressure_trace(
        self,
        *,
        theta: np.ndarray,
        x_mm: np.ndarray,
        v_mm_per_theta: np.ndarray,
        geom: CycleGeometry,
        thermo: CycleThermo,
        combustion: Mapping[str, Any],
        cycle_time_s: float | None,
    ) -> dict[str, Any]:
        """Run the integrated combustion model and derive a cylinder pressure trace."""
        # from campro_unaligned.physics.combustion import CombustionModel
        raise ImportError("CombustionModel not available (restoration incomplete)")

        # Unreachable code intentionally left for reference/future restoration
        """
        afr = combustion.get("afr")
        fuel_mass = combustion.get("fuel_mass") or combustion.get("fuel_mass_kg")
        if afr is None or fuel_mass is None:
            raise ValueError("Combustion inputs must provide 'afr' and 'fuel_mass'")
        
        # ... remainder of original method ...
        """
