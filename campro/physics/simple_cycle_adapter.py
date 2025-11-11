from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

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
        self._combustion_defaults: Dict[str, Any] = dict(combustion_defaults or {})
        self._combustion_model: Any | None = None
        self._last_combustion_config: Dict[str, Any] | None = None
        # Gain-scheduled base pressure table: (phi, fuel_mass) -> (alpha, beta)
        self._gain_table: Dict[Tuple[float, float], Tuple[float, float]] = {}

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

        combustion_data: Dict[str, Any] | None = None
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

        result: Dict[str, Any] = {
            "p": p_smooth,
            "slope": s,
            "imep": imep,
            "p_raw": p_net_kpa,
            "p_comb": p_cyl_abs_kpa,
            "p_bounce": p_bounce,
            "cycle_work_j": cycle_work_j,
            "fuel_multiplier": float(fuel_multiplier),  # Preserve for fallback path in _pressure_ratio
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
        dth[0] = (theta[1] - theta[-1] + 2 * np.pi)
        dth[-1] = (theta[0] - theta[-2] + 2 * np.pi)
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
    
    def _get_scheduled_base_pressure(self, phi: float | None, fuel_mass: float | None) -> Tuple[float, float]:
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
        phi_idx = np.searchsorted(phi_points, phi_clamped, side='right') - 1
        phi_idx = np.clip(phi_idx, 0, len(phi_points) - 2)
        fuel_idx = np.searchsorted(fuel_points, fuel_clamped, side='right') - 1
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
    ) -> Dict[str, Any]:
        """Run the integrated combustion model and derive a cylinder pressure trace."""
        from campro.physics.combustion import CombustionModel

        afr = combustion.get("afr")
        fuel_mass = combustion.get("fuel_mass") or combustion.get("fuel_mass_kg")
        if afr is None or fuel_mass is None:
            raise ValueError("Combustion inputs must provide 'afr' and 'fuel_mass'")

        if cycle_time_s is None:
            cycle_time_s = combustion.get("cycle_time_s") or combustion.get("cycle_time")
        if cycle_time_s is None:
            raise ValueError("cycle_time_s is required for combustion-aware evaluation")
        cycle_time_s = float(cycle_time_s)

        theta = np.asarray(theta, dtype=float)
        theta_deg = np.degrees(theta)
        theta_deg = np.mod(theta_deg, 360.0)

        # Map θ to time assuming constant angular velocity over the cycle
        time_s = (theta / (2.0 * np.pi)) * cycle_time_s
        time_s -= time_s[0]

        # Piston velocity in m/s: compute gradient first as default
        # Gradient: dx/dt where x is in m and time_s is in s
        piston_speed_m_per_s = np.gradient(np.asarray(x_mm, dtype=float) * 1e-3, time_s, edge_order=1)
        
        # Override with explicit value if provided
        if "piston_speed_override" in combustion:
            override_val = np.asarray(combustion.get("piston_speed_override"), dtype=float)
            if override_val.shape == time_s.shape and override_val.size > 0:
                piston_speed_m_per_s = override_val

        omega_deg_per_s = np.full_like(theta_deg, 360.0 / max(cycle_time_s, 1e-9))

        area_m2 = float(geom.area_mm2) * 1e-6
        bore_m = float(np.sqrt(max(area_m2, 1e-12) * 4.0 / np.pi))
        stroke_m = float(np.ptp(x_mm) * 1e-3) or (float(np.max(x_mm) - np.min(x_mm)) * 1e-3)
        clearance_volume_m3 = float(geom.Vc_mm3) * 1e-9

        config = {
            "fuel_type": combustion.get("fuel_type", self._combustion_defaults.get("fuel_type", "diesel")),
            "afr": float(afr),
            "bore_m": max(bore_m, 1e-4),
            "stroke_m": max(stroke_m, 1e-4),
            "clearance_volume_m3": max(clearance_volume_m3, 1e-9),
            "fuel_mass_kg": float(fuel_mass),
            "cycle_time_s": cycle_time_s,
            "initial_temperature_K": float(
                combustion.get(
                    "initial_temperature_K",
                    self._combustion_defaults.get("initial_temperature_K", 900.0),
                )
            ),
            "initial_pressure_Pa": float(
                combustion.get(
                    "initial_pressure_Pa",
                    self._combustion_defaults.get("initial_pressure_Pa", thermo.p_atm_kpa * 1e3),
                )
            ),
            "target_mfb": float(combustion.get("target_mfb", self._combustion_defaults.get("target_mfb", 0.99))),
            "m_wiebe": float(combustion.get("m_wiebe", self._combustion_defaults.get("m_wiebe", 2.0))),
            "k_turb": float(combustion.get("k_turb", self._combustion_defaults.get("k_turb", 0.3))),
            "c_burn": float(combustion.get("c_burn", self._combustion_defaults.get("c_burn", 3.0))),
            "turbulence_exponent": float(
                combustion.get(
                    "turbulence_exponent",
                    self._combustion_defaults.get("turbulence_exponent", 0.7),
                )
            ),
            "min_flame_speed": float(
                combustion.get("min_flame_speed", self._combustion_defaults.get("min_flame_speed", 0.2))
            ),
            "heating_value_override": combustion.get(
                "heating_value_override",
                self._combustion_defaults.get("heating_value_override"),
            ),
            "phi_override": combustion.get("phi_override", self._combustion_defaults.get("phi_override")),
        }

        if self._combustion_model is None:
            self._combustion_model = CombustionModel()
            self._last_combustion_config = None

        # Reconfigure model when parameters change
        if self._last_combustion_config != config:
            self._combustion_model.configure(**config)
            self._last_combustion_config = dict(config)

        ignition_time_s = combustion.get("ignition_time_s")
        ignition_theta_deg = combustion.get("ignition_theta_deg")
        if ignition_time_s is None and ignition_theta_deg is None:
            ignition_theta_deg = combustion.get("ignition_deg", -5.0)

        initial_pressure_pa = float(config.get("initial_pressure_Pa", thermo.p_atm_kpa * 1e3))
        gamma_comb = float(
            combustion.get(
                "gamma_combustion",
                self._combustion_defaults.get("gamma_combustion", 1.32),
            )
        )
        gamma_comb = max(gamma_comb, 1.001)

        sim_inputs: Dict[str, Any] = {
            "time_s": time_s,
            "theta_deg": theta_deg,
            "piston_speed_m_per_s": piston_speed_m_per_s,
            "omega_deg_per_s": omega_deg_per_s,
        }
        if ignition_time_s is not None:
            sim_inputs["ignition_time_s"] = float(ignition_time_s)
        if ignition_theta_deg is not None:
            sim_inputs["ignition_theta_deg"] = float(ignition_theta_deg)
        # Pass injector delay parameters if provided
        injector_delay_s = combustion.get("injector_delay_s")
        injector_delay_deg = combustion.get("injector_delay_deg")
        if injector_delay_s is not None:
            sim_inputs["injector_delay_s"] = float(injector_delay_s)
        if injector_delay_deg is not None:
            sim_inputs["injector_delay_deg"] = float(injector_delay_deg)

        result = self._combustion_model.simulate(sim_inputs)
        if result.status != PhysicsStatus.COMPLETED:
            raise RuntimeError(f"Combustion model failed: {result.error_message}")
        data = result.data

        heat_per_deg = data.get("heat_release_per_deg")
        if heat_per_deg is None:
            heat_rate = data.get("heat_release_rate_W")
            if heat_rate is None:
                raise RuntimeError("Combustion model output missing heat release data")
            heat_per_deg = np.divide(
                heat_rate,
                np.maximum(omega_deg_per_s, 1e-9),
                out=np.zeros_like(heat_rate),
            )

        volume_mm3 = geom.Vc_mm3 + geom.area_mm2 * np.asarray(x_mm, dtype=float)
        volume_m3 = np.maximum(volume_mm3 * 1e-9, 1e-12)
        dV_dtheta = np.gradient(volume_m3, theta, edge_order=1)
        heat_per_rad = np.asarray(heat_per_deg, dtype=float) * (180.0 / np.pi)

        theta_shift = np.roll(theta, -1)
        dtheta = theta_shift - theta
        dtheta[-1] = (theta[0] + 2.0 * np.pi) - theta[-1]

        p_cyl_pa = np.empty_like(theta, dtype=float)
        p_cyl_pa[0] = initial_pressure_pa
        # First law: dp/dθ = (γ-1)/V * (dQ/dθ - p*dV/dθ)
        # Units: dQ_dtheta [J/rad], dV_dtheta [m³/rad], p [Pa], V [m³]
        # (dQ_dtheta - p*dV_dtheta) [J/rad], divided by V [m³] gives [Pa/rad]
        gamma_comb_adjusted = max(gamma_comb - 1.0, 0.001)  # Safety: ensure (γ-1) >= 0.001
        for i in range(len(theta) - 1):
            dQ_dtheta = heat_per_rad[i]  # [J/rad]
            dVdtheta = dV_dtheta[i]  # [m³/rad]
            p_curr = max(p_cyl_pa[i], 1.0)  # [Pa]
            # Discrete form: dp/dθ = (γ-1)/V * (dQ/dθ - p*dV/dθ)
            dp_dtheta = gamma_comb_adjusted / max(volume_m3[i], 1e-12) * (dQ_dtheta - p_curr * dVdtheta)
            p_next = p_curr + dp_dtheta * dtheta[i]
            if not np.isfinite(p_next):
                p_next = p_curr
            p_cyl_pa[i + 1] = max(p_next, 1.0)

        # Close the loop by relaxing toward initial pressure for periodic consistency
        p_cyl_pa[-1] = 0.5 * (p_cyl_pa[-1] + p_cyl_pa[0])
        p_cyl_kpa = p_cyl_pa / 1e3

        ca_markers = {
            "CA10": data.get("CA10_deg"),
            "CA50": data.get("CA50_deg"),
            "CA90": data.get("CA90_deg"),
            "CA100": data.get("CA100_deg"),
        }

        return {
            "p_cyl_kpa": p_cyl_kpa,
            "ca_markers": ca_markers,
            "mass_fraction_burned": data.get("mass_fraction_burned"),
            "heat_release_per_deg": heat_per_deg,
            "initial_pressure_pa": initial_pressure_pa,
            "gamma_combustion": gamma_comb,
        }
