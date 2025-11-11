"""
Integrated combustion model for free-piston optimization.

This module provides a combustion model that converts ignition timing,
cylinder geometry, AFR, and piston speed information into a complete burn
profile suitable for coupling with the collocation-based motion optimizer.
It produces mass-fraction-burned evolution, heat-release rates, and CA10/50/90/100
markers while supporting a CasADi-compatible symbolic interface for IPOPT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from campro.freepiston.core.chem import (
    CombustionParameters,
    create_combustion_parameters,
    laminar_flame_speed,
)
from campro.logging import get_logger

from ..base.model import BasePhysicsModel
from ..base.result import PhysicsResult

log = get_logger(__name__)

CASADI_EPS = 1e-9

STOICH_AFR: Dict[str, float] = {
    "gasoline": 14.7,
    "diesel": 14.5,
    "natural_gas": 17.2,
    "hydrogen": 34.3,
}


@dataclass
class CombustionConfig:
    """Configuration container for the combustion model."""

    fuel_type: str
    afr: float
    bore_m: float
    stroke_m: float
    clearance_volume_m3: float
    fuel_mass_kg: float
    cycle_time_s: float
    initial_temperature_K: float
    initial_pressure_Pa: float = 1e5
    target_mfb: float = 0.99
    m_wiebe: float = 2.0
    k_turb: float = 0.3
    c_burn: float = 3.0
    turbulence_exponent: float = 0.7
    min_flame_speed: float = 0.1
    heating_value_override: float | None = None
    phi_override: float | None = None
    injector_delay_s: float = 0.0  # Injector delay [s] - shifts effective combustion start
    injector_delay_deg: float | None = None  # Injector delay [deg] - alternative to injector_delay_s


@dataclass
class CombustionOutputs:
    """Structured outputs from a combustion simulation."""

    time_s: np.ndarray
    theta_deg: np.ndarray | None
    mass_fraction_burned: np.ndarray
    heat_release_rate_W: np.ndarray
    heat_release_per_deg: np.ndarray | None
    burn_duration_s: float
    burn_duration_deg: float | None
    ignition_time_s: float
    ignition_theta_deg: float | None
    ca10_deg: float | None
    ca50_deg: float | None
    ca90_deg: float | None
    ca100_deg: float | None
    duration_profile: np.ndarray
    params: Dict[str, float] = field(default_factory=dict)


class CombustionModel(BasePhysicsModel):
    """Integrated combustion model with numeric and symbolic interfaces."""

    def __init__(self) -> None:
        super().__init__(name="CombustionModel")
        self.config: CombustionConfig | None = None
        self.params: CombustionParameters | None = None
        self._phi: float | None = None
        self._a_wiebe: float | None = None
        self._piston_area_m2: float | None = None
        self._clearance_height_m: float | None = None
        self._laminar_speed_m_per_s: float | None = None
        self._lhv: float | None = None
        self._stoich_afr: float | None = None

    # ------------------------------------------------------------------ #
    # Configuration and validation
    # ------------------------------------------------------------------ #
    def configure(self, **kwargs: Any) -> None:
        """Configure the combustion model using keyword arguments."""

        cfg = CombustionConfig(**kwargs)
        fuel_key = cfg.fuel_type.lower()
        if fuel_key not in STOICH_AFR:
            raise ValueError(f"Unsupported fuel type '{cfg.fuel_type}'")

        if cfg.afr <= 0:
            raise ValueError("AFR must be positive")

        if cfg.cycle_time_s <= 0:
            raise ValueError("cycle_time_s must be positive")

        if cfg.fuel_mass_kg <= 0:
            raise ValueError("fuel_mass_kg must be positive")

        params = create_combustion_parameters(fuel_key)
        self.params = params
        self.config = cfg
        self._stoich_afr = STOICH_AFR[fuel_key]

        # Geometric properties
        radius = cfg.bore_m / 2.0
        self._piston_area_m2 = math.pi * radius * radius
        self._clearance_height_m = cfg.clearance_volume_m3 / max(
            self._piston_area_m2, 1e-12,
        )

        # Combustion properties
        phi = cfg.phi_override
        if phi is None:
            phi = self._stoich_afr / cfg.afr
        if phi <= 0:
            raise ValueError("Equivalence ratio (phi) must be positive")
        self._phi = phi

        params.m_wiebe = cfg.m_wiebe
        # Compute a_Wiebe to reach the requested target burn fraction
        if cfg.target_mfb >= 1.0:
            cfg.target_mfb = 0.999999
        self._a_wiebe = -math.log(max(1.0 - cfg.target_mfb, 1e-6))

        # Nominal laminar flame speed at initial state
        self._laminar_speed_m_per_s = float(
            laminar_flame_speed(
                phi=phi,
                T=cfg.initial_temperature_K,
                p=cfg.initial_pressure_Pa,
                params=params,
            ),
        )

        # Lower heating value
        if cfg.heating_value_override is not None:
            self._lhv = cfg.heating_value_override
        else:
            self._lhv = params.LHV_fuel

        self._is_configured = True
        log.info(
            "CombustionModel configured: fuel=%s AFR=%.3f phi=%.3f m=%.2f target=%.3f",
            cfg.fuel_type,
            cfg.afr,
            self._phi,
            cfg.m_wiebe,
            cfg.target_mfb,
        )

    # ------------------------------------------------------------------ #
    # Public simulation entrypoint
    # ------------------------------------------------------------------ #
    def simulate(self, inputs: dict[str, Any], **_: Any) -> PhysicsResult:
        """Run the combustion model for a given set of inputs."""
        self._validate_inputs(inputs)
        result = self._start_simulation()

        try:
            outputs = self._compute_numeric(inputs)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.error("Combustion simulation failed: %s", exc)
            return self._finish_simulation(result, data={}, error_message=str(exc))

        data = {
            "time_s": outputs.time_s,
            "theta_deg": outputs.theta_deg,
            "mass_fraction_burned": outputs.mass_fraction_burned,
            "heat_release_rate_W": outputs.heat_release_rate_W,
            "heat_release_per_deg": outputs.heat_release_per_deg,
            "burn_duration_s": outputs.burn_duration_s,
            "burn_duration_deg": outputs.burn_duration_deg,
            "ignition_time_s": outputs.ignition_time_s,
            "ignition_theta_deg": outputs.ignition_theta_deg,
            "CA10_deg": outputs.ca10_deg,
            "CA50_deg": outputs.ca50_deg,
            "CA90_deg": outputs.ca90_deg,
            "CA100_deg": outputs.ca100_deg,
            "duration_profile": outputs.duration_profile,
            "params": outputs.params,
        }

        return self._finish_simulation(
            result,
            data=data,
            convergence_info={"status": "computed"},
        )

    # ------------------------------------------------------------------ #
    # Symbolic interface
    # ------------------------------------------------------------------ #
    def symbolic_heat_release(
        self,
        ca: Any,
        time_s: Any,
        piston_speed_m_per_s: Any,
        ignition_time_s: Any,
        omega_deg_per_s: Any | None = None,
    ) -> Dict[str, Any]:
        """
        Build CasADi expressions for the heat-release model.

        Parameters
        ----------
        ca : module
            CasADi module (typically `casadi` imported as `ca`).
        time_s : MX/SX
            Symbolic time at which to evaluate the burn state [s].
        piston_speed_m_per_s : MX/SX
            Symbolic piston speed used for turbulence correlation [m/s].
        ignition_time_s : MX/SX
            Symbolic ignition timing [s].
        omega_deg_per_s : MX/SX, optional
            Symbolic angular speed used for dQ/dθ conversion [deg/s].

        Returns
        -------
        Dict[str, Any]
            Keys: `mfb`, `burn_time`, `heat_release_rate`, `heat_release_per_deg`,
            `burn_duration_deg`.
        """
        if not self._is_configured or self.config is None or self.params is None:
            raise RuntimeError("CombustionModel must be configured before use")

        cfg = self.config
        params = self.params
        ca_fmax = ca.fmax

        S_L = float(max(self._laminar_speed_m_per_s or 1e-3, 1e-3))
        clearance_h = float(self._clearance_height_m or 1e-6)
        min_speed = float(cfg.min_flame_speed)
        alpha = float(params.alpha_turbulence)
        k_turb = float(cfg.k_turb)
        exponent = float(cfg.turbulence_exponent)
        a_wiebe = float(self._a_wiebe or 5.0)
        m_wiebe = float(cfg.m_wiebe)
        q_total = float(cfg.fuel_mass_kg * (self._lhv or params.LHV_fuel))

        u_turb = k_turb * ca.fabs(piston_speed_m_per_s)
        S_ratio = u_turb / ca_fmax(S_L, CASADI_EPS)
        S_T = S_L * (1.0 + alpha * ca.power(S_ratio, exponent))
        S_T = ca_fmax(S_T, min_speed)
        burn_time = cfg.c_burn * clearance_h / ca_fmax(S_T, CASADI_EPS)

        # Note: injector_delay handling in symbolic interface requires passing
        # effective_ignition_time_s = ignition_time_s - injector_delay_s
        # This is handled at the caller level; here we use ignition_time_s directly
        tau_raw = (time_s - ignition_time_s) / ca_fmax(burn_time, CASADI_EPS)
        tau = ca.if_else(
            tau_raw < 0.0,
            0.0,
            ca.if_else(tau_raw > 1.0, 1.0, tau_raw),
        )
        exp_term = ca.exp(-a_wiebe * ca.power(tau, m_wiebe + 1.0))
        mfb = 1.0 - exp_term

        base = (
            a_wiebe
            * (m_wiebe + 1.0)
            / ca_fmax(burn_time, CASADI_EPS)
            * ca.power(ca.fmax(tau, CASADI_EPS), m_wiebe)
            * exp_term
        )
        dxb_dt = ca.if_else(
            tau_raw < 0.0,
            0.0,
            ca.if_else(tau_raw > 1.0, 0.0, base),
        )
        heat_release_rate = dxb_dt * q_total

        if omega_deg_per_s is not None:
            heat_release_per_deg = heat_release_rate / ca_fmax(
                omega_deg_per_s,
                CASADI_EPS,
            )
            burn_duration_deg = burn_time * omega_deg_per_s
        else:
            heat_release_per_deg = None
            burn_duration_deg = None

        return {
            "mfb": mfb,
            "burn_time": burn_time,
            "heat_release_rate": heat_release_rate,
            "heat_release_per_deg": heat_release_per_deg,
            "burn_duration_deg": burn_duration_deg,
        }

    # ------------------------------------------------------------------ #
    # Internal numeric helpers
    # ------------------------------------------------------------------ #
    def _compute_numeric(self, inputs: dict[str, Any]) -> CombustionOutputs:
        cfg = self.config
        params = self.params
        if cfg is None or params is None:
            raise RuntimeError("CombustionModel must be configured before simulate")

        time_s: np.ndarray = np.asarray(inputs.get("time_s"))
        if time_s.ndim != 1:
            raise ValueError("time_s must be a 1D array")
        piston_speed = np.asarray(inputs.get("piston_speed_m_per_s"))
        if piston_speed.shape != time_s.shape:
            raise ValueError("piston_speed_m_per_s must match time_s shape")

        theta_deg_input = inputs.get("theta_deg")
        if theta_deg_input is not None:
            theta_deg = np.asarray(theta_deg_input)
            if theta_deg.shape != time_s.shape:
                raise ValueError("theta_deg must match time_s shape")
        else:
            theta_deg = None

        omega_deg_per_s_input = inputs.get("omega_deg_per_s")
        if omega_deg_per_s_input is not None:
            omega_deg_per_s = np.asarray(omega_deg_per_s_input)
            if omega_deg_per_s.shape != time_s.shape:
                raise ValueError("omega_deg_per_s must match time_s shape")
        else:
            omega_deg_per_s = None

        ignition_time_s: float | None = inputs.get("ignition_time_s")
        ignition_theta_deg: float | None = inputs.get("ignition_theta_deg")
        
        # Get injector delay from inputs (takes precedence) or config
        injector_delay_s_input: float | None = inputs.get("injector_delay_s")
        injector_delay_deg_input: float | None = inputs.get("injector_delay_deg")
        injector_delay_s: float = 0.0
        if injector_delay_s_input is not None:
            injector_delay_s = float(injector_delay_s_input)
        elif injector_delay_deg_input is not None and theta_deg is not None and time_s is not None:
            # Convert injector_delay_deg to time
            omega_avg = 360.0 / cfg.cycle_time_s if cfg else 360.0 / max(time_s[-1] - time_s[0], 1e-9)
            injector_delay_s = float(injector_delay_deg_input) / omega_avg
        elif cfg and cfg.injector_delay_s is not None:
            injector_delay_s = float(cfg.injector_delay_s)
        elif cfg and cfg.injector_delay_deg is not None and theta_deg is not None and time_s is not None:
            # Convert from config
            omega_avg = 360.0 / cfg.cycle_time_s
            injector_delay_s = float(cfg.injector_delay_deg) / omega_avg

        if ignition_time_s is None:
            if ignition_theta_deg is None or theta_deg is None:
                raise ValueError(
                    "Either ignition_time_s or (ignition_theta_deg and theta_deg) must be provided",
                )
            ignition_time_s = float(
                np.interp(
                    ignition_theta_deg,
                    theta_deg,
                    time_s,
                ),
            )
        if ignition_theta_deg is None and theta_deg is not None:
            ignition_theta_deg = float(
                np.interp(
                    ignition_time_s,
                    time_s,
                    theta_deg,
                ),
            )
        
        # Apply injector delay: shift effective ignition time earlier by delay
        # Positive delay means fuel injected earlier, so combustion can start earlier
        effective_ignition_time_s = ignition_time_s - injector_delay_s

        duration_profile = self._duration_profile_from_speed(piston_speed)
        # Use effective ignition time for burn duration lookup
        burn_duration_s = float(
            np.interp(effective_ignition_time_s, time_s, duration_profile),
        )

        if omega_deg_per_s is None and theta_deg is not None:
            # Numerical derivative for ω when only θ provided
            omega_deg_per_s = np.gradient(theta_deg, time_s, edge_order=1)
        elif omega_deg_per_s is None:
            omega_deg_per_s = np.full_like(time_s, 360.0 / cfg.cycle_time_s)

        burn_duration_deg = float(burn_duration_s * np.interp(
            effective_ignition_time_s,
            time_s,
            omega_deg_per_s,
        ))
        
        # Compute effective ignition angle for output
        effective_ignition_theta_deg: float | None = None
        if theta_deg is not None:
            effective_ignition_theta_deg = float(
                np.interp(
                    effective_ignition_time_s,
                    time_s,
                    theta_deg,
                ),
            )

        mfb, heat_rate, heat_per_deg = self._burn_profile(
            time_s=time_s,
            ignition_time_s=effective_ignition_time_s,  # Use effective ignition time
            burn_duration_s=burn_duration_s,
            omega_deg_per_s=omega_deg_per_s,
        )

        ca_points = self._extract_combustion_angles(theta_deg, mfb)

        params_out = {
            "phi": float(self._phi or 1.0),
            "m_wiebe": cfg.m_wiebe,
            "a_wiebe": float(self._a_wiebe or 5.0),
            "target_mfb": cfg.target_mfb,
            "laminar_speed": float(self._laminar_speed_m_per_s or 0.0),
        }

        return CombustionOutputs(
            time_s=time_s,
            theta_deg=theta_deg,
            mass_fraction_burned=mfb,
            heat_release_rate_W=heat_rate,
            heat_release_per_deg=heat_per_deg,
            burn_duration_s=burn_duration_s,
            burn_duration_deg=burn_duration_deg,
            ignition_time_s=effective_ignition_time_s,  # Return effective ignition time
            ignition_theta_deg=effective_ignition_theta_deg or ignition_theta_deg,
            ca10_deg=ca_points.get("CA10"),
            ca50_deg=ca_points.get("CA50"),
            ca90_deg=ca_points.get("CA90"),
            ca100_deg=ca_points.get("CA100"),
            duration_profile=np.vstack((piston_speed, duration_profile)).T,
            params=params_out,
        )

    def _duration_profile_from_speed(self, piston_speed: np.ndarray) -> np.ndarray:
        cfg = self.config
        params = self.params
        if cfg is None or params is None:
            raise RuntimeError("CombustionModel must be configured")

        S_L = max(self._laminar_speed_m_per_s or 1e-3, 1e-3)
        alpha = params.alpha_turbulence
        u_turb = cfg.k_turb * np.abs(piston_speed)
        S_T = S_L * (1.0 + alpha * np.power(u_turb / max(S_L, 1e-9), cfg.turbulence_exponent))
        S_T = np.maximum(S_T, cfg.min_flame_speed)
        clearance_h = float(self._clearance_height_m or 1e-6)
        burn_time = cfg.c_burn * clearance_h / np.maximum(S_T, 1e-9)
        return burn_time

    def _burn_profile(
        self,
        *,
        time_s: np.ndarray,
        ignition_time_s: float,
        burn_duration_s: float,
        omega_deg_per_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        if cfg is None:
            raise RuntimeError("CombustionModel must be configured")

        a = float(self._a_wiebe or 5.0)
        m = float(cfg.m_wiebe)
        tau_raw = (time_s - ignition_time_s) / max(burn_duration_s, 1e-9)
        tau = np.clip(tau_raw, 0.0, 1.0)
        exp_term = np.exp(-a * np.power(tau, m + 1.0))
        mfb = 1.0 - exp_term

        active = (tau_raw >= 0.0) & (tau_raw <= 1.0)
        dxb_dt = np.zeros_like(time_s)
        if np.any(active):
            dxb_dt[active] = (
                a
                * (m + 1.0)
                / max(burn_duration_s, 1e-9)
                * np.power(tau[active], m)
                * exp_term[active]
            )

        q_total = cfg.fuel_mass_kg * float(self._lhv or 0.0)
        heat_rate = dxb_dt * q_total
        heat_per_deg = np.divide(
            heat_rate,
            np.maximum(omega_deg_per_s, 1e-9),
            out=np.zeros_like(heat_rate),
        )
        return mfb, heat_rate, heat_per_deg

    def _extract_combustion_angles(
        self,
        theta_deg: np.ndarray | None,
        mfb: np.ndarray,
    ) -> Dict[str, float | None]:
        if theta_deg is None:
            return {"CA10": None, "CA50": None, "CA90": None, "CA100": None}

        targets = {
            "CA10": 0.10,
            "CA50": 0.50,
            "CA90": 0.90,
            "CA100": 0.99,
        }
        ca_values: Dict[str, float | None] = {}

        for name, frac in targets.items():
            if mfb[-1] < frac:
                ca_values[name] = None
                continue
            try:
                ca_values[name] = float(
                    np.interp(frac, mfb, theta_deg),
                )
            except Exception:  # pragma: no cover - fallback guard
                ca_values[name] = None

        return ca_values

