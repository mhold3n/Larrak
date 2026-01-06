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
from typing import Any

import numpy as np

from campro.constants import (
    DEFAULT_BURN_COEF_C,
    DEFAULT_TURBULENCE_FACTOR_K,
    DEFAULT_WIEBE_M,
    MIN_FLAME_SPEED,
    TURBULENCE_EXPONENT,
)
from campro.logging import get_logger
from campro.materials.fuels import get_stoich_afr
from campro.physics.combustion.symbolic import symbolic_heat_release as sym_hr

from .base import BasePhysicsModel, PhysicsResult
from .chem import CombustionParameters, create_combustion_parameters, laminar_flame_speed

log = get_logger(__name__)


# CASADI_EPS = 1e-9  <- Removed, using CASADI_EPSILON


def _get_stoich_afr(fuel_type: str) -> float:
    """Get stoichiometric AFR from materials database.

    Args:
        fuel_type: Fuel type name

    Returns:
        Stoichiometric air-fuel ratio (kg air / kg fuel)
    """
    try:
        afr, _ = get_stoich_afr(fuel_type)
        return afr
    except KeyError:
        # Fallback for unknown fuels
        log.warning(f"Unknown fuel '{fuel_type}', using gasoline stoich AFR")
        afr, _ = get_stoich_afr("gasoline")
        return afr


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
    initial_temperature_k: float
    initial_pressure_pa: float = 1e5
    target_mfb: float = 0.99
    m_wiebe: float = DEFAULT_WIEBE_M.value
    k_turb: float = DEFAULT_TURBULENCE_FACTOR_K.value
    c_burn: float = DEFAULT_BURN_COEF_C.value
    turbulence_exponent: float = TURBULENCE_EXPONENT.value
    min_flame_speed: float = MIN_FLAME_SPEED.value
    heating_value_override: float | None = None
    phi_override: float | None = None
    injector_delay_s: float = 0.0  # Injector delay [s] - shifts effective combustion start
    injector_delay_deg: float | None = (
        None  # Injector delay [deg] - alternative to injector_delay_s
    )


@dataclass
class CombustionOutputs:
    """Structured outputs from a combustion simulation."""

    time_s: np.ndarray
    theta_deg: np.ndarray | None
    mass_fraction_burned: np.ndarray
    heat_release_rate_w: np.ndarray
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
    params: dict[str, float] = field(default_factory=dict)


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
        # Validation - try to get stoich AFR to verify fuel is known
        try:
            stoich_afr = _get_stoich_afr(fuel_key)
        except KeyError:
            raise ValueError(f"Unsupported fuel type '{cfg.fuel_type}'") from None

        if cfg.afr <= 0:
            raise ValueError("AFR must be positive")

        if cfg.cycle_time_s <= 0:
            raise ValueError("cycle_time_s must be positive")

        if cfg.fuel_mass_kg < 0:
            raise ValueError("fuel_mass_kg must be non-negative")

        params = create_combustion_parameters(fuel_key)
        self.params = params
        self.config = cfg
        self._stoich_afr = stoich_afr

        # Geometric properties
        radius = cfg.bore_m / 2.0
        self._piston_area_m2 = math.pi * radius * radius
        self._clearance_height_m = cfg.clearance_volume_m3 / max(
            self._piston_area_m2,
            1e-12,
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
                temperature=cfg.initial_temperature_k,
                pressure=cfg.initial_pressure_pa,
                params=params,
            ),
        )

        # Lower heating value
        if cfg.heating_value_override is not None:
            self._lhv = cfg.heating_value_override
        else:
            self._lhv = params.lower_heating_value_fuel

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
            "heat_release_rate_W": outputs.heat_release_rate_w,
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
    ) -> dict[str, Any]:
        """
        Build CasADi expressions for the heat-release model.
        Delegates to `campro.physics.combustion.symbolic.symbolic_heat_release`.
        """
        if not self._is_configured or self.config is None or self.params is None:
            raise RuntimeError("CombustionModel must be configured before use")

        # Unpack configuration
        cfg = self.config
        params = self.params

        return sym_hr(
            time_s=time_s,
            piston_speed_m_per_s=piston_speed_m_per_s,
            ignition_time_s=ignition_time_s,
            laminar_speed_m_per_s=float(max(self._laminar_speed_m_per_s or 1e-3, 1e-3)),
            clearance_height_m=float(self._clearance_height_m or 1e-6),
            fuel_mass_kg=cfg.fuel_mass_kg,
            lower_heating_value=float(self._lhv or params.lower_heating_value_fuel),
            k_turb=float(cfg.k_turb),
            turbulence_exponent=float(cfg.turbulence_exponent),
            min_flame_speed=float(cfg.min_flame_speed),
            alpha_turbulence=float(params.alpha_turbulence),
            c_burn=float(cfg.c_burn),
            m_wiebe=float(cfg.m_wiebe),
            a_wiebe=float(self._a_wiebe or 5.0),
            omega_deg_per_s=omega_deg_per_s,
        )

    # ------------------------------------------------------------------ #
    # Internal numeric helpers
    # ------------------------------------------------------------------ #
    def _compute_numeric(self, inputs: dict[str, Any]) -> CombustionOutputs:
        cfg = self.config
        params = self.params
        if cfg is None or params is None:
            raise RuntimeError("CombustionModel must be configured before simulate")

        time_s, piston_speed, theta_deg, omega_deg_per_s = self._resolve_kinematics(inputs)

        (
            effective_ignition_time_s,
            effective_ignition_theta_deg,
            ignition_theta_deg,  # Needed for outputs
        ) = self._resolve_ignition_and_delay(inputs, time_s, theta_deg)

        if omega_deg_per_s is None and theta_deg is not None:
            omega_deg_per_s = np.gradient(theta_deg, time_s, edge_order=1)
        elif omega_deg_per_s is None and cfg is not None:
            omega_deg_per_s = np.full_like(time_s, 360.0 / cfg.cycle_time_s)

        # --------------------------------------------------------
        # Compute Burn Duration Profile (from turbulence)
        # --------------------------------------------------------
        duration_profile = self._duration_profile_from_speed(piston_speed)

        # Burn duration at the effective ignition time
        burn_duration_s = float(
            np.interp(effective_ignition_time_s, time_s, duration_profile),
        )

        omega_interp = omega_deg_per_s if omega_deg_per_s is not None else np.zeros_like(time_s)
        burn_duration_deg = float(
            burn_duration_s
            * np.interp(
                effective_ignition_time_s,
                time_s,
                omega_interp,
            )
        )

        mfb, heat_rate, heat_per_deg = self._burn_profile(
            time_s=time_s,
            ignition_time_s=effective_ignition_time_s,
            burn_duration_s=burn_duration_s,
            omega_deg_per_s=omega_interp,
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
            heat_release_rate_w=heat_rate,
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

        s_l_val = max(self._laminar_speed_m_per_s or 1e-3, 1e-3)
        alpha = params.alpha_turbulence
        u_turb = cfg.k_turb * np.abs(piston_speed)
        s_t_val = s_l_val * (
            1.0 + alpha * np.power(u_turb / max(s_l_val, 1e-9), cfg.turbulence_exponent)
        )
        s_t_val = np.maximum(s_t_val, cfg.min_flame_speed)
        clearance_h = float(self._clearance_height_m or 1e-6)
        burn_time = cfg.c_burn * clearance_h / np.maximum(s_t_val, 1e-9)
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
    ) -> dict[str, float | None]:
        if theta_deg is None:
            return {"CA10": None, "CA50": None, "CA90": None, "CA100": None}

        targets = {
            "CA10": 0.10,
            "CA50": 0.50,
            "CA90": 0.90,
            "CA100": 0.99,
        }
        ca_values: dict[str, float | None] = {}

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

    def _resolve_kinematics(
        self,
        inputs: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        time_s: np.ndarray = np.asarray(inputs.get("time_s"))
        if time_s.ndim != 1:
            raise ValueError("time_s must be a 1D array")
        piston_speed = np.asarray(inputs.get("piston_speed_m_per_s"))
        if piston_speed.shape != time_s.shape:
            raise ValueError("piston_speed_m_per_s must match time_s shape")

        theta_deg_input = inputs.get("theta_deg")
        theta_deg: np.ndarray | None = None
        if theta_deg_input is not None:
            theta_deg = np.asarray(theta_deg_input)
            if theta_deg.shape != time_s.shape:
                raise ValueError("theta_deg must match time_s shape")

        omega_input = inputs.get("omega_deg_per_s")
        omega_deg_per_s: np.ndarray | None = None
        if omega_input is not None:
            omega_deg_per_s = np.asarray(omega_input)
            if omega_deg_per_s.shape != time_s.shape:
                raise ValueError("omega_deg_per_s must match time_s shape")

        return time_s, piston_speed, theta_deg, omega_deg_per_s

    def _resolve_ignition_and_delay(
        self,
        inputs: dict[str, Any],
        time_s: np.ndarray,
        theta_deg: np.ndarray | None,
    ) -> tuple[float, float | None, float | None]:
        cfg = self.config

        # Injector Delay Logic
        injector_delay_s_input = inputs.get("injector_delay_s")
        injector_delay_deg_input = inputs.get("injector_delay_deg")
        injector_delay_s: float = 0.0

        if injector_delay_s_input is not None:
            injector_delay_s = float(injector_delay_s_input)
        elif injector_delay_deg_input is not None and theta_deg is not None:
            omega_avg = (
                360.0 / cfg.cycle_time_s if cfg else 360.0 / max(time_s[-1] - time_s[0], 1e-9)
            )
            injector_delay_s = float(injector_delay_deg_input) / omega_avg
        elif cfg and cfg.injector_delay_s is not None:
            injector_delay_s = float(cfg.injector_delay_s)
        elif cfg and cfg.injector_delay_deg is not None and theta_deg is not None:
            omega_avg = 360.0 / cfg.cycle_time_s
            injector_delay_s = float(cfg.injector_delay_deg) / omega_avg

        # Ignition Timing Logic
        ignition_time_s: float | None = inputs.get("ignition_time_s")
        ignition_theta_deg: float | None = inputs.get("ignition_theta_deg")

        if ignition_time_s is None:
            if ignition_theta_deg is None or theta_deg is None:
                raise ValueError(
                    "Either ignition_time_s or (ignition_theta_deg and theta_deg) must be provided",
                )
            ignition_time_s = float(np.interp(ignition_theta_deg, theta_deg, time_s))

        if ignition_theta_deg is None and theta_deg is not None:
            # Back-calculate angle from time if missing
            ignition_theta_deg = float(np.interp(ignition_time_s, time_s, theta_deg))

        # Effective Ignition Time
        effective_ignition_time_s = ignition_time_s - injector_delay_s

        # Effective Ignition Angle
        effective_ignition_theta_deg: float | None = None
        if theta_deg is not None:
            effective_ignition_theta_deg = float(
                np.interp(effective_ignition_time_s, time_s, theta_deg)
            )

        return effective_ignition_time_s, effective_ignition_theta_deg, ignition_theta_deg
