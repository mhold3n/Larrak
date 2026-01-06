"""
Symbolic Combustion Model logic for CasADi.

This module contains the functional core of the combustion model using CasADi symbolics.
It is decoupled from the stateful CombustionModel class to facilitate direct usage in NLPs.
"""

from __future__ import annotations

from typing import Any

import casadi as ca

from campro.constants import CASADI_PHYSICS_EPSILON


def symbolic_heat_release(
    time_s: Any,
    piston_speed_m_per_s: Any,
    ignition_time_s: Any,
    laminar_speed_m_per_s: float,
    clearance_height_m: float,
    fuel_mass_kg: float,
    lower_heating_value: float,
    k_turb: float,
    turbulence_exponent: float,
    min_flame_speed: float,
    alpha_turbulence: float,
    c_burn: float,
    m_wiebe: float,
    a_wiebe: float,
    omega_deg_per_s: Any | None = None,
) -> dict[str, Any]:
    """
    Build CasADi expressions for the heat-release model.

    Args:
        time_s: Symbolic time [s]
        piston_speed_m_per_s: Symbolic piston speed [m/s]
        ignition_time_s: Symbolic ignition timing [s]
        laminar_speed_m_per_s: Laminar flame speed [m/s]
        clearance_height_m: Clearance height [m]
        fuel_mass_kg: Fuel mass [kg]
        lower_heating_value: LHV [J/kg]
        k_turb: Turbulence multipling factor
        turbulence_exponent: Turbulence exponent
        min_flame_speed: Minimum flame speed [m/s]
        alpha_turbulence: Turbulence alpha parameter
        c_burn: Burn coefficient
        m_wiebe: Wiebe form factor
        a_wiebe: Wiebe efficiency factor
        omega_deg_per_s: Optional angular speed for degree-based outputs [deg/s]

    Returns:
        Dict with keys: mfb, burn_time, heat_release_rate, heat_release_per_deg.
    """
    ca_fmax = ca.fmax
    eps = CASADI_PHYSICS_EPSILON.value

    # Validate/Clean inputs (floats)
    s_l_val = float(max(laminar_speed_m_per_s or 1e-3, 1e-3))
    clearance_h = float(clearance_height_m or 1e-6)
    min_speed = float(min_flame_speed)
    alpha = float(alpha_turbulence)
    k_turb_val = float(k_turb)
    exponent = float(turbulence_exponent)
    a_val = float(a_wiebe)
    m_val = float(m_wiebe)
    q_total = float(fuel_mass_kg * lower_heating_value)

    # 1. Turbulent Flame Speed
    u_turb = k_turb_val * ca.fabs(piston_speed_m_per_s)
    v_ratio = u_turb / ca_fmax(s_l_val, eps)
    s_t_val = s_l_val * (1.0 + alpha * ca.power(v_ratio, exponent))
    s_t_val = ca_fmax(s_t_val, min_speed)

    # 2. Burn Duration
    burn_time = c_burn * clearance_h / ca_fmax(s_t_val, eps)

    # 3. Wiebe Function
    # Normalized time tau
    tau_raw = (time_s - ignition_time_s) / ca_fmax(burn_time, eps)

    # Clip tau to [0, 1] for MFB, but handle derivative carefully
    tau = ca.if_else(
        tau_raw < 0.0,
        0.0,
        ca.if_else(tau_raw > 1.0, 1.0, tau_raw),
    )

    exp_term = ca.exp(-a_val * ca.power(tau, m_val + 1.0))
    mfb = 1.0 - exp_term

    # 4. Heat Release Rate (dQ/dt)
    # dMFB/dt = dMFB/dtau * dtau/dt
    # dtau/dt = 1/burn_time

    base_rate = (
        a_val
        * (m_val + 1.0)
        / ca_fmax(burn_time, eps)
        * ca.power(ca.fmax(tau, eps), m_val)
        * exp_term
    )

    # Zero out rate outside burn window
    dxb_dt = ca.if_else(
        tau_raw < 0.0,
        0.0,
        ca.if_else(tau_raw > 1.0, 0.0, base_rate),
    )

    heat_release_rate = dxb_dt * q_total

    # 5. Degree-based outputs
    heat_release_per_deg = None
    burn_duration_deg = None

    if omega_deg_per_s is not None:
        heat_release_per_deg = heat_release_rate / ca_fmax(omega_deg_per_s, eps)
        burn_duration_deg = burn_time * omega_deg_per_s

    return {
        "mfb": mfb,
        "burn_time": burn_time,
        "heat_release_rate": heat_release_rate,
        "heat_release_per_deg": heat_release_per_deg,
        "burn_duration_deg": burn_duration_deg,
    }
