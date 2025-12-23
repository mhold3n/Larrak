"""
Geometry-informed pressure ratio template for Phase-1 optimization.

This module computes explicit PR templates that are informed by engine geometry
(stroke, bore, compression ratio) and optimized for thermal/expansion efficiency,
replacing the implicit seed-derived PR targets.
"""

from __future__ import annotations

import numpy as np

from campro.logging import get_logger
from campro.materials.gases import get_gamma

log = get_logger(__name__)


def compute_pr_template(
    theta: np.ndarray,
    stroke_mm: float,
    bore_mm: float,
    clearance_volume_mm3: float,
    compression_ratio: float,
    p_load_kpa: float,
    p_cc_kpa: float,
    p_env_kpa: float,
    *,
    expansion_efficiency_target: float = 0.85,
    pr_peak_scale: float = 1.5,
    gamma: float | None = None,
    evo_deg: float = 540.0,
    tdc_deg: float = 180.0,
    combustion_start_deg: float = -5.0,
    combustion_duration_deg: float = 25.0,
) -> np.ndarray:
    """Compute geometry-informed PR template optimized for expansion efficiency.

    The template is designed to:
    - Follow isentropic compression during compression phase
    - Peak during combustion phase (TDC ± 20°)
    - Remain relatively flat during expansion (ideal isentropic expansion)
    - Taper down near EVO (exhaust valve opening)

    Parameters
    ----------
    theta : np.ndarray
        Crank angle array [rad], 0..2π, monotonically increasing
    stroke_mm : float
        Stroke length [mm]
    bore_mm : float
        Bore diameter [mm]
    clearance_volume_mm3 : float
        Clearance volume [mm^3]
    compression_ratio : float
        Compression ratio (V_max/V_min)
    p_load_kpa : float
        Load pressure [kPa]
    p_cc_kpa : float
        Crankcase pressure [kPa]
    p_env_kpa : float
        Environment pressure [kPa]
    expansion_efficiency_target : float, optional
        Target expansion efficiency (0-1), default 0.85
        Lower values account for heat loss and non-ideal expansion
    pr_peak_scale : float, optional
        Peak PR scaling factor relative to baseline, default 1.5
    gamma : float, optional
        Specific heat ratio. If None, uses air at 300K (~1.4).
    evo_deg : float, optional
        Exhaust valve opening angle [deg], default 540°
    tdc_deg : float, optional
        Top dead center angle [deg], default 180°
    combustion_start_deg : float, optional
        Combustion start angle [deg], default -5° (5° before TDC)
    combustion_duration_deg : float, optional
        Combustion duration [deg], default 25°

    Returns
    -------
    np.ndarray
        PR template Π(θ) [unitless], same shape as theta
    """
    if gamma is None:
        gamma = get_gamma("air", 300.0)

    theta = np.asarray(theta, dtype=float)
    # Convert to degrees, handling wrap-around properly
    theta_deg_raw = np.degrees(theta)
    # Normalize to 0-360 range for consistent phase identification
    theta_deg = theta_deg_raw % 360.0
    # Handle negative angles that wrap to positive (e.g., -5° becomes 355°)
    # For combustion phase detection, we need to handle angles near TDC that might be negative
    theta_deg_orig = theta_deg_raw.copy()  # Keep original for reference

    # Geometry calculations
    area_mm2 = np.pi * (bore_mm / 2.0) ** 2
    stroke_volume_mm3 = area_mm2 * stroke_mm
    v_max_mm3 = clearance_volume_mm3 + stroke_volume_mm3
    v_min_mm3 = clearance_volume_mm3

    # Ideal volume profile (sinusoidal approximation for free-piston)
    # For free-piston, we use a simplified model: x(θ) = 0.5*stroke*(1 - cos(θ))
    # Volume: V(θ) = Vc + A*x(θ)
    x_ideal = 0.5 * stroke_mm * (1.0 - np.cos(theta))
    v_ideal_mm3 = clearance_volume_mm3 + area_mm2 * x_ideal

    # Surface-to-volume ratio (affects heat loss)
    # S/V = 2*π*r^2 + 2*π*r*h (cylinder) / (π*r^2*h)
    # For expansion phase, we care about heat loss penalty
    bore_m = bore_mm * 1e-3
    stroke_m = stroke_mm * 1e-3
    radius_m = bore_m / 2.0
    sv_ratio = (2.0 * np.pi * radius_m * (radius_m + stroke_m)) / (np.pi * radius_m**2 * stroke_m)
    heat_loss_factor = 1.0 - 0.1 * sv_ratio  # Rough heat loss penalty (0-10% reduction)

    # Denominator for PR (constant across cycle)
    p_bounce_base = 0.0  # Will be computed per theta if needed, but for template we use base
    denom_base_kpa = p_load_kpa + p_cc_kpa + p_env_kpa + p_bounce_base

    # Initialize PR template
    pi_template = np.ones_like(theta_deg, dtype=float)

    # Phase identification
    # Handle negative combustion start angles (e.g., -5° before TDC)
    combustion_start_wrapped = (tdc_deg + combustion_start_deg) % 360.0
    combustion_end_wrapped = (tdc_deg + combustion_start_deg + combustion_duration_deg) % 360.0

    compression_phase = (theta_deg >= 0.0) & (theta_deg < tdc_deg)
    # Handle combustion phase that might wrap around 0° (if start is negative)
    if combustion_start_wrapped > combustion_end_wrapped:
        # Wraps around 360° boundary
        combustion_phase = (theta_deg >= combustion_start_wrapped) | (
            theta_deg < combustion_end_wrapped
        )
    else:
        combustion_phase = (theta_deg >= combustion_start_wrapped) & (
            theta_deg < combustion_end_wrapped
        )
    expansion_phase = (theta_deg >= tdc_deg) & (theta_deg < evo_deg) & (~combustion_phase)
    evo_phase = theta_deg >= evo_deg

    # 1. Compression Phase (0° → TDC): Isentropic compression
    if np.any(compression_phase):
        v_comp = v_ideal_mm3[compression_phase]
        v_comp = np.maximum(v_comp, v_min_mm3 * 1.001)  # Avoid division by zero
        # Isentropic compression: P ∝ (V_max/V)^(γ)
        # PR = p_cyl / p_denom, so we need to model p_cyl
        # For template: p_cyl ∝ (V_max/V)^γ
        pr_compression = (v_max_mm3 / v_comp) ** (gamma - 1.0)
        # Normalize to baseline (start of compression) - ensure we have at least one point
        if len(pr_compression) > 0 and pr_compression[0] > 0.0:
            pr_compression = pr_compression / pr_compression[0]
        pi_template[compression_phase] = pr_compression

    # 2. Combustion Phase (TDC ± combustion_duration): PR peak
    if np.any(combustion_phase):
        # Find peak location (center of combustion)
        peak_deg = tdc_deg + combustion_start_deg + combustion_duration_deg / 2.0
        theta_comb = theta_deg[combustion_phase]
        peak_distance = np.abs(theta_comb - peak_deg)
        # Normalized distance from peak (0 to 1)
        peak_dist_norm = peak_distance / (combustion_duration_deg / 2.0)
        peak_dist_norm = np.clip(peak_dist_norm, 0.0, 1.0)

        # Gaussian-like peak shape
        peak_shape = np.exp(-2.0 * peak_dist_norm**2)

        # Get compression PR at TDC as baseline
        tdc_idx = np.argmin(np.abs(theta_deg - tdc_deg))
        pr_base_tdc = pi_template[tdc_idx] if tdc_idx < len(pi_template) else 1.0

        # Scale peak by pr_peak_scale
        pr_peak = pr_base_tdc * pr_peak_scale
        pr_combustion = pr_base_tdc + (pr_peak - pr_base_tdc) * peak_shape
        pi_template[combustion_phase] = pr_combustion

    # 3. Expansion Phase (TDC → EVO): Flat PR (ideal isentropic expansion)
    if np.any(expansion_phase):
        # Start from peak PR at end of combustion
        comb_end_deg = tdc_deg + combustion_start_deg + combustion_duration_deg
        comb_end_idx = np.argmin(np.abs(theta_deg - comb_end_deg))
        pr_expansion_start = (
            pi_template[comb_end_idx] if comb_end_idx < len(pi_template) else pr_peak_scale
        )

        # Ideal isentropic expansion would maintain PR constant
        # But we account for expansion efficiency (heat loss, friction)
        # Lower efficiency → PR decreases during expansion
        v_exp = v_ideal_mm3[expansion_phase]
        v_exp_start = v_ideal_mm3[comb_end_idx] if comb_end_idx < len(v_ideal_mm3) else v_min_mm3
        # Get end volume from expansion phase indices
        exp_indices = np.where(expansion_phase)[0]
        if len(exp_indices) > 0:
            v_exp_end = v_ideal_mm3[exp_indices[-1]]
        else:
            v_exp_end = v_max_mm3

        # Ideal isentropic: PR constant
        # Non-ideal: PR decreases with expansion efficiency
        # PR(θ) = PR_start * (1 - (1-η_exp) * (V(θ) - V_start)/(V_end - V_start))
        v_normalized = (v_exp - v_exp_start) / max(v_exp_end - v_exp_start, 1e-9)
        v_normalized = np.clip(v_normalized, 0.0, 1.0)

        # Apply expansion efficiency (heat loss reduces PR during expansion)
        pr_expansion = pr_expansion_start * (
            1.0 - (1.0 - expansion_efficiency_target) * v_normalized * heat_loss_factor
        )
        pi_template[expansion_phase] = pr_expansion

    # 4. Near EVO (EVO → 720°): Taper down PR as exhaust opens
    if np.any(evo_phase):
        # Get PR at EVO as starting point
        evo_idx = np.argmin(np.abs(theta_deg - evo_deg))
        pr_evo_start = pi_template[evo_idx] if evo_idx < len(pi_template) else 1.0

        # Exponential decay toward ambient PR
        theta_evo = theta_deg[evo_phase]
        evo_distance = theta_evo - evo_deg
        evo_distance_norm = evo_distance / (360.0 - evo_deg)  # Normalize to 0-1
        evo_distance_norm = np.clip(evo_distance_norm, 0.0, 1.0)

        # Target PR at end (ambient, normalized)
        pr_ambient = (p_env_kpa + p_cc_kpa) / max(denom_base_kpa, 1e-6)
        pr_evo = pr_evo_start * np.exp(-3.0 * evo_distance_norm) + pr_ambient * (
            1.0 - np.exp(-3.0 * evo_distance_norm)
        )
        pi_template[evo_phase] = pr_evo

    # Ensure PR is positive and smooth
    pi_template = np.maximum(pi_template, 0.1)

    # Apply light smoothing to reduce discontinuities at phase boundaries
    pi_template = _smooth_phase_boundaries(pi_template, theta_deg, tdc_deg, evo_deg)

    return pi_template


def _smooth_phase_boundaries(
    pi_template: np.ndarray, theta_deg: np.ndarray, tdc_deg: float, evo_deg: float
) -> np.ndarray:
    """Apply smoothing at phase boundaries to reduce discontinuities.

    Parameters
    ----------
    pi_template : np.ndarray
        PR template values
    theta_deg : np.ndarray
        Crank angles in degrees
    tdc_deg : float
        Top dead center angle
    evo_deg : float
        Exhaust valve opening angle

    Returns
    -------
    np.ndarray
        Smoothed PR template
    """
    # Simple moving average filter at boundaries
    window_size = 5
    half_window = window_size // 2

    pi_smooth = pi_template.copy()
    n = len(pi_template)

    # Smooth around TDC
    tdc_idx = np.argmin(np.abs(theta_deg - tdc_deg))
    for i in range(max(0, tdc_idx - half_window), min(n, tdc_idx + half_window + 1)):
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        pi_smooth[i] = np.mean(pi_template[start_idx:end_idx])

    # Smooth around EVO
    evo_idx = np.argmin(np.abs(theta_deg - evo_deg))
    for i in range(max(0, evo_idx - half_window), min(n, evo_idx + half_window + 1)):
        start_idx = max(0, i - half_window)
        end_idx = min(n, i + half_window + 1)
        pi_smooth[i] = np.mean(pi_template[start_idx:end_idx])

    return pi_smooth
