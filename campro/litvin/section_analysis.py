"""Section analysis for piecewise gear optimization based on combustion timing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq

from campro.freepiston.core.chem import CombustionParameters, wiebe_function
from campro.logging import get_logger
from campro.utils.progress_logger import ProgressLogger

log = get_logger(__name__)

__all__ = [
    "SectionBoundaries",
    "compute_combustion_timing",
    "identify_combustion_sections",
    "get_section_boundaries",
]


@dataclass
class SectionBoundaries:
    """Section boundaries for piecewise optimization."""

    ca10_theta: float  # CA10 position [deg]
    ca50_theta: float  # CA50 position [deg]
    ca90_theta: float  # CA90 position [deg]
    ca100_theta: float  # CA100 position [deg]
    bdc_theta: float  # BDC position [deg]
    tdc_theta: float  # TDC position [deg]
    sections: dict[str, tuple[float, float]]  # Section name -> (theta_start, theta_end)


def compute_combustion_timing(
    theta: np.ndarray,
    position: np.ndarray,
    combustion_params: CombustionParameters,
) -> tuple[float, float, float, float]:
    """Compute CA10/CA50/CA90/CA100 from combustion model.

    Uses Wiebe function to compute mass fraction burned at each theta,
    then inverts to find theta positions where mass fraction = 0.10, 0.50, 0.90, 1.00.

    Parameters
    ----------
    theta : np.ndarray
        Cam angle array [deg]
    position : np.ndarray
        Piston position array [mm] (for validation/logging)
    combustion_params : CombustionParameters
        Combustion model parameters

    Returns
    -------
    tuple[float, float, float, float]
        CA10, CA50, CA90, CA100 theta positions [deg]
    """
    comb_logger = ProgressLogger("COMBUSTION", flush_immediately=True)
    comb_logger.step(1, 4, "Computing mass fraction burned curve")

    # Compute mass fraction burned at each theta
    mass_fraction = np.zeros_like(theta)
    for i, th in enumerate(theta):
        mass_fraction[i] = wiebe_function(
            theta=th,
            theta_start=combustion_params.theta_start,
            theta_duration=combustion_params.theta_duration,
            m=combustion_params.m_wiebe,
            a=combustion_params.a_wiebe,
        )

    comb_logger.info(
        f"Mass fraction range: {np.min(mass_fraction):.3f} to {np.max(mass_fraction):.3f}"
    )
    comb_logger.info(
        f"Combustion start: {combustion_params.theta_start:.2f}° "
        f"duration: {combustion_params.theta_duration:.2f}°"
    )

    comb_logger.step(2, 4, "Finding CA10/CA50/CA90/CA100 positions")

    # Find theta where mass fraction equals target values using root finding
    def find_ca_target(target_fraction: float) -> float:
        """Find theta where mass fraction equals target."""
        # Find valid range for root finding
        valid_indices = np.where(
            (mass_fraction >= target_fraction - 0.01)
            & (mass_fraction <= target_fraction + 0.01)
        )[0]

        if len(valid_indices) == 0:
            # Combustion outside range - extrapolate or use bounds
            if target_fraction < 0.01:
                return combustion_params.theta_start
            elif target_fraction > 0.99:
                return combustion_params.theta_start + combustion_params.theta_duration
            else:
                # Interpolate
                idx_below = np.where(mass_fraction < target_fraction)[0]
                idx_above = np.where(mass_fraction > target_fraction)[0]
                if len(idx_below) > 0 and len(idx_above) > 0:
                    idx_below = idx_below[-1]
                    idx_above = idx_above[0]
                    # Linear interpolation
                    frac = (target_fraction - mass_fraction[idx_below]) / (
                        mass_fraction[idx_above] - mass_fraction[idx_below]
                    )
                    return theta[idx_below] + frac * (theta[idx_above] - theta[idx_below])
                else:
                    return combustion_params.theta_start

        # Use root finding in valid range
        idx_start = max(0, valid_indices[0] - 5)
        idx_end = min(len(theta) - 1, valid_indices[-1] + 5)
        theta_min = theta[idx_start]
        theta_max = theta[idx_end]

        def residual(th: float) -> float:
            return wiebe_function(
                theta=th,
                theta_start=combustion_params.theta_start,
                theta_duration=combustion_params.theta_duration,
                m=combustion_params.m_wiebe,
                a=combustion_params.a_wiebe,
            ) - target_fraction

        try:
            ca_value = brentq(residual, theta_min, theta_max, xtol=0.1)
            return ca_value
        except ValueError:
            # Fallback to linear interpolation
            idx_below = np.where(mass_fraction < target_fraction)[0]
            idx_above = np.where(mass_fraction > target_fraction)[0]
            if len(idx_below) > 0 and len(idx_above) > 0:
                idx_below = idx_below[-1]
                idx_above = idx_above[0]
                frac = (target_fraction - mass_fraction[idx_below]) / (
                    mass_fraction[idx_above] - mass_fraction[idx_below]
                )
                return theta[idx_below] + frac * (theta[idx_above] - theta[idx_below])
            else:
                return combustion_params.theta_start

    ca10 = find_ca_target(0.10)
    ca50 = find_ca_target(0.50)
    ca90 = find_ca_target(0.90)
    ca100 = find_ca_target(1.00)

    comb_logger.info(f"CA10: {ca10:.2f}°, CA50: {ca50:.2f}°, CA90: {ca90:.2f}°, CA100: {ca100:.2f}°")
    comb_logger.step_complete("CA timing computation", 0.0)

    # Validate ordering
    if not (ca10 <= ca50 <= ca90 <= ca100):
        comb_logger.warning(
            f"CA values not in order: CA10={ca10:.2f}, CA50={ca50:.2f}, "
            f"CA90={ca90:.2f}, CA100={ca100:.2f}"
        )

    return ca10, ca50, ca90, ca100


def identify_combustion_sections(
    primary_data: dict[str, Any],
    combustion_params: CombustionParameters | None = None,
) -> SectionBoundaries:
    """Identify all 6 combustion sections from primary optimization data.

    Parameters
    ----------
    primary_data : dict[str, Any]
        Primary optimization data with 'theta' and 'position' arrays. If
        available, may also contain a 'ca_markers' dict with keys
        {'CA10','CA50','CA90','CA100'} from the integrated combustion model.
    combustion_params : CombustionParameters, optional
        Combustion model parameters (fallback when CA markers not provided)

    Returns
    -------
    SectionBoundaries
        Section boundaries with all 6 sections defined
    """
    from campro.utils.progress_logger import ProgressLogger

    comb_logger = ProgressLogger("COMBUSTION", flush_immediately=True)
    comb_logger.step(1, 3, "Extracting motion law data")

    # Extract motion law data
    theta = primary_data.get("theta")
    position = primary_data.get("position")

    if theta is None or position is None:
        raise ValueError("Primary data must contain 'theta' and 'position' arrays")

    # Convert to numpy arrays if needed
    theta = np.asarray(theta)
    position = np.asarray(position)

    if len(theta) != len(position):
        raise ValueError("theta and position arrays must have same length")

    comb_logger.info(f"Motion law: {len(theta)} points, theta range: {np.min(theta):.2f}° to {np.max(theta):.2f}°")
    comb_logger.step_complete("Data extraction", 0.0)

    ca_markers = primary_data.get("ca_markers")
    if ca_markers:
        comb_logger.step(2, 3, "Using CA markers from integrated combustion model")

        def _extract(marker: str) -> float:
            if marker not in ca_markers or ca_markers[marker] is None:
                raise ValueError(f"CA marker '{marker}' missing from primary data")
            return float(ca_markers[marker])

        ca10 = _extract("CA10")
        ca50 = _extract("CA50")
        ca90 = _extract("CA90")
        ca100 = _extract("CA100")
        comb_logger.info(
            f"CA markers provided: CA10={ca10:.2f}°, CA50={ca50:.2f}°, "
            f"CA90={ca90:.2f}°, CA100={ca100:.2f}°"
        )
        comb_logger.step_complete("Combustion timing (integrated)", 0.0)
    else:
        if combustion_params is None:
            raise ValueError(
                "Combustion parameters must be provided when CA markers are unavailable",
            )
        comb_logger.step(2, 3, "Computing combustion timing via legacy Wiebe model")
        ca10, ca50, ca90, ca100 = compute_combustion_timing(
            theta, position, combustion_params
        )

    # Identify BDC (max position) and TDC (min position)
    bdc_idx = np.argmax(position)
    tdc_idx = np.argmin(position)
    bdc_theta = theta[bdc_idx]
    tdc_theta = theta[tdc_idx]

    comb_logger.info(f"BDC: {bdc_theta:.2f}° (position={position[bdc_idx]:.2f}mm)")
    comb_logger.info(f"TDC: {tdc_theta:.2f}° (position={position[tdc_idx]:.2f}mm)")
    comb_logger.step_complete("Section identification", 0.0)

    # Define all 6 sections
    comb_logger.step(3, 3, "Defining section boundaries")
    sections = {
        "CA10-CA50": (ca10, ca50),
        "CA50-CA90": (ca50, ca90),
        "CA90-CA100": (ca90, ca100),
        "CA100-BDC": (ca100, bdc_theta),
        "BDC-TDC": (bdc_theta, tdc_theta),
        "TDC-CA10": (tdc_theta, ca10),
    }

    # Validate section ordering
    section_list = [
        ("CA10-CA50", ca10, ca50),
        ("CA50-CA90", ca50, ca90),
        ("CA90-CA100", ca90, ca100),
        ("CA100-BDC", ca100, bdc_theta),
        ("BDC-TDC", bdc_theta, tdc_theta),
        ("TDC-CA10", tdc_theta, ca10),
    ]

    for name, start, end in section_list:
        # Handle 0° as 360° for wraparound calculations (cycle is 1-360°)
        end_normalized = 360.0 if end == 0.0 else end
        
        if start > end_normalized:
            # Handle wrap-around: any section that crosses 360°/1° boundary
            # Normalize: start to end of cycle (360°), then 1° to end
            span = (360.0 - start) + end_normalized
            comb_logger.info(
                f"Section {name} wraps around: {start:.2f}° to 360° then 1° to {end_normalized:.2f}° ({span:.2f}° span)"
            )
        else:
            span = end_normalized - start
            comb_logger.info(f"Section {name}: {start:.2f}° to {end_normalized:.2f}° ({span:.2f}° span)")

    comb_logger.step_complete("Section boundaries", 0.0)

    return SectionBoundaries(
        ca10_theta=ca10,
        ca50_theta=ca50,
        ca90_theta=ca90,
        ca100_theta=ca100,
        bdc_theta=bdc_theta,
        tdc_theta=tdc_theta,
        sections=sections,
    )


def get_section_boundaries(
    theta: np.ndarray,
    position: np.ndarray,
    section_boundaries: SectionBoundaries,
) -> dict[str, tuple[int, int]]:
    """Map section boundaries to theta array indices.

    Parameters
    ----------
    theta : np.ndarray
        Cam angle array [deg]
    position : np.ndarray
        Piston position array [mm]
    section_boundaries : SectionBoundaries
        Section boundaries with theta positions

    Returns
    -------
    dict[str, tuple[int, int]]
        Section name -> (start_index, end_index) in theta array
    """
    indices = {}
    for name, (theta_start, theta_end) in section_boundaries.sections.items():
        # Find closest indices
        idx_start = np.argmin(np.abs(theta - theta_start))
        idx_end = np.argmin(np.abs(theta - theta_end))

        # Handle wrap-around for any section that crosses 0°/360° boundary
        if theta_start > theta_end:
            # For wraparound sections, indices are still valid
            # The optimization code will handle the wraparound logic
            indices[name] = (idx_start, idx_end)
        else:
            indices[name] = (idx_start, idx_end)

    return indices
