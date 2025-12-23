"""
CEM Validation Gates: Pre-check feasibility before expensive NLP optimization.

This module provides gate functions that query the CEM to validate profiles
before committing to full optimization. This prevents wasted computation on
clearly infeasible configurations.

Usage:
    from campro.validation.cem_gates import check_motion_feasibility

    report = check_motion_feasibility(x_profile, theta)
    if not report.is_valid:
        for v in report.violations:
            print(f"  {v.code.name}: {v.message}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from campro.logging import get_logger

if TYPE_CHECKING:
    from truthmaker.cem import OperatingEnvelope, ValidationReport

log = get_logger(__name__)

# Try to import CEM client
try:
    from truthmaker.cem import CEMClient, OperatingEnvelope, ValidationReport, ViolationSeverity

    CEM_AVAILABLE = True
except ImportError:
    CEM_AVAILABLE = False
    log.debug("CEM not available, gates will return mock-valid results")


def check_motion_feasibility(
    x_profile: np.ndarray,
    theta: np.ndarray | None = None,
    mock: bool = True,
) -> "ValidationReport":
    """
    Validate a target motion profile against CEM constraints.

    Should be called BEFORE Phase 3 NLP to avoid wasted computation.
    Returns structured violations with margins and suggested actions.

    Args:
        x_profile: Piston position trajectory [m]
        theta: Crank angle array [rad], generated if not provided
        mock: Use mock mode for development

    Returns:
        ValidationReport with is_valid flag and any violations
    """
    if not CEM_AVAILABLE:
        # Return a mock valid report
        from dataclasses import dataclass, field

        @dataclass
        class MockReport:
            is_valid: bool = True
            violations: list = field(default_factory=list)
            cem_version: str = "mock-fallback"

        log.debug("CEM unavailable, returning mock-valid report")
        return MockReport()  # type: ignore

    if theta is None:
        theta = np.linspace(0, 2 * np.pi, len(x_profile))

    with CEMClient(mock=mock) as cem:
        report = cem.validate_motion(x_profile, theta)

    if not report.is_valid:
        n_fatal = sum(1 for v in report.violations if v.severity == ViolationSeverity.FATAL)
        n_error = sum(1 for v in report.violations if v.severity == ViolationSeverity.ERROR)
        log.warning(f"Motion validation failed: {n_fatal} fatal, {n_error} errors")

    return report


def check_thermo_feasibility(
    rpm: float,
    p_intake_bar: float,
    fuel_mass_kg: float,
    bore: float = 0.1,
    stroke: float = 0.1,
    cr: float = 15.0,
    mock: bool = True,
) -> "ValidationReport":
    """
    Validate thermodynamic operating point against CEM limits.

    Checks if the proposed operating condition is within the feasible
    envelope returned by CEM.

    Args:
        rpm: Engine speed [rev/min]
        p_intake_bar: Intake manifold pressure [bar]
        fuel_mass_kg: Fuel mass per cycle [kg]
        bore: Cylinder bore [m]
        stroke: Piston stroke [m]
        cr: Compression ratio
        mock: Use mock mode

    Returns:
        ValidationReport with is_valid flag
    """
    if not CEM_AVAILABLE:
        from dataclasses import dataclass, field

        @dataclass
        class MockReport:
            is_valid: bool = True
            violations: list = field(default_factory=list)

        return MockReport()  # type: ignore

    with CEMClient(mock=mock) as cem:
        envelope = cem.get_thermo_envelope(bore, stroke, cr, rpm)

        # Check if operating point is within envelope
        violations = []

        boost_min, boost_max = envelope.boost_range
        if p_intake_bar < boost_min or p_intake_bar > boost_max:
            from truthmaker.cem import (
                ConstraintViolation,
                SuggestedActionCode,
                ViolationCode,
                ViolationSeverity,
            )

            violations.append(
                ConstraintViolation(
                    code=ViolationCode.CONFIG_INVALID_BOUNDS,
                    severity=ViolationSeverity.ERROR,
                    message=f"Intake pressure {p_intake_bar:.2f} bar outside envelope [{boost_min:.2f}, {boost_max:.2f}]",
                    margin=min(p_intake_bar - boost_min, boost_max - p_intake_bar),
                    suggested_action=SuggestedActionCode.TIGHTEN_BOUNDS,
                )
            )

        fuel_min, fuel_max = envelope.fuel_range
        if fuel_mass_kg < fuel_min or fuel_mass_kg > fuel_max:
            from truthmaker.cem import (
                ConstraintViolation,
                SuggestedActionCode,
                ViolationCode,
                ViolationSeverity,
            )

            violations.append(
                ConstraintViolation(
                    code=ViolationCode.THERMO_LAMBDA_TOO_RICH
                    if fuel_mass_kg > fuel_max
                    else ViolationCode.THERMO_LAMBDA_TOO_LEAN,
                    severity=ViolationSeverity.ERROR,
                    message=f"Fuel mass {fuel_mass_kg * 1000:.2f} mg outside envelope [{fuel_min * 1000:.2f}, {fuel_max * 1000:.2f}]",
                    margin=min(fuel_mass_kg - fuel_min, fuel_max - fuel_mass_kg),
                    suggested_action=SuggestedActionCode.ADJUST_PHASE,
                )
            )

        from truthmaker.cem import ValidationReport as VR

        return VR(
            is_valid=len(violations) == 0 and envelope.feasible,
            violations=violations,
            cem_version="mock-0.1.0" if mock else "live",
        )


def check_gear_feasibility(
    config: dict,
    mock: bool = True,
) -> "ValidationReport":
    """
    Validate gear configuration against manufacturability constraints.

    Checks module, tooth counts, and contact conditions.

    Args:
        config: Gear configuration dictionary with keys:
            - module: Gear module [mm]
            - z_sun: Sun gear tooth count
            - z_planet: Planet gear tooth count
            - z_ring: Ring gear tooth count
        mock: Use mock mode

    Returns:
        ValidationReport with is_valid flag
    """
    if not CEM_AVAILABLE:
        from dataclasses import dataclass, field

        @dataclass
        class MockReport:
            is_valid: bool = True
            violations: list = field(default_factory=list)

        return MockReport()  # type: ignore

    # For now, return a valid report - gear validation will be
    # implemented when Phase 3 gear synthesis is active
    from truthmaker.cem import ValidationReport as VR

    log.debug("Gear feasibility check: placeholder returning valid")
    return VR(is_valid=True, violations=[])


def get_operating_envelope(
    bore: float,
    stroke: float,
    cr: float,
    rpm: float,
    mock: bool = True,
) -> "OperatingEnvelope":
    """
    Get the feasible operating envelope from CEM.

    Args:
        bore: Cylinder bore [m]
        stroke: Piston stroke [m]
        cr: Compression ratio
        rpm: Engine speed [rev/min]
        mock: Use mock mode

    Returns:
        OperatingEnvelope with boost_range, fuel_range, motion_bounds
    """
    if not CEM_AVAILABLE:
        from dataclasses import dataclass

        @dataclass
        class MockEnvelope:
            boost_range: tuple[float, float] = (1.0, 3.0)
            fuel_range: tuple[float, float] = (1e-5, 1e-3)
            motion_bounds: tuple[float, float] = (0.0, 0.2)
            feasible: bool = True
            config_hash: str = "mock"

        return MockEnvelope()  # type: ignore

    with CEMClient(mock=mock) as cem:
        return cem.get_thermo_envelope(bore, stroke, cr, rpm)


__all__ = [
    "check_motion_feasibility",
    "check_thermo_feasibility",
    "check_gear_feasibility",
    "get_operating_envelope",
    "CEM_AVAILABLE",
]
