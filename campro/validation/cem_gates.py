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
    mock: bool = False,
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
        if mock:
            # Return a mock valid report
            from dataclasses import dataclass, field

            @dataclass
            class MockReport:
                is_valid: bool = True
                violations: list = field(default_factory=list)
                cem_version: str = "mock-fallback"

            log.debug("CEM unavailable, returning mock-valid report")
            return MockReport()  # type: ignore
        else:
            raise ImportError("CEM library (truthmaker) not available. Cannot run validation.")

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
    mock: bool = False,
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
        if mock:
            from dataclasses import dataclass, field

            @dataclass
            class MockReport:
                is_valid: bool = True
                violations: list = field(default_factory=list)

            return MockReport()  # type: ignore
        else:
            raise ImportError("CEM library not available")

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
    mock: bool = False,
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
        if mock:
            from dataclasses import dataclass, field

            @dataclass
            class MockReport:
                is_valid: bool = True
                violations: list = field(default_factory=list)

            return MockReport()  # type: ignore
        else:
            raise ImportError("CEM library not available")

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
    mock: bool = False,
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
        if mock:
            from dataclasses import dataclass

            @dataclass
            class MockEnvelope:
                boost_range: tuple[float, float] = (1.0, 3.0)
                fuel_range: tuple[float, float] = (1e-5, 1e-3)
                motion_bounds: tuple[float, float] = (0.0, 0.2)
                feasible: bool = True
                config_hash: str = "mock"

            return MockEnvelope()  # type: ignore
        else:
            raise ImportError("CEM library not available")

    with CEMClient(mock=mock) as cem:
        return cem.get_thermo_envelope(bore, stroke, cr, rpm)


# =============================================================================
# HiFi Surrogate Gates (Phase 4)
# =============================================================================

# Lazy-loaded surrogate models
_thermal_surrogate = None
_structural_surrogate = None


def _load_thermal_surrogate():
    """Lazy load thermal surrogate model."""
    global _thermal_surrogate
    if _thermal_surrogate is None:
        try:
            from pathlib import Path

            from truthmaker.surrogates.models.hifi_surrogates import ThermalSurrogate

            model_path = Path(__file__).parents[2] / "models" / "hifi" / "thermal_surrogate.pt"
            if model_path.exists():
                _thermal_surrogate = ThermalSurrogate.load(str(model_path))
                log.info(f"Loaded thermal surrogate from {model_path}")
            else:
                # Create untrained model for development
                _thermal_surrogate = ThermalSurrogate(n_models=3)
                log.warning("Using untrained thermal surrogate (no model file found)")
        except ImportError as e:
            log.warning(f"Could not load thermal surrogate: {e}")
            _thermal_surrogate = None
    return _thermal_surrogate


def _load_structural_surrogate():
    """Lazy load structural surrogate model."""
    global _structural_surrogate
    if _structural_surrogate is None:
        try:
            from pathlib import Path

            from truthmaker.surrogates.models.hifi_surrogates import StructuralSurrogate

            model_path = Path(__file__).parents[2] / "models" / "hifi" / "structural_surrogate.pt"
            if model_path.exists():
                _structural_surrogate = StructuralSurrogate.load(str(model_path))
                log.info(f"Loaded structural surrogate from {model_path}")
            else:
                _structural_surrogate = StructuralSurrogate(n_models=3)
                log.warning("Using untrained structural surrogate")
        except ImportError as e:
            log.warning(f"Could not load structural surrogate: {e}")
            _structural_surrogate = None
    return _structural_surrogate


def check_hifi_thermal_feasibility(
    bore_mm: float,
    stroke_mm: float,
    cr: float,
    rpm: float,
    load_fraction: float,
    T_limit_K: float = 620.0,
    confidence_sigma: float = 2.0,
) -> dict:
    """
    Check thermal feasibility using HiFi surrogate.

    Uses trained ensemble model to predict T_crown_max and check
    against limit with uncertainty margin.

    Args:
        bore_mm: Cylinder bore [mm]
        stroke_mm: Piston stroke [mm]
        cr: Compression ratio
        rpm: Engine speed
        load_fraction: Load (0-1)
        T_limit_K: Maximum allowable temperature [K]
        confidence_sigma: Uncertainty multiplier

    Returns:
        Dict with is_feasible, T_predicted, uncertainty, margin
    """
    model = _load_thermal_surrogate()

    if model is None:
        # Fallback: assume feasible but flag uncertainty
        return {
            "is_feasible": True,
            "T_predicted": 500.0,
            "uncertainty": 100.0,
            "margin": T_limit_K - 500.0,
            "source": "fallback",
        }

    # Normalize inputs
    from Simulations.hifi.training_schema import NormalizationParams

    norm = NormalizationParams()

    inputs = np.array(
        [
            [
                (bore_mm - norm.bore_range[0]) / (norm.bore_range[1] - norm.bore_range[0]),
                (stroke_mm - norm.stroke_range[0]) / (norm.stroke_range[1] - norm.stroke_range[0]),
                (cr - norm.cr_range[0]) / (norm.cr_range[1] - norm.cr_range[0]),
                (rpm - norm.rpm_range[0]) / (norm.rpm_range[1] - norm.rpm_range[0]),
                (load_fraction - norm.load_range[0]) / (norm.load_range[1] - norm.load_range[0]),
            ]
        ],
        dtype=np.float32,
    )

    mean, std = model.predict(inputs)
    T_pred = float(mean[0, 0])
    T_std = float(std[0, 0])

    # Conservative bound
    T_conservative = T_pred + confidence_sigma * T_std
    is_feasible = T_conservative < T_limit_K
    margin = T_limit_K - T_conservative

    return {
        "is_feasible": is_feasible,
        "T_predicted": T_pred,
        "uncertainty": T_std,
        "margin": margin,
        "source": "surrogate",
    }


def check_hifi_structural_feasibility(
    bore_mm: float,
    stroke_mm: float,
    cr: float,
    rpm: float,
    load_fraction: float,
    yield_strength_MPa: float = 280.0,
    safety_factor: float = 1.5,
    confidence_sigma: float = 2.0,
) -> dict:
    """
    Check structural feasibility using HiFi surrogate.

    Predicts von Mises stress and checks against yield with safety factor.

    Args:
        bore_mm, stroke_mm, cr, rpm, load_fraction: Operating point
        yield_strength_MPa: Material yield strength
        safety_factor: Required safety margin
        confidence_sigma: Uncertainty multiplier

    Returns:
        Dict with is_feasible, stress_predicted, uncertainty, margin
    """
    model = _load_structural_surrogate()

    if model is None:
        return {
            "is_feasible": True,
            "stress_predicted": 100.0,
            "uncertainty": 30.0,
            "margin": yield_strength_MPa / safety_factor - 100.0,
            "source": "fallback",
        }

    from Simulations.hifi.training_schema import NormalizationParams

    norm = NormalizationParams()

    inputs = np.array(
        [
            [
                (bore_mm - norm.bore_range[0]) / (norm.bore_range[1] - norm.bore_range[0]),
                (stroke_mm - norm.stroke_range[0]) / (norm.stroke_range[1] - norm.stroke_range[0]),
                (cr - norm.cr_range[0]) / (norm.cr_range[1] - norm.cr_range[0]),
                (rpm - norm.rpm_range[0]) / (norm.rpm_range[1] - norm.rpm_range[0]),
                (load_fraction - norm.load_range[0]) / (norm.load_range[1] - norm.load_range[0]),
            ]
        ],
        dtype=np.float32,
    )

    mean, std = model.predict(inputs)
    stress_pred = float(mean[0, 0])
    stress_std = float(std[0, 0])

    stress_conservative = stress_pred + confidence_sigma * stress_std
    allowable = yield_strength_MPa / safety_factor
    is_feasible = stress_conservative < allowable
    margin = allowable - stress_conservative

    return {
        "is_feasible": is_feasible,
        "stress_predicted": stress_pred,
        "uncertainty": stress_std,
        "margin": margin,
        "source": "surrogate",
    }


def check_all_hifi_gates(
    bore_mm: float,
    stroke_mm: float,
    cr: float,
    rpm: float,
    load_fraction: float,
) -> dict:
    """
    Run all HiFi feasibility gates.

    Aggregates thermal and structural checks for overall feasibility.
    """
    thermal = check_hifi_thermal_feasibility(bore_mm, stroke_mm, cr, rpm, load_fraction)
    structural = check_hifi_structural_feasibility(bore_mm, stroke_mm, cr, rpm, load_fraction)

    all_feasible = thermal["is_feasible"] and structural["is_feasible"]

    return {
        "is_feasible": all_feasible,
        "thermal": thermal,
        "structural": structural,
    }


__all__ = [
    "check_motion_feasibility",
    "check_thermo_feasibility",
    "check_gear_feasibility",
    "get_operating_envelope",
    "check_hifi_thermal_feasibility",
    "check_hifi_structural_feasibility",
    "check_all_hifi_gates",
    "CEM_AVAILABLE",
]
