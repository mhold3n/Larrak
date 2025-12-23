"""Safety factor and design margin utilities.

This module provides engineering safety factors and design margins for
mechanical design calculations, following industry best practices.

Key concepts:
    - Safety Factor (SF): Ratio of allowable to actual stress/load
    - Design Margin: Additional capacity beyond nominal requirements
    - Knock-down Factor: Reduction factor for uncertainties

Reference Standards:
    - ASME BPVC Section VIII (Pressure Vessels)
    - FKM Guideline (German mechanical fatigue design)
    - Eurocode 3 (Structural steel design)
    - MIL-HDBK-5J (Metallic materials for aerospace)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

log = logging.getLogger(__name__)


# =============================================================================
# Safety Factor Categories
# =============================================================================


class FailureMode(Enum):
    """Failure mode categories."""

    YIELDING = auto()  # Plastic deformation
    FRACTURE = auto()  # Brittle fracture
    FATIGUE = auto()  # Cyclic loading failure
    BUCKLING = auto()  # Stability failure
    CREEP = auto()  # Time-dependent deformation
    WEAR = auto()  # Surface degradation
    CORROSION = auto()  # Chemical degradation


class LoadType(Enum):
    """Load type categories."""

    STATIC = auto()  # Constant or slowly varying
    DYNAMIC = auto()  # Time-varying (cyclic)
    IMPULSE = auto()  # Short-duration transient
    THERMAL = auto()  # Temperature-induced
    COMBINED = auto()  # Multiple simultaneous loads


class ConsequenceLevel(Enum):
    """Consequence of failure categories."""

    NEGLIGIBLE = auto()  # Minor inconvenience
    MARGINAL = auto()  # Repairable damage
    CRITICAL = auto()  # Major damage, possible injury
    CATASTROPHIC = auto()  # Loss of life or total destruction


# =============================================================================
# Safety Factor Dataclass
# =============================================================================


@dataclass
class SafetyFactor:
    """Engineering safety factor with metadata.

    Attributes:
        value: Numeric safety factor value (must be >= 1.0)
        failure_mode: Type of failure this factor guards against
        load_type: Type of loading condition
        consequence: Consequence level if failure occurs
        source: Reference standard or source document
        notes: Additional design considerations
        uncertainty_factor: Factor accounting for material/load uncertainty
    """

    value: float
    failure_mode: FailureMode
    load_type: LoadType = LoadType.STATIC
    consequence: ConsequenceLevel = ConsequenceLevel.MARGINAL
    source: str = ""
    notes: str = ""
    uncertainty_factor: float = 1.0

    def __post_init__(self) -> None:
        """Validate safety factor."""
        if self.value < 1.0:
            raise ValueError(f"Safety factor must be >= 1.0, got {self.value}")

    @property
    def total_factor(self) -> float:
        """Total factor including uncertainty."""
        return self.value * self.uncertainty_factor

    def apply_to_allowable(self, allowable: float) -> float:
        """Compute design allowable from material allowable.

        Args:
            allowable: Material allowable stress/load

        Returns:
            Design allowable (allowable / SF)
        """
        return allowable / self.total_factor

    def check_margin(self, actual: float, allowable: float) -> tuple[bool, float]:
        """Check if design margin is satisfied.

        Args:
            actual: Actual stress/load
            allowable: Material allowable

        Returns:
            Tuple of (passes, margin_ratio)
        """
        design_allowable = self.apply_to_allowable(allowable)
        margin = design_allowable - actual
        ratio = actual / design_allowable if design_allowable > 0 else float("inf")
        return ratio <= 1.0, 1.0 - ratio


# =============================================================================
# Standard Safety Factors
# =============================================================================

# Pressure vessel design (ASME BPVC Section VIII Div. 1)
SF_PRESSURE_VESSEL_YIELD = SafetyFactor(
    value=1.5,
    failure_mode=FailureMode.YIELDING,
    load_type=LoadType.STATIC,
    consequence=ConsequenceLevel.CRITICAL,
    source="ASME BPVC Section VIII, Division 1",
    notes="Design stress = Sy/1.5 for yield-based design",
)

SF_PRESSURE_VESSEL_ULTIMATE = SafetyFactor(
    value=3.5,
    failure_mode=FailureMode.FRACTURE,
    load_type=LoadType.STATIC,
    consequence=ConsequenceLevel.CRITICAL,
    source="ASME BPVC Section VIII, Division 1",
    notes="Design stress = Su/3.5 for ultimate-based design",
)

# Fatigue design (FKM Guideline)
SF_FATIGUE_NORMAL = SafetyFactor(
    value=1.5,
    failure_mode=FailureMode.FATIGUE,
    load_type=LoadType.DYNAMIC,
    consequence=ConsequenceLevel.MARGINAL,
    source="FKM Richtlinie, 6th Edition",
    notes="For normal operation with well-characterized loads",
)

SF_FATIGUE_CRITICAL = SafetyFactor(
    value=2.0,
    failure_mode=FailureMode.FATIGUE,
    load_type=LoadType.DYNAMIC,
    consequence=ConsequenceLevel.CRITICAL,
    source="FKM Richtlinie, 6th Edition",
    notes="For safety-critical components",
)

# Buckling (Eurocode approach)
SF_BUCKLING = SafetyFactor(
    value=1.35,
    failure_mode=FailureMode.BUCKLING,
    load_type=LoadType.STATIC,
    consequence=ConsequenceLevel.MARGINAL,
    source="Eurocode 3, EN 1993-1-1",
    notes="Partial factor for permanent actions",
    uncertainty_factor=1.1,  # Additional factor for imperfections
)

# IC Engine specific safety factors
SF_PISTON_FATIGUE = SafetyFactor(
    value=2.0,
    failure_mode=FailureMode.FATIGUE,
    load_type=LoadType.DYNAMIC,
    consequence=ConsequenceLevel.CRITICAL,
    source="Heywood, IC Engine Fundamentals",
    notes="Piston crown fatigue under cyclic thermal and pressure loading",
)

SF_COMBUSTION_PEAK_PRESSURE = SafetyFactor(
    value=1.3,
    failure_mode=FailureMode.YIELDING,
    load_type=LoadType.IMPULSE,
    consequence=ConsequenceLevel.CRITICAL,
    source="SAE J1349",
    notes="Factor applied to predicted peak cylinder pressure",
)


# =============================================================================
# Design Margin Tracking
# =============================================================================


@dataclass
class DesignMargin:
    """Track design margin for a specific parameter.

    Attributes:
        name: Parameter name
        actual: Actual/calculated value
        allowable: Allowable limit
        sf: Applied safety factor
        unit: Unit of measurement
    """

    name: str
    actual: float
    allowable: float
    sf: SafetyFactor
    unit: str = ""

    @property
    def design_allowable(self) -> float:
        """Design allowable after safety factor."""
        return self.sf.apply_to_allowable(self.allowable)

    @property
    def utilization(self) -> float:
        """Utilization ratio (actual / design_allowable)."""
        if self.design_allowable <= 0:
            return float("inf")
        return self.actual / self.design_allowable

    @property
    def margin_percent(self) -> float:
        """Margin percentage (positive = within limit)."""
        return (1.0 - self.utilization) * 100.0

    @property
    def passes(self) -> bool:
        """Check if design margin is satisfied."""
        return self.utilization <= 1.0

    def summary(self) -> str:
        """Generate summary string."""
        status = "✓ PASS" if self.passes else "✗ FAIL"
        return (
            f"{self.name}: {self.actual:.3g} / {self.design_allowable:.3g} {self.unit} "
            f"({self.utilization:.1%} utilization, {self.margin_percent:+.1f}% margin) "
            f"[SF={self.sf.value}] {status}"
        )


class DesignMarginReport:
    """Collect and report design margins for a component."""

    def __init__(self, component_name: str) -> None:
        """Initialize report.

        Args:
            component_name: Name of component being analyzed
        """
        self.component_name = component_name
        self.margins: list[DesignMargin] = []

    def add_margin(
        self,
        name: str,
        actual: float,
        allowable: float,
        sf: SafetyFactor,
        unit: str = "",
    ) -> DesignMargin:
        """Add a design margin check.

        Args:
            name: Parameter name
            actual: Actual value
            allowable: Allowable limit
            sf: Safety factor to apply
            unit: Unit of measurement

        Returns:
            Created DesignMargin
        """
        margin = DesignMargin(name, actual, allowable, sf, unit)
        self.margins.append(margin)
        return margin

    @property
    def all_pass(self) -> bool:
        """Check if all margins pass."""
        return all(m.passes for m in self.margins)

    @property
    def critical_utilization(self) -> float:
        """Maximum utilization across all margins."""
        if not self.margins:
            return 0.0
        return max(m.utilization for m in self.margins)

    def generate_report(self) -> str:
        """Generate full text report."""
        lines = [
            "=" * 70,
            f"Design Margin Report: {self.component_name}",
            "=" * 70,
        ]

        # Sort by utilization (most critical first)
        sorted_margins = sorted(self.margins, key=lambda m: -m.utilization)

        for margin in sorted_margins:
            lines.append(margin.summary())

        lines.append("-" * 70)
        status = "ALL PASS" if self.all_pass else "FAILURES DETECTED"
        lines.append(f"Overall: {status} (Critical utilization: {self.critical_utilization:.1%})")
        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# Recommended Safety Factors by Application
# =============================================================================


def get_recommended_sf(
    failure_mode: FailureMode,
    consequence: ConsequenceLevel,
    load_type: LoadType = LoadType.STATIC,
) -> SafetyFactor:
    """Get recommended safety factor for given conditions.

    Args:
        failure_mode: Type of failure to guard against
        consequence: Consequence level if failure occurs
        load_type: Type of loading

    Returns:
        Recommended SafetyFactor
    """
    # Base factors by failure mode
    base_factors = {
        FailureMode.YIELDING: 1.5,
        FailureMode.FRACTURE: 2.5,
        FailureMode.FATIGUE: 1.5,
        FailureMode.BUCKLING: 1.35,
        FailureMode.CREEP: 2.0,
        FailureMode.WEAR: 1.25,
        FailureMode.CORROSION: 1.5,
    }

    # Consequence multipliers
    consequence_mult = {
        ConsequenceLevel.NEGLIGIBLE: 0.9,
        ConsequenceLevel.MARGINAL: 1.0,
        ConsequenceLevel.CRITICAL: 1.25,
        ConsequenceLevel.CATASTROPHIC: 1.5,
    }

    # Load type multipliers
    load_mult = {
        LoadType.STATIC: 1.0,
        LoadType.DYNAMIC: 1.2,
        LoadType.IMPULSE: 1.3,
        LoadType.THERMAL: 1.1,
        LoadType.COMBINED: 1.4,
    }

    base = base_factors.get(failure_mode, 2.0)
    cons_m = consequence_mult.get(consequence, 1.0)
    load_m = load_mult.get(load_type, 1.0)

    final_value = base * cons_m * load_m

    return SafetyFactor(
        value=round(final_value, 2),
        failure_mode=failure_mode,
        load_type=load_type,
        consequence=consequence,
        source="Derived from engineering best practices",
        notes=f"Base={base}, consequence_mult={cons_m}, load_mult={load_m}",
    )
