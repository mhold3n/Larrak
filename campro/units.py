"""Engineering units and typed physical constants.

All physical constants are documented with:
- SI units
- Source citation
- Validity range (if applicable)
- Uncertainty (if applicable)

This module provides the foundation for dimensional consistency and
traceability required by mechanical engineering industry standards.

Usage:
    from campro.units import PhysicalConstant, TOLERANCE_POSITION

    # Access value directly
    tol = TOLERANCE_POSITION.value

    # Get value with uncertainty bounds
    val, unc = TOLERANCE_POSITION.with_uncertainty()

    # Check validity range
    if TOLERANCE_POSITION.is_valid(x):
        ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstant:
    """Typed physical constant with engineering metadata.

    Attributes:
        value: Numerical value of the constant
        unit: SI unit string (e.g., "m", "kg/s", "J/(mol·K)")
        source: Citation or reference for the value
        uncertainty: Absolute uncertainty (same units as value)
        valid_range: (min, max) tuple for validity checking
        notes: Additional documentation

    Example:
        >>> STOICH_AFR_GASOLINE = PhysicalConstant(
        ...     value=14.7,
        ...     unit="kg_air/kg_fuel",
        ...     source="Heywood, ICE Fundamentals, Table 3.4",
        ...     uncertainty=0.1,
        ...     valid_range=(14.5, 14.9),
        ... )
    """

    value: float
    unit: str
    source: str
    uncertainty: float | None = None
    valid_range: tuple[float, float] | None = None
    notes: str = ""

    def with_uncertainty(self) -> tuple[float, float]:
        """Return (value, absolute_uncertainty)."""
        return self.value, self.uncertainty or 0.0

    def relative_uncertainty(self) -> float:
        """Return relative uncertainty as fraction (0-1)."""
        if self.uncertainty is None or self.value == 0:
            return 0.0
        return abs(self.uncertainty / self.value)

    def is_valid(self, test_value: float) -> bool:
        """Check if a value falls within the valid range."""
        if self.valid_range is None:
            return True
        return self.valid_range[0] <= test_value <= self.valid_range[1]

    def bounds(self) -> tuple[float, float]:
        """Return (lower, upper) bounds including uncertainty."""
        unc = self.uncertainty or 0.0
        return (self.value - unc, self.value + unc)

    def __float__(self) -> float:
        """Allow direct use in calculations."""
        return self.value

    def __mul__(self, other: float | int | PhysicalConstant) -> float:
        if isinstance(other, PhysicalConstant):
            return self.value * other.value
        return self.value * other

    def __rmul__(self, other: float | int) -> float:
        return self.value * other

    def __truediv__(self, other: float | int | PhysicalConstant) -> float:
        if isinstance(other, PhysicalConstant):
            return self.value / other.value
        return self.value / other

    def __rtruediv__(self, other: float | int) -> float:
        return other / self.value

    def __add__(self, other: float | int | PhysicalConstant) -> float:
        if isinstance(other, PhysicalConstant):
            return self.value + other.value
        return self.value + other

    def __radd__(self, other: float | int) -> float:
        return other + self.value

    def __sub__(self, other: float | int | PhysicalConstant) -> float:
        if isinstance(other, PhysicalConstant):
            return self.value - other.value
        return self.value - other

    def __rsub__(self, other: float | int) -> float:
        return other - self.value

    def __neg__(self) -> float:
        return -self.value

    def __abs__(self) -> float:
        return abs(self.value)

    def __lt__(self, other: float | int | PhysicalConstant) -> bool:
        if isinstance(other, PhysicalConstant):
            return self.value < other.value
        return self.value < other

    def __le__(self, other: float | int | PhysicalConstant) -> bool:
        if isinstance(other, PhysicalConstant):
            return self.value <= other.value
        return self.value <= other

    def __gt__(self, other: float | int | PhysicalConstant) -> bool:
        if isinstance(other, PhysicalConstant):
            return self.value > other.value
        return self.value > other

    def __ge__(self, other: float | int | PhysicalConstant) -> bool:
        if isinstance(other, PhysicalConstant):
            return self.value >= other.value
        return self.value >= other


# =============================================================================
# Mathematical Constants
# =============================================================================

PI = PhysicalConstant(
    value=math.pi,
    unit="dimensionless",
    source="Python math.pi (IEEE 754 double precision)",
    notes="Circle constant π",
)

DEG_TO_RAD = PhysicalConstant(
    value=math.pi / 180.0,
    unit="rad/deg",
    source="Definition: π/180",
    notes="Multiply degrees by this to get radians",
)

RAD_TO_DEG = PhysicalConstant(
    value=180.0 / math.pi,
    unit="deg/rad",
    source="Definition: 180/π",
    notes="Multiply radians by this to get degrees",
)


# =============================================================================
# Numerical Tolerances
# =============================================================================

TOLERANCE_GENERAL = PhysicalConstant(
    value=1e-8,
    unit="dimensionless",
    source="Internal: general-purpose numerical tolerance",
    notes="Used for equality checks and convergence criteria",
)

TOLERANCE_COLLOCATION = PhysicalConstant(
    value=1e-6,
    unit="dimensionless",
    source="Internal: collocation defect tolerance",
    notes="Relative tolerance for collocation constraint residuals",
)

TOLERANCE_GEOMETRY = PhysicalConstant(
    value=1e-6,
    unit="m",
    source="Internal: geometric tolerance",
    notes="Absolute position tolerance for profile closure checks",
)


# =============================================================================
# Thermodynamic Constants
# =============================================================================

R_UNIVERSAL = PhysicalConstant(
    value=8.31446261815324,
    unit="J/(mol·K)",
    source="CODATA 2018 - NIST",
    uncertainty=0.0,  # Exact by definition since 2019
    notes="Universal gas constant (exact value)",
)

STANDARD_PRESSURE = PhysicalConstant(
    value=101325.0,
    unit="Pa",
    source="IUPAC definition",
    notes="Standard atmospheric pressure (1 atm)",
)

STANDARD_TEMPERATURE = PhysicalConstant(
    value=298.15,
    unit="K",
    source="IUPAC definition (25°C)",
    notes="Standard reference temperature",
)

# Air Properties at Standard Conditions
R_AIR = PhysicalConstant(
    value=287.0,
    unit="J/(kg·K)",
    source="Derived: R_universal / M_air = 8.314 / 0.02897",
    uncertainty=0.1,
    notes="Specific gas constant for dry air (M_air = 28.97 g/mol)",
)

GAMMA_AIR = PhysicalConstant(
    value=1.4,
    unit="dimensionless",
    source="Heywood, ICE Fundamentals, Table A.1",
    uncertainty=0.01,
    valid_range=(1.35, 1.42),
    notes="Ratio of specific heats Cp/Cv for air at 300K; decreases with temperature",
)

CP_AIR = PhysicalConstant(
    value=1005.0,
    unit="J/(kg·K)",
    source="Heywood, ICE Fundamentals, Table A.1",
    uncertainty=5.0,
    valid_range=(1000.0, 1200.0),
    notes="Specific heat at constant pressure for air at 300K; increases with temperature",
)

CV_AIR = PhysicalConstant(
    value=718.0,
    unit="J/(kg·K)",
    source="Derived: Cp - R = 1005 - 287",
    uncertainty=5.0,
    notes="Specific heat at constant volume for air at 300K",
)

# Standard Gravity
G_STANDARD = PhysicalConstant(
    value=9.80665,
    unit="m/s²",
    source="CGPM 1901 - Standard gravity definition",
    uncertainty=0.0,  # Exact by definition
    notes="Standard acceleration due to gravity (exact value)",
)


# =============================================================================
# Combustion Properties
# =============================================================================

STOICH_AFR: dict[str, PhysicalConstant] = {
    "gasoline": PhysicalConstant(
        value=14.7,
        unit="kg_air/kg_fuel",
        source="Heywood, Internal Combustion Engine Fundamentals, Table 3.4",
        uncertainty=0.1,
        valid_range=(14.5, 14.9),
        notes="Stoichiometric AFR for typical gasoline (approximated as C8H18)",
    ),
    "diesel": PhysicalConstant(
        value=14.5,
        unit="kg_air/kg_fuel",
        source="Heywood, ICE Fundamentals, Ch. 3",
        uncertainty=0.2,
        valid_range=(14.3, 14.7),
        notes="Stoichiometric AFR for typical diesel fuel",
    ),
    "natural_gas": PhysicalConstant(
        value=17.2,
        unit="kg_air/kg_fuel",
        source="Heywood, ICE Fundamentals, Ch. 3",
        uncertainty=0.3,
        valid_range=(16.8, 17.6),
        notes="Stoichiometric AFR for natural gas (primarily CH4)",
    ),
    "hydrogen": PhysicalConstant(
        value=34.3,
        unit="kg_air/kg_fuel",
        source="Heywood, ICE Fundamentals, Ch. 3",
        uncertainty=0.1,
        valid_range=(34.0, 34.6),
        notes="Stoichiometric AFR for hydrogen (H2)",
    ),
}

LOWER_HEATING_VALUE: dict[str, PhysicalConstant] = {
    "gasoline": PhysicalConstant(
        value=43.0e6,
        unit="J/kg",
        source="Heywood, ICE Fundamentals, Table 3.3",
        uncertainty=1.0e6,
        valid_range=(42.0e6, 44.0e6),
        notes="Lower heating value at 25°C",
    ),
    "diesel": PhysicalConstant(
        value=42.5e6,
        unit="J/kg",
        source="Heywood, ICE Fundamentals, Table 3.3",
        uncertainty=1.0e6,
        valid_range=(41.5e6, 43.5e6),
        notes="Lower heating value at 25°C",
    ),
    "natural_gas": PhysicalConstant(
        value=50.0e6,
        unit="J/kg",
        source="Heywood, ICE Fundamentals, Table 3.3",
        uncertainty=2.0e6,
        valid_range=(47.0e6, 53.0e6),
        notes="Lower heating value - varies with composition",
    ),
    "hydrogen": PhysicalConstant(
        value=120.0e6,
        unit="J/kg",
        source="NIST Chemistry WebBook",
        uncertainty=0.5e6,
        notes="Lower heating value at 25°C",
    ),
}


# =============================================================================
# Fluid Properties (Reference Conditions)
# =============================================================================

DEFAULT_DISCHARGE_COEFFICIENT = PhysicalConstant(
    value=0.7,
    unit="dimensionless",
    source="Internal: typical valve discharge coefficient",
    uncertainty=0.1,
    valid_range=(0.5, 0.9),
    notes="Cd for sharp-edged orifice flow; actual value depends on geometry",
)


# =============================================================================
# Gear Geometry Constants (Litvin)
# =============================================================================

DEFAULT_PRESSURE_ANGLE = PhysicalConstant(
    value=20.0,
    unit="deg",
    source="AGMA 2001-D04 (standard pressure angle)",
    valid_range=(12.0, 35.0),
    notes="Standard pressure angle for involute gears",
)

MIN_PRESSURE_ANGLE = PhysicalConstant(
    value=12.0,
    unit="deg",
    source="AGMA 2001-D04 (minimum practical)",
    notes="Below this, risk of undercutting increases",
)

MAX_PRESSURE_ANGLE = PhysicalConstant(
    value=35.0,
    unit="deg",
    source="AGMA 2001-D04 (maximum practical)",
    notes="Above this, bearing loads become excessive",
)


# =============================================================================
# CasADi Integration Constants
# =============================================================================

CASADI_EPSILON = PhysicalConstant(
    value=1e-12,
    unit="dimensionless",
    source="Internal: CasADi domain guard",
    notes="Epsilon for sqrt, division, and other potentially singular operations",
)

CASADI_ASIN_CLAMP = PhysicalConstant(
    value=0.999999,
    unit="dimensionless",
    source="Internal: arcsin domain guard",
    notes="Clamp value for arcsin input to avoid domain errors at ±1",
)


# =============================================================================
# Helper Functions
# =============================================================================


def get_stoich_afr(fuel_type: str) -> PhysicalConstant:
    """Get stoichiometric AFR for a fuel type.

    Args:
        fuel_type: Fuel name (gasoline, diesel, natural_gas, hydrogen)

    Returns:
        PhysicalConstant with AFR value and metadata

    Raises:
        KeyError: If fuel type not recognized
    """
    fuel_key = fuel_type.lower()
    if fuel_key not in STOICH_AFR:
        valid = ", ".join(STOICH_AFR.keys())
        raise KeyError(f"Unknown fuel type '{fuel_type}'. Valid options: {valid}")
    return STOICH_AFR[fuel_key]


def get_lhv(fuel_type: str) -> PhysicalConstant:
    """Get lower heating value for a fuel type.

    Args:
        fuel_type: Fuel name (gasoline, diesel, natural_gas, hydrogen)

    Returns:
        PhysicalConstant with LHV value and metadata

    Raises:
        KeyError: If fuel type not recognized
    """
    fuel_key = fuel_type.lower()
    if fuel_key not in LOWER_HEATING_VALUE:
        valid = ", ".join(LOWER_HEATING_VALUE.keys())
        raise KeyError(f"Unknown fuel type '{fuel_type}'. Valid options: {valid}")
    return LOWER_HEATING_VALUE[fuel_key]


# =============================================================================
# Fuel Property Aliases (for direct import)
# =============================================================================

LHV_GASOLINE = LOWER_HEATING_VALUE["gasoline"]
STOICH_AFR_GASOLINE = STOICH_AFR["gasoline"]
