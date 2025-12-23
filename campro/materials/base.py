"""Base classes for temperature/pressure-dependent material properties.

This module provides the foundation for thermophysical property modeling
with engineering-grade metadata and validity checking.

Example:
    >>> from campro.materials.base import MaterialProperty, PropertySource
    >>>
    >>> source = PropertySource(
    ...     reference="NIST Chemistry WebBook",
    ...     doi="10.18434/T4D303",
    ...     t_range_k=(200.0, 6000.0),
    ...     p_range_pa=(1e3, 1e8),
    ... )
    >>>
    >>> def cp_air(t_k: float, p_pa: float) -> float:
    ...     # NASA polynomial or correlation
    ...     return 1005.0 + 0.1 * (t_k - 300.0)
    >>>
    >>> cp = MaterialProperty(
    ...     name="specific_heat_cp",
    ...     unit="J/(kg·K)",
    ...     correlation=cp_air,
    ...     source=source,
    ...     uncertainty_rel=0.02,
    ... )
    >>>
    >>> value = cp(500.0)  # Evaluate at 500 K, 1 atm
    >>> value, unc = cp.with_uncertainty(500.0)  # With uncertainty
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PropertySource:
    """Citation and validity metadata for material property data.

    Attributes:
        reference: Full citation string (author, title, year)
        doi: Digital Object Identifier if available
        year: Publication year
        t_range_k: Valid temperature range (K)
        p_range_pa: Valid pressure range (Pa)
        notes: Additional notes or caveats
    """

    reference: str
    doi: str | None = None
    year: int = 0
    t_range_k: tuple[float, float] = (200.0, 6000.0)
    p_range_pa: tuple[float, float] = (1e3, 1e8)
    notes: str = ""

    def is_valid_conditions(self, t_k: float, p_pa: float) -> bool:
        """Check if T, P are within valid range."""
        t_ok = self.t_range_k[0] <= t_k <= self.t_range_k[1]
        p_ok = self.p_range_pa[0] <= p_pa <= self.p_range_pa[1]
        return t_ok and p_ok

    def format_citation(self) -> str:
        """Format as a bibliographic citation."""
        cite = self.reference
        if self.year:
            cite = f"{cite} ({self.year})"
        if self.doi:
            cite = f"{cite}, DOI: {self.doi}"
        return cite


# Type for property correlation functions
PropertyCorrelation = Callable[[float, float], float]


@dataclass
class MaterialProperty:
    """Temperature/pressure-dependent material property with uncertainty.

    Supports evaluation at given conditions with automatic range validation
    and uncertainty propagation.

    Attributes:
        name: Property name (e.g., "specific_heat_cp")
        unit: SI unit string
        correlation: Function f(t_k, p_pa) -> property_value
        source: PropertySource with citation and validity range
        uncertainty_rel: Relative uncertainty (0-1)
        extrapolation_warning: If True, warn when extrapolating
    """

    name: str
    unit: str
    correlation: PropertyCorrelation
    source: PropertySource
    uncertainty_rel: float = 0.0
    extrapolation_warning: bool = True

    def __call__(
        self,
        t_k: float,
        p_pa: float = 101325.0,
        validate: bool = True,
    ) -> float:
        """Evaluate property at temperature T and pressure P.

        Args:
            t_k: Temperature in Kelvin
            p_pa: Pressure in Pascal (default: 1 atm)
            validate: If True, check validity range and warn

        Returns:
            Property value at given conditions

        Raises:
            ValueError: If conditions are outside validity range and
                extrapolation is not allowed
        """
        if validate:
            self._validate_range(t_k, p_pa)
        return self.correlation(t_k, p_pa)

    def with_uncertainty(
        self,
        t_k: float,
        p_pa: float = 101325.0,
    ) -> tuple[float, float]:
        """Return (value, absolute_uncertainty) at given conditions."""
        val = self(t_k, p_pa)
        return val, abs(val * self.uncertainty_rel)

    def bounds(
        self,
        t_k: float,
        p_pa: float = 101325.0,
    ) -> tuple[float, float]:
        """Return (lower_bound, upper_bound) including uncertainty."""
        val, unc = self.with_uncertainty(t_k, p_pa)
        return (val - unc, val + unc)

    def _validate_range(self, t_k: float, p_pa: float) -> None:
        """Check if conditions are within validity range."""
        if not self.source.is_valid_conditions(t_k, p_pa):
            t_min, t_max = self.source.t_range_k
            p_min, p_max = self.source.p_range_pa

            msg = (
                f"{self.name}: Conditions T={t_k:.1f}K, P={p_pa:.0f}Pa "
                f"outside valid range T=[{t_min:.0f}, {t_max:.0f}]K, "
                f"P=[{p_min:.0e}, {p_max:.0e}]Pa"
            )

            if self.extrapolation_warning:
                warnings.warn(msg, UserWarning, stacklevel=3)
            else:
                raise ValueError(msg)

    def __repr__(self) -> str:
        return (
            f"MaterialProperty({self.name!r}, unit={self.unit!r}, "
            f"uncertainty={self.uncertainty_rel:.1%})"
        )


@dataclass
class MaterialDatabase:
    """Collection of material properties for a substance.

    Provides a convenient container for related properties of a single
    material (e.g., all thermophysical properties of air).

    Attributes:
        name: Material name
        properties: Dict mapping property names to MaterialProperty objects
        metadata: Additional material metadata
    """

    name: str
    properties: dict[str, MaterialProperty] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_property(self, prop: MaterialProperty) -> None:
        """Add a property to the database."""
        self.properties[prop.name] = prop

    def get(
        self,
        prop_name: str,
        t_k: float,
        p_pa: float = 101325.0,
    ) -> float:
        """Get property value at given conditions.

        Args:
            prop_name: Property name (e.g., "specific_heat_cp")
            t_k: Temperature in Kelvin
            p_pa: Pressure in Pascal

        Returns:
            Property value

        Raises:
            KeyError: If property not found
        """
        if prop_name not in self.properties:
            available = ", ".join(self.properties.keys())
            raise KeyError(
                f"Property '{prop_name}' not found for {self.name}. Available: {available}"
            )
        return self.properties[prop_name](t_k, p_pa)

    def get_with_uncertainty(
        self,
        prop_name: str,
        t_k: float,
        p_pa: float = 101325.0,
    ) -> tuple[float, float]:
        """Get property value and uncertainty at given conditions."""
        if prop_name not in self.properties:
            raise KeyError(f"Property '{prop_name}' not found for {self.name}")
        return self.properties[prop_name].with_uncertainty(t_k, p_pa)

    def list_properties(self) -> list[str]:
        """List available property names."""
        return list(self.properties.keys())

    def __repr__(self) -> str:
        props = ", ".join(self.properties.keys())
        return f"MaterialDatabase({self.name!r}, properties=[{props}])"


# =============================================================================
# NASA Polynomial Support (for gas properties)
# =============================================================================


@dataclass
class NASAPolynomialCoeffs:
    """NASA 7-coefficient polynomial for thermodynamic properties.

    Standard form for Cp/R, H/RT, S/R as functions of temperature.
    Two sets of coefficients: low-T (200-1000K) and high-T (1000-6000K).

    Reference:
        McBride, B.J. and Gordon, S., "Computer Program for Calculation
        of Complex Chemical Equilibrium Compositions and Applications",
        NASA RP-1311, 1996.
    """

    species: str
    t_mid: float  # Transition temperature (typically 1000 K)

    # High temperature coefficients (t_mid to t_high)
    a_high: tuple[float, float, float, float, float, float, float]
    t_high: float  # Upper limit (typically 6000 K)

    # Low temperature coefficients (t_low to t_mid)
    a_low: tuple[float, float, float, float, float, float, float]
    t_low: float  # Lower limit (typically 200 K)

    source: PropertySource | None = None

    def cp_over_r(self, t: float) -> float:
        """Compute Cp/R at temperature T."""
        a = self._select_coeffs(t)
        return a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3 + a[4] * t**4

    def h_over_rt(self, t: float) -> float:
        """Compute H/(R*T) at temperature T."""
        a = self._select_coeffs(t)
        return a[0] + a[1] * t / 2 + a[2] * t**2 / 3 + a[3] * t**3 / 4 + a[4] * t**4 / 5 + a[5] / t

    def s_over_r(self, t: float) -> float:
        """Compute S/R at temperature T."""
        a = self._select_coeffs(t)
        import math

        return (
            a[0] * math.log(t)
            + a[1] * t
            + a[2] * t**2 / 2
            + a[3] * t**3 / 3
            + a[4] * t**4 / 4
            + a[6]
        )

    def _select_coeffs(self, t: float) -> tuple[float, float, float, float, float, float, float]:
        """Select appropriate coefficient set based on temperature."""
        if t < self.t_low or t > self.t_high:
            warnings.warn(
                f"{self.species}: T={t:.1f}K outside valid range "
                f"[{self.t_low:.0f}, {self.t_high:.0f}]K",
                UserWarning,
            )

        if t >= self.t_mid:
            return self.a_high
        return self.a_low


def create_nasa_cp_property(
    coeffs: NASAPolynomialCoeffs,
    molecular_weight: float,  # kg/mol
    uncertainty_rel: float = 0.01,
) -> MaterialProperty:
    """Create a Cp MaterialProperty from NASA polynomial coefficients.

    Args:
        coeffs: NASA polynomial coefficients
        molecular_weight: Molecular weight in kg/mol
        uncertainty_rel: Relative uncertainty (default 1%)

    Returns:
        MaterialProperty for specific heat Cp
    """
    from campro.units import R_UNIVERSAL

    r = R_UNIVERSAL.value  # J/(mol·K)

    def cp_correlation(t_k: float, p_pa: float) -> float:  # noqa: ARG001
        """Cp in J/(kg·K)."""
        # Cp/R from polynomial, convert to J/(kg·K)
        cp_over_r = coeffs.cp_over_r(t_k)
        return cp_over_r * r / molecular_weight

    source = coeffs.source or PropertySource(
        reference=f"NASA polynomials for {coeffs.species}",
        t_range_k=(coeffs.t_low, coeffs.t_high),
    )

    return MaterialProperty(
        name=f"cp_{coeffs.species}",
        unit="J/(kg·K)",
        correlation=cp_correlation,
        source=source,
        uncertainty_rel=uncertainty_rel,
    )
