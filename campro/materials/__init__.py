"""Materials subpackage for temperature/pressure-dependent properties.

This package provides validated material property data with:
- Temperature and pressure dependencies
- Source citations and validity ranges
- Uncertainty quantification
- Integration with campro physics models

Modules:
    base: Base classes for material properties
    fuels: Fuel property database (planned)
    gases: Gas property correlations (planned)
"""

from __future__ import annotations

from campro.materials.base import (
    MaterialDatabase,
    MaterialProperty,
    NASAPolynomialCoeffs,
    PropertySource,
    create_nasa_cp_property,
)

__all__ = [
    "MaterialDatabase",
    "MaterialProperty",
    "NASAPolynomialCoeffs",
    "PropertySource",
    "create_nasa_cp_property",
]
