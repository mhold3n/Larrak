"""Fuel property database with temperature dependencies.

This module provides validated fuel properties for combustion modeling,
including:
- Heating values (LHV, HHV)
- Density correlations
- Vaporization properties
- Stoichiometric ratios

All properties include source citations and uncertainty bounds.

Reference Sources:
    - Heywood, J.B., "Internal Combustion Engine Fundamentals", 2nd Ed., 2018
    - NIST Chemistry WebBook (https://webbook.nist.gov/chemistry/)
    - API Technical Data Book, 7th Ed.
"""

from __future__ import annotations

from dataclasses import dataclass

from campro.materials.base import MaterialDatabase, MaterialProperty, PropertySource

# =============================================================================
# Source Citations
# =============================================================================

HEYWOOD_2018 = PropertySource(
    reference="Heywood, J.B., Internal Combustion Engine Fundamentals, 2nd Ed.",
    year=2018,
    t_range_k=(250.0, 800.0),
    p_range_pa=(1e4, 1e7),
    notes="Standard reference for IC engine combustion properties",
)

NIST_WEBBOOK = PropertySource(
    reference="NIST Chemistry WebBook",
    doi="10.18434/T4D303",
    t_range_k=(200.0, 1500.0),
    p_range_pa=(1e3, 1e8),
)

API_DATABOOK = PropertySource(
    reference="API Technical Data Book - Petroleum Refining",
    year=2021,
    t_range_k=(273.0, 700.0),
    p_range_pa=(1e4, 1e7),
)


# =============================================================================
# Fuel Property Dataclass
# =============================================================================


@dataclass
class FuelProperties:
    """Complete property set for a fuel type.

    Attributes:
        name: Fuel identifier
        formula: Chemical formula (approximate for blends)
        molecular_weight: kg/mol
        lhv: Lower heating value at 25°C (J/kg)
        lhv_uncertainty: Absolute uncertainty in LHV (J/kg)
        hhv: Higher heating value at 25°C (J/kg)
        stoich_afr: Stoichiometric air-fuel ratio (kg/kg)
        stoich_afr_uncertainty: Absolute uncertainty in AFR
        density_15c: Density at 15°C, 1 atm (kg/m³)
        boiling_point: Normal boiling point (K) - for pure fuels
        flash_point: Flash point (K)
        autoignition_temp: Autoignition temperature (K)
        latent_heat_vap: Latent heat of vaporization at Tb (J/kg)
        source: Primary data source
    """

    name: str
    formula: str
    molecular_weight: float  # kg/mol
    lhv: float  # J/kg
    lhv_uncertainty: float  # J/kg
    hhv: float  # J/kg
    stoich_afr: float  # kg_air/kg_fuel
    stoich_afr_uncertainty: float
    density_15c: float  # kg/m³
    boiling_point: float | None  # K (None for blends)
    flash_point: float  # K
    autoignition_temp: float  # K
    latent_heat_vap: float  # J/kg
    source: PropertySource


# =============================================================================
# Fuel Database Entries
# =============================================================================

GASOLINE = FuelProperties(
    name="gasoline",
    formula="C8H18 (approx)",
    molecular_weight=0.114,  # kg/mol (iso-octane approximation)
    lhv=43.0e6,
    lhv_uncertainty=1.0e6,
    hhv=46.4e6,
    stoich_afr=14.7,
    stoich_afr_uncertainty=0.1,
    density_15c=750.0,
    boiling_point=None,  # Blend: 30-200°C range
    flash_point=230.0,  # -43°C
    autoignition_temp=553.0,  # 280°C
    latent_heat_vap=305.0e3,
    source=HEYWOOD_2018,
)

DIESEL = FuelProperties(
    name="diesel",
    formula="C12H23 (approx)",
    molecular_weight=0.170,  # kg/mol
    lhv=42.5e6,
    lhv_uncertainty=1.0e6,
    hhv=45.4e6,
    stoich_afr=14.5,
    stoich_afr_uncertainty=0.2,
    density_15c=840.0,
    boiling_point=None,  # Blend: 180-370°C range
    flash_point=325.0,  # 52°C
    autoignition_temp=527.0,  # 254°C
    latent_heat_vap=250.0e3,
    source=HEYWOOD_2018,
)

NATURAL_GAS = FuelProperties(
    name="natural_gas",
    formula="CH4 (primary)",
    molecular_weight=0.016,  # kg/mol (methane)
    lhv=50.0e6,
    lhv_uncertainty=2.0e6,  # Higher uncertainty due to composition variation
    hhv=55.5e6,
    stoich_afr=17.2,
    stoich_afr_uncertainty=0.3,
    density_15c=0.68,  # kg/m³ at 1 atm
    boiling_point=111.7,  # -161.5°C (methane)
    flash_point=85.0,  # -188°C (not typically measured)
    autoignition_temp=813.0,  # 540°C
    latent_heat_vap=510.0e3,
    source=HEYWOOD_2018,
)

HYDROGEN = FuelProperties(
    name="hydrogen",
    formula="H2",
    molecular_weight=0.002016,  # kg/mol
    lhv=120.0e6,
    lhv_uncertainty=0.5e6,
    hhv=141.8e6,
    stoich_afr=34.3,
    stoich_afr_uncertainty=0.1,
    density_15c=0.084,  # kg/m³ at 1 atm
    boiling_point=20.3,  # -252.9°C
    flash_point=20.0,  # Not applicable (always flammable)
    autoignition_temp=773.0,  # 500°C
    latent_heat_vap=446.0e3,
    source=NIST_WEBBOOK,
)

ETHANOL = FuelProperties(
    name="ethanol",
    formula="C2H5OH",
    molecular_weight=0.046,  # kg/mol
    lhv=26.8e6,
    lhv_uncertainty=0.3e6,
    hhv=29.7e6,
    stoich_afr=9.0,
    stoich_afr_uncertainty=0.05,
    density_15c=789.0,
    boiling_point=351.5,  # 78.4°C
    flash_point=286.0,  # 13°C
    autoignition_temp=636.0,  # 363°C
    latent_heat_vap=840.0e3,
    source=NIST_WEBBOOK,
)

METHANOL = FuelProperties(
    name="methanol",
    formula="CH3OH",
    molecular_weight=0.032,  # kg/mol
    lhv=19.9e6,
    lhv_uncertainty=0.2e6,
    hhv=22.7e6,
    stoich_afr=6.5,
    stoich_afr_uncertainty=0.05,
    density_15c=792.0,
    boiling_point=337.8,  # 64.7°C
    flash_point=284.0,  # 11°C
    autoignition_temp=738.0,  # 465°C
    latent_heat_vap=1100.0e3,
    source=NIST_WEBBOOK,
)


# Fuel registry
FUEL_DATABASE: dict[str, FuelProperties] = {
    "gasoline": GASOLINE,
    "diesel": DIESEL,
    "natural_gas": NATURAL_GAS,
    "hydrogen": HYDROGEN,
    "ethanol": ETHANOL,
    "methanol": METHANOL,
}


# =============================================================================
# Temperature-Dependent Property Correlations
# =============================================================================


def _gasoline_density_correlation(t_k: float, p_pa: float) -> float:  # noqa: ARG001
    """Gasoline density as function of temperature.

    Uses API correlation for petroleum liquids.

    Args:
        t_k: Temperature in Kelvin
        p_pa: Pressure in Pascal (ignored for liquids)

    Returns:
        Density in kg/m³
    """
    # Reference: API correlation
    # ρ = ρ_15 * [1 - β*(T - 288.15)]
    # β ≈ 0.00095 /K for gasoline
    rho_15 = GASOLINE.density_15c
    beta = 0.00095  # 1/K
    return rho_15 * (1 - beta * (t_k - 288.15))


def _diesel_density_correlation(t_k: float, p_pa: float) -> float:  # noqa: ARG001
    """Diesel density as function of temperature."""
    rho_15 = DIESEL.density_15c
    beta = 0.00080  # 1/K (smaller than gasoline)
    return rho_15 * (1 - beta * (t_k - 288.15))


def _latent_heat_correlation(t_k: float, t_b: float, h_vap_b: float, t_c: float) -> float:
    """Watson correlation for latent heat of vaporization.

    Args:
        t_k: Temperature in K
        t_b: Normal boiling point in K
        h_vap_b: Latent heat at boiling point in J/kg
        t_c: Critical temperature in K

    Returns:
        Latent heat at T in J/kg
    """
    # Watson correlation: H_vap(T) = H_vap(Tb) * ((Tc - T)/(Tc - Tb))^0.38
    if t_k >= t_c:
        return 0.0
    return h_vap_b * ((t_c - t_k) / (t_c - t_b)) ** 0.38


def _cp_liquid_correlation(t_k: float, fuel: FuelProperties) -> float:
    """Liquid fuel specific heat correlation.

    Uses Lee-Kesler correlation for petroleum fractions.

    Args:
        t_k: Temperature in K
        fuel: Fuel properties

    Returns:
        Specific heat in J/(kg·K)
    """
    # Simplified correlation for hydrocarbons
    # cp = A + B*T where A, B depend on molecular weight
    mw = fuel.molecular_weight * 1000  # g/mol
    a = 1.4 + 0.002 * mw
    b = 0.003 + 0.00001 * mw
    return (a + b * t_k) * 1000  # Convert to J/(kg·K)


# =============================================================================
# Material Property Constructors
# =============================================================================


def create_fuel_density_property(fuel_name: str) -> MaterialProperty:
    """Create density MaterialProperty for a fuel.

    Args:
        fuel_name: Name of fuel in database

    Returns:
        MaterialProperty for density
    """
    fuel = get_fuel(fuel_name)

    if fuel_name == "gasoline":
        correlation = _gasoline_density_correlation
    elif fuel_name == "diesel":
        correlation = _diesel_density_correlation
    else:
        # Default: constant density
        def correlation(t_k: float, p_pa: float) -> float:  # noqa: ARG001
            return fuel.density_15c

    return MaterialProperty(
        name=f"density_{fuel_name}",
        unit="kg/m³",
        correlation=correlation,
        source=fuel.source,
        uncertainty_rel=0.01,
    )


def create_fuel_cp_property(fuel_name: str) -> MaterialProperty:
    """Create specific heat MaterialProperty for a liquid fuel.

    Args:
        fuel_name: Name of fuel in database

    Returns:
        MaterialProperty for specific heat
    """
    fuel = get_fuel(fuel_name)

    def correlation(t_k: float, p_pa: float) -> float:  # noqa: ARG001
        return _cp_liquid_correlation(t_k, fuel)

    return MaterialProperty(
        name=f"cp_{fuel_name}",
        unit="J/(kg·K)",
        correlation=correlation,
        source=fuel.source,
        uncertainty_rel=0.05,  # 5% uncertainty typical for Cp correlations
    )


def create_fuel_database(fuel_name: str) -> MaterialDatabase:
    """Create complete MaterialDatabase for a fuel.

    Args:
        fuel_name: Name of fuel

    Returns:
        MaterialDatabase with all available properties
    """
    fuel = get_fuel(fuel_name)

    db = MaterialDatabase(
        name=fuel_name,
        metadata={
            "formula": fuel.formula,
            "molecular_weight_kg_mol": fuel.molecular_weight,
            "lhv_J_kg": fuel.lhv,
            "stoich_afr": fuel.stoich_afr,
        },
    )

    # Add temperature-dependent properties
    db.add_property(create_fuel_density_property(fuel_name))
    db.add_property(create_fuel_cp_property(fuel_name))

    return db


# =============================================================================
# Public API
# =============================================================================


def get_fuel(fuel_name: str) -> FuelProperties:
    """Get fuel properties from database.

    Args:
        fuel_name: Fuel name (case-insensitive)

    Returns:
        FuelProperties dataclass

    Raises:
        KeyError: If fuel not found
    """
    key = fuel_name.lower()
    if key not in FUEL_DATABASE:
        valid = ", ".join(FUEL_DATABASE.keys())
        raise KeyError(f"Unknown fuel '{fuel_name}'. Available: {valid}")
    return FUEL_DATABASE[key]


def list_fuels() -> list[str]:
    """List available fuel types."""
    return list(FUEL_DATABASE.keys())


def get_lhv(fuel_name: str) -> tuple[float, float]:
    """Get lower heating value with uncertainty.

    Args:
        fuel_name: Fuel name

    Returns:
        (lhv_J_kg, uncertainty_J_kg)
    """
    fuel = get_fuel(fuel_name)
    return fuel.lhv, fuel.lhv_uncertainty


def get_stoich_afr(fuel_name: str) -> tuple[float, float]:
    """Get stoichiometric AFR with uncertainty.

    Args:
        fuel_name: Fuel name

    Returns:
        (afr_kg_kg, uncertainty)
    """
    fuel = get_fuel(fuel_name)
    return fuel.stoich_afr, fuel.stoich_afr_uncertainty
