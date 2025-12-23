"""Gas property correlations using NASA polynomials.

This module provides thermodynamic properties for common gases used in
combustion simulations, based on NASA 7-coefficient polynomials.

Reference Sources:
    - McBride, B.J., Gordon, S., "Computer Program for Calculation of
      Complex Chemical Equilibrium Compositions and Applications",
      NASA RP-1311, 1996.
    - Burcat, A., Ruscic, B., "Third Millennium Ideal Gas and Condensed
      Phase Thermochemical Database", ANL-05/20, 2005.

Properties available:
    - Specific heat at constant pressure (Cp)
    - Enthalpy (H)
    - Entropy (S)
    - Gibbs free energy (G)
"""

from __future__ import annotations

import math

from campro.materials.base import (
    MaterialDatabase,
    MaterialProperty,
    NASAPolynomialCoeffs,
    PropertySource,
)

# =============================================================================
# Source Citations
# =============================================================================

NASA_RP1311 = PropertySource(
    reference="McBride & Gordon, NASA RP-1311",
    year=1996,
    t_range_k=(200.0, 6000.0),
    p_range_pa=(1e3, 1e8),
    notes="NASA Glenn thermodynamic database",
)

BURCAT_ANL = PropertySource(
    reference="Burcat & Ruscic, Third Millennium Database, ANL-05/20",
    year=2005,
    t_range_k=(200.0, 6000.0),
    p_range_pa=(1e3, 1e8),
)


# =============================================================================
# NASA Polynomial Coefficients
# =============================================================================

# Air (approximated as 79% N2 + 21% O2 by volume)
# Using property-weighted average approach
N2_COEFFS = NASAPolynomialCoeffs(
    species="N2",
    t_mid=1000.0,
    a_high=(
        2.92664000e00,
        1.48797700e-03,
        -5.68476100e-07,
        1.00970400e-10,
        -6.75335100e-15,
        -9.22797700e02,
        5.98052800e00,
    ),
    t_high=6000.0,
    a_low=(
        3.29867700e00,
        1.40824000e-03,
        -3.96322200e-06,
        5.64151500e-09,
        -2.44485500e-12,
        -1.02090000e03,
        3.95037200e00,
    ),
    t_low=200.0,
    source=NASA_RP1311,
)

O2_COEFFS = NASAPolynomialCoeffs(
    species="O2",
    t_mid=1000.0,
    a_high=(
        3.28253800e00,
        1.48308800e-03,
        -7.57966700e-07,
        2.09470600e-10,
        -2.16717200e-14,
        -1.08845800e03,
        5.45323100e00,
    ),
    t_high=6000.0,
    a_low=(
        3.21293600e00,
        1.12748600e-03,
        -5.75615000e-07,
        1.31387700e-09,
        -8.76855400e-13,
        -1.00524900e03,
        6.03473800e00,
    ),
    t_low=200.0,
    source=NASA_RP1311,
)

CO2_COEFFS = NASAPolynomialCoeffs(
    species="CO2",
    t_mid=1000.0,
    a_high=(
        4.63659500e00,
        2.74131900e-03,
        -9.95828500e-07,
        1.60373000e-10,
        -9.16103200e-15,
        -4.90249300e04,
        -1.93534900e00,
    ),
    t_high=6000.0,
    a_low=(
        2.35677400e00,
        8.98459700e-03,
        -7.12356300e-06,
        2.45919000e-09,
        -1.43699500e-13,
        -4.83719700e04,
        9.90105200e00,
    ),
    t_low=200.0,
    source=NASA_RP1311,
)

H2O_COEFFS = NASAPolynomialCoeffs(
    species="H2O",
    t_mid=1000.0,
    a_high=(
        2.67214600e00,
        3.05629300e-03,
        -8.73026000e-07,
        1.20099600e-10,
        -6.39161800e-15,
        -2.98992100e04,
        6.86281700e00,
    ),
    t_high=6000.0,
    a_low=(
        3.38684200e00,
        3.47498200e-03,
        -6.35469600e-06,
        6.96858100e-09,
        -2.50658800e-12,
        -3.02081100e04,
        2.59023300e00,
    ),
    t_low=200.0,
    source=NASA_RP1311,
)

H2_COEFFS = NASAPolynomialCoeffs(
    species="H2",
    t_mid=1000.0,
    a_high=(
        2.99142300e00,
        7.00064400e-04,
        -5.63382900e-08,
        -9.23157800e-12,
        1.58275200e-15,
        -8.35034000e02,
        -1.35511000e00,
    ),
    t_high=6000.0,
    a_low=(
        3.29812400e00,
        8.24944200e-04,
        -8.14301500e-07,
        -9.47543400e-11,
        4.13487200e-13,
        -1.01252100e03,
        -3.29409400e00,
    ),
    t_low=200.0,
    source=NASA_RP1311,
)

CO_COEFFS = NASAPolynomialCoeffs(
    species="CO",
    t_mid=1000.0,
    a_high=(
        3.02507800e00,
        1.44268900e-03,
        -5.63082800e-07,
        1.01858100e-10,
        -6.91095200e-15,
        -1.42683500e04,
        6.10821800e00,
    ),
    t_high=6000.0,
    a_low=(
        3.26245200e00,
        1.51194100e-03,
        -3.88175500e-06,
        5.58194400e-09,
        -2.47495100e-12,
        -1.43105400e04,
        4.84889700e00,
    ),
    t_low=200.0,
    source=NASA_RP1311,
)

CH4_COEFFS = NASAPolynomialCoeffs(
    species="CH4",
    t_mid=1000.0,
    a_high=(
        1.65326200e00,
        1.00263100e-02,
        -3.31661200e-06,
        5.36483100e-10,
        -3.14696900e-14,
        -1.00095900e04,
        9.90506300e00,
    ),
    t_high=6000.0,
    a_low=(
        5.14987600e00,
        -1.36709100e-02,
        4.91800400e-05,
        -4.84743000e-08,
        1.66693900e-11,
        -1.02465900e04,
        -4.64130400e00,
    ),
    t_low=200.0,
    source=NASA_RP1311,
)

# Registry of gas coefficients
GAS_COEFFS: dict[str, NASAPolynomialCoeffs] = {
    "N2": N2_COEFFS,
    "O2": O2_COEFFS,
    "CO2": CO2_COEFFS,
    "H2O": H2O_COEFFS,
    "H2": H2_COEFFS,
    "CO": CO_COEFFS,
    "CH4": CH4_COEFFS,
}

# Molecular weights in kg/mol
MOLECULAR_WEIGHTS: dict[str, float] = {
    "N2": 0.028014,
    "O2": 0.031999,
    "CO2": 0.044009,
    "H2O": 0.018015,
    "H2": 0.002016,
    "CO": 0.028010,
    "CH4": 0.016043,
    "air": 0.028964,  # Standard dry air
}


# =============================================================================
# Universal Gas Constant
# =============================================================================

R_UNIVERSAL = 8.31446261815324  # J/(mol·K) - CODATA 2018 exact value


# =============================================================================
# Property Calculations
# =============================================================================


def get_cp(species: str, t_k: float) -> float:
    """Get specific heat at constant pressure.

    Args:
        species: Gas species name (N2, O2, CO2, H2O, etc.)
        t_k: Temperature in Kelvin

    Returns:
        Cp in J/(kg·K)
    """
    if species == "air":
        # Air as 79% N2 + 21% O2 by mole fraction
        # For mixture: Cp_mix = (y_N2 * Cp_N2 * MW_N2 + y_O2 * Cp_O2 * MW_O2) / MW_air
        # where y is mole fraction and Cp is in J/(mol·K)
        cp_n2_mass = get_cp("N2", t_k)  # J/(kg·K)
        cp_o2_mass = get_cp("O2", t_k)  # J/(kg·K)

        # Convert to molar basis
        cp_n2_mol = cp_n2_mass * MOLECULAR_WEIGHTS["N2"]  # J/(mol·K)
        cp_o2_mol = cp_o2_mass * MOLECULAR_WEIGHTS["O2"]  # J/(mol·K)

        # Molar average
        cp_mix_mol = 0.79 * cp_n2_mol + 0.21 * cp_o2_mol  # J/(mol·K)

        # Convert back to mass basis
        return cp_mix_mol / MOLECULAR_WEIGHTS["air"]  # J/(kg·K)

    if species not in GAS_COEFFS:
        raise KeyError(f"Unknown species '{species}'. Available: {list(GAS_COEFFS.keys())}")

    coeffs = GAS_COEFFS[species]
    mw = MOLECULAR_WEIGHTS[species]

    # Cp/R from NASA polynomial
    cp_over_r = coeffs.cp_over_r(t_k)

    # Convert to J/(kg·K)
    return cp_over_r * R_UNIVERSAL / mw


def get_enthalpy(species: str, t_k: float) -> float:
    """Get specific enthalpy.

    Args:
        species: Gas species name
        t_k: Temperature in Kelvin

    Returns:
        Enthalpy in J/kg (relative to 298.15 K reference)
    """
    if species not in GAS_COEFFS:
        raise KeyError(f"Unknown species '{species}'")

    coeffs = GAS_COEFFS[species]
    mw = MOLECULAR_WEIGHTS[species]

    # H/(R*T) from NASA polynomial
    h_over_rt = coeffs.h_over_rt(t_k)

    # Convert to J/kg
    return h_over_rt * R_UNIVERSAL * t_k / mw


def get_entropy(species: str, t_k: float, p_pa: float = 101325.0) -> float:
    """Get specific entropy.

    Args:
        species: Gas species name
        t_k: Temperature in Kelvin
        p_pa: Pressure in Pascal

    Returns:
        Entropy in J/(kg·K)
    """
    if species not in GAS_COEFFS:
        raise KeyError(f"Unknown species '{species}'")

    coeffs = GAS_COEFFS[species]
    mw = MOLECULAR_WEIGHTS[species]

    # S/R from NASA polynomial (at standard pressure)
    s_over_r = coeffs.s_over_r(t_k)

    # Pressure correction: S = S° - R*ln(P/P°)
    p_ref = 101325.0  # Standard pressure
    s_over_r -= math.log(p_pa / p_ref)

    # Convert to J/(kg·K)
    return s_over_r * R_UNIVERSAL / mw


def get_gamma(species: str, t_k: float) -> float:
    """Get ratio of specific heats (gamma = Cp/Cv).

    Args:
        species: Gas species name
        t_k: Temperature in Kelvin

    Returns:
        Gamma (dimensionless)
    """
    if species not in MOLECULAR_WEIGHTS:
        raise KeyError(f"Unknown species '{species}'")

    mw = MOLECULAR_WEIGHTS[species]
    cp = get_cp(species, t_k)

    # For ideal gas: Cv = Cp - R/M
    r_specific = R_UNIVERSAL / mw  # J/(kg·K)
    cv = cp - r_specific

    return cp / cv


# =============================================================================
# MaterialProperty Constructors
# =============================================================================


def create_gas_cp_property(species: str) -> MaterialProperty:
    """Create Cp MaterialProperty for a gas species.

    Args:
        species: Gas species name

    Returns:
        MaterialProperty for specific heat
    """
    if species not in GAS_COEFFS and species != "air":
        raise KeyError(f"Unknown species '{species}'")

    source = GAS_COEFFS.get(species, N2_COEFFS).source if species != "air" else NASA_RP1311

    def correlation(t_k: float, p_pa: float) -> float:  # noqa: ARG001
        return get_cp(species, t_k)

    return MaterialProperty(
        name=f"cp_{species}",
        unit="J/(kg·K)",
        correlation=correlation,
        source=source,
        uncertainty_rel=0.01,  # 1% uncertainty for NASA polynomials
    )


def create_gas_gamma_property(species: str) -> MaterialProperty:
    """Create gamma (Cp/Cv) MaterialProperty for a gas species.

    Args:
        species: Gas species name

    Returns:
        MaterialProperty for gamma
    """
    source = GAS_COEFFS.get(species, N2_COEFFS).source if species != "air" else NASA_RP1311

    def correlation(t_k: float, p_pa: float) -> float:  # noqa: ARG001
        return get_gamma(species, t_k)

    return MaterialProperty(
        name=f"gamma_{species}",
        unit="dimensionless",
        correlation=correlation,
        source=source,
        uncertainty_rel=0.01,
    )


def create_gas_database(species: str) -> MaterialDatabase:
    """Create complete MaterialDatabase for a gas species.

    Args:
        species: Gas species name

    Returns:
        MaterialDatabase with thermodynamic properties
    """
    db = MaterialDatabase(
        name=species,
        metadata={
            "molecular_weight_kg_mol": MOLECULAR_WEIGHTS.get(species, 0.0),
            "type": "ideal_gas",
        },
    )

    db.add_property(create_gas_cp_property(species))
    db.add_property(create_gas_gamma_property(species))

    return db


# =============================================================================
# Public API
# =============================================================================


def list_gases() -> list[str]:
    """List available gas species."""
    return list(GAS_COEFFS.keys()) + ["air"]


def get_molecular_weight(species: str) -> float:
    """Get molecular weight in kg/mol."""
    if species not in MOLECULAR_WEIGHTS:
        raise KeyError(f"Unknown species '{species}'. Available: {list(MOLECULAR_WEIGHTS.keys())}")
    return MOLECULAR_WEIGHTS[species]
