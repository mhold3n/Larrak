"""
CEM Rule Framework: Extensible validation rules for constraint enforcement.

This module defines the base abstractions for validation rules that can be
evaluated during optimization. Rules are organized by category and can be
registered dynamically via the rule registry.

Categories:
- THERMODYNAMIC: Pressure, temperature, efficiency limits
- MECHANICAL: Stress, fatigue, gear contact forces
- TRIBOLOGICAL: Film thickness, wear, scuffing protection
- MANUFACTURING: Tolerance stack-ups, surface finish, cost gates
- EMISSIONS: Regulatory compliance (NOx, CO, PM)
- ACOUSTIC: NVH and noise limits
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuleCategory(Enum):
    """Categories for grouping validation rules."""

    THERMODYNAMIC = "thermo"  # P_max, T_max, efficiency limits
    MECHANICAL = "mech"  # Stress, fatigue, gear contact
    TRIBOLOGICAL = "tribo"  # Film thickness, wear, scuffing
    MANUFACTURING = "mfg"  # Tolerance, surface finish, cost
    EMISSIONS = "emissions"  # NOx, CO, PM limits
    ACOUSTIC = "acoustic"  # NVH, noise limits


class RuleSeverity(Enum):
    """Severity level when rule is violated."""

    INFO = 0  # Informational only
    WARN = 1  # Warning, optimization can continue
    ERROR = 2  # Error, result may be suboptimal
    FATAL = 3  # Fatal, optimization should abort


@dataclass
class RuleResult:
    """Result of evaluating a single rule."""

    rule_name: str
    passed: bool
    margin: float = 0.0  # Positive = within limit, negative = violated
    message: str = ""
    severity: RuleSeverity = RuleSeverity.INFO
    context: dict[str, Any] = field(default_factory=dict)


class RuleBase(ABC):
    """
    Abstract base class for all CEM validation rules.

    Subclasses must implement:
    - category: The rule's category for filtering
    - name: Human-readable rule identifier
    - evaluate: Logic to check the rule against a context
    """

    category: RuleCategory
    name: str
    default_severity: RuleSeverity = RuleSeverity.ERROR

    @abstractmethod
    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        """
        Evaluate this rule against the given context.

        Args:
            context: Dictionary containing relevant state/parameters

        Returns:
            RuleResult with pass/fail status and margin
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(category={self.category.value})"


# =============================================================================
# THERMODYNAMIC RULES (Stubs)
# =============================================================================


class MaxCrownTemperatureRule(RuleBase):
    """Checks piston crown temperature stays below material limit."""

    category = RuleCategory.THERMODYNAMIC
    name = "max_crown_temperature"

    def __init__(self, limit_k: float = 573.0):  # 300°C default
        self.limit_k = limit_k

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("MaxCrownTemperatureRule scheduled for future sprint")


class MaxCylinderPressureRule(RuleBase):
    """Checks peak cylinder pressure stays within structural limits."""

    category = RuleCategory.THERMODYNAMIC
    name = "max_cylinder_pressure"

    def __init__(self, limit_bar: float = 200.0):
        self.limit_bar = limit_bar

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("MaxCylinderPressureRule scheduled for future sprint")


class MinThermalEfficiencyRule(RuleBase):
    """Checks brake thermal efficiency meets minimum target."""

    category = RuleCategory.THERMODYNAMIC
    name = "min_thermal_efficiency"

    def __init__(self, min_efficiency: float = 0.35):
        self.min_efficiency = min_efficiency

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("MinThermalEfficiencyRule scheduled for future sprint")


# =============================================================================
# MECHANICAL RULES (Stubs)
# =============================================================================


class MaxContactStressRule(RuleBase):
    """Checks gear contact stress stays below Hertzian limit."""

    category = RuleCategory.MECHANICAL
    name = "max_contact_stress"

    def __init__(self, limit_mpa: float = 1500.0):
        self.limit_mpa = limit_mpa

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("MaxContactStressRule scheduled for future sprint")


class FatigueSafetyFactorRule(RuleBase):
    """Checks fatigue safety factor exceeds minimum."""

    category = RuleCategory.MECHANICAL
    name = "fatigue_safety_factor"

    def __init__(self, min_sf: float = 2.0):
        self.min_sf = min_sf

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("FatigueSafetyFactorRule scheduled for future sprint")


# =============================================================================
# TRIBOLOGICAL RULES (Stubs)
# =============================================================================


class MinFilmThicknessRule(RuleBase):
    """Checks EHL film thickness ratio (lambda) exceeds minimum."""

    category = RuleCategory.TRIBOLOGICAL
    name = "min_film_thickness"

    def __init__(self, min_lambda: float = 1.5):
        self.min_lambda = min_lambda

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("MinFilmThicknessRule scheduled for future sprint")


class ScuffingProtectionRule(RuleBase):
    """Checks flash temperature stays below scuffing threshold."""

    category = RuleCategory.TRIBOLOGICAL
    name = "scuffing_protection"

    def __init__(self, max_flash_temp_k: float = 423.0):  # 150°C
        self.max_flash_temp_k = max_flash_temp_k

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("ScuffingProtectionRule scheduled for future sprint")


# =============================================================================
# MANUFACTURING RULES (Stubs)
# =============================================================================


class CycleToleranceRule(RuleBase):
    """Checks motion profile is within manufacturing tolerance."""

    category = RuleCategory.MANUFACTURING
    name = "cycle_tolerance"

    def __init__(self, tolerance_mm: float = 0.01):
        self.tolerance_mm = tolerance_mm

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("CycleToleranceRule scheduled for future sprint")


class SurfaceFinishRule(RuleBase):
    """Checks required surface finish is achievable."""

    category = RuleCategory.MANUFACTURING
    name = "surface_finish"

    def __init__(self, max_ra_um: float = 0.4):
        self.max_ra_um = max_ra_um

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("SurfaceFinishRule scheduled for future sprint")


# =============================================================================
# EMISSIONS RULES (Stubs)
# =============================================================================


class NOxEmissionRule(RuleBase):
    """Checks NOx emissions stay within regulatory limit."""

    category = RuleCategory.EMISSIONS
    name = "nox_emission"

    def __init__(self, limit_g_kwh: float = 0.4):  # Euro VI approx
        self.limit_g_kwh = limit_g_kwh

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("NOxEmissionRule scheduled for future sprint")


class PMEmissionRule(RuleBase):
    """Checks particulate matter emissions stay within limit."""

    category = RuleCategory.EMISSIONS
    name = "pm_emission"

    def __init__(self, limit_g_kwh: float = 0.01):
        self.limit_g_kwh = limit_g_kwh

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("PMEmissionRule scheduled for future sprint")


# =============================================================================
# ACOUSTIC RULES (Stubs)
# =============================================================================


class MaxNoiseRule(RuleBase):
    """Checks gear noise stays within NVH target."""

    category = RuleCategory.ACOUSTIC
    name = "max_noise"

    def __init__(self, limit_dba: float = 75.0):
        self.limit_dba = limit_dba

    def evaluate(self, context: dict[str, Any]) -> RuleResult:
        raise NotImplementedError("MaxNoiseRule scheduled for future sprint")


__all__ = [
    # Enums
    "RuleCategory",
    "RuleSeverity",
    # Base classes
    "RuleBase",
    "RuleResult",
    # Thermodynamic
    "MaxCrownTemperatureRule",
    "MaxCylinderPressureRule",
    "MinThermalEfficiencyRule",
    # Mechanical
    "MaxContactStressRule",
    "FatigueSafetyFactorRule",
    # Tribological
    "MinFilmThicknessRule",
    "ScuffingProtectionRule",
    # Manufacturing
    "CycleToleranceRule",
    "SurfaceFinishRule",
    # Emissions
    "NOxEmissionRule",
    "PMEmissionRule",
    # Acoustic
    "MaxNoiseRule",
]
