"""Design utilities for mechanical engineering calculations.

Provides safety factors, design margins, and structural analysis tools.
"""

from campro.design.safety import (
    SF_BUCKLING,
    SF_COMBUSTION_PEAK_PRESSURE,
    SF_FATIGUE_CRITICAL,
    SF_FATIGUE_NORMAL,
    SF_PISTON_FATIGUE,
    SF_PRESSURE_VESSEL_ULTIMATE,
    SF_PRESSURE_VESSEL_YIELD,
    ConsequenceLevel,
    DesignMargin,
    DesignMarginReport,
    FailureMode,
    LoadType,
    SafetyFactor,
    get_recommended_sf,
)

__all__ = [
    "ConsequenceLevel",
    "DesignMargin",
    "DesignMarginReport",
    "FailureMode",
    "LoadType",
    "SafetyFactor",
    "SF_BUCKLING",
    "SF_COMBUSTION_PEAK_PRESSURE",
    "SF_FATIGUE_CRITICAL",
    "SF_FATIGUE_NORMAL",
    "SF_PISTON_FATIGUE",
    "SF_PRESSURE_VESSEL_ULTIMATE",
    "SF_PRESSURE_VESSEL_YIELD",
    "get_recommended_sf",
]
