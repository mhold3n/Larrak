"""
IO Schema for Larrak Phase 4 Simulations.
Defines the strict contract between the 0D Optimizer and 3D Solvers.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator


# --- 1. Geometry Inputs ---
class GeometryConfig(BaseModel):
    """Geometric parameters defining the engine."""

    bore: float = Field(..., gt=0.0, description="Cylinder bore [m]")
    stroke: float = Field(..., gt=0.0, description="Piston stroke [m]")
    conrod: float = Field(..., gt=0.0, description="Connecting rod length [m]")
    compression_ratio: float = Field(..., gt=1.0, description="Geometric CR")
    # Derived/Detailed params for FEA
    liner_thickness: float = 0.005  # [m]
    piston_clearance: float = 50e-6  # [m]


# --- 2. Operating Conditions ---
class OperatingPoint(BaseModel):
    """Steady-state operating conditions."""

    rpm: float = Field(..., gt=0.0)
    lambda_val: float = Field(1.0, description="Lambda (Air/Fuel Ratio)")
    p_intake: float = Field(..., gt=0.0, description="Intake Manifold Pressure [Pa]")
    T_intake: float = Field(..., gt=0.0, description="Intake Manifold Temperature [K]")
    T_coolant: float = Field(360.0, description="Coolant Temperature [K]")
    T_oil: float = Field(360.0, description="Oil Temperature [K]")


# --- 3. Boundary Conditions (Time-Resolved) ---
class BoundaryConditions(BaseModel):
    """Cycle-resolved data arrays (Functions of Crank Angle)."""

    crank_angle: List[float] = Field(..., description="Crank angle array [deg or rad]")
    pressure_gas: List[float] = Field(..., description="Cylinder Pressure [Pa]")
    temperature_gas: List[float] = Field(..., description="Bulk Gas Temperature [K]")
    heat_transfer_coeff: Optional[List[float]] = Field(None, description="Woschni HTC [W/m2K]")
    piston_speed: Optional[List[float]] = Field(None, description="Piston Velocity [m/s]")

    @model_validator(mode="before")
    @classmethod
    def check_lengths(cls, values):
        """Ensure all arrays are the same length."""
        ca = values.get("crank_angle")
        pg = values.get("pressure_gas")
        if ca and pg and len(ca) != len(pg):
            raise ValueError(f"Length Mismatch: CA={len(ca)}, Pressure={len(pg)}")
        return values


# --- 4. Main Input Bundle ---
class SimulationInput(BaseModel):
    """Complete Input Package for a Phase 4 Simulation Run."""

    run_id: str = Field(..., description="Unique Hash ID for this run")
    solver_settings: Dict[str, Any] = Field(default_factory=dict)
    geometry: GeometryConfig
    operating_point: OperatingPoint
    boundary_conditions: BoundaryConditions


# --- 5. Output Bundle ---
class SimulationOutput(BaseModel):
    """Standardized Output from any Phase 4 Solver."""

    run_id: str
    success: bool = True
    # Scalar Results (Feasibility Gates)
    T_crown_max: Optional[float] = None
    T_liner_max: Optional[float] = None
    max_von_mises: Optional[float] = None
    friction_fmep: Optional[float] = None
    # Fitted Calibration Maps (e.g., {'A': 0.5, 'B': 0.01})
    calibration_params: Dict[str, float] = Field(default_factory=dict)
    # Metrics
    fit_residual: float = 0.0
    computation_time: float = 0.0
