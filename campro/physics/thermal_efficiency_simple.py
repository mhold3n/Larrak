"""
Simplified thermal efficiency model for FPE optimization.

This module implements physics-based thermal efficiency calculations
with key constraints from free-piston engine literature, using
simplified models that are suitable for optimization while
maintaining physical fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from casadi import *

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class ThermalEfficiencyConfig:
    """Configuration for thermal efficiency model."""

    # Physical constants
    gamma: float = 1.4  # Specific heat ratio
    clearance: float = 0.002  # 2mm clearance
    R_gas: float = 287.0  # Gas constant (J/kg/K)

    # Efficiency targets from FPE literature
    efficiency_target: float = 0.55  # 55% target from FPE studies
    compression_ratio_range: tuple[float, float] = (20.0, 70.0)

    # Heat transfer parameters (simplified Woschni correlation)
    # Scaled down to ensure penalties don't dominate Otto efficiency (typically 0.5-0.7)
    heat_transfer_coeff: float = 0.01
    woschni_factor: float = 0.05

    # Mechanical loss parameters
    friction_coeff: float = 0.01
    viscous_damping: float = 0.01


class SimplifiedThermalModel:
    """
    Simplified thermal efficiency model for FPE optimization.

    Implements key physics from FPE literature:
    - Otto cycle efficiency with variable compression ratio
    - Heat loss modeling (simplified Woschni correlation)
    - Mechanical losses
    - Pressure rate constraints
    """

    def __init__(self, config: ThermalEfficiencyConfig | None = None):
        """
        Initialize thermal efficiency model.

        Parameters
        ----------
        config : Optional[ThermalEfficiencyConfig]
            Model configuration
        """
        self.config = config or ThermalEfficiencyConfig()

        log.info(
            f"Initialized SimplifiedThermalModel: "
            f"efficiency_target={self.config.efficiency_target}, "
            f"CR_range={self.config.compression_ratio_range}",
        )

    def compute_compression_ratio(
        self, position: Any, clearance: float | None = None,
    ) -> Any:
        """
        Compute compression ratio from piston position.

        Parameters
        ----------
        position : Any
            Piston position (CasADi variable or numpy array)
        clearance : Optional[float]
            Clearance distance (uses config default if None)

        Returns
        -------
        Any
            Compression ratio
        """
        if clearance is None:
            clearance = self.config.clearance

        # CR = V_max / V_min = (stroke + clearance) / clearance
        return (position + clearance) / clearance

    def compute_otto_efficiency(self, compression_ratio: Any) -> Any:
        """
        Compute Otto cycle thermal efficiency.

        Parameters
        ----------
        compression_ratio : Any
            Compression ratio (CasADi variable or numpy array)

        Returns
        -------
        Any
            Otto cycle efficiency: 1 - 1/CR^(γ-1)
        """
        return 1 - 1 / (compression_ratio ** (self.config.gamma - 1))

    def compute_heat_loss_penalty(self, velocity: Any, position: Any) -> Any:
        """
        Compute heat loss penalty using simplified Woschni correlation.

        Parameters
        ----------
        velocity : Any
            Piston velocity in per-degree units (m/deg)
            Note: Coefficients are scaled for per-degree units
        position : Any
            Piston position (m)

        Returns
        -------
        Any
            Heat loss penalty
        """
        # Simplified Woschni correlation
        # Heat transfer coefficient proportional to velocity
        heat_transfer = self.config.heat_transfer_coeff * velocity**2

        # Additional penalty for high velocity at TDC (more heat loss)
        tdc_penalty = self.config.woschni_factor * velocity**2 * (1 - position)

        return heat_transfer + tdc_penalty

    def compute_mechanical_losses(self, velocity: Any, acceleration: Any) -> Any:
        """
        Compute mechanical losses.

        Parameters
        ----------
        velocity : Any
            Piston velocity in per-degree units (m/deg)
            Note: Coefficients are scaled for per-degree units
        acceleration : Any
            Piston acceleration in per-degree units (m/deg²)
            Note: Coefficients are scaled for per-degree units

        Returns
        -------
        Any
            Mechanical loss penalty
        """
        # Friction losses (proportional to velocity squared)
        friction_loss = self.config.friction_coeff * velocity**2

        # Viscous damping (proportional to acceleration)
        viscous_loss = self.config.viscous_damping * acceleration**2

        return friction_loss + viscous_loss

    def compute_thermal_efficiency(
        self, position: Any, velocity: Any, acceleration: Any,
    ) -> Any:
        """
        Compute total thermal efficiency.

        Parameters
        ----------
        position : Any
            Piston position in meters (m)
        velocity : Any
            Piston velocity in per-degree units (m/deg)
        acceleration : Any
            Piston acceleration in per-degree units (m/deg²)

        Returns
        -------
        Any
            Total thermal efficiency
        """
        # Compression ratio
        cr = self.compute_compression_ratio(position)

        # Otto cycle efficiency
        otto_eff = self.compute_otto_efficiency(cr)

        # Heat loss penalty
        heat_loss = self.compute_heat_loss_penalty(velocity, position)

        # Mechanical losses
        mech_loss = self.compute_mechanical_losses(velocity, acceleration)

        # Total efficiency
        total_eff = otto_eff - heat_loss - mech_loss

        return total_eff

    def add_compression_ratio_constraints(self, opti: Any, position: Any) -> None:
        """
        Add compression ratio constraints to optimization problem.
        
        Enforces limits at extremes (BDC and TDC) only, not at every point.
        At BDC: position[0] = 0, so cr = 1.0 (minimum)
        At TDC: position[-1] = stroke, so cr = (stroke + clearance)/clearance (maximum)

        Parameters
        ----------
        opti : Any
            CasADi Opti stack
        position : Any
            Piston position variable (array or CasADi variable)
        """
        # Compression ratio limits from FPE literature
        cr_min, cr_max = self.config.compression_ratio_range
        clearance = self.config.clearance
        
        # Compute CR at TDC (maximum displacement)
        # position[-1] is TDC (maximum displacement, enforce maximum limit here)
        # The geometric constraint (position[-1] == stroke) already determines the achievable CR
        # No minimum CR constraint needed - geometry determines the minimum achievable CR
        cr_end = self.compute_compression_ratio(position[-1], clearance)  # TDC: maximum CR
        
        # Enforce maximum CR limit only (geometry determines minimum)
        opti.subject_to(cr_end <= cr_max)  # Maximum CR enforced at TDC
        
        # For interior points, only enforce upper bound to prevent over-compression mid-stroke
        # Since cr(theta) is monotonic between endpoints, the TDC check guarantees
        # the entire trace stays within [1.0, cr_end]
        if hasattr(position, '__len__') and len(position) > 2:
            for i in range(1, len(position) - 1):  # Skip endpoints
                cr = self.compute_compression_ratio(position[i], clearance)
                opti.subject_to(cr <= cr_max)  # Prevent over-compression
                opti.subject_to(cr >= 1.0)  # Physical lower bound (always satisfied at BDC)

    def add_pressure_rate_constraints(
        self, opti: Any, acceleration: Any, dtheta: float, max_rate: float = 1000.0,
    ) -> None:
        """
        Add pressure rate constraints to avoid diesel knock.

        Parameters
        ----------
        opti : Any
            CasADi Opti stack
        acceleration : Any
            Piston acceleration variable in per-degree units (m/deg²)
        dtheta : float
            Angular step size (deg) - note: changed from dt to dtheta for per-degree formulation
        max_rate : float
            Maximum pressure rate (Pa/ms)
        """
        # Pressure rate constraint (simplified)
        # Limit acceleration rate to avoid excessive pressure rise
        # Note: acceleration is in per-degree units, so rate is computed per degree
        # For typical engine speeds, need to convert to per-second rate
        # Assuming ~360 deg/s: rate_per_s = rate_per_deg * 360
        for i in range(len(acceleration) - 1):
            pressure_rate_per_deg = abs(acceleration[i + 1] - acceleration[i]) / dtheta
            # Convert to per-second rate (assuming 360 deg/s typical)
            pressure_rate_per_s = pressure_rate_per_deg * 360.0
            opti.subject_to(pressure_rate_per_s <= max_rate)

    def add_temperature_constraints(
        self, opti: Any, velocity: Any, max_temp_rise: float = 1000.0,
    ) -> None:
        """
        Add temperature rise constraints.

        Parameters
        ----------
        opti : Any
            CasADi Opti stack
        velocity : Any
            Piston velocity variable in per-degree units (m/deg)
        max_temp_rise : float
            Maximum temperature rise (K)
        """
        # Simplified temperature constraint
        # Limit velocity to control temperature rise
        # Note: velocity is in per-degree units, so constraint is scaled accordingly
        # For typical engine speeds (~360 deg/s), 1 m/deg ≈ 360 m/s
        # Constraint scaled to work with per-degree units
        velocity_limit = max_temp_rise / 3600.0  # Scaled for per-degree units
        for v in velocity:
            opti.subject_to(abs(v) <= velocity_limit)

    def compute_efficiency_objective(
        self, position: Any, velocity: Any, acceleration: Any,
    ) -> Any:
        """
        Compute thermal efficiency objective for optimization.

        Parameters
        ----------
        position : Any
            Piston position (CasADi variable or numpy array)
        velocity : Any
            Piston velocity (CasADi variable or numpy array)
        acceleration : Any
            Piston acceleration (CasADi variable or numpy array)

        Returns
        -------
        Any
            Thermal efficiency objective (negative for minimization)
        """
        # Compute thermal efficiency
        thermal_eff = self.compute_thermal_efficiency(position, velocity, acceleration)

        # Return negative efficiency for minimization
        return -thermal_eff

    def add_physics_constraints(
        self, opti: Any, position: Any, velocity: Any, acceleration: Any, dtheta: float,
    ) -> None:
        """
        Add all physics constraints to optimization problem.

        Parameters
        ----------
        opti : Any
            CasADi Opti stack
        position : Any
            Piston position variable (m)
        velocity : Any
            Piston velocity variable in per-degree units (m/deg)
        acceleration : Any
            Piston acceleration variable in per-degree units (m/deg²)
        dtheta : float
            Angular step size (deg) - note: changed from dt to dtheta for per-degree formulation
        """
        # Compression ratio constraints
        self.add_compression_ratio_constraints(opti, position)

        # Pressure rate constraints (using angular step)
        self.add_pressure_rate_constraints(opti, acceleration, dtheta)

        # Temperature constraints
        self.add_temperature_constraints(opti, velocity)

    def evaluate_efficiency(
        self, position: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate thermal efficiency for given motion profile.

        Parameters
        ----------
        position : np.ndarray
            Position profile
        velocity : np.ndarray
            Velocity profile
        acceleration : np.ndarray
            Acceleration profile

        Returns
        -------
        Dict[str, float]
            Efficiency metrics
        """
        # Compute compression ratio
        cr = self.compute_compression_ratio(position)

        # Otto cycle efficiency
        otto_eff = self.compute_otto_efficiency(cr)

        # Heat loss penalty
        heat_loss = self.compute_heat_loss_penalty(velocity, position)

        # Mechanical losses
        mech_loss = self.compute_mechanical_losses(velocity, acceleration)

        # Total efficiency
        total_eff = otto_eff - heat_loss - mech_loss

        return {
            "compression_ratio": float(np.mean(cr)),
            "otto_efficiency": float(np.mean(otto_eff)),
            "heat_loss_penalty": float(np.mean(heat_loss)),
            "mechanical_loss": float(np.mean(mech_loss)),
            "total_efficiency": float(np.mean(total_eff)),
            "efficiency_target": self.config.efficiency_target,
            "efficiency_achieved": float(np.mean(total_eff))
            >= self.config.efficiency_target,
        }

    def get_efficiency_target(self) -> float:
        """Get the efficiency target from configuration."""
        return self.config.efficiency_target

    def update_config(self, **kwargs) -> None:
        """
        Update model configuration.

        Parameters
        ----------
        **kwargs
            Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                log.debug(f"Updated {key} = {value}")
            else:
                log.warning(f"Unknown configuration parameter: {key}")
