from dataclasses import dataclass


@dataclass
class GearGeometry:
    """Fixed Geometric Parameters for Conjugate Gear Set."""
    ratio: float = 2.0      # Target Gear Ratio (Ring/Planet)
    
    # Envelope Constraints (mm)
    max_radius: float = 80.0
    min_radius: float = 20.0
    
    # Optimization Weights (Penalty factors)
    w_tracking: float = 1.0  # Motion Law Tracking
    w_smooth: float = 0.5    # Acceleration Smoothness
    w_slip: float = 10.0     # Slip Ratio regularity
    w_curvature: float = 5.0 # Curvature continuity
    
    # Solver
    n_points: int = 360      # Grid Resolution (1 deg)

class Phase3Config:
    """Master Configuration for Phase 3 Adaptive Loop."""
    gear = GearGeometry()
    
    # Validation Thresholds
    tracking_error_margin_deg: float = 0.5 
    interference_tolerance: float = 1e-4

# Global Instance
GEAR_CONFIG = Phase3Config()
