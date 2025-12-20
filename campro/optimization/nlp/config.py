from dataclasses import dataclass, field
import numpy as np

@dataclass
class EngineGeometry:
    """Fixed Geometric Parameters defining the Physical Target."""
    bore: float = 0.1      # Meters
    stroke: float = 0.2    # Meters
    cr: float = 15.0       # Compression Ratio
    conrod: float = 0.4    # Meters
    
    # Valve Timings (Fixed for Phase 1)
    intake_open: float = float(np.radians(340.0))
    intake_dur: float = float(np.radians(220.0))
    exhaust_open: float = float(np.radians(140.0))
    exhaust_dur: float = float(np.radians(220.0))

@dataclass
class SimulationRanges:
    """User-Defined Scope for Motion Study."""
    # User Request: RPM 10-10000
    rpm_min: float = 10.0
    rpm_max: float = 10000.0
    rpm_steps: int = 15  # Moderate resolution for test
    
    # User Request: Air 10mg - 12000mg
    # Mapped to Boost roughly 0.01 Bar to 8.0 Bar
    boost_min: float = 0.01 # Bar ( Vacuum)
    boost_max: float = 8.0  # Bar (High Boost)
    boost_steps: int = 15
    
    # User Request: Fuel 1mg - 500mg
    fuel_min: float = 1.0 # mg
    fuel_max: float = 500.0 # mg
    fuel_steps: int = 15

    # Logic for filtering (0.7 - 1.5 Lambda)
    lambda_min: float = 0.7
    lambda_max: float = 1.5


class Phase1Config:
    """Master Configuration for Phase 1 Adaptive Loop."""
    geometry = EngineGeometry()
    ranges = SimulationRanges()
    
    # Validation Thresholds
    error_margin_percent: float = 5.0 # Recursion Trigger
    
    @property
    def rpm_grid(self):
        return np.linspace(self.ranges.rpm_min, self.ranges.rpm_max, self.ranges.rpm_steps)
        
    @property
    def boost_grid(self):
        return np.linspace(self.ranges.boost_min, self.ranges.boost_max, self.ranges.boost_steps)

    @property
    def fuel_grid(self):
        return np.linspace(self.ranges.fuel_min, self.ranges.fuel_max, self.ranges.fuel_steps)

# Global Instance
CONFIG = Phase1Config()
