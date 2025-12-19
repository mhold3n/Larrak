"""Base abstractions for Larrak Physics Simulations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

from pydantic import BaseModel, Field

class SimulationConfig(BaseModel):
    """Configuration container for any simulation."""
    dt: float = Field(..., gt=0.0)
    t_end: float = Field(..., gt=0.0)
    params: Dict[str, Any] = Field(default_factory=dict)

class BaseSimulation(ABC):
    """Abstract Base Class for all physics simulations (FEA, CFD, Kinetics)."""
    
    def __init__(self, name: str, config: SimulationConfig):
        self.name = name
        self.config = config
        self.results: Dict[str, Any] = {}
        self.t = 0.0
        
    @abstractmethod
    def setup(self):
        """Initialize solvers, meshes, and boundary conditions."""
        pass
    
    @abstractmethod
    def step(self, dt: float):
        """Advance the simulation by one time step."""
        pass
        
    def solve(self):
        """Run the full simulation loop."""
        self.setup()
        print(f"[{self.name}] Starting simulation...")
        while self.t < self.config.t_end:
            self.step(self.config.dt)
            self.t += self.config.dt
        print(f"[{self.name}] Simulation Complete.")

    @abstractmethod
    def post_process(self) -> Dict[str, Any]:
        """Process and return results."""
        return self.results
