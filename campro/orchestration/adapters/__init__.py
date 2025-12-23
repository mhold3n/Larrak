"""
Orchestration adapters connecting existing modules to orchestrator protocols.
"""

from campro.orchestration.adapters.cem_adapter import CEM_AVAILABLE, CEMClientAdapter
from campro.orchestration.adapters.simulation_adapter import (
    MockSimulationAdapter,
    PhysicsSimulationAdapter,
)
from campro.orchestration.adapters.solver_adapter import IPOPTSolverAdapter, SimpleSolverAdapter
from campro.orchestration.adapters.surrogate_adapter import (
    SURROGATE_AVAILABLE,
    EnsembleSurrogateAdapter,
    MockSurrogateAdapter,
)

__all__ = [
    # CEM
    "CEM_AVAILABLE",
    "CEMClientAdapter",
    # Simulation
    "MockSimulationAdapter",
    "PhysicsSimulationAdapter",
    # Solver
    "IPOPTSolverAdapter",
    "SimpleSolverAdapter",
    # Surrogate
    "EnsembleSurrogateAdapter",
    "MockSurrogateAdapter",
    "SURROGATE_AVAILABLE",
]
