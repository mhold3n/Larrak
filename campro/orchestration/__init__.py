"""
Orchestration module for CEM-gated optimization.

Implements the architecture where CEM gates expensive evaluations,
surrogate provides cheap predictions, and solver operates under trust regions.

Key components:
- Orchestrator: Main control loop
- BudgetManager: Allocates simulation calls strategically
- TrustRegion: Bounds solver steps based on uncertainty
- EvaluationCache: Memoizes expensive evaluations
"""

from campro.orchestration.budget import (
    BudgetAllocation,
    BudgetManager,
    BudgetState,
    SelectionStrategy,
)
from campro.orchestration.cache import CacheEntry, CacheStats, EvaluationCache
from campro.orchestration.orchestrator import (
    CEMInterface,
    OrchestrationConfig,
    OrchestrationResult,
    Orchestrator,
    SimulationInterface,
    SolverInterface,
    SurrogateInterface,
)
from campro.orchestration.trust_region import TrustRegion, TrustRegionConfig

__all__ = [
    # Budget
    "BudgetAllocation",
    "BudgetManager",
    "BudgetState",
    "SelectionStrategy",
    # Cache
    "CacheEntry",
    "CacheStats",
    "EvaluationCache",
    # Orchestrator
    "CEMInterface",
    "OrchestrationConfig",
    "OrchestrationResult",
    "Orchestrator",
    "SimulationInterface",
    "SolverInterface",
    "SurrogateInterface",
    # Trust Region
    "TrustRegion",
    "TrustRegionConfig",
]
