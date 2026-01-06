"""
CEM-Gated Orchestrator: Main control loop for optimization.

Implements the architecture where CEM gates access to expensive simulation,
surrogate provides cheap predictions, and solver operates under trust regions.

Usage:
    from campro.orchestration import Orchestrator

    orch = Orchestrator(cem_client, surrogate, solver, simulation)
    result = orch.optimize(initial_params, budget=1000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from campro.logging import get_logger
from campro.orchestration.budget import BudgetManager
from campro.orchestration.cache import EvaluationCache
from campro.orchestration.provenance import ProvenanceClient
from campro.orchestration.trust_region import TrustRegion
from provenance.execution_events import (
    EventType,
    emit_event,
    module_end,
    module_start,
    step_end,
    step_start,
)

log = get_logger(__name__)


class CEMInterface(Protocol):
    """Protocol for CEM layer."""

    def generate_batch(self, params: dict[str, Any], n: int) -> list[dict[str, Any]]:
        """Generate batch of feasible candidates."""
        ...

    def check_feasibility(
        self, candidate: dict[str, Any], run_id: str | None = None
    ) -> tuple[bool, float]:
        """Check feasibility and return (is_feasible, score)."""
        ...

    def repair(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Project candidate back to feasible manifold."""
        ...

    def adapt_rules(
        self, truth_data: list[tuple[dict[str, Any], float]], run_id: str | None = None
    ) -> Any:
        """Adapt rule parameters based on HiFi simulation results (optional)."""
        ...


class SurrogateInterface(Protocol):
    """Protocol for surrogate layer."""

    def predict(self, candidates: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """Predict objectives and return (predictions, uncertainty)."""
        ...

    def update(self, data: list[tuple[dict[str, Any], float]]) -> None:
        """Update surrogate with new truth data."""
        ...


class SolverInterface(Protocol):
    """Protocol for solver layer."""

    def refine(
        self,
        candidate: dict[str, Any],
        objective_fn: Any,
        max_step: np.ndarray,
    ) -> dict[str, Any]:
        """Refine candidate under step constraints."""
        ...


class SimulationInterface(Protocol):
    """Protocol for expensive simulation layer."""

    def evaluate(self, candidate: dict[str, Any]) -> float:
        """Evaluate candidate with expensive simulation."""
        ...


@dataclass
class OrchestrationResult:
    """Result from orchestration run."""

    best_candidate: dict[str, Any]
    best_objective: float
    n_sim_calls: int
    n_surrogate_calls: int
    n_iterations: int
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def efficiency(self) -> float:
        """Ratio of surrogate to sim calls."""
        if self.n_sim_calls == 0:
            return float("inf")
        return self.n_surrogate_calls / self.n_sim_calls


@dataclass
class OrchestrationConfig:
    """Configuration for orchestration loop."""

    # Budget
    total_sim_budget: int = 1000
    batch_size: int = 50

    # Convergence
    max_iterations: int = 100
    convergence_threshold: float = 1e-4
    patience: int = 10  # Iterations without improvement before stopping

    # Trust region
    initial_trust_radius: float = 0.1

    # Surrogate update frequency
    retrain_every: int = 5  # Iterations between surrogate retraining

    # Provenance
    use_provenance: bool = True


class Orchestrator:
    """
    Main orchestration loop for CEM-gated optimization.

    Control flow:
    1. CEM generates feasible batch
    2. Surrogate predicts objectives
    3. Solver refines (trust-bounded)
    4. Budget selects for truth evaluation
    5. Update surrogate
    6. Repeat until convergence or budget exhausted
    """

    def __init__(
        self,
        cem: CEMInterface,
        surrogate: SurrogateInterface,
        solver: SolverInterface,
        simulation: SimulationInterface,
        config: OrchestrationConfig | None = None,
    ):
        """
        Initialize orchestrator.

        Args:
            cem: CEM layer (feasibility + generation)
            surrogate: Surrogate layer (cheap predictions)
            solver: Solver layer (local refinement)
            simulation: Simulation layer (expensive truth)
            config: Orchestration configuration
        """
        self.cem = cem
        self.surrogate = surrogate
        self.solver = solver
        self.simulation = simulation
        self.config = config or OrchestrationConfig()

        # Components
        self.budget = BudgetManager(self.config.total_sim_budget)
        self.trust_region = TrustRegion()
        self.cache = EvaluationCache()
        self.provenance = ProvenanceClient(use_provenance=self.config.use_provenance)

        # State
        self._best_candidate: dict[str, Any] | None = None
        self._best_objective: float = float("-inf")
        self._n_surrogate_calls = 0
        self._history: list[dict[str, Any]] = []

    def optimize(
        self,
        initial_params: dict[str, Any],
    ) -> OrchestrationResult:
        """
        Run optimization loop.

        Args:
            initial_params: Starting parameters for generation

        Returns:
            OrchestrationResult with best candidate and statistics
        """
        # Provenance: Start run
        run_id = self.provenance.start_run(initial_params)
        if run_id:
            log.info(f"Provenance tracking enabled. Run ID: {run_id}")

        log.info("Starting CEM-gated optimization")
        log.info(f"Budget: {self.config.total_sim_budget} sim calls")

        emit_event(
            EventType.RUN_START,
            "ORCH",
            metadata={"run_id": run_id or "unknown", "config": str(self.config)},
        )

        iteration = 0
        no_improvement_count = 0

        while not self.budget.exhausted() and iteration < self.config.max_iterations:
            # Check for stop signal from dashboard
            if os.environ.get("ORCHESTRATOR_STOP_SIGNAL") == "1":
                log.warning("Orchestrator loop aborted due to stop signal")
                break

            iteration += 1

            step_start(iteration)

            # Step 1: CEM generates feasible batch
            module_start("CEM", step="generate_batch")
            candidates = self.cem.generate_batch(initial_params, n=self.config.batch_size)
            module_end("CEM", candidates=len(candidates))

            if not candidates:
                log.warning("CEM generated no candidates, stopping")
                break

            # Step 2: Surrogate predicts
            module_start("SUR", step="predict")
            predictions, uncertainty = self.surrogate.predict(candidates)
            self._n_surrogate_calls += len(candidates)
            module_end("SUR", predictions=len(predictions))

            # Get CEM feasibility scores
            feasibility = np.array([self.cem.check_feasibility(c)[1] for c in candidates])

            # Step 3: Solver refines top candidates (trust-bounded)
            module_start("SOL", step="local_refinement")
            refined_candidates = self._refine_candidates(candidates, predictions, uncertainty)

            # Repredict after refinement
            if refined_candidates:
                ref_pred, ref_unc = self.surrogate.predict(refined_candidates)
                self._n_surrogate_calls += len(refined_candidates)
            else:
                refined_candidates = candidates
                ref_pred, ref_unc = predictions, uncertainty
            module_end("SOL", refined=len(refined_candidates))

            # Step 4: Budget selects for truth evaluation
            module_start("SEL", step="pick_for_eval")
            selected_indices = self.budget.select(
                refined_candidates,
                ref_pred,
                ref_unc,
                feasibility[: len(refined_candidates)]
                if len(feasibility) >= len(refined_candidates)
                else None,
            )
            module_end("SEL", selected=len(selected_indices))

            # Step 5: Expensive simulation (sparse)
            truth_data = []
            if selected_indices:
                module_start("CCX", step="run_simulation")
                for idx in selected_indices:
                    candidate = refined_candidates[idx]

                    module_start("CACHE", candidate_id=str(candidate.get("id", idx)))
                    # Check cache first
                    result, was_cached = self.cache.get_or_compute(
                        candidate, lambda c: self.simulation.evaluate(c)
                    )
                    module_end("CACHE", was_cached=was_cached)

                    truth_data.append((candidate, result))

                    # Update best
                    if result > self._best_objective:
                        self._best_objective = result
                        self._best_candidate = candidate.copy()
                        no_improvement_count = 0
                        log.info(f"Iter {iteration}: New best = {result:.6f}")
                module_end("CCX", simulations=len(selected_indices))
            else:
                # Still need to end the module/step if we skipped it?
                # Actually if we don't start it we don't end it.
                pass

            # Update trust region based on prediction accuracy
            if truth_data and selected_indices:
                predicted_best = ref_pred[selected_indices[0]]
                actual_best = truth_data[0][1]
                self.trust_region.update(
                    float(predicted_best),
                    actual_best,
                    float(ref_unc[selected_indices[0]]) if len(ref_unc) > 0 else 0.0,
                )

            # Step 6: Update surrogate periodically
            if iteration % self.config.retrain_every == 0 and truth_data:
                module_start("PROV", step="update_surrogate")
                self.surrogate.update(truth_data)
                log.debug(f"Surrogate updated with {len(truth_data)} samples")
                module_end("PROV", updated=True)

            # Step 7: Adapt CEM rules based on HiFi feedback
            # Step 7: Adapt CEM rules based on HiFi feedback
            if truth_data and hasattr(self.cem, "adapt_rules"):
                try:
                    module_start("PROV", step="adapt_rules")
                    run_id = getattr(self.provenance, "run_id", None)
                    # Helper function in CEM will handle None, but explicit cast satisfies linter if run_id is expected str
                    adaptation_report = self.cem.adapt_rules(
                        truth_data, run_id=str(run_id) if run_id else None
                    )
                    if adaptation_report and getattr(adaptation_report, "any_adapted", False):
                        log.info(f"CEM adapted {adaptation_report.total_rules_adapted} rules")
                    module_end("PROV", adapted=True)
                except Exception as e:
                    log.debug(f"CEM adapt_rules skipped: {e}")
                    module_end("PROV", adapted=False, error=str(e))

            # Record history
            self._history.append(
                {
                    "iteration": iteration,
                    "n_candidates": len(candidates),
                    "n_evaluated": len(selected_indices),
                    "best_predicted": float(ref_pred.max()) if len(ref_pred) > 0 else 0,
                    "best_actual": self._best_objective,
                    "budget_remaining": self.budget.remaining(),
                }
            )

            # Check convergence
            no_improvement_count += 1

            step_end(iteration, best_objective=self._best_objective)

            if no_improvement_count >= self.config.patience:
                log.info(f"Converged after {self.config.patience} iterations without improvement")
                break

        # Final validation
        if self._best_candidate is not None:
            validation_indices = self.budget.select_for_validation(
                [self._best_candidate], np.array([self._best_objective])
            )
            if validation_indices:
                validated = self.simulation.evaluate(self._best_candidate)
                log.info(f"Final validation: {validated:.6f}")

        # Provenance: End run
        # Provenance: End run
        self.provenance.end_run(status="COMPLETED" if self._best_candidate else "FAILED")
        self.provenance.close()
        emit_event(EventType.RUN_END, "ORCH", metadata={"success": True})

        return OrchestrationResult(
            best_candidate=self._best_candidate or {},
            best_objective=self._best_objective,
            n_sim_calls=self.budget.state.used,
            n_surrogate_calls=self._n_surrogate_calls,
            n_iterations=iteration,
            history=self._history,
        )

    def _refine_candidates(
        self,
        candidates: list[dict[str, Any]],
        predictions: np.ndarray,
        uncertainty: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Refine top candidates using solver with trust region."""
        # Select top candidates for refinement
        n_refine = min(10, len(candidates))
        top_indices = np.argsort(predictions)[-n_refine:][::-1]

        refined = []
        for idx in top_indices:
            candidate = candidates[idx]
            unc = uncertainty[idx] if len(uncertainty) > idx else 0.0

            # Compute max step from trust region
            max_step = self.trust_region.bound_step(
                np.ones(10) * 0.1,  # Default step proposal
                unc,
            )

            try:
                # Use surrogate as objective for solver
                def surrogate_obj(c):
                    pred, _ = self.surrogate.predict([c])
                    return float(pred[0])

                refined_candidate = self.solver.refine(
                    candidate,
                    surrogate_obj,
                    max_step,
                )

                # Repair if needed
                is_feasible, _ = self.cem.check_feasibility(refined_candidate)
                if not is_feasible:
                    refined_candidate = self.cem.repair(refined_candidate)

                refined.append(refined_candidate)

            except Exception as e:
                log.debug(f"Refinement failed for candidate {idx}: {e}")
                refined.append(candidate)  # Keep original

        return refined

    def get_statistics(self) -> dict[str, Any]:
        """Get orchestration statistics."""
        return {
            "budget": self.budget.get_statistics(),
            "trust_region": self.trust_region.get_statistics(),
            "cache": self.cache.get_statistics(),
            "best_objective": self._best_objective,
            "n_surrogate_calls": self._n_surrogate_calls,
            "efficiency": self._n_surrogate_calls / max(1, self.budget.state.used),
        }


__all__ = [
    "CEMInterface",
    "OrchestrationConfig",
    "OrchestrationResult",
    "Orchestrator",
    "SimulationInterface",
    "SolverInterface",
    "SurrogateInterface",
]
