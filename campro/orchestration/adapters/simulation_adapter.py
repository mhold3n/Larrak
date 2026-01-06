"""
Simulation Adapter: Wraps expensive physics simulation as SimulationInterface.

Adapts the existing physics kernels for truth evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from campro.logging import get_logger
from campro.orchestration.orchestrator import SimulationInterface

log = get_logger(__name__)


class PhysicsSimulationAdapter:
    """
    Adapter wrapping physics simulation as SimulationInterface.

    Provides truth evaluations using the full physics solver.
    This is the "expensive" path - use sparingly.
    """

    def __init__(
        self,
        use_full_physics: bool = True,
        cache_results: bool = True,
    ):
        """
        Initialize adapter.

        Args:
            use_full_physics: Use full 1D physics (True) or 0D approximation (False)
            cache_results: Whether to cache simulation results
        """
        self.use_full_physics = use_full_physics
        self.cache_results = cache_results
        self._cache: dict[str, float] = {}
        self._call_count = 0

    def evaluate(
        self,
        candidate: dict[str, Any],
    ) -> float:
        """
        Evaluate candidate with expensive physics simulation.

        Args:
            candidate: Candidate design dict

        Returns:
            Objective value (higher = better)
        """
        self._call_count += 1

        # Check cache
        if self.cache_results:
            cache_key = self._compute_cache_key(candidate)
            if cache_key in self._cache:
                log.debug("Simulation cache hit")
                return self._cache[cache_key]

        try:
            if self.use_full_physics:
                # Validation: Ensure physics binaries exist
                import shutil

                if not shutil.which("ccx") and not shutil.which("openfoam"):
                    # We check for one or the other depending on solver, but generally at least one must exist
                    # For now, just warn if neither, or distinct logic.
                    # Assuming 'ccx' is main one for now based on context.
                    if not shutil.which("ccx"):
                        raise RuntimeError(
                            "CalculiX binary ('ccx') not found in PATH. Cannot run full physics."
                        )

                result = self._run_full_simulation(candidate)
            else:
                result = self._run_0d_simulation(candidate)

            # Cache result
            if self.cache_results:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            # Re-raise configuration errors, swallow only runtime physics failures
            if "not found" in str(e):
                raise e
            log.warning(f"Simulation failed: {e}")
            return float("-inf")  # Failed simulations are worst

    def _run_full_simulation(
        self,
        candidate: dict[str, Any],
    ) -> float:
        """Run full physics simulation."""
        from campro.optimization.driver import solve_cycle

        params = self._candidate_to_params(candidate)
        params["use_0d_model"] = False

        result = solve_cycle(params)

        # Extract objective (thermal efficiency)
        if hasattr(result, "objective_value"):
            return result.objective_value
        elif (
            hasattr(result, "performance_metrics")
            and "thermal_efficiency" in result.performance_metrics
        ):
            return result.performance_metrics["thermal_efficiency"]
        else:
            return 0.0

    def _run_0d_simulation(
        self,
        candidate: dict[str, Any],
    ) -> float:
        """Run simplified 0D physics simulation."""
        # Extract key parameters
        rpm = candidate.get("rpm", 3000.0)
        p_intake = candidate.get("p_intake_bar", 1.5)
        fuel_mass = candidate.get("fuel_mass_kg", 5e-5)
        cr = candidate.get("cr", 15.0)

        # Simple Otto cycle efficiency model + losses
        gamma = 1.3
        eta_otto = 1 - (1 / cr) ** (gamma - 1)

        # Friction losses
        fmep_factor = 0.05 + 0.02 * (rpm / 3000)
        eta_mech = 1 - fmep_factor

        # Combustion efficiency
        phi = fuel_mass / 5e-5
        eta_comb = 0.95 if 0.8 < phi < 1.2 else 0.80

        # Boost benefit
        boost_factor = min(1.1, 1.0 + 0.02 * (p_intake - 1.0))

        eta = eta_otto * eta_mech * eta_comb * boost_factor

        return float(eta)

    def _candidate_to_params(
        self,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert candidate dict to solver params."""
        params = candidate.copy()

        # Remove internal keys
        params = {k: v for k, v in params.items() if not k.startswith("_")}

        return params

    def _compute_cache_key(
        self,
        candidate: dict[str, Any],
    ) -> str:
        """Compute cache key for candidate."""
        import hashlib
        import json

        # Extract key parameters for hashing
        key_params = {
            k: v
            for k, v in candidate.items()
            if isinstance(v, (int, float, str)) and not k.startswith("_")
        }

        json_str = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()[:16]

    @property
    def call_count(self) -> int:
        """Get number of simulation calls made."""
        return self._call_count

    def reset_stats(self) -> None:
        """Reset call statistics."""
        self._call_count = 0
        self._cache.clear()


class MockSimulationAdapter:
    """Mock simulation for testing (cheap, random noise)."""

    def __init__(self, noise_scale: float = 0.05):
        self.noise_scale = noise_scale
        self._call_count = 0

    def evaluate(self, candidate: dict[str, Any]) -> float:
        self._call_count += 1

        # Base value from candidate
        p_intake = candidate.get("p_intake_bar", 1.5)
        fuel = candidate.get("fuel_mass_kg", 5e-5)

        base = 0.38 + 0.02 * p_intake - 0.1 * abs(fuel - 5e-5) / 5e-5
        noise = np.random.normal(0, self.noise_scale)

        return float(base + noise)

    @property
    def call_count(self) -> int:
        return self._call_count


__all__ = [
    "MockSimulationAdapter",
    "PhysicsSimulationAdapter",
]
