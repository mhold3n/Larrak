"""HiFi Simulation Adapter for Orchestrator.

Implements SimulationInterface protocol to enable orchestrator
to dispatch expensive HiFi simulations when surrogate uncertainty
exceeds threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from campro.logging import get_logger
from campro.orchestration.cache import EvaluationCache

log = get_logger(__name__)


class SimulationInterface(Protocol):
    """Protocol for expensive simulation layer."""

    def evaluate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Evaluate candidate with expensive simulation."""
        ...


@dataclass
class HiFiSimulationConfig:
    """Configuration for HiFi simulation dispatch."""

    # Which solvers to run
    run_thermal: bool = True
    run_structural: bool = True
    run_flow: bool = False

    # Uncertainty threshold for triggering HiFi
    uncertainty_threshold: float = 50.0  # K for thermal, MPa for structural

    # Caching
    cache_dir: str = ".cache/hifi"
    use_cache: bool = True

    # Timeouts
    solver_timeout_s: int = 3600


class HiFiSimulationAdapter:
    """
    Adapter that dispatches to HiFi solvers when needed.

    Implements SimulationInterface for use with Orchestrator.

    Decision logic:
    1. Query surrogate for prediction + uncertainty
    2. If uncertainty < threshold: return surrogate prediction
    3. If uncertainty >= threshold: run actual CFD/FEA
    """

    def __init__(self, config: HiFiSimulationConfig | None = None):
        self.config = config or HiFiSimulationConfig()

        # Initialize cache
        if self.config.use_cache:
            cache_path = Path(self.config.cache_dir) / "hifi_cache.pkl"
            self._cache = EvaluationCache(persist_path=cache_path)
        else:
            self._cache = None

        # Lazy load adapters
        self._thermal_adapter = None
        self._structural_adapter = None

    def evaluate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate candidate with HiFi simulation.

        Args:
            candidate: Dict with bore_mm, stroke_mm, cr, rpm, load_fraction

        Returns:
            Dict with T_crown_max, von_mises_max, and other outputs
        """
        # Check cache first
        if self._cache:
            cached_result = self._cache.get(candidate)
            if cached_result is not None:
                log.debug(f"HiFi cache hit for {candidate.get('run_id', 'unknown')}")
                return cached_result

        result = {
            "success": True,
            "source": "hifi",
            "thermal": None,
            "structural": None,
        }

        # Run thermal CFD if enabled
        if self.config.run_thermal:
            thermal_result = self._run_thermal(candidate)
            result["thermal"] = thermal_result
            result["T_crown_max"] = thermal_result.get("T_crown_max")

        # Run structural FEA if enabled
        if self.config.run_structural:
            structural_result = self._run_structural(candidate)
            result["structural"] = structural_result
            result["von_mises_max"] = structural_result.get("von_mises_max")

        # Cache result
        if self._cache:
            self._cache.put(candidate, result)

        return result

    def _run_thermal(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Run thermal CFD (CHT) simulation."""
        try:
            from Simulations.hifi import ConjugateHTAdapter
            from Simulations.hifi.example_inputs import create_simulation_input

            if self._thermal_adapter is None:
                self._thermal_adapter = ConjugateHTAdapter()

            sim_input = create_simulation_input(
                run_id=candidate.get("run_id", "thermal_run"),
                bore_mm=candidate.get("bore_mm", 85),
                stroke_mm=candidate.get("stroke_mm", 90),
                rpm=candidate.get("rpm", 3000),
                load_fraction=candidate.get("load_fraction", 1.0),
                compression_ratio=candidate.get("cr", 12.5),
            )

            self._thermal_adapter.load_input(sim_input)
            output = self._thermal_adapter.solve_steady_state()

            return {
                "success": output.success,
                "T_crown_max": output.T_crown_max,
                "calibration_params": output.calibration_params,
            }

        except Exception as e:
            log.error(f"Thermal simulation failed: {e}")
            return {"success": False, "error": str(e)}

    def _run_structural(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Run structural FEA simulation."""
        try:
            from Simulations.hifi import StructuralFEAAdapter
            from Simulations.hifi.example_inputs import create_simulation_input

            if self._structural_adapter is None:
                self._structural_adapter = StructuralFEAAdapter()

            sim_input = create_simulation_input(
                run_id=candidate.get("run_id", "structural_run"),
                bore_mm=candidate.get("bore_mm", 85),
                stroke_mm=candidate.get("stroke_mm", 90),
                rpm=candidate.get("rpm", 3000),
                load_fraction=candidate.get("load_fraction", 1.0),
                compression_ratio=candidate.get("cr", 12.5),
            )

            self._structural_adapter.load_input(sim_input)
            output = self._structural_adapter.solve_steady_state()

            return {
                "success": output.success,
                "von_mises_max": output.max_von_mises,
                "calibration_params": output.calibration_params,
            }

        except Exception as e:
            log.error(f"Structural simulation failed: {e}")
            return {"success": False, "error": str(e)}

    def should_dispatch_hifi(
        self, surrogate_uncertainty: float, threshold: float | None = None
    ) -> bool:
        """
        Decide if HiFi simulation should be dispatched.

        Args:
            surrogate_uncertainty: Uncertainty from surrogate prediction
            threshold: Override threshold

        Returns:
            True if HiFi should be run
        """
        thresh = threshold or self.config.uncertainty_threshold
        return surrogate_uncertainty >= thresh

    def get_statistics(self) -> dict[str, Any]:
        """Get cache and simulation statistics."""
        stats = {"adapter_type": "hifi"}
        if self._cache:
            stats["cache"] = self._cache.get_statistics()
        return stats


__all__ = [
    "HiFiSimulationAdapter",
    "HiFiSimulationConfig",
]
