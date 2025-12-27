"""Runtime Weaviate Logger: Logs orchestration state to Weaviate collections.

Provides decorators and helpers for automatically logging:
- BudgetSnapshot: after budget allocations
- CacheEntry: on cache hits/misses
- TrustRegionLog: after trust region adjustments
- OptimizationStep: each iteration

Usage:
    from provenance.runtime_logger import RuntimeLogger

    logger = RuntimeLogger(client)
    logger.log_budget_snapshot(budget_manager, run_id)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

# Conditional Weaviate import
try:
    import weaviate

    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None  # type: ignore[assignment]


class RuntimeLogger:
    """Logs orchestration runtime state to Weaviate."""

    def __init__(self, client: Any | None = None):
        """Initialize logger with optional Weaviate client."""
        self.client = client
        self._run_id: str | None = None

    def set_run_id(self, run_id: str) -> None:
        """Set current run ID for all subsequent logs."""
        self._run_id = run_id

    def _timestamp(self) -> str:
        """Get RFC3339 timestamp for Weaviate DATE type."""
        return datetime.now(timezone.utc).isoformat()

    def _generate_uuid(self, *parts: str) -> str:
        """Generate deterministic UUID from parts."""
        combined = ":".join(parts)
        return hashlib.md5(combined.encode()).hexdigest()

    def log_budget_snapshot(
        self,
        total: int,
        spent: int,
        remaining: int,
        allocation: dict[str, float] | None = None,
    ) -> str | None:
        """Log a budget snapshot.

        Args:
            total: Total budget
            spent: Budget spent so far
            remaining: Budget remaining
            allocation: Optional allocation breakdown

        Returns:
            UUID of created entry if successful
        """
        if not self.client or not WEAVIATE_AVAILABLE:
            return None

        try:
            collection = self.client.collections.get("BudgetSnapshot")
            timestamp = self._timestamp()
            snapshot_id = self._generate_uuid("budget", str(self._run_id), timestamp)

            collection.data.insert(
                properties={
                    "snapshot_id": snapshot_id,
                    "timestamp": timestamp,
                    "total_budget": total,
                    "spent_budget": spent,
                    "remaining_budget": remaining,
                    "allocation_json": json.dumps(allocation or {}),
                },
                uuid=snapshot_id,
            )
            return snapshot_id
        except Exception:
            return None

    def log_cache_entry(
        self,
        param_hash: str,
        was_hit: bool,
        compute_time_ms: float = 0,
        result_uuid: str | None = None,
    ) -> str | None:
        """Log a cache access.

        Args:
            param_hash: Hash of cached parameters
            was_hit: Whether this was a cache hit
            compute_time_ms: Time to compute if miss
            result_uuid: Optional UUID of HiFiSimulationRun result

        Returns:
            UUID of created entry if successful
        """
        if not self.client or not WEAVIATE_AVAILABLE:
            return None

        try:
            collection = self.client.collections.get("CacheEntry")
            timestamp = self._timestamp()
            entry_id = self._generate_uuid("cache", param_hash, timestamp)

            properties = {
                "entry_id": entry_id,
                "param_hash": param_hash,
                "created_at": timestamp,
                "last_accessed": timestamp,
                "access_count": 1,
                "was_hit": was_hit,
                "compute_time_ms": compute_time_ms,
            }

            collection.data.insert(properties=properties, uuid=entry_id)
            return entry_id
        except Exception:
            return None

    def log_trust_region(
        self,
        radius_before: float,
        radius_after: float,
        action: str,  # "expand", "contract", "hold"
        predicted_improvement: float = 0,
        actual_improvement: float = 0,
        ratio: float = 0,
        step_uuid: str | None = None,
    ) -> str | None:
        """Log a trust region adjustment.

        Args:
            radius_before: Radius before adjustment
            radius_after: Radius after adjustment
            action: Type of adjustment
            predicted_improvement: Predicted improvement
            actual_improvement: Actual improvement
            ratio: Agreement ratio
            step_uuid: Optional UUID of OptimizationStep

        Returns:
            UUID of created entry if successful
        """
        if not self.client or not WEAVIATE_AVAILABLE:
            return None

        try:
            collection = self.client.collections.get("TrustRegionLog")
            timestamp = self._timestamp()
            log_id = self._generate_uuid("tr", str(self._run_id), timestamp)

            collection.data.insert(
                properties={
                    "log_id": log_id,
                    "timestamp": timestamp,
                    "radius_before": radius_before,
                    "radius_after": radius_after,
                    "action": action,
                    "predicted_improvement": predicted_improvement,
                    "actual_improvement": actual_improvement,
                    "ratio": ratio,
                },
                uuid=log_id,
            )
            return log_id
        except Exception:
            return None

    def log_optimization_step(
        self,
        iteration: int,
        candidates_generated: int,
        candidates_refined: int,
        candidates_selected: int,
        budget_remaining: int,
        best_objective: float,
        best_params: dict[str, Any] | None = None,
        duration_ms: float = 0,
    ) -> str | None:
        """Log an optimization step.

        Args:
            iteration: Current iteration number
            candidates_generated: Number of candidates from CEM
            candidates_refined: Number refined by solver
            candidates_selected: Number selected for truth eval
            budget_remaining: Remaining simulation budget
            best_objective: Best objective value so far
            best_params: Optional best parameters as dict
            duration_ms: Time for this step

        Returns:
            UUID of created entry if successful
        """
        if not self.client or not WEAVIATE_AVAILABLE:
            return None

        try:
            collection = self.client.collections.get("OptimizationStep")
            timestamp = self._timestamp()
            step_id = self._generate_uuid("step", str(self._run_id), str(iteration))

            collection.data.insert(
                properties={
                    "step_id": step_id,
                    "iteration": iteration,
                    "timestamp": timestamp,
                    "candidates_generated": candidates_generated,
                    "candidates_refined": candidates_refined,
                    "candidates_selected": candidates_selected,
                    "budget_remaining": budget_remaining,
                    "best_objective": best_objective,
                    "best_params_json": json.dumps(best_params or {}),
                    "duration_ms": duration_ms,
                },
                uuid=step_id,
            )
            return step_id
        except Exception:
            return None


# Singleton for easy access
_logger: RuntimeLogger | None = None


def get_runtime_logger(client: Any | None = None) -> RuntimeLogger:
    """Get or create the runtime logger singleton.

    Args:
        client: Optional Weaviate client (first call only)

    Returns:
        RuntimeLogger instance
    """
    global _logger
    if _logger is None:
        _logger = RuntimeLogger(client)
    return _logger


def init_runtime_logger(client: Any) -> RuntimeLogger:
    """Initialize the runtime logger with a client.

    Args:
        client: Weaviate client

    Returns:
        RuntimeLogger instance
    """
    global _logger
    _logger = RuntimeLogger(client)
    return _logger
