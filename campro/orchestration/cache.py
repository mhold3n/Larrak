"""
Evaluation Cache: Memoizes expensive simulation calls.

Prevents redundant computation by caching results keyed by a stable
hash of the input parameters.

Usage:
    from campro.orchestration import EvaluationCache

    cache = EvaluationCache()
    result = cache.get_or_compute(params, expensive_fn)
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""

    result: Any
    param_hash: str
    timestamp: float = 0.0
    hit_count: int = 0


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class EvaluationCache:
    """
    Memoization cache for expensive evaluations.

    Features:
    - Stable hashing of numpy arrays and nested structures
    - LRU eviction when max size exceeded
    - Optional disk persistence
    - Statistics tracking
    """

    def __init__(
        self,
        max_size: int = 10000,
        persist_path: Path | str | None = None,
    ):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries before eviction
            persist_path: Optional path to persist cache to disk
        """
        self.max_size = max_size
        self.persist_path = Path(persist_path) if persist_path else None

        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # For LRU
        self._stats = CacheStats()

        # Load from disk if exists
        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()

    def get(self, params: dict[str, Any]) -> Any | None:
        """
        Get cached result if available.

        Args:
            params: Parameter dictionary

        Returns:
            Cached result or None if not found
        """
        param_hash = self._compute_hash(params)

        if param_hash in self._cache:
            entry = self._cache[param_hash]
            entry.hit_count += 1
            self._stats.hits += 1

            # Update access order for LRU
            if param_hash in self._access_order:
                self._access_order.remove(param_hash)
            self._access_order.append(param_hash)

            return entry.result

        self._stats.misses += 1
        return None

    def put(self, params: dict[str, Any], result: Any) -> None:
        """
        Store result in cache.

        Args:
            params: Parameter dictionary
            result: Result to cache
        """
        param_hash = self._compute_hash(params)

        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            self._evict_lru()

        import time

        self._cache[param_hash] = CacheEntry(
            result=result,
            param_hash=param_hash,
            timestamp=time.time(),
        )
        self._access_order.append(param_hash)

    def get_or_compute(
        self,
        params: dict[str, Any],
        compute_fn: Callable[[dict[str, Any]], Any],
    ) -> tuple[Any, bool]:
        """
        Get from cache or compute and cache.

        Args:
            params: Parameter dictionary
            compute_fn: Function to compute result if not cached

        Returns:
            Tuple of (result, was_cached)
        """
        cached = self.get(params)
        if cached is not None:
            return cached, True

        result = compute_fn(params)
        self.put(params, result)
        return result, False

    def _compute_hash(self, params: dict[str, Any]) -> str:
        """Compute stable hash of parameters."""
        # Convert to JSON-serializable form
        serializable = self._to_serializable(params)
        json_str = json.dumps(serializable, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _to_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable form."""
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist(), "__dtype__": str(obj.dtype)}
        elif isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
            self._stats.evictions += 1

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()

    def save_to_disk(self) -> None:
        """Save cache to disk."""
        if self.persist_path is None:
            return

        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(
                {
                    "cache": self._cache,
                    "access_order": self._access_order,
                },
                f,
            )
        log.info(f"Saved cache ({len(self._cache)} entries) to {self.persist_path}")

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            self._cache = data["cache"]
            self._access_order = data["access_order"]
            log.info(f"Loaded cache ({len(self._cache)} entries) from {self.persist_path}")
        except Exception as e:
            log.warning(f"Failed to load cache: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": self._stats.hit_rate,
            "evictions": self._stats.evictions,
        }


__all__ = [
    "CacheEntry",
    "CacheStats",
    "EvaluationCache",
]
