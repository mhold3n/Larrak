"""
Base storage classes and interfaces.

This module defines the fundamental storage system for optimization results,
providing a consistent interface for storing, retrieving, and sharing results
between different optimization components.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


class StorageStatus(Enum):
    """Status of storage operations."""

    PENDING = "pending"
    STORED = "stored"
    RETRIEVED = "retrieved"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class StorageResult:
    """Result of a storage operation."""

    # Storage metadata
    storage_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)

    # Storage status
    status: StorageStatus = StorageStatus.PENDING

    # Stored data
    data: dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optimization context (NEW)
    constraints: dict[str, Any] | None = None
    optimization_rules: dict[str, Any] | None = None
    solver_settings: dict[str, Any] | None = None

    # Access information
    access_count: int = 0
    last_accessed: float | None = None

    # Expiration
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if the stored result has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def is_accessible(self) -> bool:
        """Check if the result is accessible (not expired and not failed)."""
        return self.status == StorageStatus.STORED and not self.is_expired()

    def mark_accessed(self) -> None:
        """Mark the result as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()

    def get_age(self) -> float:
        """Get the age of the stored result in seconds."""
        return time.time() - self.timestamp


class BaseStorage(ABC):
    """
    Base class for all storage systems.

    Provides a common interface for storing and retrieving optimization
    results across different storage backends (memory, file, database).
    """

    def __init__(self, name: str = "BaseStorage"):
        self.name = name
        self._storage: dict[str, StorageResult] = {}
        self._access_history: list[
            tuple[str, float, str]
        ] = []  # (id, timestamp, operation)

    @abstractmethod
    def store(
        self,
        key: str,
        data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        constraints: dict[str, Any] | None = None,
        optimization_rules: dict[str, Any] | None = None,
        solver_settings: dict[str, Any] | None = None,
        expires_in: float | None = None,
    ) -> StorageResult:
        """
        Store optimization result data.

        Args:
            key: Unique key for the stored data
            data: Data to store
            metadata: Optional metadata
            constraints: Optional constraints used in optimization
            optimization_rules: Optional optimization rules and parameters
            solver_settings: Optional solver settings and configuration
            expires_in: Optional expiration time in seconds

        Returns:
            StorageResult object
        """

    @abstractmethod
    def retrieve(self, key: str) -> StorageResult | None:
        """
        Retrieve stored optimization result.

        Args:
            key: Key of the data to retrieve

        Returns:
            StorageResult object or None if not found
        """

    @abstractmethod
    def remove(self, key: str) -> bool:
        """
        Remove stored data.

        Args:
            key: Key of the data to remove

        Returns:
            True if removed successfully, False otherwise
        """

    def list_keys(self) -> list[str]:
        """List all stored keys."""
        return list(self._storage.keys())

    def list_accessible_keys(self) -> list[str]:
        """List all accessible (non-expired) keys."""
        return [key for key, result in self._storage.items() if result.is_accessible()]

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        expired_keys = [
            key for key, result in self._storage.items() if result.is_expired()
        ]

        for key in expired_keys:
            self.remove(key)

        log.info(f"Cleaned up {len(expired_keys)} expired entries")
        return len(expired_keys)

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        total_entries = len(self._storage)
        accessible_entries = len(self.list_accessible_keys())
        expired_entries = total_entries - accessible_entries

        # Calculate total access count
        total_accesses = sum(result.access_count for result in self._storage.values())

        # Calculate average age
        if self._storage:
            avg_age = np.mean([result.get_age() for result in self._storage.values()])
        else:
            avg_age = 0.0

        return {
            "total_entries": total_entries,
            "accessible_entries": accessible_entries,
            "expired_entries": expired_entries,
            "total_accesses": total_accesses,
            "average_age_seconds": avg_age,
            "storage_name": self.name,
        }

    def _log_access(self, key: str, operation: str) -> None:
        """Log storage access."""
        self._access_history.append((key, time.time(), operation))

        # Keep only last 1000 access records
        if len(self._access_history) > 1000:
            self._access_history = self._access_history[-1000:]

    def get_access_history(
        self, limit: int | None = None,
    ) -> list[tuple[str, float, str]]:
        """Get access history."""
        if limit is None:
            return self._access_history.copy()
        return self._access_history[-limit:] if limit > 0 else []
