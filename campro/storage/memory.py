"""
In-memory storage implementation.

This module provides an in-memory storage system for optimization results,
suitable for temporary storage and sharing between optimization components
within the same process.
"""

from typing import Dict, List, Optional, Any
import time

from .base import BaseStorage, StorageResult, StorageStatus
from campro.logging import get_logger

log = get_logger(__name__)


class MemoryStorage(BaseStorage):
    """
    In-memory storage for optimization results.
    
    This storage system keeps all data in memory, making it fast for
    temporary storage and sharing between optimization components.
    """
    
    def __init__(self, name: str = "MemoryStorage", max_entries: int = 1000):
        super().__init__(name)
        self.max_entries = max_entries
    
    def store(self, key: str, data: Dict[str, Any], 
              metadata: Optional[Dict[str, Any]] = None,
              constraints: Optional[Dict[str, Any]] = None,
              optimization_rules: Optional[Dict[str, Any]] = None,
              solver_settings: Optional[Dict[str, Any]] = None,
              expires_in: Optional[float] = None) -> StorageResult:
        """
        Store optimization result data in memory.
        
        Args:
            key: Unique key for the stored data
            data: Data to store
            metadata: Optional metadata
            expires_in: Optional expiration time in seconds
            
        Returns:
            StorageResult object
        """
        # Check if we need to clean up space
        if len(self._storage) >= self.max_entries:
            self._cleanup_oldest()
        
        # Create storage result
        result = StorageResult(
            data=data,
            metadata=metadata or {},
            constraints=constraints,
            optimization_rules=optimization_rules,
            solver_settings=solver_settings,
            status=StorageStatus.STORED
        )
        
        # Set expiration if specified
        if expires_in is not None:
            result.expires_at = time.time() + expires_in
        
        # Store the result
        self._storage[key] = result
        
        # Log the operation
        self._log_access(key, "store")
        
        log.debug(f"Stored data with key '{key}' in {self.name}")
        return result
    
    def retrieve(self, key: str) -> Optional[StorageResult]:
        """
        Retrieve stored optimization result from memory.
        
        Args:
            key: Key of the data to retrieve
            
        Returns:
            StorageResult object or None if not found/expired
        """
        if key not in self._storage:
            log.debug(f"Key '{key}' not found in {self.name}")
            return None
        
        result = self._storage[key]
        
        # Check if expired
        if result.is_expired():
            log.debug(f"Key '{key}' has expired in {self.name}")
            result.status = StorageStatus.EXPIRED
            return None
        
        # Mark as accessed
        result.mark_accessed()
        
        # Log the operation
        self._log_access(key, "retrieve")
        
        log.debug(f"Retrieved data with key '{key}' from {self.name}")
        return result
    
    def remove(self, key: str) -> bool:
        """
        Remove stored data from memory.
        
        Args:
            key: Key of the data to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        if key not in self._storage:
            log.debug(f"Key '{key}' not found for removal in {self.name}")
            return False
        
        del self._storage[key]
        
        # Log the operation
        self._log_access(key, "remove")
        
        log.debug(f"Removed data with key '{key}' from {self.name}")
        return True
    
    def _cleanup_oldest(self) -> None:
        """Remove the oldest entry to make space."""
        if not self._storage:
            return
        
        # Find the oldest entry
        oldest_key = min(self._storage.keys(), 
                        key=lambda k: self._storage[k].timestamp)
        
        log.debug(f"Cleaning up oldest entry '{oldest_key}' to make space")
        self.remove(oldest_key)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        total_size = 0
        data_sizes = {}
        
        for key, result in self._storage.items():
            # Estimate size of stored data
            size = self._estimate_data_size(result.data)
            total_size += size
            data_sizes[key] = size
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'entry_count': len(self._storage),
            'data_sizes': data_sizes
        }
    
    def _estimate_data_size(self, data: Dict[str, Any]) -> int:
        """Estimate the size of data in bytes."""
        import sys
        
        total_size = 0
        for key, value in data.items():
            # Size of key
            total_size += sys.getsizeof(key)
            
            # Size of value
            if hasattr(value, 'nbytes'):  # numpy array
                total_size += value.nbytes
            else:
                total_size += sys.getsizeof(value)
        
        return total_size
