"""
Optimization result registry system.

This module provides a centralized registry for managing optimization results
across different optimization components, enabling cascaded optimization
where secondary optimizers can access results from primary optimizers.
"""

from typing import Dict, List, Optional, Any, Union
import time

from .base import BaseStorage, StorageResult, StorageStatus
from .memory import MemoryStorage
from campro.logging import get_logger

log = get_logger(__name__)


class OptimizationRegistry:
    """
    Centralized registry for optimization results.
    
    This registry manages optimization results from different optimizers,
    enabling cascaded optimization where secondary optimizers can access
    and use results from primary optimizers.
    """
    
    def __init__(self, storage: Optional[BaseStorage] = None):
        self.storage = storage or MemoryStorage("OptimizationRegistry")
        self._optimization_chains: Dict[str, List[str]] = {}  # chain_id -> [optimizer_ids]
        self._optimizer_results: Dict[str, str] = {}  # optimizer_id -> storage_key
    
    def register_optimizer(self, optimizer_id: str, chain_id: Optional[str] = None) -> None:
        """
        Register an optimizer in the registry.
        
        Args:
            optimizer_id: Unique identifier for the optimizer
            chain_id: Optional chain identifier for cascaded optimization
        """
        if chain_id is None:
            chain_id = "default"
        
        if chain_id not in self._optimization_chains:
            self._optimization_chains[chain_id] = []
        
        if optimizer_id not in self._optimization_chains[chain_id]:
            self._optimization_chains[chain_id].append(optimizer_id)
            log.info(f"Registered optimizer '{optimizer_id}' in chain '{chain_id}'")
    
    def store_result(self, optimizer_id: str, result_data: Dict[str, Any],
                    metadata: Optional[Dict[str, Any]] = None,
                    constraints: Optional[Dict[str, Any]] = None,
                    optimization_rules: Optional[Dict[str, Any]] = None,
                    solver_settings: Optional[Dict[str, Any]] = None,
                    expires_in: Optional[float] = None) -> StorageResult:
        """
        Store optimization result for an optimizer.
        
        Args:
            optimizer_id: Identifier of the optimizer
            result_data: Optimization result data
            metadata: Optional metadata
            expires_in: Optional expiration time in seconds
            
        Returns:
            StorageResult object
        """
        # Create storage key
        storage_key = f"{optimizer_id}_{int(time.time())}"
        
        # Add optimizer metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            'optimizer_id': optimizer_id,
            'timestamp': time.time(),
            'chain_id': self._get_chain_id(optimizer_id)
        })
        
        # Store the result
        storage_result = self.storage.store(
            key=storage_key,
            data=result_data,
            metadata=metadata,
            constraints=constraints,
            optimization_rules=optimization_rules,
            solver_settings=solver_settings,
            expires_in=expires_in
        )
        
        # Track the result
        self._optimizer_results[optimizer_id] = storage_key
        
        log.info(f"Stored result for optimizer '{optimizer_id}' with key '{storage_key}'")
        return storage_result
    
    def get_result(self, optimizer_id: str) -> Optional[StorageResult]:
        """
        Get the latest result for an optimizer.
        
        Args:
            optimizer_id: Identifier of the optimizer
            
        Returns:
            StorageResult object or None if not found
        """
        if optimizer_id not in self._optimizer_results:
            log.debug(f"No result found for optimizer '{optimizer_id}'")
            return None
        
        storage_key = self._optimizer_results[optimizer_id]
        result = self.storage.retrieve(storage_key)
        
        if result is None:
            log.warning(f"Result for optimizer '{optimizer_id}' not accessible (expired or removed)")
            # Remove from tracking
            del self._optimizer_results[optimizer_id]
        
        return result
    
    def get_chain_results(self, chain_id: str) -> Dict[str, StorageResult]:
        """
        Get all results for a specific optimization chain.
        
        Args:
            chain_id: Chain identifier
            
        Returns:
            Dictionary mapping optimizer_id to StorageResult
        """
        if chain_id not in self._optimization_chains:
            log.debug(f"No chain found with id '{chain_id}'")
            return {}
        
        results = {}
        for optimizer_id in self._optimization_chains[chain_id]:
            result = self.get_result(optimizer_id)
            if result is not None:
                results[optimizer_id] = result
        
        return results
    
    def get_available_results(self, optimizer_id: str) -> Dict[str, StorageResult]:
        """
        Get all available results that an optimizer can access.
        
        Args:
            optimizer_id: Identifier of the requesting optimizer
            
        Returns:
            Dictionary mapping optimizer_id to StorageResult
        """
        chain_id = self._get_chain_id(optimizer_id)
        if chain_id is None:
            return {}
        
        # Get all results from the same chain
        chain_results = self.get_chain_results(chain_id)
        
        # Filter to only include results from optimizers that run before this one
        optimizer_index = self._optimization_chains[chain_id].index(optimizer_id)
        available_results = {}
        
        for i, other_optimizer_id in enumerate(self._optimization_chains[chain_id]):
            if i < optimizer_index and other_optimizer_id in chain_results:
                available_results[other_optimizer_id] = chain_results[other_optimizer_id]
        
        return available_results
    
    def clear_chain(self, chain_id: str) -> int:
        """
        Clear all results for a specific chain.
        
        Args:
            chain_id: Chain identifier
            
        Returns:
            Number of results cleared
        """
        if chain_id not in self._optimization_chains:
            return 0
        
        cleared_count = 0
        for optimizer_id in self._optimization_chains[chain_id]:
            if optimizer_id in self._optimizer_results:
                storage_key = self._optimizer_results[optimizer_id]
                if self.storage.remove(storage_key):
                    cleared_count += 1
                del self._optimizer_results[optimizer_id]
        
        log.info(f"Cleared {cleared_count} results for chain '{chain_id}'")
        return cleared_count
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        storage_stats = self.storage.get_storage_stats()
        
        return {
            'storage_stats': storage_stats,
            'total_chains': len(self._optimization_chains),
            'total_optimizers': len(self._optimizer_results),
            'chains': {
                chain_id: {
                    'optimizer_count': len(optimizers),
                    'optimizers': optimizers
                }
                for chain_id, optimizers in self._optimization_chains.items()
            }
        }
    
    def _get_chain_id(self, optimizer_id: str) -> Optional[str]:
        """Get the chain ID for an optimizer."""
        for chain_id, optimizers in self._optimization_chains.items():
            if optimizer_id in optimizers:
                return chain_id
        return None
