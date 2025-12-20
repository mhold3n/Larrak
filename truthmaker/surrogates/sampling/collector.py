"""
CEM Data Collector: Automatic training data collection from CEM evaluations.

Logs (x, y_low, margins, regime_id, constraints) tuples for surrogate training.
Supports incremental CSV storage and multi-fidelity schema.
"""

from __future__ import annotations

import os
import csv
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from truthmaker.cem import (
    CEMClient,
    ValidationReport,
    OperatingRegime,
)


@dataclass
class TrainingSample:
    """Single training sample with CEM metadata."""
    # Timestamp
    timestamp: str
    
    # Design inputs (flattened)
    x_inputs: Dict[str, float]
    
    # Low-fidelity outputs (from reduced models)
    y_low: Dict[str, float]
    
    # High-fidelity outputs (optional, for Δ-learning)
    y_high: Optional[Dict[str, float]] = None
    
    # CEM metadata
    margins: Dict[str, float] = field(default_factory=dict)
    constraint_codes: List[int] = field(default_factory=list)
    regime_id: int = 0
    is_valid: bool = True
    cem_version: str = ""
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Flatten for CSV storage."""
        row = {
            'timestamp': self.timestamp,
            'regime_id': self.regime_id,
            'is_valid': self.is_valid,
            'cem_version': self.cem_version,
            'constraint_codes': json.dumps(self.constraint_codes),
        }
        
        # Flatten inputs with prefix
        for k, v in self.x_inputs.items():
            row[f'x_{k}'] = v
        
        # Flatten y_low outputs
        for k, v in self.y_low.items():
            row[f'y_low_{k}'] = v
        
        # Flatten y_high if available
        if self.y_high:
            for k, v in self.y_high.items():
                row[f'y_high_{k}'] = v
        
        # Flatten margins
        for k, v in self.margins.items():
            row[f'margin_{k}'] = v
        
        return row


class CEMDataCollector:
    """
    Collects training data from CEM evaluations for surrogate training.
    
    Features:
    - Incremental CSV storage (append mode)
    - Multi-fidelity support (y_low now, y_high later)
    - Automatic metadata extraction from ValidationReport
    
    Usage:
        collector = CEMDataCollector("training_data.csv")
        
        # During evaluation loop:
        report = cem.validate_motion(x_profile)
        metadata = cem.extract_training_metadata(report)
        collector.log_evaluation(
            x_inputs={'rpm': 2000, 'boost': 1.5},
            y_low={'efficiency': 0.42, 'p_max': 80.0},
            cem_metadata=metadata
        )
        
        # Later, add high-fidelity labels:
        collector.log_high_fidelity(
            timestamp="2024-01-01T12:00:00",
            y_high={'efficiency': 0.45, 'p_max': 85.0}
        )
    """
    
    def __init__(
        self, 
        output_path: str,
        auto_flush: bool = True,
        flush_interval: int = 10
    ):
        """
        Initialize collector.
        
        Args:
            output_path: Path to CSV file for storage
            auto_flush: Whether to flush after each write
            flush_interval: If not auto_flush, flush every N samples
        """
        self.output_path = Path(output_path)
        self.auto_flush = auto_flush
        self.flush_interval = flush_interval
        
        self._buffer: List[Dict[str, Any]] = []
        self._fieldnames: Optional[List[str]] = None
        self._sample_count = 0
        
        # Create directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing fieldnames if file exists
        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            with open(self.output_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                self._fieldnames = reader.fieldnames
    
    def log_evaluation(
        self,
        x_inputs: Dict[str, float],
        y_low: Dict[str, float],
        cem_metadata: Dict[str, Any],
        timestamp: Optional[str] = None
    ) -> None:
        """
        Log a training sample from a CEM evaluation.
        
        Args:
            x_inputs: Design input variables (e.g., {'rpm': 2000, 'boost': 1.5})
            y_low: Low-fidelity outputs (e.g., {'efficiency': 0.42})
            cem_metadata: Output from CEMClient.extract_training_metadata()
            timestamp: Optional timestamp, defaults to now
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        sample = TrainingSample(
            timestamp=timestamp,
            x_inputs=x_inputs,
            y_low=y_low,
            margins=cem_metadata.get('margins', {}),
            constraint_codes=cem_metadata.get('constraint_codes', []),
            regime_id=cem_metadata.get('regime_id', 0),
            is_valid=cem_metadata.get('is_valid', True),
            cem_version=cem_metadata.get('cem_version', ''),
        )
        
        self._write_sample(sample)
    
    def log_high_fidelity(
        self,
        timestamp: str,
        y_high: Dict[str, float]
    ) -> None:
        """
        Add high-fidelity labels to an existing sample (for Δ-learning).
        
        Note: This requires rewriting the CSV or using a separate HiFi file.
        For simplicity, this implementation appends to a separate _hifi.csv file.
        
        Args:
            timestamp: Timestamp of the original sample to match
            y_high: High-fidelity outputs
        """
        hifi_path = self.output_path.with_suffix('.hifi.csv')
        
        row = {
            'timestamp': timestamp,
            **{f'y_high_{k}': v for k, v in y_high.items()}
        }
        
        # Check if file exists to determine if we need headers
        write_header = not hifi_path.exists() or hifi_path.stat().st_size == 0
        
        with open(hifi_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    
    def _write_sample(self, sample: TrainingSample) -> None:
        """Write a sample to the CSV file."""
        row = sample.to_flat_dict()
        
        # Update fieldnames if we see new columns
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        else:
            # Extend fieldnames for any new columns
            for key in row.keys():
                if key not in self._fieldnames:
                    self._fieldnames.append(key)
        
        self._buffer.append(row)
        self._sample_count += 1
        
        if self.auto_flush or self._sample_count % self.flush_interval == 0:
            self._flush()
    
    def _flush(self) -> None:
        """Flush buffer to disk."""
        if not self._buffer:
            return
        
        # Check if we need to write header
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerows(self._buffer)
        
        self._buffer.clear()
    
    def get_sample_count(self) -> int:
        """Return total number of samples collected (including existing)."""
        if not self.output_path.exists():
            return self._sample_count
        
        with open(self.output_path, 'r', newline='') as f:
            return sum(1 for _ in f) - 1 + len(self._buffer)  # -1 for header
    
    def close(self) -> None:
        """Flush remaining buffer and close."""
        self._flush()
    
    def __enter__(self) -> 'CEMDataCollector':
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
