"""
Sampling Utilities for Design of Experiments (DOE).
"""

import numpy as np
import itertools
from typing import List, Dict, Any

def generate_grid_sample(ranges: Dict[str, List[float]]) -> List[Dict[str, float]]:
    """
    Generate a Full Factorial Grid sample.
    
    Args:
        ranges: Dict mapping variable name to list of levels.
                e.g. {"rpm": [1000, 2000], "load": [0.5, 1.0]}
                
    Returns:
        List of dictionaries defining the DOE points.
    """
    keys = list(ranges.keys())
    values = list(ranges.values())
    
    samples = []
    for combination in itertools.product(*values):
        point = dict(zip(keys, combination))
        samples.append(point)
        
    return samples

def generate_latin_hypercube_sample(
    ranges: Dict[str, tuple[float, float]], 
    n_samples: int,
    seed: int = 42
) -> List[Dict[str, float]]:
    """
    Generate a Latin Hypercube Sample (LHS).
    Ensures each variable is sampled `n_samples` times across its range.
    
    Args:
        ranges: Dict mapping variable name to (min, max) tuple.
        n_samples: Number of points to generate.
        seed: Random seed.
        
    Returns:
        List of dictionaries defining the DOE points.
    """
    rng = np.random.default_rng(seed)
    keys = list(ranges.keys())
    n_vars = len(keys)
    
    # 1. Generate Stratified Samples (0..1)
    # Shape: (n_samples, n_vars)
    # Each column is a permutation of linspace(0, 1, n) + jitter
    
    raw_samples = np.zeros((n_samples, n_vars))
    
    interval_size = 1.0 / n_samples
    
    for i in range(n_vars):
        # Stratified bins
        bins = np.linspace(0, 1.0 - interval_size, n_samples)
        # Add jitter within bin
        jitter = rng.uniform(0, interval_size, n_samples)
        # Combine
        col = bins + jitter
        # Shuffle
        rng.shuffle(col)
        raw_samples[:, i] = col
        
    # 2. Scale to Ranges
    samples = []
    for r in range(n_samples):
        point = {}
        for c, key in enumerate(keys):
            min_val, max_val = ranges[key]
            norm_val = raw_samples[r, c]
            # IDW Scale
            val = min_val + norm_val * (max_val - min_val)
            point[key] = float(val)
        samples.append(point)
        
    return samples

def generate_stratified_training_set(bounds: Dict[str, tuple[float, float]], n_points: int) -> List[Dict[str, float]]:
    """
    Generates a robust training set that includes Boundaries + LHS interior.
    Crucial for surrogates to avoid extrapolation.
    """
    # 1. Corners (Full Factorial of Bounds)
    # If 2 vars, 4 corners. If 3 vars, 8 corners.
    corners = []
    keys = list(bounds.keys())
    ranges = {k: [v[0], v[1]] for k, v in bounds.items()}
    corners = generate_grid_sample(ranges)
    
    # 2. Interior (LHS)
    n_interior = max(0, n_points - len(corners))
    if n_interior > 0:
        interior = generate_latin_hypercube_sample(bounds, n_interior)
        return corners + interior
    
    return corners[:n_points]
