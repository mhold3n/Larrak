"""
CEM-Constrained Sampling: Generate training samples within feasible envelopes.

Provides Sobol and Latin Hypercube sampling constrained to CEM feasible regions.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Dict, Any
import numpy as np

try:
    from scipy.stats.qmc import Sobol, LatinHypercube
    HAS_SCIPY_QMC = True
except ImportError:
    HAS_SCIPY_QMC = False

from truthmaker.cem import CEMClient, OperatingEnvelope


def sample_feasible_envelope(
    cem: CEMClient,
    n_samples: int,
    bounds: Dict[str, Tuple[float, float]],
    method: Literal['sobol', 'lhs', 'random'] = 'sobol',
    seed: Optional[int] = None,
    engine_geometry: Optional[Dict[str, float]] = None,
    rpm_range: Tuple[float, float] = (800, 5000),
) -> np.ndarray:
    """
    Sample from CEM-defined feasible region, not the full hypercube.
    
    Args:
        cem: CEM client instance
        n_samples: Number of samples to generate
        bounds: Dict mapping variable names to (min, max) bounds
            e.g., {'rpm': (800, 5000), 'boost_bar': (0.5, 4.0), 'fuel_mg': (10, 200)}
        method: 'sobol', 'lhs', or 'random'
        seed: Random seed for reproducibility
        engine_geometry: Optional dict with 'bore_m', 'stroke_m', 'compression_ratio'
            for envelope generation
        rpm_range: RPM range over which to sample
        
    Returns:
        np.ndarray of shape (n_samples, n_dims) with feasible samples
    """
    var_names = list(bounds.keys())
    n_dims = len(var_names)
    
    # Generate unit hypercube samples
    if method == 'sobol':
        if not HAS_SCIPY_QMC:
            raise ImportError("scipy.stats.qmc required for Sobol sampling")
        sampler = Sobol(d=n_dims, scramble=True, seed=seed)
        samples_unit = sampler.random(n_samples)
    elif method == 'lhs':
        if not HAS_SCIPY_QMC:
            raise ImportError("scipy.stats.qmc required for LHS sampling")
        sampler = LatinHypercube(d=n_dims, seed=seed)
        samples_unit = sampler.random(n_samples)
    else:  # random
        rng = np.random.default_rng(seed)
        samples_unit = rng.random((n_samples, n_dims))
    
    # Scale to bounds
    lower = np.array([bounds[v][0] for v in var_names])
    upper = np.array([bounds[v][1] for v in var_names])
    samples = lower + samples_unit * (upper - lower)
    
    # If we have engine geometry, use CEM to get tighter bounds per operating point
    if engine_geometry is not None and cem is not None:
        samples = _filter_by_envelope(
            samples, var_names, cem, engine_geometry, rpm_range
        )
    
    return samples


def _filter_by_envelope(
    samples: np.ndarray,
    var_names: list,
    cem: CEMClient,
    engine_geometry: Dict[str, float],
    rpm_range: Tuple[float, float]
) -> np.ndarray:
    """Filter samples to those within CEM thermo envelope."""
    # Find variable indices
    rpm_idx = var_names.index('rpm') if 'rpm' in var_names else None
    boost_idx = var_names.index('boost_bar') if 'boost_bar' in var_names else None
    fuel_idx = var_names.index('fuel_mg') if 'fuel_mg' in var_names else None
    
    if rpm_idx is None:
        return samples  # No RPM to filter on
    
    feasible_mask = np.ones(len(samples), dtype=bool)
    
    for i, sample in enumerate(samples):
        rpm = sample[rpm_idx]
        
        # Get envelope for this RPM
        envelope = cem.get_thermo_envelope(
            bore=engine_geometry.get('bore_m', 0.08),
            stroke=engine_geometry.get('stroke_m', 0.15),
            cr=engine_geometry.get('compression_ratio', 12.0),
            rpm=rpm
        )
        
        if not envelope.feasible:
            feasible_mask[i] = False
            continue
        
        # Check boost bounds
        if boost_idx is not None:
            boost = sample[boost_idx]
            if not (envelope.boost_range[0] <= boost <= envelope.boost_range[1]):
                feasible_mask[i] = False
                continue
        
        # Check fuel bounds
        if fuel_idx is not None:
            fuel = sample[fuel_idx]
            if not (envelope.fuel_range[0] <= fuel <= envelope.fuel_range[1]):
                feasible_mask[i] = False
    
    return samples[feasible_mask]


def densify_near_constraints(
    samples: np.ndarray,
    var_names: list,
    cem: CEMClient,
    n_additional: int = 50,
    margin_threshold: float = 0.1,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add samples near constraint boundaries for better boundary learning.
    
    Args:
        samples: Existing samples array of shape (n, d)
        var_names: Names of variables
        cem: CEM client for validation
        n_additional: Number of additional samples to generate
        margin_threshold: Relative margin threshold for "near boundary"
        seed: Random seed
        
    Returns:
        Extended samples array with additional boundary samples
    """
    rng = np.random.default_rng(seed)
    
    # Find samples near constraints
    boundary_samples = []
    
    for sample in samples:
        # Create input dict for motion validation (simplified)
        # In practice, you'd convert sample to a motion profile
        # For now, we identify boundary samples and perturb them
        pass  # Placeholder - requires domain-specific motion generation
    
    # For now, just perturb existing samples slightly
    if len(samples) > 0:
        # Pick random samples to perturb
        n_to_perturb = min(n_additional, len(samples))
        indices = rng.choice(len(samples), n_to_perturb, replace=False)
        
        # Perturb by small amount (5% of range)
        perturbed = samples[indices].copy()
        ranges = samples.max(axis=0) - samples.min(axis=0) + 1e-6
        noise = rng.uniform(-0.05, 0.05, perturbed.shape) * ranges
        perturbed += noise
        
        boundary_samples = perturbed
    
    if len(boundary_samples) > 0:
        return np.vstack([samples, boundary_samples])
    return samples


def generate_stratified_doe(
    n_samples_per_regime: Dict[int, int],
    bounds: Dict[str, Tuple[float, float]],
    rpm_idle: float = 800.0,
    rpm_cruise_max: float = 3000.0,
    method: Literal['sobol', 'lhs', 'random'] = 'lhs',
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate stratified DOE with samples per operating regime.
    
    Args:
        n_samples_per_regime: Dict mapping regime_id to sample count
            e.g., {1: 50, 2: 200, 3: 100}  # idle, cruise, full_load
        bounds: Variable bounds
        rpm_idle: Threshold for idle regime
        rpm_cruise_max: Threshold for cruise regime (above = full_load)
        method: Sampling method
        seed: Random seed
        
    Returns:
        Tuple of (samples, regime_ids)
    """
    all_samples = []
    all_regimes = []
    
    var_names = list(bounds.keys())
    rpm_idx = var_names.index('rpm') if 'rpm' in var_names else None
    
    rng = np.random.default_rng(seed)
    
    for regime_id, n_samples in n_samples_per_regime.items():
        # Adjust RPM bounds based on regime
        regime_bounds = bounds.copy()
        
        if rpm_idx is not None and 'rpm' in regime_bounds:
            rpm_min, rpm_max = regime_bounds['rpm']
            
            if regime_id == 1:  # idle
                regime_bounds['rpm'] = (rpm_min, min(rpm_idle, rpm_max))
            elif regime_id == 2:  # cruise
                regime_bounds['rpm'] = (max(rpm_min, rpm_idle), min(rpm_cruise_max, rpm_max))
            elif regime_id == 3:  # full_load
                regime_bounds['rpm'] = (max(rpm_min, rpm_cruise_max), rpm_max)
        
        # Generate samples for this regime
        samples = sample_feasible_envelope(
            cem=None,  # Don't filter by envelope here
            n_samples=n_samples,
            bounds=regime_bounds,
            method=method,
            seed=rng.integers(0, 2**31) if seed else None
        )
        
        all_samples.append(samples)
        all_regimes.extend([regime_id] * len(samples))
    
    return np.vstack(all_samples), np.array(all_regimes)
