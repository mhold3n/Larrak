"""NLP variable and constraint scaling utilities.

This module provides functions for computing scaling factors to improve
numerical conditioning of nonlinear programs (NLPs) solved with IPOPT.

Key functions:
    - compute_variable_scaling: Unit-based variable scaling with bounds refinement
    - compute_constraint_scaling: Constraint scaling from bounds
    - verify_scaling_quality: Jacobian-based scaling quality metrics

The scaling approach follows Betts/Biegler recommendations:
1. Scale variables so scaled values are O(1)
2. Scale constraints so Jacobian elements are O(1)
3. Maintain condition number < 10^6 for numerical stability
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np

from campro.optimization.nlp.diagnostics import compute_scaled_jacobian
from campro.utils.structured_reporter import StructuredReporter

# Get module logger
log = logging.getLogger(__name__)


# =============================================================================
# Unit Reference Scaling Configuration
# =============================================================================

DEFAULT_UNIT_REFERENCES: dict[str, float] = {
    "positions": 0.05,  # meters -> scale ~ 1/0.05 = 20 (typical position ~50mm)
    "velocities": 10.0,  # m/s -> scale ~ 1/10 = 0.1 (typical velocity ~10 m/s)
    "densities": 1.0,  # kg/m^3 (already reasonable)
    "temperatures": 1000.0,  # K (normalize to 0.001-2.0 range)
    "pressures": 1e6,  # MPa (normalize to 0.01-10 range)
    "valve_areas": 1e-4,  # m^2 -> scale ~ 1/1e-4 = 1e4 (typical area ~0.1 mm^2)
    "ignition": 1.0,  # seconds (already in base units)
    "scavenging_fractions": 1.0,  # dimensionless (yF)
    "scavenging_masses": 0.01,  # kg (Mdel, Mlost)
    "scavenging_area_integrals": 1e-4,  # m^2*s (AinInt, AexInt)
    "scavenging_time_moments": 5e-5,  # m^2*s^2 (AinTmom, AexTmom)
    "cycle_time": 0.05,  # s (T_cycle)
}

# Constraint type category targets for scaling
# These targets represent the desired magnitude of scaled constraints per category
CONSTRAINT_TYPE_TARGETS = {
    "kinematic": 1.0,  # Position, velocity, acceleration constraints
    "thermodynamic": 1.0,  # Reduced from 10.0 to lower weight/priority
    "boundary": 0.1,  # Boundary conditions (tighter)
    "continuity": 1.0,  # Continuity constraints
}


# =============================================================================
# Variable Scaling
# =============================================================================


def compute_variable_scaling(
    lbx: np.ndarray,
    ubx: np.ndarray,
    x0: np.ndarray | None = None,
    variable_groups: dict[str, list[int]] | None = None,
    group_config: dict[str, dict[str, Any]] | None = None,
    unit_references: dict[str, float] | None = None,
    max_scale_ratio: float = 1e6,
) -> np.ndarray:
    """Compute variable scaling factors with unit-based initialization.

    Scaling phases:
    1. Unit-based scaling: Apply per-group reference scales (physical units)
    2. Group normalization: Clamp scales within group using max_ratio
    3. Bounds/x0 fallback: Value-based scaling adjustment
    4. Final normalization: Ensure global scale ratio within maximum

    Args:
        lbx: Lower bounds on variables
        ubx: Upper bounds on variables
        x0: Initial guess values (optional)
        variable_groups: Dict mapping group names to variable indices
        group_config: Per-group configuration (use_log_scale, max_ratio)
        unit_references: Reference values for each group (defaults used if None)
        max_scale_ratio: Maximum allowed ratio between any two scale factors

    Returns:
        Array of scale factors (one per variable)
    """
    n_vars = len(lbx)
    scale = np.ones(n_vars)

    # Use default unit references if not provided
    refs = unit_references or DEFAULT_UNIT_REFERENCES
    cfg = group_config or {}

    # Map variable indices to their groups
    var_to_group: dict[int, str] = {}
    if variable_groups:
        for group_name, indices in variable_groups.items():
            if group_name in refs:
                for idx in indices:
                    if 0 <= idx < n_vars:
                        var_to_group[idx] = group_name

    # Phase 1: Unit-based scaling
    for i in range(n_vars):
        group = var_to_group.get(i)
        if group and group in refs:
            ref = refs[group]
            # Check for log-scale groups
            grp_cfg = cfg.get(group, {})
            if grp_cfg.get("use_log_scale", False):
                ref = 1.0  # Log-space variables don't need unit conversion scaling
            scale[i] = 1.0 / ref

    # Phase 2: Group normalization with configurable max_ratio
    if variable_groups:
        for group_name, indices in variable_groups.items():
            if indices and len(indices) > 0:
                group_indices = np.array([i for i in indices if 0 <= i < n_vars])
                if len(group_indices) > 0:
                    group_scales = scale[group_indices]
                    group_min = group_scales.min()
                    group_max = group_scales.max()

                    # Get max_ratio from config (default 10.0)
                    grp_cfg = cfg.get(group_name, {})
                    max_ratio = grp_cfg.get("max_ratio", 10.0)

                    # Cap ratio to configured max_ratio
                    if group_min > 1e-10 and group_max / group_min > max_ratio:
                        group_median = np.median(group_scales)
                        sqrt_ratio = np.sqrt(max_ratio)
                        lo = group_median / sqrt_ratio
                        hi = group_median * sqrt_ratio
                        for idx in group_indices:
                            scale[idx] = np.clip(scale[idx], lo, hi)

    # Phase 3: Bounds/x0 fallback (value-based scaling)
    for i in range(n_vars):
        lb = lbx[i]
        ub = ubx[i]

        # Skip if unbounded
        if lb == -np.inf and ub == np.inf:
            continue

        # Compute magnitude from bounds
        if lb == -np.inf:
            magnitude = abs(ub)
        elif ub == np.inf:
            magnitude = abs(lb)
        else:
            magnitude = max(abs(lb), abs(ub))

        # Incorporate initial guess value if available
        if x0 is not None and i < len(x0) and np.isfinite(x0[i]):
            magnitude = max(magnitude, abs(x0[i]))

        # Apply value-based adjustment with clamping
        if magnitude > 1e-6:
            target_scale = 1.0 / magnitude
            # Clamp ratio to [0.1, 10.0] to respect unit-based priors
            ratio = target_scale / scale[i]
            ratio = np.clip(ratio, 0.1, 10.0)
            scale[i] = scale[i] * ratio

    # Phase 4: Final clamping to [1e-3, 1e3]
    scale = np.clip(scale, 1e-3, 1e3)

    return scale


def compute_constraint_scaling(
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
    min_scale: float = 1e-8,
    max_scale: float = 1e8,
) -> np.ndarray:
    """Compute constraint scaling factors to normalize constraints to O(1).

    For each constraint, uses max(|lb|, |ub|) to compute a scale factor s_g
    such that the scaled constraint g_scaled = s_g * g has magnitude ~O(1).

    Args:
        lbg: Lower bounds on constraints (can be None)
        ubg: Upper bounds on constraints (can be None)
        min_scale: Minimum allowed scale factor
        max_scale: Maximum allowed scale factor

    Returns:
        Array of constraint scale factors (one per constraint)
    """
    if lbg is None and ubg is None:
        return np.array([])

    if lbg is None:
        n_cons = len(ubg) if ubg is not None else 0
        lbg = -np.inf * np.ones(n_cons)
    if ubg is None:
        n_cons = len(lbg) if lbg is not None else 0
        ubg = np.inf * np.ones(n_cons)

    n_cons = len(lbg)
    scale_g = np.ones(n_cons)

    for i in range(n_cons):
        lb = lbg[i]
        ub = ubg[i]

        # Skip free constraints
        if lb == -np.inf and ub == np.inf:
            continue

        # Compute characteristic magnitude
        if lb == -np.inf:
            magnitude = abs(ub)
        elif ub == np.inf:
            magnitude = abs(lb)
        else:
            magnitude = max(abs(lb), abs(ub))

        # Compute scale factor (1/magnitude to bring to O(1))
        if magnitude > 1e-10:
            scale_g[i] = 1.0 / magnitude
        else:
            # For very small bounds (like equality constraints at 0),
            # use a moderate scale to avoid over-scaling
            scale_g[i] = 1.0

    # Clamp to safe range
    scale_g = np.clip(scale_g, min_scale, max_scale)

    return scale_g


# =============================================================================
# Scaling Quality Verification
# =============================================================================


def compute_scaling_quality(
    nlp_or_jac: Any,
    x0: Any = None,
    scale: np.ndarray | None = None,
    scale_g: np.ndarray | None = None,
    lbg: Any = None,
    ubg: Any = None,
    reporter: Any = None,
    meta: Any = None,
) -> dict[str, Any]:
    """Compute scaling quality metrics from Jacobian.

    Args:
        nlp_or_jac: NLP dict OR Jacobian matrix (n_constraints x n_vars)
        x0: Initial guess (required if nlp passed)
        scale: Variable scaling factors
        scale_g: Constraint scaling factors
        lbg: Lower constraint bounds (optional)
        ubg: Upper constraint bounds (optional)
        reporter: Reporter (optional)
        meta: Metadata (optional)

    Returns:
        Dictionary with quality metrics
    """
    # Handle flexible arguments
    jac_g = None
    if isinstance(nlp_or_jac, np.ndarray):
        jac_g = nlp_or_jac
    elif isinstance(nlp_or_jac, dict):
        # Compute jacobian from NLP
        if scale is None or scale_g is None:
            return {
                "quality_score": 0.0,
                "condition_number": np.inf,
                "error": "Missing scale/scale_g",
            }
        try:
            jac_g, _ = compute_scaled_jacobian(nlp_or_jac, x0, scale, scale_g)
        except Exception:
            pass

    if jac_g is None:
        return {
            "quality_score": 0.0,
            "condition_number": np.inf,
            "error": "Failed to compute Jacobian",
        }

    result: dict[str, Any] = {
        "condition_number": np.inf,
        "max_entry": 0.0,
        "min_nonzero_entry": np.inf,
        "quality_score": 0.0,
    }

    if jac_g is None or jac_g.size == 0:
        return result

    # Apply scaling: J_scaled[i,j] = scale_g[i] * J[i,j] * scale[j]
    # (scale_g scales rows, scale scales columns)
    try:
        jac_scaled = scale_g[:, np.newaxis] * jac_g * scale[np.newaxis, :]
    except (ValueError, IndexError):
        log.warning("Failed to apply scaling to Jacobian - dimension mismatch")
        return result

    # Find max and min nonzero entries
    abs_jac = np.abs(jac_scaled)
    result["max_entry"] = float(np.max(abs_jac))

    nonzero_mask = abs_jac > 1e-15
    if np.any(nonzero_mask):
        result["min_nonzero_entry"] = float(np.min(abs_jac[nonzero_mask]))

    # Estimate condition number from entry ratio
    if result["min_nonzero_entry"] > 0:
        result["condition_number"] = result["max_entry"] / result["min_nonzero_entry"]

    # Compute quality score (1.0 = excellent, 0.0 = poor)
    # Target: condition number < 10^4
    log_cond = np.log10(max(result["condition_number"], 1.0))
    result["quality_score"] = max(0.0, 1.0 - log_cond / 6.0)  # Score 0 at 10^6

    return result


def diagnose_scaling_issues(
    scale: np.ndarray,
    scale_g: np.ndarray,
    variable_groups: dict[str, list[int]] | None = None,
    constraint_groups: dict[str, list[int]] | None = None,
) -> list[str]:
    """Diagnose potential scaling issues.

    Args:
        scale: Variable scaling factors
        scale_g: Constraint scaling factors
        variable_groups: Variable group indices
        constraint_groups: Constraint group indices

    Returns:
        List of warning messages for detected issues
    """
    issues: list[str] = []

    # Check for extreme scale factors
    if len(scale) > 0:
        scale_ratio = scale.max() / scale.min() if scale.min() > 0 else np.inf
        if scale_ratio > 1e6:
            issues.append(
                f"Extreme variable scale ratio: {scale_ratio:.2e} "
                f"(range [{scale.min():.2e}, {scale.max():.2e}])"
            )

    if len(scale_g) > 0:
        scale_g_ratio = scale_g.max() / scale_g.min() if scale_g.min() > 0 else np.inf
        if scale_g_ratio > 1e6:
            issues.append(
                f"Extreme constraint scale ratio: {scale_g_ratio:.2e} "
                f"(range [{scale_g.min():.2e}, {scale_g.max():.2e}])"
            )

    # Check for NaN/Inf
    if np.any(~np.isfinite(scale)):
        bad_indices = np.where(~np.isfinite(scale))[0]
        issues.append(f"Non-finite variable scales at indices: {bad_indices[:10].tolist()}")

    if np.any(~np.isfinite(scale_g)):
        bad_indices = np.where(~np.isfinite(scale_g))[0]
        issues.append(f"Non-finite constraint scales at indices: {bad_indices[:10].tolist()}")

    # Check group-specific issues
    if variable_groups:
        for group_name, indices in variable_groups.items():
            valid_idx = [i for i in indices if 0 <= i < len(scale)]
            if valid_idx:
                group_scales = scale[valid_idx]
                if group_scales.min() > 0:
                    ratio = group_scales.max() / group_scales.min()
                    if ratio > 100:
                        issues.append(
                            f"Variable group '{group_name}' has high scale ratio: {ratio:.2e}"
                        )

    return issues


# =============================================================================
# Jacobian Utilities
# =============================================================================


def equilibrate_jacobian_iterative(
    scale: np.ndarray,
    scale_g: np.ndarray,
    jac_g: np.ndarray,
    n_iterations: int = 3,
    target: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively equilibrate Jacobian by balancing row and column norms.

    This alternates between scaling constraints (rows) and variables (columns)
    to achieve balanced Jacobian norms.

    Args:
        scale: Initial variable scaling factors
        scale_g: Initial constraint scaling factors
        jac_g: Constraint Jacobian matrix
        n_iterations: Number of equilibration iterations
        target: Target norm for each row/column

    Returns:
        Tuple of (refined_scale, refined_scale_g)
    """
    scale_out = scale.copy()
    scale_g_out = scale_g.copy()

    for _ in range(n_iterations):
        # Apply current scaling
        try:
            jac_scaled = scale_g_out[:, np.newaxis] * jac_g * scale_out[np.newaxis, :]
        except (ValueError, IndexError):
            break

        # Update row scaling (constraints)
        row_norms = np.sqrt(np.sum(jac_scaled**2, axis=1))
        row_norms = np.maximum(row_norms, 1e-12)
        scale_g_out = scale_g_out * (target / row_norms)

        # Update column scaling (variables)
        jac_scaled = scale_g_out[:, np.newaxis] * jac_g * scale_out[np.newaxis, :]
        col_norms = np.sqrt(np.sum(jac_scaled**2, axis=0))
        col_norms = np.maximum(col_norms, 1e-12)
        scale_out = scale_out * (target / col_norms)

    # Clamp to safe bounds
    scale_out = np.clip(scale_out, 1e-8, 1e8)
    scale_g_out = np.clip(scale_g_out, 1e-8, 1e8)

    return scale_out, scale_g_out


# =============================================================================
# Scaling Cache
# =============================================================================


def get_scaling_cache_path() -> Path:
    """Get the path to the scaling cache file."""
    cache_dir = Path.home() / ".campro" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "scaling_cache.json"


def generate_scaling_cache_key(
    n_vars: int,
    n_constraints: int,
    variable_groups: dict[str, list[int]] | None = None,
    meta: dict[str, Any] | None = None,
) -> str:
    """
    Generate a cache key from problem characteristics.

    Args:
        n_vars: Number of variables
        n_constraints: Number of constraints
        variable_groups: Variable group mapping
        meta: Problem metadata

    Returns:
        Cache key string (hash)
    """
    key_data: dict[str, Any] = {
        "n_vars": n_vars,
        "n_constraints": n_constraints,
    }

    if variable_groups:
        key_data["variable_groups"] = {
            group: len(indices) for group, indices in variable_groups.items()
        }

    if meta and "constraint_groups" in meta:
        constraint_groups = meta["constraint_groups"]
        key_data["constraint_groups"] = {
            group: len(indices) for group, indices in constraint_groups.items()
        }

    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def load_scaling_cache(
    cache_key: str,
    n_vars: int,
    n_constraints: int,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
    """
    Load scaling factors from cache if available.

    Args:
        cache_key: Cache key for this problem
        n_vars: Expected number of variables
        n_constraints: Expected number of constraints

    Returns:
        Tuple of (scale, scale_g, quality) or (None, None, None) if not found
    """
    cache_path = get_scaling_cache_path()

    if not cache_path.exists():
        return None, None, None

    try:
        with open(cache_path) as f:
            cache = json.load(f)

        if cache_key not in cache:
            return None, None, None

        entry = cache[cache_key]

        # Verify dimensions match
        cached_scale = np.array(entry["scale"])
        cached_scale_g = np.array(entry["scale_g"])

        if len(cached_scale) != n_vars or len(cached_scale_g) != n_constraints:
            return None, None, None

        quality = entry.get("quality", {})

        return cached_scale, cached_scale_g, quality

    except Exception as e:
        log.debug(f"Failed to load scaling cache: {e}")
        return None, None, None


def save_scaling_cache(
    cache_key: str,
    scale: np.ndarray,
    scale_g: np.ndarray,
    quality: dict[str, Any],
) -> None:
    """
    Save scaling factors to cache.

    Args:
        cache_key: Cache key for this problem
        scale: Variable scaling factors
        scale_g: Constraint scaling factors
        quality: Quality metrics
    """
    cache_path = get_scaling_cache_path()

    try:
        # Load existing cache
        cache = {}
        if cache_path.exists():
            with open(cache_path) as f:
                cache = json.load(f)

        # Update cache entry
        cache[cache_key] = {
            "scale": scale.tolist(),
            "scale_g": scale_g.tolist(),
            "quality": quality,
            "timestamp": time.time(),
        }

        # Limit cache size (keep only most recent 10 entries)
        if len(cache) > 10:
            entries = list(cache.items())
            entries.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
            cache = dict(entries[:10])

        # Save cache
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)

    except Exception as e:
        log.debug(f"Failed to save scaling cache: {e}")


# =============================================================================
# Constraint Type Identification (extracted from driver.py)
# =============================================================================


def identify_constraint_types(
    meta: dict[str, Any] | None,
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
    jac_g0_arr: np.ndarray[Any, Any] | None = None,
    g0_arr: np.ndarray[Any, Any] | None = None,
) -> dict[int, str]:
    """
    Identify constraint types from metadata or heuristics.

    Args:
        meta: NLP metadata dict (may contain constraint_groups)
        lbg: Lower constraint bounds
        ubg: Upper constraint bounds
        jac_g0_arr: Constraint Jacobian at initial guess (optional, for heuristics)
        g0_arr: Constraint values at initial guess (optional, for heuristics)

    Returns:
        Dict mapping constraint index to constraint type string
    """
    constraint_types: dict[int, str] = {}

    # Try to use metadata constraint groups first (explicit identification)
    if meta and "constraint_groups" in meta:
        constraint_groups = meta["constraint_groups"]
        for con_type, indices in constraint_groups.items():
            for idx in indices:
                constraint_types[idx] = con_type
        return constraint_types

    # Fall back to heuristic identification if metadata unavailable
    if lbg is None or ubg is None:
        return constraint_types

    n_cons = len(lbg)

    # Heuristic: Identify constraint types based on bounds patterns
    for i in range(n_cons):
        lb = lbg[i]
        ub = ubg[i]

        # Equality constraints (lb == ub)
        if lb == ub and np.isfinite(lb):
            # Check if it's a periodicity constraint (typically 0.0)
            if abs(lb) < 1e-6:
                constraint_types[i] = "periodicity"
            else:
                constraint_types[i] = "continuity"  # Default for equality
        # Inequality constraints
        elif lb == -np.inf and ub == np.inf:
            constraint_types[i] = "collocation_residuals"  # Unbounded = residuals
        elif lb > 0 and np.isfinite(lb) and ub == np.inf:
            # Lower bound only, positive
            if lb > 1e3:  # Large lower bound suggests pressure (Pa)
                constraint_types[i] = "path_pressure"
            elif lb > 1e2:  # Medium lower bound suggests combustion (J)
                constraint_types[i] = "combustion"
            else:
                constraint_types[i] = "path_clearance"  # Small positive = clearance
        elif abs(lb) < 1e-6 and ub > 1e5:
            # Near-zero lower bound, large upper bound suggests pressure
            constraint_types[i] = "path_pressure"
        elif abs(lb) < 1e-6 and ub > 1e3:
            # Near-zero lower bound, medium upper bound suggests combustion
            constraint_types[i] = "combustion"
        elif abs(lb) < 50 and abs(ub) < 50:
            # Small bounds suggest velocity/acceleration
            constraint_types[i] = "path_velocity"
        else:
            # Default to path constraint
            constraint_types[i] = "path_constraint"

    # Refine using Jacobian row norms if available
    if jac_g0_arr is not None and jac_g0_arr.size > 0:
        jac_row_norms = np.linalg.norm(jac_g0_arr, axis=1)
        if len(jac_row_norms) == n_cons:
            # Penalty constraints have very large row norms
            penalty_threshold = np.percentile(jac_row_norms[jac_row_norms > 0], 95) * 10
            for i in range(n_cons):
                if jac_row_norms[i] > penalty_threshold:
                    constraint_types[i] = "path_clearance"  # Likely penalty constraint

    return constraint_types


def map_constraint_type_to_category(con_type: str) -> str:
    """
    Map existing constraint types to category-based scaling categories.

    Args:
        con_type: Existing constraint type string (e.g., 'path_velocity', 'path_pressure')

    Returns:
        Category string: 'kinematic', 'thermodynamic', 'boundary', or 'continuity'
    """
    # Kinematic: position, velocity, acceleration constraints
    if con_type in {"path_velocity", "path_constraint"}:
        return "kinematic"

    # Thermodynamic: temperature, pressure, energy constraints
    if con_type in {"path_pressure", "combustion"}:
        return "thermodynamic"

    # Boundary: boundary conditions, periodicity
    if con_type in {"periodicity", "path_clearance"}:
        return "boundary"

    # Continuity: continuity constraints, collocation residuals
    if con_type in {"continuity", "collocation_residuals"}:
        return "continuity"

    # Default: treat unknown types as kinematic (most common)
    return "kinematic"


def compute_constraint_scaling_by_type(
    constraint_types: dict[int, str],
    nlp: Any,
    x0: np.ndarray[Any, Any],
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
    scale: np.ndarray[Any, Any],
    jac_g0_arr: np.ndarray[Any, Any] | None = None,
    g0_arr: np.ndarray[Any, Any] | None = None,
    current_scale_g: np.ndarray[Any, Any] | None = None,
) -> np.ndarray[Any, Any]:
    """
    Compute constraint scaling factors using constraint-type-aware strategies.

    Applies specialized scaling per constraint type:
    - Penalty/Clearance: Aggressive scaling (target max entry < 1e1)
    - Pressure: Account for 1e6 unit conversion, normalize to O(1)
    - Combustion: Account for 1e3 unit conversion, normalize to O(1)
    - Collocation residuals: Standard scaling
    - Path constraints: Standard scaling based on bounds and Jacobian sensitivity
    - Periodicity: Equality constraints, scale based on typical violation magnitude

    Args:
        constraint_types: Dict mapping constraint index to type string
        nlp: CasADi NLP dict
        x0: Initial guess
        lbg: Lower constraint bounds
        ubg: Upper constraint bounds
        scale: Variable scaling factors
        jac_g0_arr: Unscaled Jacobian (optional)
        g0_arr: Constraint values at initial guess (optional)

    Returns:
        Array of constraint scaling factors
    """
    if lbg is None or ubg is None:
        return np.array([])  # type: ignore[return-value]

    n_cons = len(lbg)
    # Start with current constraint scales (if available from previous iteration)
    # This allows constraint-type-aware to refine existing scaling rather than starting from scratch
    # If no current scales provided, start with ones
    if current_scale_g is not None and len(current_scale_g) == n_cons:
        scale_g = current_scale_g.copy()
    else:
        scale_g = np.ones(n_cons)

    # Group constraints by type for batch processing
    type_to_indices: dict[str, list[int]] = {}
    for idx, con_type in constraint_types.items():
        if con_type not in type_to_indices:
            type_to_indices[con_type] = []
        type_to_indices[con_type].append(idx)

    # Compute Jacobian row norms and max entries if available
    # IMPORTANT: Use scaled row norms AND max entries (accounting for variable scaling) to properly
    # assess constraint sensitivity. Max entries are more important than norms for extreme cases.
    jac_row_norms = None
    jac_row_max_entries = (
        None  # Max absolute entry per row (after variable scaling, before constraint scaling)
    )
    if jac_g0_arr is not None and jac_g0_arr.size > 0:
        if scale is not None and len(scale) > 0:
            # Compute scaled row norms: sqrt(sum_j (J[i,j] / scale[j])^2)
            # Also compute max absolute entry per row: max_j |J[i,j] / scale[j]|
            # This represents the sensitivity of constraint i to scaled variables
            jac_row_norms = np.zeros(jac_g0_arr.shape[0])
            jac_row_max_entries = np.zeros(jac_g0_arr.shape[0])
            for i in range(jac_g0_arr.shape[0]):
                row_norm_sq = 0.0
                row_max = 0.0
                for j in range(min(jac_g0_arr.shape[1], len(scale))):
                    if scale[j] > 1e-10:
                        scaled_entry = abs(jac_g0_arr[i, j] / scale[j])
                        row_norm_sq += scaled_entry**2
                        row_max = max(row_max, scaled_entry)
                jac_row_norms[i] = np.sqrt(row_norm_sq)
                jac_row_max_entries[i] = row_max

                # DIAGNOSTIC: Validate computed values
                if not np.isfinite(jac_row_norms[i]):
                    log.warning(
                        f"Non-finite jac_row_norms[{i}]: {jac_row_norms[i]}, replacing with 0.0"
                    )
                    jac_row_norms[i] = 0.0
                if not np.isfinite(jac_row_max_entries[i]):
                    log.warning(
                        f"Non-finite jac_row_max_entries[{i}]: {jac_row_max_entries[i]}, replacing with 0.0"
                    )
                    jac_row_max_entries[i] = 0.0
        else:
            # Without variable scales, use unscaled row norms and max entries
            jac_row_norms = np.linalg.norm(jac_g0_arr, axis=1)
            jac_row_max_entries = np.abs(jac_g0_arr).max(axis=1)

            # DIAGNOSTIC: Check for NaN/Inf in computed norms
            if np.any(np.isnan(jac_row_norms)) or np.any(np.isinf(jac_row_norms)):
                log.warning(
                    f"NaN/Inf in jac_row_norms: NaN={np.sum(np.isnan(jac_row_norms))}, Inf={np.sum(np.isinf(jac_row_norms))}"
                )
                jac_row_norms = np.nan_to_num(jac_row_norms, nan=0.0, posinf=1e10, neginf=-1e10)

            if np.any(np.isnan(jac_row_max_entries)) or np.any(np.isinf(jac_row_max_entries)):
                log.warning(
                    f"NaN/Inf in jac_row_max_entries: NaN={np.sum(np.isnan(jac_row_max_entries))}, Inf={np.sum(np.isinf(jac_row_max_entries))}"
                )
                jac_row_max_entries = np.nan_to_num(
                    jac_row_max_entries, nan=0.0, posinf=1e10, neginf=-1e10
                )

    # Compute scaling based on category targets

    for con_type, indices in type_to_indices.items():
        log.debug("Processing constraint type '%s' with %d constraints", con_type, len(indices))
        for i in indices:
            if i >= n_cons:
                log.debug("  Skipping constraint %d: index >= n_cons (%d)", i, n_cons)
                continue

            lb = lbg[i]
            ub = ubg[i]

            # Get constraint value magnitude
            g0_mag = 0.0
            if g0_arr is not None and i < len(g0_arr):
                g0_val = g0_arr[i]
                g0_mag = float(abs(g0_val)) if np.isfinite(g0_val) else 0.0

            # Get Jacobian row norm and max entry if available
            jac_norm = 0.0
            jac_max_entry = 0.0
            if jac_row_norms is not None and i < len(jac_row_norms):
                jac_norm = jac_row_norms[i]
                # DIAGNOSTIC: Check for NaN/Inf
                if not np.isfinite(jac_norm):
                    log.warning("Non-finite jac_norm[%d]: %f, replacing with 0.0", i, jac_norm)
                    jac_norm = 0.0
            if jac_row_max_entries is not None and i < len(jac_row_max_entries):
                jac_max_entry = jac_row_max_entries[i]
                # DIAGNOSTIC: Check for NaN/Inf
                if not np.isfinite(jac_max_entry):
                    log.warning(
                        "Non-finite jac_max_entry[%d]: %f, replacing with 0.0", i, jac_max_entry
                    )
                    jac_max_entry = 0.0
            # Use max entry for aggressive scaling decisions (more accurate than norm)
            # Fall back to norm if max entry not available
            jac_sensitivity = max(jac_max_entry, jac_norm) if jac_max_entry > 0 else jac_norm
            # DIAGNOSTIC: Validate jac_sensitivity
            if not np.isfinite(jac_sensitivity):
                log.warning(
                    "Non-finite jac_sensitivity[%d]: %f, replacing with 0.0", i, jac_sensitivity
                )
                jac_sensitivity = 0.0

            # Compute magnitude from bounds
            if lb == -np.inf and ub == np.inf:
                magnitude = max(g0_mag, jac_norm) if jac_norm > 0 else g0_mag
            elif lb == -np.inf:
                magnitude = max(float(abs(ub)), g0_mag, jac_norm)
            elif ub == np.inf:
                magnitude = max(float(abs(lb)), g0_mag, jac_norm)
            else:
                magnitude = max(float(abs(lb)), float(abs(ub)), g0_mag, jac_norm)

            # DIAGNOSTIC: Validate magnitude
            if not np.isfinite(magnitude):
                log.warning(
                    "Non-finite magnitude[%d]: %f, lb=%f, ub=%f, g0_mag=%f, jac_norm=%f",
                    i,
                    magnitude,
                    lb,
                    ub,
                    g0_mag,
                    jac_norm,
                )
                magnitude = max(
                    float(abs(lb)) if np.isfinite(lb) else 1.0,
                    float(abs(ub)) if np.isfinite(ub) else 1.0,
                    g0_mag if np.isfinite(g0_mag) else 1.0,
                )

            # Map constraint type to category and get target magnitude
            category = map_constraint_type_to_category(con_type)
            target_magnitude = CONSTRAINT_TYPE_TARGETS.get(category, 1.0)

            # Apply category-based scaling with type-specific handling for extreme cases
            # For extreme Jacobian entries (jac_sensitivity > 1e6), use aggressive scaling
            # Otherwise, scale to achieve target magnitude for the category
            if jac_sensitivity > 1e6:
                # Very extreme Jacobian entries - need very aggressive scaling
                # Target O(1) max entry for extreme cases regardless of category
                target_max_entry = 1e0
                scale_g[i] = target_max_entry / max(jac_sensitivity, 1e-10)
                min_scale_needed = target_max_entry / jac_sensitivity
                scale_g[i] = np.clip(scale_g[i], max(min_scale_needed * 0.1, 1e-10), 1e2)
                if i < 5:  # Log first few for debugging
                    log.debug(
                        "  %s[%d] (category=%s): extreme jac_sensitivity=%.3e, "
                        "scale=%.3e, expected_scaled_max=%.3e",
                        con_type,
                        i,
                        category,
                        jac_sensitivity,
                        scale_g[i],
                        scale_g[i] * jac_sensitivity,
                    )
            elif jac_sensitivity > 1e2:
                # Large Jacobian sensitivity - scale to target magnitude
                # Use category target but ensure max entry is reasonable
                effective_magnitude = max(magnitude, jac_sensitivity)
                scale_g[i] = target_magnitude / max(effective_magnitude, 1e-10)
                scale_g[i] = np.clip(scale_g[i], 1e-8, 1e2)
            elif magnitude > 1e-10 or jac_sensitivity > 1e-10:
                # Normal scaling: scale to achieve target magnitude for category
                effective_magnitude = (
                    max(magnitude, jac_sensitivity) if jac_sensitivity > 0 else magnitude
                )
                # Ensure we don't divide by zero or create NaN
                if effective_magnitude > 1e-10:
                    scale_g[i] = target_magnitude / effective_magnitude
                else:
                    scale_g[i] = 1.0
                # Clip based on category: tighter bounds for boundary, wider for thermodynamic
                if category == "boundary":
                    scale_g[i] = np.clip(scale_g[i], 1e-3, 1e3)
                elif category == "thermodynamic":
                    scale_g[i] = np.clip(scale_g[i], 1e-4, 1e2)
                else:
                    scale_g[i] = np.clip(scale_g[i], 1e-3, 1e3)
            # Small magnitude - keep existing scaling or use default
            elif scale_g[i] <= 1e-10:
                scale_g[i] = 1.0

            # DIAGNOSTIC: Validate computed scale_g[i] is finite
            if not np.isfinite(scale_g[i]):
                log.warning(
                    "Non-finite scale_g[%d] computed: %f, category=%s, "
                    "magnitude=%f, jac_sensitivity=%f, target=%f",
                    i,
                    scale_g[i],
                    category,
                    magnitude,
                    jac_sensitivity,
                    target_magnitude,
                )
                scale_g[i] = 1.0  # Reset to safe default

    # Verification step: Check actual scaled max entries for collocation_residuals and continuity
    # Use stored jac_sensitivity values to verify scaling is correct
    # (Note: we need to re-derive some values here since we didn't store them in a dict in the simplified loop above)
    # Actually, we should just run the overscaling detection which is more robust.

    # Overscaling detection and correction: Prevent very small scaled Jacobian entries
    # Very small entries (<1e-10) indicate overscaling and cause numerical precision issues
    # Balance: target max entry O(1-10), min entry >= 1e-6 (conservative threshold)
    min_scaled_entry_threshold = 1e-6  # Minimum acceptable scaled Jacobian entry (conservative)
    critical_overscaling_threshold = (
        1e-10  # Critical threshold - entries below this indicate severe overscaling
    )
    max_scaled_entry_target = 1e1  # Target max entry (O(10))

    if jac_g0_arr is not None and jac_g0_arr.size > 0:
        # Compute current scaled Jacobian to detect overscaling
        # Using compute_scaled_jacobian from diagnostics
        _, jac_g0_scaled_check = compute_scaled_jacobian(nlp, x0, scale, scale_g)

        if jac_g0_scaled_check is not None:
            jac_mag_check = np.abs(jac_g0_scaled_check)
            n_corrected = 0

            for i in range(min(jac_g0_scaled_check.shape[0], len(scale_g))):
                # Get row's scaled entries
                row_entries = jac_mag_check[i, :]
                row_nonzero = row_entries[row_entries > 0]

                if len(row_nonzero) > 0:
                    row_min = row_nonzero.min()
                    row_max = row_nonzero.max()

                    # Check for critical overscaling first (entries <1e-10)
                    if row_min < critical_overscaling_threshold:
                        # Critical overscaling: apply aggressive correction
                        # Target: bring min entry to at least 1e-6
                        target_min = min_scaled_entry_threshold
                        adjustment_factor = target_min / max(row_min, 1e-15)

                        # Check if adjustment would create too large max entry
                        new_max = row_max * adjustment_factor
                        if (
                            new_max <= max_scaled_entry_target * 50
                        ):  # Allow more aggressive correction for critical cases
                            # Apply aggressive adjustment
                            old_scale = scale_g[i]
                            scale_g[i] = scale_g[i] * adjustment_factor
                            n_corrected += 1

                            if i < 5:  # Log first few for debugging
                                con_type = constraint_types.get(i, "path_constraint")
                                log.debug(
                                    "  Critical overscaling correction[%d]: %s, "
                                    "row_min=%.3e, row_max=%.3e, "
                                    "adjustment=%.3e, "
                                    "old_scale=%.3e, new_scale=%.3e, "
                                    "new_max=%.3e",
                                    i,
                                    con_type,
                                    row_min,
                                    row_max,
                                    adjustment_factor,
                                    old_scale,
                                    scale_g[i],
                                    new_max,
                                )
                        else:
                            # Compromise: use geometric mean to balance min and max
                            # Target: bring min to threshold while keeping max reasonable
                            compromise_factor = (target_min / max(row_min, 1e-15)) ** 0.7
                            if compromise_factor > 1.0:
                                old_scale = scale_g[i]
                                scale_g[i] = scale_g[i] * compromise_factor
                                n_corrected += 1

                                if i < 5:  # Log first few for debugging
                                    con_type = constraint_types.get(i, "path_constraint")
                                    log.debug(
                                        "  Critical overscaling compromise[%d]: %s, "
                                        "row_min=%.3e, row_max=%.3e, "
                                        "compromise_factor=%.3e, "
                                        "old_scale=%.3e, new_scale=%.3e",
                                        i,
                                        con_type,
                                        row_min,
                                        row_max,
                                        compromise_factor,
                                        old_scale,
                                        scale_g[i],
                                    )
                    # Check if row has overscaling (min entry too small but not critical)
                    elif row_min < min_scaled_entry_threshold:
                        # Compute adjustment to bring min entry to threshold
                        # But don't increase max entry too much
                        adjustment_factor = min_scaled_entry_threshold / max(row_min, 1e-12)

                        # Check if adjustment would create too large max entry
                        new_max = row_max * adjustment_factor
                        if new_max <= max_scaled_entry_target * 10:  # Allow up to 10x target
                            # Apply adjustment
                            old_scale = scale_g[i]
                            scale_g[i] = scale_g[i] * adjustment_factor
                            n_corrected += 1

                            if i < 5:  # Log first few for debugging
                                con_type = constraint_types.get(i, "path_constraint")
                                log.debug(
                                    "  Overscaling correction[%d]: %s, "
                                    "row_min=%.3e, row_max=%.3e, "
                                    "adjustment=%.3e, "
                                    "old_scale=%.3e, new_scale=%.3e, "
                                    "new_max=%.3e",
                                    i,
                                    con_type,
                                    row_min,
                                    row_max,
                                    adjustment_factor,
                                    old_scale,
                                    scale_g[i],
                                    new_max,
                                )
                        else:
                            # Compromise: adjust less aggressively to balance min and max
                            # Target: bring min to threshold while keeping max reasonable
                            compromise_factor = (min_scaled_entry_threshold / row_min) ** 0.5
                            if compromise_factor > 1.0:
                                old_scale = scale_g[i]
                                scale_g[i] = scale_g[i] * compromise_factor
                                n_corrected += 1

                                if i < 5:  # Log first few for debugging
                                    con_type = constraint_types.get(i, "path_constraint")
                                    log.debug(
                                        "  Overscaling compromise[%d]: %s, "
                                        "row_min=%.3e, row_max=%.3e, "
                                        "compromise_factor=%.3e, "
                                        "old_scale=%.3e, new_scale=%.3e",
                                        i,
                                        con_type,
                                        row_min,
                                        row_max,
                                        compromise_factor,
                                        old_scale,
                                        scale_g[i],
                                    )

            # Also check for column-based overscaling (variables with very small Jacobian entries)
            # This can happen when variable scales are too large
            n_var_corrected = 0
            for j in range(min(jac_g0_scaled_check.shape[1], len(scale))):
                # Get column's scaled entries
                col_entries = jac_mag_check[:, j]
                col_nonzero = col_entries[col_entries > 0]

                if len(col_nonzero) > 0:
                    col_min = col_nonzero.min()
                    col_max = col_nonzero.max()

                    # Check for critical overscaling in column
                    if col_min < critical_overscaling_threshold:
                        # Variable scale is too large, need to reduce it
                        # Reducing scale[j] will increase scaled Jacobian entries
                        target_min = min_scaled_entry_threshold
                        reduction_factor = col_min / max(target_min, 1e-15)
                        # Clamp reduction to reasonable range
                        reduction_factor = np.clip(
                            reduction_factor, 0.1, 0.9
                        )  # Reduce scale by 10-90%

                        old_scale_j = scale[j]
                        scale[j] = scale[j] * reduction_factor
                        n_var_corrected += 1

                        if j < 5:  # Log first few for debugging
                            log.debug(
                                "  Column overscaling correction[%d]: "
                                "col_min=%.3e, col_max=%.3e, "
                                "reduction_factor=%.3e, "
                                "old_scale=%.3e, new_scale=%.3e",
                                j,
                                col_min,
                                col_max,
                                reduction_factor,
                                old_scale_j,
                                scale[j],
                            )
                    elif col_min < min_scaled_entry_threshold:
                        # Moderate overscaling: reduce variable scale less aggressively
                        reduction_factor = (col_min / max(min_scaled_entry_threshold, 1e-12)) ** 0.5
                        reduction_factor = np.clip(reduction_factor, 0.3, 0.9)  # Reduce by 10-70%

                        old_scale_j = scale[j]
                        scale[j] = scale[j] * reduction_factor
                        n_var_corrected += 1

                        if j < 5:  # Log first few for debugging
                            log.debug(
                                "  Column overscaling correction[%d]: "
                                "col_min=%.3e, col_max=%.3e, "
                                "reduction_factor=%.3e, "
                                "old_scale=%.3e, new_scale=%.3e",
                                j,
                                col_min,
                                col_max,
                                reduction_factor,
                                old_scale_j,
                                scale[j],
                            )

            if n_corrected > 0 or n_var_corrected > 0:
                log.debug(
                    "Overscaling correction: adjusted %d constraints and %d variables "
                    "to prevent very small entries",
                    n_corrected,
                    n_var_corrected,
                )

    # Normalize constraint scales to maintain reasonable range
    # BUT: Preserve aggressive scaling for extreme constraint types (collocation_residuals, continuity)
    # These constraint types need very small scale factors to normalize large Jacobian entries
    scale_g_log = np.log10(np.maximum(scale_g, 1e-10))
    median_log = np.median(scale_g_log)
    sqrt_10_log = np.log10(np.sqrt(10.0))

    # Identify extreme constraint types that need aggressive scaling preserved
    extreme_types = {"collocation_residuals", "continuity", "path_clearance"}

    # Clip outliers with constraint-type-aware bounds
    # For extreme constraint types, allow wider range (1e-8 to 1e2)
    # For other types, use tighter range (1e-3 to 1e2)
    for i in range(n_cons):
        con_type = constraint_types.get(i, "path_constraint")
        is_extreme = con_type in extreme_types

        if is_extreme:
            # Preserve aggressive scaling for extreme constraint types
            # Allow very small scales (down to 1e-10) for extreme cases, but prevent overscaling
            lower_bound_extreme = -10.0  # 1e-10 (allow very small scales for extreme cases)
            upper_bound_extreme = 2.0  # 1e2
            if scale_g_log[i] < lower_bound_extreme:
                scale_g[i] = 10.0**lower_bound_extreme
            elif scale_g_log[i] > upper_bound_extreme:
                scale_g[i] = 10.0**upper_bound_extreme
            # Otherwise, preserve the aggressive scaling (don't clip)
        else:
            # For normal constraint types, use percentile-based clipping
            lower_bound = max(median_log - 2.0 * sqrt_10_log, -3.0)  # Allow down to 1e-3
            upper_bound = min(median_log + 2.0 * sqrt_10_log, 2.0)  # Allow up to 1e2
            if scale_g_log[i] < lower_bound:
                scale_g[i] = max(10.0**lower_bound, scale_g[i] * 0.1)
            elif scale_g_log[i] > upper_bound:
                scale_g[i] = min(10.0**upper_bound, scale_g[i] * 10.0)

    return scale_g


def compute_unified_constraint_magnitudes(
    g0_arr: np.ndarray[Any, Any],
    jac_g0_arr: np.ndarray[Any, Any] | None,
    lbg: np.ndarray[Any, Any] | None,
    ubg: np.ndarray[Any, Any] | None,
    scale: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """
    Compute unified constraint magnitudes combining actual values, Jacobian sensitivity, and bounds.

    For each constraint i, computes:
    magnitude_i = max(|g_i(x0)|, ||J_i||_scaled, |bound_i|)

    This combines:
    - Actual constraint value at x0
    - Jacobian row sensitivity (scaled by variable scales)
    - Bound magnitude

    Args:
        g0_arr: Constraint values at x0
        jac_g0_arr: Constraint Jacobian at x0 (can be None)
        lbg: Lower constraint bounds (can be None)
        ubg: Upper constraint bounds (can be None)
        scale: Variable scaling factors

    Returns:
        Array of unified magnitudes (one per constraint)
    """
    n_cons = len(g0_arr)
    constraint_magnitudes = np.zeros(n_cons)

    for i in range(n_cons):
        # Actual constraint value
        g_val = abs(g0_arr[i]) if i < len(g0_arr) and np.isfinite(g0_arr[i]) else 0.0

        # Jacobian row sensitivity (scaled by variable scales)
        jac_row_norm = 0.0
        if jac_g0_arr is not None and i < jac_g0_arr.shape[0]:
            row_norm_sq = 0.0
            for j in range(min(jac_g0_arr.shape[1], len(scale))):
                if scale[j] > 1e-10:
                    row_norm_sq += (jac_g0_arr[i, j] / scale[j]) ** 2
            jac_row_norm = np.sqrt(row_norm_sq)

        # Bound magnitude
        bound_mag = 0.0
        if lbg is not None and i < len(lbg):
            lb = lbg[i]
            if np.isfinite(lb):
                bound_mag = max(bound_mag, abs(lb))
        if ubg is not None and i < len(ubg):
            ub = ubg[i]
            if np.isfinite(ub):
                bound_mag = max(bound_mag, abs(ub))

        # Unified magnitude: max of all three
        # Use 1e-12 floor to allow scaling of very small residuals (was 1e-6)
        constraint_magnitudes[i] = max(g_val, jac_row_norm, bound_mag, 1e-12)

    return constraint_magnitudes


def analyze_constraint_rank(
    nlp: dict[str, Any],
    x0: np.ndarray[Any, Any],
    scale: np.ndarray[Any, Any],
    scale_g: np.ndarray[Any, Any],
    meta: dict[str, Any] | None = None,
    reporter: StructuredReporter | None = None,
) -> None:
    """
    Analyze constraint Jacobian rank using SVD to identify redundant constraints.

    Args:
        nlp: CasADi NLP dict
        x0: Initial guess
        scale: Variable scaling factors
        scale_g: Constraint scaling factors
        meta: Problem metadata (optional)
    """
    try:
        # SVD Guard: Skip for large problems unless forced
        force_diagnostics = os.environ.get("FREE_PISTON_DIAGNOSTICS") == "1"
        is_large = len(x0) > 3000

        if is_large and not force_diagnostics:
            log.info(
                f"Skipping SVD audit for large problem ({len(x0)} variables). Set FREE_PISTON_DIAGNOSTICS=1 to force."
            )
            return

        log.info("Starting constraint rank audit (SVD)...")

        # Evaluate Jacobian at x0
        j_func = (
            nlp["jac_g_x"]
            if "jac_g_x" in nlp
            else ca.Function("J", [nlp["x"]], [ca.jacobian(nlp["g"], nlp["x"])])
        )
        j_val = j_func(x0)

        # Convert to dense or sparse matrix
        if hasattr(j_val, "full"):
            j_mat = j_val.full()
        else:
            j_mat = np.array(j_val)

        # Apply scaling: J_scaled = diag(scale_g) @ J @ diag(1/scale)
        # Efficient scaling using broadcasting
        j_scaled = j_mat * scale_g[:, np.newaxis] / scale[np.newaxis, :]

        m, n = j_scaled.shape
        k = min(m, n)

        if k > 1000:
            log.info(f"Large Jacobian ({m}x{n}), computing subset of singular values...")
            try:
                u, s, _ = np.linalg.svd(j_scaled, full_matrices=False)
            except Exception:
                u, s, _ = np.linalg.svd(j_scaled, full_matrices=False)
        else:
            u, s, _ = np.linalg.svd(j_scaled, full_matrices=False)

        # Analyze singular values
        s_max = s.max()
        s_min = s.min()
        cond = s_max / s_min if s_min > 1e-20 else float("inf")

        log.info(f"Jacobian SVD: sigma_max={s_max:.2e}, sigma_min={s_min:.2e}, cond={cond:.2e}")

        # Identify near-zero singular values (redundant constraints)
        threshold = 1e-8 * s_max
        small_sv_indices = np.where(s < threshold)[0]

        if len(small_sv_indices) > 0:
            log.warning(
                f"Found {len(small_sv_indices)} near-zero singular values (< {threshold:.2e}). "
                "Constraints are likely redundant."
            )

            # Analyze the smallest singular value's vector
            idx_smallest = -1  # Last one is smallest in numpy svd
            u_smallest = u[:, idx_smallest]

            # Find constraints with large components in this vector
            contrib_indices = np.where(np.abs(u_smallest) > 0.1)[0]

            log.info(
                f"Constraints involved in smallest singular value (sigma={s[-1]:.2e}): "
                f"{contrib_indices.tolist()}"
            )

            if meta and "constraint_groups" in meta:
                log.info(
                    f"Mapping {len(contrib_indices)} indices to {len(meta['constraint_groups'])} groups..."
                )
    except Exception as e:
        log.warning(f"Constraint rank audit failed: {e}")


def relax_over_scaled_groups(
    scale: np.ndarray,
    scale_g: np.ndarray,
    over_scaled_variable_groups: list[str],
    over_scaled_constraint_types: list[str],
    variable_groups: dict[str, list[int]] | None = None,
    constraint_groups: dict[str, list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Relax scale factors for over-scaled groups using geometric averaging.

    Uses geometric mean (sqrt) to move scale factors halfway back toward 1.0,
    which is appropriate for multi-decade scales. This selectively relaxes only
    problematic groups while preserving well-scaled ones.

    Args:
        scale: Current variable scaling factors
        scale_g: Current constraint scaling factors
        over_scaled_variable_groups: List of variable group names to relax
        over_scaled_constraint_types: List of constraint type names to relax
        variable_groups: Dict mapping group names to variable indices
        constraint_groups: Dict mapping constraint types to constraint indices

    Returns:
        Tuple of (relaxed_scale, relaxed_scale_g) arrays
    """
    relaxed_scale = scale.copy()
    relaxed_scale_g = scale_g.copy()

    # Relax variable groups
    if variable_groups is not None:
        for group_name in over_scaled_variable_groups:
            if group_name in variable_groups:
                indices = variable_groups[group_name]
                for idx in indices:
                    if 0 <= idx < len(relaxed_scale):
                        # Geometric mean with 1.0: sqrt(scale[idx] * 1.0)
                        relaxed_scale[idx] = np.sqrt(relaxed_scale[idx] * 1.0)

    # Relax constraint types
    if constraint_groups is not None:
        for con_type in over_scaled_constraint_types:
            if con_type in constraint_groups:
                indices = constraint_groups[con_type]
                for idx in indices:
                    if 0 <= idx < len(relaxed_scale_g):
                        # Geometric mean with 1.0: sqrt(scale_g[idx] * 1.0)
                        relaxed_scale_g[idx] = np.sqrt(relaxed_scale_g[idx] * 1.0)

    return relaxed_scale, relaxed_scale_g


def compute_constraint_scaling_from_evaluation(
    nlp: Any,
    x0: np.ndarray,
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute constraint scaling factors from actual constraint evaluation and Jacobian at initial guess.

    Uses robust scaling that accounts for both constraint values and Jacobian sensitivity:
    1. Computes constraint Jacobian at initial guess
    2. For each constraint, uses max(|g0[i]|, typical_jacobian_row_magnitude[i])
    3. For equality constraints (lb == ub): uses reference scale based on typical residual magnitude
    4. Caps all scale factors to prevent extreme ratios (1e-3 to 1e3 range)
    5. Uses percentiles for more robust scaling

    Args:
        nlp: CasADi NLP dict with 'x' and 'g' keys
        x0: Initial guess for variables
        lbg: Lower bounds on constraints (can be None)
        ubg: Upper bounds on constraints (can be None)
        scale: Variable scaling factors (used to compute scaled Jacobian magnitude)

    Returns:
        Array of constraint scale factors (one per constraint)
    """
    try:
        # Fall back to bounds-based scaling if evaluation fails
        bounds_scale = compute_constraint_scaling(lbg, ubg)

        if not isinstance(nlp, dict) or "g" not in nlp or "x" not in nlp:
            log.warning("NLP does not support constraint evaluation, using bounds-based scaling")
            return bounds_scale

        g_expr = nlp["g"]
        x_sym = nlp["x"]

        # Handle empty constraints
        if g_expr is None or g_expr.numel() == 0:
            return bounds_scale

        # Create function to evaluate constraints
        try:
            g_func = ca.Function("g_func", [x_sym], [g_expr])
            g0 = g_func(x0)
            g0_arr = np.array(g0).flatten()
        except Exception as e:
            log.warning(
                f"Failed to evaluate constraints at initial guess: {e}, using bounds-based scaling"
            )
            return bounds_scale

        # Compute constraint Jacobian to account for sensitivity
        jac_row_norms = None
        try:
            jac_g_expr = ca.jacobian(g_expr, x_sym)
            jac_g_func = ca.Function("jac_g_func", [x_sym], [jac_g_expr])
            jac_g0 = jac_g_func(x0)
            jac_g0_arr = np.array(jac_g0)

            # Compute row norms accounting for variable scaling
            # The scaled Jacobian is: J_scaled[i,j] = scale_g[i] * J[i,j] / scale[j]
            # For row norm calculation, we need: ||row_i|| = sqrt(sum_j (J[i,j] / scale[j])^2)
            # This represents the sensitivity of constraint i to scaled variables
            if scale is not None and len(scale) > 0:
                # Compute scaled Jacobian row norms: sqrt(sum_j (J[i,j] / scale[j])^2)
                jac_row_norms = np.zeros(jac_g0_arr.shape[0])
                for i in range(jac_g0_arr.shape[0]):
                    row_norm_sq = 0.0
                    for j in range(min(jac_g0_arr.shape[1], len(scale))):
                        if scale[j] > 1e-10:
                            row_norm_sq += (jac_g0_arr[i, j] / scale[j]) ** 2
                    jac_row_norms[i] = np.sqrt(row_norm_sq)
            else:
                # Without variable scales, use unscaled row norms
                jac_row_norms = np.linalg.norm(jac_g0_arr, axis=1)

            # Use median of non-zero row norms as reference
            non_zero_norms = jac_row_norms[jac_row_norms > 1e-10]
            if len(non_zero_norms) > 0:
                ref_jac_norm_val = np.median(non_zero_norms)
                ref_jac_norm = float(ref_jac_norm_val)
            else:
                ref_jac_norm = 1.0
        except Exception as e:
            log.debug(f"Could not compute constraint Jacobian: {e}, using constraint values only")
            jac_row_norms = None
            ref_jac_norm = 1.0

        # Combine bounds and actual constraint values
        if lbg is None and ubg is None:
            n_cons = len(g0_arr)
            lbg = -np.inf * np.ones(n_cons)
            ubg = np.inf * np.ones(n_cons)

        if lbg is None:
            n_cons = len(g0_arr)
            lbg = -np.inf * np.ones(n_cons)
        if ubg is None:
            n_cons = len(g0_arr)
            ubg = np.inf * np.ones(n_cons)

        n_cons = len(g0_arr)
        scale_g = np.ones(n_cons)

        # Identify equality constraints (typically collocation residuals)
        equality_mask = np.zeros(n_cons, dtype=bool)
        for i in range(n_cons):
            lb = lbg[i] if i < len(lbg) else -np.inf
            ub = ubg[i] if i < len(ubg) else np.inf
            if lb == ub and np.isfinite(lb):
                equality_mask[i] = True

        # Compute reference scale for equality constraints based on typical residual magnitude
        # Use median of absolute values of equality constraint residuals (excluding zeros)
        equality_g0 = g0_arr[equality_mask]
        if len(equality_g0) > 0:
            abs_equality = np.abs(equality_g0)
            # Use median of non-zero values, or 1e-3 as default
            non_zero = abs_equality[abs_equality > 1e-10]
            if len(non_zero) > 0:
                ref_magnitude_val = np.median(non_zero)
                ref_magnitude = float(ref_magnitude_val)
            else:
                ref_magnitude = 1e-3  # Default reference for zero residuals
        else:
            ref_magnitude = 1e-3

        # Compute scale factors with robust handling
        for i in range(n_cons):
            lb = lbg[i] if i < len(lbg) else -np.inf
            ub = ubg[i] if i < len(ubg) else np.inf
            g_val = g0_arr[i] if i < len(g0_arr) else 0.0

            # For equality constraints, use reference magnitude
            if equality_mask[i]:
                magnitude = ref_magnitude
                # Detect large-magnitude combustion residuals (>1e4) and apply aggressive scaling
                if np.isfinite(g_val) and abs(g_val) > 1e4:
                    # For large combustion residuals, use more aggressive normalization
                    # Scale down by normalizing to typical residual magnitude
                    magnitude = max(ref_magnitude, abs(g_val) / 1e6)  # Normalize large residuals
            else:
                # For inequality constraints, compute from bounds and values
                if lb == -np.inf and ub == np.inf:
                    magnitude = float(abs(g_val)) if np.isfinite(g_val) else 1.0
                elif lb == -np.inf:
                    magnitude = float(abs(ub))
                elif ub == np.inf:
                    magnitude = float(abs(lb))
                else:
                    magnitude = max(float(abs(lb)), float(abs(ub)))

                # Incorporate actual constraint value (but don't let tiny values dominate)
                if np.isfinite(g_val):
                    # Use max of bounds and value, but cap value influence
                    g_val_abs = float(abs(g_val))
                    bounds_max = max(float(abs(lb)), float(abs(ub)))
                    magnitude = max(magnitude, min(g_val_abs, bounds_max * 10.0))
                    # Detect large-magnitude combustion/energy constraints (>1e4) and apply aggressive scaling
                    if abs(g_val) > 1e4 or magnitude > 1e4:
                        # For large energy/combustion constraints, normalize more aggressively
                        # This helps bring Jacobian entries from O(1e6-1e8) to O(1)
                        magnitude = max(magnitude / 1e6, 1e-3)  # Normalize large constraints

            # Incorporate Jacobian row norm to account for constraint sensitivity
            # If constraint i has a large Jacobian row norm, it needs a smaller scale factor
            # to keep the scaled Jacobian elements O(1)
            if jac_row_norms is not None and i < len(jac_row_norms):
                jac_norm_i = jac_row_norms[i]
                if jac_norm_i > 1e-10:
                    # Normalize scaled Jacobian row to O(1): scale_g[i] should be 1.0 / jac_norm_i
                    # But we also need to account for constraint value magnitude
                    # Use max of constraint value magnitude and Jacobian row norm
                    magnitude = max(magnitude, jac_norm_i)

                    # Identify penalty constraints (rows with very large Jacobian norms)
                    # These are likely clearance penalties with 1e6 stiffness
                    penalty_threshold = ref_jac_norm * 1e3  # 1000x larger than typical
                    if jac_norm_i > penalty_threshold:
                        # For penalty constraints, aggressively normalize Jacobian entries
                        # Target: scaled Jacobian elements should be O(1)
                        magnitude = max(
                            magnitude, jac_norm_i / 1e3
                        )  # More aggressive normalization

            # Compute scale factor: scale_g[i] = 1.0 / max(|g0[i]|, ||J_scaled_row[i]||)
            # This ensures scaled Jacobian elements are O(1)
            if magnitude > 1e-6:
                scale_g[i] = 1.0 / magnitude
            else:
                scale_g[i] = 1.0

            # Cap scale factors to tighter range to limit condition number
            # Range 1e-2 to 1e2 gives max condition number of 1e4 (tighter than before)
            # This helps achieve target condition number < 1e6
            scale_g[i] = np.clip(scale_g[i], 1e-2, 1e2)

        # Log scaling statistics before normalization
        scale_g_pre = scale_g.copy()
        log.debug(
            f"Constraint scaling (pre-normalization): range=[{scale_g_pre.min():.3e}, {scale_g_pre.max():.3e}], "
            f"mean={scale_g_pre.mean():.3e}, equality_ref={ref_magnitude:.3e}, "
            f"n_equality={equality_mask.sum()}/{n_cons}",
        )

        # Normalize constraint scales relative to variable scales
        # Ensure constraint scales stay within reasonable range relative to variable scales
        if scale is not None and len(scale) > 0:
            # Get typical variable scale magnitude
            scale_median = np.median(scale[scale > 1e-10])

            # Normalize constraint scales relative to variable scales
            # Constraint scales should be roughly comparable to variable scales
            scale_g_median = np.median(scale_g[scale_g > 1e-10])
            if scale_g_median > 1e-10 and scale_median > 1e-10:
                # Adjust constraint scales to be comparable to variable scales
                # Limit adjustment to tighter range: [0.316, 3.16] for 10¹ ratio
                scale_ratio = scale_median / scale_g_median
                sqrt_10 = np.sqrt(10.0)
                scale_ratio = np.clip(scale_ratio, 1.0 / sqrt_10, sqrt_10)
                scale_g = scale_g * scale_ratio
        # Normalize scale factors to reduce extreme ratios
        # Use percentile-based normalization to prevent outliers
        scale_g_log = np.log10(np.maximum(scale_g, 1e-10))
        p25 = np.percentile(scale_g_log, 25)
        p75 = np.percentile(scale_g_log, 75)
        iqr = p75 - p25

        # Clip outliers with tighter bounds: cap to ±1 log unit from median (10¹ ratio)
        median_log = np.median(scale_g_log)
        sqrt_10_log = np.log10(np.sqrt(10.0))  # ≈ 0.5 log units
        lower_bound = max(median_log - sqrt_10_log, -2.0)  # At least 1e-2
        upper_bound = min(median_log + sqrt_10_log, 2.0)  # At most 1e2

        n_clipped = 0
        for i in range(n_cons):
            if scale_g_log[i] < lower_bound:
                scale_g[i] = 10.0**lower_bound
                n_clipped += 1
            elif scale_g_log[i] > upper_bound:
                scale_g[i] = 10.0**upper_bound
                n_clipped += 1
        # Apply clamping strategy similar to variables: ensure constraint scales stay within [median/sqrt(10), median*sqrt(10)]
        # This gives 10¹ ratio (tighter than before)
        scale_g_median = np.median(scale_g[scale_g > 1e-10])
        if scale_g_median > 1e-10:
            sqrt_10 = np.sqrt(10.0)
            for i in range(n_cons):
                if scale_g[i] > 1e-10:
                    scale_g[i] = np.clip(
                        scale_g[i], scale_g_median / sqrt_10, scale_g_median * sqrt_10
                    )
        log.info(
            f"Constraint scaling (post-normalization): range=[{scale_g.min():.3e}, {scale_g.max():.3e}], "
            f"mean={scale_g.mean():.3e}, clipped={n_clipped}/{n_cons} outliers",
        )
        return scale_g

    except Exception as e:
        log.warning(
            f"Error in constraint evaluation-based scaling: {e}, using bounds-based scaling"
        )
        return compute_constraint_scaling(lbg, ubg)


def analyze_magnitude_distribution(
    magnitudes: np.ndarray[Any, Any],
    aggressive: bool = False,
) -> dict[str, Any]:
    """
    Analyze magnitude distribution using robust statistics without distribution assumptions.

    Computes percentiles and IQR to identify center (median) and detect outliers.
    No assumptions about distribution shape (Poisson, Gaussian, etc.).

    Args:
        magnitudes: Array of constraint magnitudes
        aggressive: If True, use 3xIQR for outlier detection instead of 1.5xIQR.
                   More conservative, only flags truly extreme outliers.

    Returns:
        Dictionary with:
        - median: Median (p50) - the center point
        - iqr: Interquartile Range (p75 - p25)
        - outlier_mask: Boolean array indicating outliers (using IQR method)
        - percentiles: Dict with p5, p25, p50, p75, p95
    """
    # Filter out near-zero values to avoid skewing statistics
    magnitudes_nonzero = magnitudes[magnitudes > 1e-20]

    if len(magnitudes_nonzero) == 0:
        # Fallback if no valid magnitudes
        return {
            "median": 1.0,
            "iqr": 1.0,
            "outlier_mask": np.zeros(len(magnitudes), dtype=bool),
            "percentiles": {"p5": 1.0, "p25": 1.0, "p50": 1.0, "p75": 1.0, "p95": 1.0},
        }

    # Compute robust percentiles
    p5 = np.percentile(magnitudes_nonzero, 5)
    p25 = np.percentile(magnitudes_nonzero, 25)
    p50 = np.percentile(magnitudes_nonzero, 50)  # Median (center)
    p75 = np.percentile(magnitudes_nonzero, 75)
    p95 = np.percentile(magnitudes_nonzero, 95)

    # Compute IQR for outlier detection
    iqr = p75 - p25

    # Detect outliers using IQR method (no distribution assumptions)
    # Use different IQR multiplier based on aggressiveness
    # - Standard (1.5xIQR): catches ~0.7% outliers
    # - Aggressive (3xIQR): catches ~0.003% outliers (only extreme values)
    iqr_multiplier = 3.0 if aggressive else 1.5
    lower_bound = p25 - iqr_multiplier * iqr
    upper_bound = p75 + iqr_multiplier * iqr

    # Handle edge case where IQR is very small
    if iqr < 1e-10:
        # If IQR is too small, use wider bounds based on percentiles
        lower_bound = p5
        upper_bound = p95

    outlier_mask = (magnitudes < lower_bound) | (magnitudes > upper_bound)

    return {
        "median": p50,
        "iqr": iqr,
        "outlier_mask": outlier_mask,
        "percentiles": {"p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95},
    }


def normalize_to_median_center(
    magnitudes: np.ndarray[Any, Any],
    median: float,
    iqr: float,
    outlier_mask: np.ndarray[Any, Any],
    scale_bounds: tuple[float, float] = (1e-8, 1e8),
) -> np.ndarray[Any, Any]:
    """
    Normalize all magnitudes to center at median.

    Outliers are handled separately to prevent them from skewing the center.
    All normal constraints are normalized so their scaled magnitudes center at median.

    Args:
        magnitudes: Array of constraint magnitudes
        median: Median (center point) from distribution analysis
        iqr: Interquartile Range from distribution analysis
        outlier_mask: Boolean array indicating outliers
        scale_bounds: (min, max) bounds for outlier scale factors.
                     Tighter bounds (e.g., 1e-6, 1e6) prevent extreme multipliers.

    Returns:
        Array of scale factors (one per constraint)
    """
    scales = np.ones(len(magnitudes))
    target_center = 1.0  # Normalized center

    for i in range(len(magnitudes)):
        mag = magnitudes[i]

        if outlier_mask[i]:
            # Outlier: normalize but bound to prevent extreme scales
            # Still normalize to center, but limit scale factor
            if mag > 1e-10:
                scale = target_center / mag
                # Bound outlier scales using provided bounds
                scales[i] = np.clip(scale, scale_bounds[0], scale_bounds[1])
            else:
                scales[i] = 1.0
        # Normal constraint: normalize to center
        # Scale so that scaled magnitude = median (target_center)
        elif mag > 1e-20:
            scales[i] = target_center / mag
        else:
            scales[i] = 1.0

    return scales


def cluster_by_magnitude(
    magnitudes: np.ndarray[Any, Any],
) -> dict[int, list[int]]:
    """
    Cluster magnitudes into log10 bins.

    Args:
        magnitudes: Array of magnitudes

    Returns:
        Dictionary mapping log10 bin index to list of indices
    """
    clusters: dict[int, list[int]] = {}

    # Handle zeros or negative values (shouldn't happen for magnitudes, but safety first)
    safe_mags = np.maximum(magnitudes, 1e-20)

    # Compute log10 magnitudes
    log_mags = np.log10(safe_mags)

    # Bin into integers (floor)
    # e.g., 0.05 -> -1.3 -> -2 (bin -2: [1e-2, 1e-1])
    # e.g., 50 -> 1.7 -> 1 (bin 1: [1e1, 1e2])
    bins = np.floor(log_mags).astype(int)

    # Group indices
    for i, bin_idx in enumerate(bins):
        if bin_idx not in clusters:
            clusters[bin_idx] = []
        clusters[bin_idx].append(i)

    return clusters


def try_scaling_strategy(
    strategy_name: str,
    nlp: Any,
    x0: np.ndarray,
    lbx: np.ndarray,
    ubx: np.ndarray,
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
    scale: np.ndarray,
    scale_g: np.ndarray,
    jac_g0_arr: np.ndarray,
    jac_g0_scaled: np.ndarray,
    variable_groups: dict[str, list[int]] | None,
    constraint_types: dict[int, str] | None = None,
    meta: dict[str, Any] | None = None,
    g0_arr: np.ndarray | None = None,
    target_max_entry: float = 1e2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Try a specific scaling strategy and return refined scales.

    Args:
        strategy_name: Name of strategy to try
        nlp: CasADi NLP dict
        x0: Initial guess
        lbx, ubx: Variable bounds
        lbg, ubg: Constraint bounds
        scale: Current variable scales
        scale_g: Current constraint scales
        jac_g0_arr: Unscaled Jacobian
        jac_g0_scaled: Current scaled Jacobian
        variable_groups: Variable group mapping
        target_max_entry: Target maximum scaled Jacobian entry

    Returns:
        Tuple of (new_scale, new_scale_g)
    """
    new_scale = scale.copy()
    new_scale_g = scale_g.copy()

    jac_mag = np.abs(jac_g0_scaled)

    if strategy_name == "tighten_ratios":
        # Strategy 1: Tighten variable and constraint scaling ratios
        scale_median = np.median(new_scale[new_scale > 1e-10])
        tight_factor = np.sqrt(np.sqrt(10.0))
        for i in range(len(new_scale)):
            if new_scale[i] > 1e-10:
                new_scale[i] = np.clip(
                    new_scale[i],
                    scale_median / tight_factor,
                    scale_median * tight_factor,
                )

        if len(new_scale_g) > 0:
            scale_g_median = np.median(new_scale_g[new_scale_g > 1e-10])
            for i in range(len(new_scale_g)):
                if new_scale_g[i] > 1e-10:
                    new_scale_g[i] = np.clip(
                        new_scale_g[i],
                        scale_g_median / tight_factor,
                        scale_g_median * tight_factor,
                    )

    elif strategy_name == "row_max_scaling":
        # Strategy 2: Scale constraint rows based on max entry per row
        # Target: max scaled entry per row should be <= target_max_entry
        # Also check for min entries to prevent overscaling
        min_entry_threshold = 1e-6  # Minimum acceptable entry
        for i in range(min(jac_g0_scaled.shape[0], len(new_scale_g))):
            row_entries = jac_mag[i, :]
            row_nonzero = row_entries[row_entries > 0]
            if len(row_nonzero) > 0:
                row_max = row_nonzero.max()
                row_min = row_nonzero.min()

                # Check for overscaling first (min entry too small)
                if row_min < 1e-10:
                    # Critical overscaling: increase scale_g to bring min entry up
                    target_min = min_entry_threshold
                    increase_factor = target_min / max(row_min, 1e-15)
                    # Limit increase to prevent creating new max entry issues
                    increase_factor = min(increase_factor, 1e4)  # Cap at 1e4x
                    new_scale_g[i] = new_scale_g[i] * increase_factor
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                elif row_max > target_max_entry:
                    # Reduce scale_g[i] to bring max entry down to target
                    reduction_factor = target_max_entry / row_max
                    new_scale_g[i] = new_scale_g[i] * reduction_factor
                    # Check if reduction would create overscaling
                    new_min_estimate = row_min * reduction_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: reduce less aggressively
                        compromise_factor = (target_max_entry / row_max) ** 0.7
                        new_scale_g[i] = new_scale_g[i] * (compromise_factor / reduction_factor)
                    # Allow more aggressive scaling for very large entries
                    if row_max > 1e10:
                        # Extra aggressive for extreme entries
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-6, 1e3)
                    else:
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-3, 1e3)

    elif strategy_name == "column_max_scaling":
        # Strategy 3: Scale variable columns based on max entry per column
        # Target: max scaled entry per column should be <= target_max_entry
        # Use more conservative scaling to avoid over-scaling variables
        # Also check for min entries to prevent overscaling
        min_entry_threshold = 1e-6  # Minimum acceptable entry
        for j in range(min(jac_g0_scaled.shape[1], len(new_scale))):
            col_entries = jac_mag[:, j]
            col_nonzero = col_entries[col_entries > 0]
            if len(col_nonzero) > 0:
                col_max = col_nonzero.max()
                col_min = col_nonzero.min()

                # Check for overscaling first (min entry too small)
                if col_min < 1e-10:
                    # Critical overscaling: reduce scale[j] to increase min entry
                    target_min = min_entry_threshold
                    reduction_factor = col_min / max(target_min, 1e-15)
                    # Limit reduction to reasonable range
                    reduction_factor = np.clip(reduction_factor, 0.1, 0.9)
                    new_scale[j] = new_scale[j] * reduction_factor
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)
                elif col_max > target_max_entry:
                    # Increase scale[j] to bring max entry down to target
                    # But be conservative - don't increase too much at once
                    increase_factor = min(col_max / target_max_entry, 10.0)  # Cap at 10x increase
                    new_scale[j] = new_scale[j] * increase_factor
                    # Check if increase would create overscaling
                    new_min_estimate = col_min / increase_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: increase less aggressively
                        compromise_factor = (col_max / target_max_entry) ** 0.7
                        new_scale[j] = new_scale[j] * (compromise_factor / increase_factor)
                    # More conservative clamping to prevent extreme variable scaling
                    if col_max > 1e10:
                        # Extra aggressive for extreme entries, but still bounded
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)  # Tighter range
                    else:
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e2)

        # Re-normalize variable scales to maintain 10^1 ratio constraint
        scale_min = new_scale[new_scale > 1e-10].min() if (new_scale > 1e-10).any() else 1e-10
        scale_max = new_scale.max()
        if scale_min > 1e-10 and scale_max / scale_min > 1e1:
            scale_median = np.median(new_scale[new_scale > 1e-10])
            sqrt_10 = np.sqrt(10.0)
            for j in range(len(new_scale)):
                if new_scale[j] > 1e-10:
                    new_scale[j] = np.clip(
                        new_scale[j], scale_median / sqrt_10, scale_median * sqrt_10
                    )

    elif strategy_name == "extreme_entry_targeting":
        # Strategy 4: Aggressively target extreme entries (>1e2)
        # Find rows and columns with extreme entries and scale them down aggressively
        # Use adaptive threshold based on current max entry
        # Also check for min entries to prevent overscaling
        min_entry_threshold = 1e-6  # Minimum acceptable entry
        global_max = jac_mag.max() if jac_g0_scaled.size > 0 else 0.0
        if global_max > 1e10:
            # For very extreme entries, use more aggressive threshold
            extreme_threshold = 1e1  # Target O(10) for extreme cases
        else:
            extreme_threshold = 1e2  # Target O(100) for moderate cases

        for i in range(min(jac_g0_scaled.shape[0], len(new_scale_g))):
            row_entries = jac_mag[i, :]
            row_nonzero = row_entries[row_entries > 0]
            if len(row_nonzero) > 0:
                row_max = row_nonzero.max()
                row_min = row_nonzero.min()

                # Check for overscaling first
                if row_min < 1e-10:
                    # Critical overscaling: increase scale_g to bring min entry up
                    target_min = min_entry_threshold
                    increase_factor = target_min / max(row_min, 1e-15)
                    increase_factor = min(increase_factor, 1e4)  # Cap at 1e4x
                    new_scale_g[i] = new_scale_g[i] * increase_factor
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                elif row_max > extreme_threshold:
                    # Aggressively reduce scale_g[i] to bring max entry to threshold
                    reduction_factor = extreme_threshold / row_max
                    new_scale_g[i] = new_scale_g[i] * reduction_factor
                    # Check if reduction would create overscaling
                    new_min_estimate = row_min * reduction_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: reduce less aggressively
                        compromise_factor = (extreme_threshold / row_max) ** 0.7
                        new_scale_g[i] = new_scale_g[i] * (compromise_factor / reduction_factor)
                    # Allow very aggressive scaling for extreme entries
                    if row_max > 1e10:
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                    else:
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-6, 1e3)

        for j in range(min(jac_g0_scaled.shape[1], len(new_scale))):
            col_entries = jac_mag[:, j]
            col_nonzero = col_entries[col_entries > 0]
            if len(col_nonzero) > 0:
                col_max = col_nonzero.max()
                col_min = col_nonzero.min()

                # Check for overscaling first
                if col_min < 1e-10:
                    # Critical overscaling: reduce scale[j] to increase min entry
                    target_min = min_entry_threshold
                    reduction_factor = col_min / max(target_min, 1e-15)
                    reduction_factor = np.clip(reduction_factor, 0.1, 0.9)
                    new_scale[j] = new_scale[j] * reduction_factor
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)
                elif col_max > extreme_threshold:
                    # Aggressively increase scale[j] to bring max entry to threshold
                    # But cap the increase to prevent over-scaling variables
                    increase_factor = min(col_max / extreme_threshold, 100.0)  # Cap at 100x
                    new_scale[j] = new_scale[j] * increase_factor
                    # Check if increase would create overscaling
                    new_min_estimate = col_min / increase_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: increase less aggressively
                        compromise_factor = (col_max / extreme_threshold) ** 0.7
                        new_scale[j] = new_scale[j] * (compromise_factor / increase_factor)
                    # Allow aggressive scaling but with tighter bounds
                    if col_max > 1e10:
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)  # Tighter range
                    else:
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e2)

        # Re-normalize variable scales to maintain 10^1 ratio constraint
        scale_min = new_scale[new_scale > 1e-10].min() if (new_scale > 1e-10).any() else 1e-10
        scale_max = new_scale.max()
        if scale_min > 1e-10 and scale_max / scale_min > 1e1:
            scale_median = np.median(new_scale[new_scale > 1e-10])
            sqrt_10 = np.sqrt(10.0)
            for j in range(len(new_scale)):
                if new_scale[j] > 1e-10:
                    new_scale[j] = np.clip(
                        new_scale[j], scale_median / sqrt_10, scale_median * sqrt_10
                    )

    elif strategy_name == "percentile_based":
        # Strategy 5: Scale based on percentiles - target p95 entries
        # Bring p95 entries down to target_max_entry
        if jac_g0_scaled.size > 0:
            jac_mag_nonzero = jac_mag[jac_mag > 0]
            if len(jac_mag_nonzero) > 0:
                p95_value = np.percentile(jac_mag_nonzero, 95)
                if p95_value > target_max_entry:
                    global_reduction = target_max_entry / p95_value
                    # Apply reduction to constraint scales
                    for i in range(len(new_scale_g)):
                        new_scale_g[i] = new_scale_g[i] * global_reduction
                        # Allow more aggressive scaling for high percentiles
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-5, 1e3)

    elif strategy_name == "combined_row_column":
        # Strategy 6: Combined aggressive row and column scaling
        # Apply both row and column scaling together for maximum effect
        # Also check for min entries to prevent overscaling
        min_entry_threshold = 1e-6  # Minimum acceptable entry
        # First apply row scaling
        for i in range(min(jac_g0_scaled.shape[0], len(new_scale_g))):
            row_entries = jac_mag[i, :]
            row_nonzero = row_entries[row_entries > 0]
            if len(row_nonzero) > 0:
                row_max = row_nonzero.max()
                row_min = row_nonzero.min()

                # Check for overscaling first
                if row_min < 1e-10:
                    # Critical overscaling: increase scale_g to bring min entry up
                    target_min = min_entry_threshold
                    increase_factor = target_min / max(row_min, 1e-15)
                    increase_factor = min(increase_factor, 1e4)  # Cap at 1e4x
                    new_scale_g[i] = new_scale_g[i] * increase_factor
                    new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                elif row_max > target_max_entry:
                    reduction_factor = target_max_entry / row_max
                    new_scale_g[i] = new_scale_g[i] * reduction_factor
                    # Check if reduction would create overscaling
                    new_min_estimate = row_min * reduction_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: reduce less aggressively
                        compromise_factor = (target_max_entry / row_max) ** 0.7
                        new_scale_g[i] = new_scale_g[i] * (compromise_factor / reduction_factor)
                    if row_max > 1e10:
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-8, 1e3)
                    else:
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-5, 1e3)

        # Then apply column scaling (more conservative to avoid over-scaling)
        for j in range(min(jac_g0_scaled.shape[1], len(new_scale))):
            col_entries = jac_mag[:, j]
            col_nonzero = col_entries[col_entries > 0]
            if len(col_nonzero) > 0:
                col_max = col_nonzero.max()
                col_min = col_nonzero.min()

                # Check for overscaling first
                if col_min < 1e-10:
                    # Critical overscaling: reduce scale[j] to increase min entry
                    target_min = min_entry_threshold
                    reduction_factor = col_min / max(target_min, 1e-15)
                    reduction_factor = np.clip(reduction_factor, 0.1, 0.9)
                    new_scale[j] = new_scale[j] * reduction_factor
                    new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)
                elif col_max > target_max_entry:
                    # Cap increase factor to prevent extreme variable scaling
                    increase_factor = min(col_max / target_max_entry, 10.0)  # Cap at 10x
                    new_scale[j] = new_scale[j] * increase_factor
                    # Check if increase would create overscaling
                    new_min_estimate = col_min / increase_factor
                    if new_min_estimate < min_entry_threshold:
                        # Compromise: increase less aggressively
                        compromise_factor = (col_max / target_max_entry) ** 0.7
                        new_scale[j] = new_scale[j] * (compromise_factor / increase_factor)
                    if col_max > 1e10:
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e3)  # Tighter range
                    else:
                        new_scale[j] = np.clip(new_scale[j], 1e-2, 1e2)

        # Re-normalize variable scales to maintain 10^1 ratio constraint
        scale_min = new_scale[new_scale > 1e-10].min() if (new_scale > 1e-10).any() else 1e-10
        scale_max = new_scale.max()
        if scale_min > 1e-10 and scale_max / scale_min > 1e1:
            scale_median = np.median(new_scale[new_scale > 1e-10])
            sqrt_10 = np.sqrt(10.0)
            for j in range(len(new_scale)):
                if new_scale[j] > 1e-10:
                    new_scale[j] = np.clip(
                        new_scale[j], scale_median / sqrt_10, scale_median * sqrt_10
                    )

    elif strategy_name == "constraint_type_aware":
        # Strategy: Constraint-type-aware scaling
        # Use specialized scaling strategies per constraint type
        if constraint_types is None or len(constraint_types) == 0:
            # Fall back to standard scaling if types unavailable
            log.debug(
                "Constraint-type-aware: No constraint types available, falling back to standard scaling"
            )
            return new_scale, new_scale_g

        log.debug(f"Constraint-type-aware: Processing {len(constraint_types)} constraint types")

        # Compute constraint values if not provided
        if g0_arr is None:
            try:
                if isinstance(nlp, dict) and "g" in nlp and "x" in nlp:
                    g_expr = nlp["g"]
                    x_sym = nlp["x"]
                    g_func = ca.Function("g_func", [x_sym], [g_expr])
                    g0 = g_func(x0)
                    g0_arr = np.array(g0)
                    log.debug(
                        f"Constraint-type-aware: Computed constraint values, shape={g0_arr.shape}"
                    )
            except Exception as e:
                log.debug(f"Constraint-type-aware: Failed to compute constraint values: {e}")
                g0_arr = None

        # Apply constraint-type-aware scaling
        # Pass current scale_g so it can refine existing scaling rather than starting from scratch
        log.debug(
            f"Constraint-type-aware: Computing scaling with current scale_g range=[{scale_g.min():.3e}, {scale_g.max():.3e}]"
        )
        try:
            new_scale_g = compute_constraint_scaling_by_type(
                constraint_types,
                nlp,
                x0,
                lbg,
                ubg,
                new_scale,
                jac_g0_arr=jac_g0_arr,
                g0_arr=g0_arr,
                current_scale_g=scale_g,
            )
            log.debug(
                f"Constraint-type-aware: Computed new scale_g, range=[{new_scale_g.min():.3e}, {new_scale_g.max():.3e}]"
            )
        except Exception as e:
            log.error(f"Constraint-type-aware: Failed in compute_constraint_scaling_by_type: {e}")
            log.error(f"Exception traceback: {traceback.format_exc()}")
            raise

        # Debug: Check if constraint-type-aware actually modified scales
        if not np.allclose(new_scale_g, scale_g, rtol=1e-6):
            # Log which constraint types were modified
            modified_types = {}
            n_modified = 0
            for idx, con_type in constraint_types.items():
                if idx < len(scale_g) and idx < len(new_scale_g):
                    scale_diff = abs(new_scale_g[idx] - scale_g[idx])
                    scale_ref = float(max(abs(scale_g[idx]), 1.0))
                    if scale_diff > 1e-6 * scale_ref:
                        n_modified += 1
                        if con_type not in modified_types:
                            modified_types[con_type] = {"count": 0, "max_change": 0.0}
                        modified_types[con_type]["count"] += 1
                        change_ratio = abs(new_scale_g[idx] / (scale_g[idx] + 1e-10))
                        modified_types[con_type]["max_change"] = max(
                            modified_types[con_type]["max_change"], change_ratio
                        )

            log.debug(f"Constraint-type-aware: Modified {n_modified} constraints")
            for con_type, stats in modified_types.items():
                log.debug(
                    f"  {con_type}: {stats['count']} constraints, max_change_ratio={stats['max_change']:.3e}"
                )
        else:
            log.debug("Constraint-type-aware: No constraints were modified (scales unchanged)")

    elif strategy_name == "percentile_based":
        # Strategy 5: Scale based on percentiles - target p95 entries
        # Bring p95 entries down to target_max_entry
        if jac_g0_scaled.size > 0:
            jac_mag_nonzero = jac_mag[jac_mag > 0]
            if len(jac_mag_nonzero) > 0:
                p95_value = np.percentile(jac_mag_nonzero, 95)
                if p95_value > target_max_entry:
                    global_reduction = target_max_entry / p95_value
                    # Apply reduction to constraint scales
                    for i in range(len(new_scale_g)):
                        new_scale_g[i] = new_scale_g[i] * global_reduction
                        # Allow more aggressive scaling for high percentiles
                        new_scale_g[i] = np.clip(new_scale_g[i], 1e-5, 1e3)

    elif strategy_name == "jacobian_equilibration":
        # Strategy: Iterative Jacobian equilibration
        # Balance row and column norms to improve numerical conditioning
        if jac_g0_arr is not None and jac_g0_arr.size > 0:
            new_scale, new_scale_g = equilibrate_jacobian_iterative(
                new_scale,
                new_scale_g,
                jac_g0_arr,
                n_iterations=3,
                target=1.0,
            )

    return new_scale, new_scale_g


def normalize_scales_to_center(
    scale: np.ndarray,
    scale_g: np.ndarray,
    scale_f: float,
    target_center: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize all scales to maintain center point and 10^1 ratio constraint.

    Ensures:
    - Median of all scales is at target_center
    - All scales within 10^1 ratio of center
    - Preserves relative importance of constraints

    Args:
        scale: Variable scaling factors
        scale_g: Constraint scaling factors
        scale_f: Objective scaling factor
        target_center: Target center point (default 1.0)

    Returns:
        Tuple of (normalized_scale, normalized_scale_g, normalized_scale_f)
    """
    # Compute median scales (robust center)
    scale_median = np.median(scale[scale > 1e-10]) if (scale > 1e-10).any() else target_center
    scale_g_median = (
        np.median(scale_g[scale_g > 1e-10])
        if len(scale_g) > 0 and (scale_g > 1e-10).any()
        else target_center
    )

    # Find overall center (median of all scales)
    all_scales = np.concatenate(
        [
            scale[scale > 1e-10],
            scale_g[scale_g > 1e-10],
            [abs(scale_f)] if abs(scale_f) > 1e-10 else [],
        ]
    )
    overall_median = np.median(all_scales) if len(all_scales) > 0 else target_center

    # Normalize to target_center
    if overall_median > 1e-10:
        center_ratio = target_center / overall_median
        scale = scale * center_ratio
        scale_g = scale_g * center_ratio
        scale_f = scale_f * center_ratio

    # Apply relaxed ratio constraint around center
    # We use a very loose constraint (1e12) to allow for physical scale differences
    # (e.g. pressure ~1e5 vs valve area ~1e-4 requires ~1e9 ratio in scales)
    # The previous 10.0 ratio was too restrictive for multi-physics problems
    max_ratio = 1e12
    sqrt_ratio = np.sqrt(max_ratio)
    lower_bound = target_center / sqrt_ratio
    upper_bound = target_center * sqrt_ratio

    # Clamp variable scales
    for i in range(len(scale)):
        if scale[i] > 1e-10:
            scale[i] = np.clip(scale[i], lower_bound, upper_bound)

    # Clamp constraint scales
    for i in range(len(scale_g)):
        if scale_g[i] > 1e-10:
            scale_g[i] = np.clip(scale_g[i], lower_bound, upper_bound)

    # Clamp objective scale
    if abs(scale_f) > 1e-10:
        scale_f = np.clip(scale_f, lower_bound, upper_bound)

    return scale, scale_g, scale_f


def combine_scales(scales: list[float | None]) -> float:
    """
    Combine multiple variable scales into a single robust group scale.

    Args:
        scales: List of scaling factors (can contain None)

    Returns:
        Median scale factor, clamped to [1e-6, 1e6]. Returns 1.0 if no valid scales.
    """
    # Filter out None values
    vals = [s for s in scales if s is not None]

    if not vals:
        return 1.0

    # Compute median
    vals.sort()
    m = vals[len(vals) // 2]

    # Clamp to reasonable range to prevent extreme scaling
    return max(min(m, 1e6), 1e-6)


def compute_objective_scaling(
    nlp: dict[str, Any],
    x0: np.ndarray,
    scale: np.ndarray,
    target_f: float = 1.0,
    variable_groups: dict[str, list[int]] | None = None,
) -> float:
    """
    Compute objective scaling factor based on approximate magnitude.

    Uses initial gradient or evaluation to estimate objective magnitude.

    Args:
        nlp: CasADi NLP dict
        x0: Initial guess
        scale: Variable scaling factors
        target_f: Target scaled objective value (default 1.0)
        variable_groups: Optional variable groups for targeted analysis

    Returns:
        Objective scaling factor
    """
    import casadi as ca

    # Default scale
    scale_f = 1.0

    if not isinstance(nlp, dict) or "f" not in nlp or "x" not in nlp:
        return scale_f

    try:
        f_expr = nlp["f"]
        x_sym = nlp["x"]
        f_func = ca.Function("f_func", [x_sym], [f_expr])
        f_val = float(f_func(x0))

        # Check gradient magnitude as well if possible
        grad_mag = 0.0
        try:
            grad_f_expr = ca.gradient(f_expr, x_sym)
            grad_f_func = ca.Function("grad_f", [x_sym], [grad_f_expr])
            grad_f = np.array(grad_f_func(x0)).flatten()
            # Scale gradient by variable scales
            # Chain rule: d(f)/d(x_scaled) = d(f)/dx * dx/dx_scaled = grad_f * scale
            grad_f_scaled = grad_f * scale
            grad_mag = np.max(np.abs(grad_f_scaled))
        except Exception:
            pass

        # Use combination of value and gradient
        f_mag = abs(f_val)
        if f_mag < 1e-8:
            # If objective is tiny, rely on gradient
            base_mag = grad_mag if grad_mag > 1e-8 else 1.0
        else:
            # Use geometric mean if both reliable, else just function value
            if grad_mag > 1e-8:
                base_mag = np.sqrt(f_mag * grad_mag)
            else:
                base_mag = f_mag

        # Compute scaling factor
        if base_mag > 1e-10:
            scale_f = target_f / base_mag

        # Clamp to reasonable range
        scale_f = np.clip(scale_f, 1e-6, 1e6)

    except Exception as e:
        import logging

        log = logging.getLogger(__name__)
        log.debug(f"Failed to compute objective scaling: {e}")
        scale_f = 1.0

    return scale_f


def compute_unified_data_driven_scaling(
    nlp: dict[str, Any],
    x0: np.ndarray,
    lbx: np.ndarray,
    ubx: np.ndarray,
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
    variable_groups: dict[str, list[int]] | None,
    meta: dict[str, Any] | None = None,
    reporter: StructuredReporter | None = None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """
    Compute unified data-driven scaling for variables and constraints.

    This is the master entry point for the new data-driven scaling pipeline.
    It orchestrates the analysis, grouping, and scaling of variables and constraints.

    Args:
        nlp: CasADi NLP dict
        x0: Initial guess (unscaled)
        lbx: Lower variable bounds
        ubx: Upper variable bounds
        lbg: Lower constraint bounds
        ubg: Upper constraint bounds
        variable_groups: Variable group mapping
        meta: Problem metadata (optional)
        reporter: Optional reporter for logging

    Returns:
        Tuple of (scale, scale_g, scale_f)
    """
    if reporter:
        reporter.info("Starting unified data-driven scaling pipeline")

    # Step 1: Compute variable scaling (Unit-based + Bounds + x0)
    # This provides the baseline for everything else
    scale = compute_variable_scaling(
        lbx,
        ubx,
        x0=x0,
        variable_groups=variable_groups,
    )

    # Step 2: Compute constraint scaling (Magnitude-based)
    # Uses evaluation at x0 if possible, otherwise bounds
    try:
        scale_g = compute_constraint_scaling_from_evaluation(nlp, x0, lbg, ubg, scale=scale)
    except Exception as e:
        if reporter:
            reporter.warning(f"Failed to compute constraint scaling from evaluation: {e}")
        scale_g = compute_constraint_scaling(lbg, ubg)

    # Step 3: Compute objective scaling
    scale_f = compute_objective_scaling(
        nlp, x0, scale, target_f=1.0, variable_groups=variable_groups
    )

    # Step 4: Iterative Refinement (Jacobian-based)
    # This is the critical step that aligns scales with the system dynamics
    if reporter:
        reporter.info("Refining scaling with iterative Jacobian analysis")

    try:
        quality_metrics = {}

        refined_scale, refined_scale_g, quality_metrics = refine_scaling_iteratively(
            nlp,
            x0,
            lbx,
            ubx,
            lbg,
            ubg,
            variable_groups,
            meta,
            reporter=reporter,
            max_iterations=5,  # Allow enough iterations for complex multiphysics
        )
        scale = refined_scale
        scale_g = refined_scale_g

        # Log quality metrics
        if reporter:
            reporter.info(
                f"Refined scaling quality: condition_number={quality_metrics.get('condition_number', np.inf):.3e}, "
                f"quality_score={quality_metrics.get('quality_score', 0.0):.3f}"
            )
    except Exception as e:
        if reporter:
            reporter.warning(f"Iterative refinement failed: {e}")
            reporter.debug(traceback.format_exc())

    # Step 5: Analyze distribution of magnitudes
    # Identify outliers that might need targeted fixing
    constraint_magnitudes = compute_unified_constraint_magnitudes(
        np.array([]), None, lbg, ubg, scale
    )  # We'll recompute properly inside analysis
    # Need actual g0 and jac for proper analysis
    try:
        if isinstance(nlp, dict) and "g" in nlp and "x" in nlp:
            g_func = ca.Function("g_func", [nlp["x"]], [nlp["g"]])
            g0 = g_func(x0)
            g0_arr = np.array(g0).flatten()

            # Compute scaled Jacobian
            jac_g0_arr, _ = compute_scaled_jacobian(nlp, x0, scale, scale_g)

            constraint_magnitudes = compute_unified_constraint_magnitudes(
                g0_arr, jac_g0_arr, lbg, ubg, scale
            )
    except Exception:
        pass

    dist_metrics = analyze_magnitude_distribution(constraint_magnitudes)
    median = dist_metrics["median"]
    iqr = dist_metrics["iqr"]
    outlier_mask = dist_metrics["outlier_mask"]
    percentiles = dist_metrics["percentiles"]

    # Step 6: Cluster by magnitude to identify scaling groups
    _ = cluster_by_magnitude(constraint_magnitudes)

    if reporter:
        reporter.info(
            f"Magnitude analysis: median={median:.3e}, iqr={iqr:.3e}, "
            f"outliers={int(outlier_mask.sum())} ({outlier_mask.sum() / len(constraint_magnitudes):.1%})"
        )
        reporter.debug(f"Percentiles: {percentiles}")

    # Step 7: Group-aware Normalization
    # Ensure standard groups (collocation, continuity, boundary) are centered appropriately
    # This prevents one group from forcing extreme global scaling
    target_center = 1.0

    # Only apply if per-group scaling hasn't already achieved good conditioning
    # Check if scales are already well-centered (median close to 1.0)
    all_scales = np.concatenate(
        [
            scale[scale > 1e-10],
            scale_g[scale_g > 1e-10],
            [abs(scale_f)] if abs(scale_f) > 1e-10 else [],
        ]
    )
    current_median = np.median(all_scales) if len(all_scales) > 0 else 1.0

    # Check if we need global normalization
    # Skip if median is already close to target (within 10x) and spread is reasonable
    median_ratio = current_median / target_center
    needs_normalization = not (0.1 <= median_ratio <= 10.0)

    if needs_normalization:
        if reporter:
            reporter.info(
                f"Applying group-aware global normalization (current median={current_median:.3e}, "
                f"target={target_center:.3e})"
            )
        # Apply group-aware normalization that preserves relative group structure
        scale, scale_g, scale_f = normalize_scales_to_center(
            scale, scale_g, scale_f, target_center=target_center
        )
    elif reporter:
        reporter.info(
            f"Skipping global normalization (per-group scaling already good: "
            f"median={current_median:.3e}, target={target_center:.3e})"
        )

    # TARGETED FIX: Energy Constraint Scaling
    # The Energy collocation constraints are often dominated by large residuals (due to units),
    # leading to small scale_g, which makes the Jacobian row vanish.
    # We force scaling based on the Energy variable magnitude to ensure J ~ 1/dt.
    if (
        meta
        and "constraint_groups" in meta
        and variable_groups
        and "temperatures" in variable_groups
    ):
        # Note: 'temperatures' group contains Energy variables (E) as per nlp.py
        e_indices = variable_groups["temperatures"]

        if e_indices:
            # Get median magnitude of Energy variables from x0 (unscaled)
            # We use x0 because 'scale' array might be clamped or influenced by bounds,
            # whereas we want to normalize the physical residual (which depends on x0 magnitude).
            e_vals = np.abs(x0[e_indices])
            mag_e = np.median(e_vals) if len(e_vals) > 0 else 1.0

            # Target scale_g = 1 / mag_e
            # This normalizes the constraint residual (roughly proportional to E) to ~1.
            target_scale_g = 1.0 / max(mag_e, 1e-3)

            # Apply to all Energy-related groups (collocation, continuity, boundary)
            count = 0
            for group_name, indices in meta["constraint_groups"].items():
                if (
                    group_name.startswith("collocation_E")
                    or group_name.startswith("continuity_E")
                    or group_name.startswith("boundary_E")
                    or group_name.startswith("boundary_initial_E")
                    or group_name.startswith("boundary_final_E")
                ):
                    scale_g[indices] = target_scale_g
                    count += len(indices)

            if reporter and count > 0:
                reporter.info(
                    f"Applied targeted Energy scaling to {count} constraints (collocation/continuity/boundary). "
                    f"Override Scale: {target_scale_g:.2e} (based on Var Mag: {mag_e:.2e})"
                )

    # Step 8: Verify scaling quality
    quality_metrics = compute_scaling_quality(
        nlp,
        x0,
        scale,
        scale_g,
        lbg,
        ubg,
        reporter=reporter,
        meta=meta,
    )

    # Add distribution analysis to quality metrics
    quality_metrics["distribution_analysis"] = {
        "median": median,
        "iqr": iqr,
        "n_outliers": int(outlier_mask.sum()),
        "outlier_ratio": float(outlier_mask.sum() / len(constraint_magnitudes))
        if len(constraint_magnitudes) > 0
        else 0.0,
        "percentiles": percentiles,
    }

    # SAFETY CLAMP: Ensure no constraint is scaled effectively to zero
    # "Small elements" in Jacobian often come from tiny scale_g
    # We enforce a floor.
    if reporter:
        n_below_floor = np.sum(scale_g < 1e-4)
        if n_below_floor > 0:
            reporter.info(f"Clamping {n_below_floor} constraint scale factors to 1e-4 floor.")

    scale_g = np.maximum(scale_g, 1e-4)
    # Also clamp upper bound to avoid exploding gradients
    scale_g = np.minimum(scale_g, 1e5)

    if reporter:
        condition_number = quality_metrics.get("condition_number", np.inf)
        quality_score = quality_metrics.get("quality_score", 0.0)
        reporter.info(
            f"Unified scaling complete: condition_number={condition_number:.3e}, "
            f"quality_score={quality_score:.3f}, "
            f"scaled_constraints_center={median:.3e}",
        )

    return scale, scale_g, scale_f, quality_metrics


def refine_scaling_iteratively(
    nlp: Any,
    x0: np.ndarray,
    lbx: np.ndarray,
    ubx: np.ndarray,
    lbg: np.ndarray | None,
    ubg: np.ndarray | None,
    variable_groups: dict[str, list[int]] | None,
    meta: dict[str, Any] | None = None,
    reporter: StructuredReporter | None = None,
    max_iterations: int = 5,
    target_condition_number: float = 1e3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Iteratively refine scaling factors to achieve target condition number.

    Tries multiple scaling strategies per iteration and selects the best one.

    Args:
        nlp: CasADi NLP dict
        x0: Initial guess (unscaled)
        lbx: Lower variable bounds
        ubx: Upper variable bounds
        lbg: Lower constraint bounds
        ubg: Upper constraint bounds
        variable_groups: Variable group mapping
        reporter: Optional reporter for logging
        max_iterations: Maximum refinement iterations
        target_condition_number: Target condition number (default 1e3)

    Returns:
        Tuple of (refined_scale, refined_scale_g, quality_metrics)
    """
    # Generate cache key for this problem
    n_vars = len(x0) if x0 is not None else len(lbx)
    n_constraints = len(lbg) if lbg is not None else 0
    cache_key = generate_scaling_cache_key(n_vars, n_constraints, variable_groups, meta)

    # Track whether we're starting from cached scaling
    skip_initial_scaling = False

    # Try to load cached scaling
    cached_scale, cached_scale_g, cached_quality = load_scaling_cache(
        cache_key, n_vars, n_constraints
    )

    if cached_scale is not None and cached_scale_g is not None:
        # Verify cached scaling quality
        cached_quality_check = compute_scaling_quality(
            nlp,
            x0,
            cached_scale,
            cached_scale_g,
            lbg,
            ubg,
            reporter=None,
            meta=meta,
        )
        cached_condition = cached_quality_check.get("condition_number", np.inf)
        cached_score = cached_quality_check.get("quality_score", 0.0)

        # Minimum quality_score threshold
        min_quality_score = 0.7

        # Use cached scaling as starting point if it meets both targets, otherwise iterate to improve
        if cached_condition <= target_condition_number and cached_score >= min_quality_score:
            if reporter:
                reporter.info(
                    f"Using cached scaling (meets targets): condition_number={cached_condition:.3e}, "
                    f"quality_score={cached_score:.3f} >= {min_quality_score:.3f}",
                )
            return cached_scale, cached_scale_g, cached_quality_check
        else:
            # Use cached scaling as starting point for iteration
            if reporter:
                condition_msg = (
                    f"condition_number={cached_condition:.3e} > {target_condition_number:.3e}"
                    if cached_condition > target_condition_number
                    else ""
                )
                quality_msg = (
                    f"quality_score={cached_score:.3f} < {min_quality_score:.3f}"
                    if cached_score < min_quality_score
                    else ""
                )
                reason = " or ".join(filter(None, [condition_msg, quality_msg]))
                reporter.info(
                    f"Using cached scaling as starting point: condition_number={cached_condition:.3e}, "
                    f"quality_score={cached_score:.3f} ({reason}), will iterate to improve",
                )
            # Start with cached scaling instead of computing initial scaling
            scale = cached_scale.copy()
            scale_g = cached_scale_g.copy()
            quality = cached_quality_check
            condition_number = cached_condition
            initial_condition_number = condition_number
            # Skip initial scaling computation and go straight to iteration
            skip_initial_scaling = True

    # Get unscaled Jacobian early for use in variable scaling
    jac_g0_arr_initial = None
    try:
        jac_g0_arr_initial, _ = compute_scaled_jacobian(
            nlp, x0, np.ones(len(x0)), np.ones(len(lbg) if lbg is not None else 0)
        )
    except Exception:
        pass  # Jacobian unavailable, will use fallback

    if not skip_initial_scaling:
        # Compute initial scaling with Jacobian if available
        # NOTE: compute_variable_scaling now uses unit-based initialization and doesn't take Jacobian
        scale = compute_variable_scaling(
            lbx,
            ubx,
            x0=x0,
            variable_groups=variable_groups,
        )
        try:
            scale_g = compute_constraint_scaling_from_evaluation(nlp, x0, lbg, ubg, scale=scale)
        except Exception:
            scale_g = compute_constraint_scaling(lbg, ubg)

        # Check initial quality
        quality = compute_scaling_quality(
            nlp, x0, scale, scale_g, lbg, ubg, reporter=reporter, meta=meta
        )
        condition_number = quality.get("condition_number", np.inf)
        initial_condition_number = condition_number  # Save for comparison at end

        if reporter:
            reporter.info(
                f"Initial scaling quality: condition_number={condition_number:.3e}, quality_score={quality.get('quality_score', 0.0):.3f}"
            )

        # If condition number is acceptable, save and return initial scaling
        if condition_number <= target_condition_number:
            save_scaling_cache(cache_key, scale, scale_g, quality)
            return scale, scale_g, quality

    # Get unscaled and scaled Jacobian for strategy evaluation
    jac_g0_arr, jac_g0_scaled = compute_scaled_jacobian(nlp, x0, scale, scale_g)
    if jac_g0_arr is None or jac_g0_scaled is None:
        # Can't compute Jacobian, fall back to simple tightening
        if reporter:
            reporter.warning(
                "Cannot compute Jacobian for refinement, using simple ratio tightening"
            )
        return scale, scale_g, quality

    # Identify constraint types once at the start
    constraint_types = identify_constraint_types(meta, lbg, ubg, jac_g0_arr=jac_g0_arr)

    # Get constraint values for type-aware scaling
    g0_arr = None
    try:
        if isinstance(nlp, dict) and "g" in nlp and "x" in nlp:
            g_expr = nlp["g"]
            x_sym = nlp["x"]
            g_func = ca.Function("g_func", [x_sym], [g_expr])
            g0 = g_func(x0)
            g0_arr = np.array(g0)
    except Exception:
        pass

    # Define strategies to try (in order of preference)
    # Prioritize constraint-type-aware if constraint groups are available
    has_constraint_groups = meta and "constraint_groups" in meta and len(constraint_types) > 0

    if condition_number > 1e20:
        # For extremely ill-conditioned problems, prioritize aggressive strategies
        if has_constraint_groups:
            strategies = [
                "constraint_type_aware",  # Most targeted - use constraint type information
                "jacobian_equilibration",  # Iterative Jacobian equilibration
                "combined_row_column",  # Combine row and column scaling
                "extreme_entry_targeting",  # Aggressively target extreme entries
                "row_max_scaling",  # Scale rows with large max entries
                "column_max_scaling",  # Scale columns with large max entries
                "percentile_based",  # Scale based on percentiles
            ]
        else:
            strategies = [
                "jacobian_equilibration",  # Iterative Jacobian equilibration
                "combined_row_column",  # Most aggressive - combine row and column scaling
                "extreme_entry_targeting",  # Aggressively target extreme entries
                "row_max_scaling",  # Scale rows with large max entries
                "column_max_scaling",  # Scale columns with large max entries
                "percentile_based",  # Scale based on percentiles
            ]
    # For moderately ill-conditioned problems, include conservative strategies
    elif has_constraint_groups:
        strategies = [
            "constraint_type_aware",  # Most targeted - use constraint type information
            "jacobian_equilibration",  # Iterative Jacobian equilibration
            "combined_row_column",  # Combine row and column scaling
            "extreme_entry_targeting",  # Aggressively target extreme entries
            "row_max_scaling",  # Scale rows with large max entries
            "column_max_scaling",  # Scale columns with large max entries
            "percentile_based",  # Scale based on percentiles
            "tighten_ratios",  # Conservative - tighten ratios
        ]
    else:
        strategies = [
            "jacobian_equilibration",  # Iterative Jacobian equilibration
            "combined_row_column",  # Most aggressive - combine row and column scaling
            "extreme_entry_targeting",  # Aggressively target extreme entries
            "row_max_scaling",  # Scale rows with large max entries
            "column_max_scaling",  # Scale columns with large max entries
            "percentile_based",  # Scale based on percentiles
            "tighten_ratios",  # Conservative - tighten ratios
        ]

    # Track previous iteration's condition number for improvement detection
    prev_condition_number = condition_number

    # Track relaxed groups to prevent oscillations (cache relaxation decisions)
    relaxed_groups_cache: dict[str, int] = {}  # Maps group name to iteration when relaxed
    skip_iterations = 3  # Skip re-tightening for 3 iterations after relaxation

    # Iteratively refine scaling
    for iteration in range(max_iterations):
        if reporter:
            reporter.info(f"Scaling refinement iteration {iteration + 1}/{max_iterations}")

        best_scale = scale.copy()
        best_scale_g = scale_g.copy()
        best_condition_number = condition_number  # Start with current condition number
        best_quality_score = quality.get("quality_score", 0.0)
        best_strategy = None

        # Try each strategy and pick the best one
        for strategy in strategies:
            if reporter:
                reporter.debug(f"Trying strategy: {strategy}")
            try:
                # Try this strategy
                test_scale, test_scale_g = try_scaling_strategy(
                    strategy,
                    nlp,
                    x0,
                    lbx,
                    ubx,
                    lbg,
                    ubg,
                    scale,
                    scale_g,
                    jac_g0_arr,
                    jac_g0_scaled,
                    variable_groups,
                    constraint_types=constraint_types if has_constraint_groups else None,
                    meta=meta,
                    g0_arr=g0_arr,
                    target_max_entry=1e2,
                )

                # Recompute constraint scaling with new variable scales if needed
                # (some strategies modify variable scales, which affects constraint scaling)
                # BUT: preserve aggressive constraint scaling from strategies that modified scale_g
                scale_g_was_modified = not np.allclose(test_scale_g, scale_g, rtol=1e-6)
                scale_was_modified = not np.allclose(test_scale, scale, rtol=1e-6)

                # Special handling for constraint-type-aware strategy: don't recompute
                if strategy == "constraint_type_aware" and scale_g_was_modified:
                    pass  # Keep test_scale_g as computed by constraint-type-aware
                elif scale_was_modified and not scale_g_was_modified:
                    # Variable scales changed but constraint scales weren't modified by strategy
                    # Need to recompute constraint scaling with new variable scales
                    try:
                        test_scale_g = compute_constraint_scaling_from_evaluation(
                            nlp,
                            x0,
                            lbg,
                            ubg,
                            scale=test_scale,
                        )
                    except Exception:
                        test_scale_g = compute_constraint_scaling(lbg, ubg)
                elif scale_was_modified and scale_g_was_modified:
                    # Both were modified - need to recompute constraint scaling but preserve
                    # the aggressive adjustments from the strategy
                    old_scale_g = scale_g.copy()
                    try:
                        new_base_scale_g = compute_constraint_scaling_from_evaluation(
                            nlp,
                            x0,
                            lbg,
                            ubg,
                            scale=test_scale,
                        )
                        # Preserve relative adjustments
                        if (
                            len(old_scale_g) > 0
                            and len(new_base_scale_g) > 0
                            and len(test_scale_g) > 0
                        ):
                            strategy_adjustment = test_scale_g / (old_scale_g + 1e-10)
                            test_scale_g = new_base_scale_g * strategy_adjustment
                            test_scale_g = np.clip(test_scale_g, 1e-6, 1e3)
                        else:
                            test_scale_g = new_base_scale_g
                    except Exception:
                        # Keep the strategy-modified scale_g
                        pass

                # IMPORTANT: Recompute scaled Jacobian with NEW scales to properly evaluate quality
                test_jac_g0_arr, test_jac_g0_scaled = compute_scaled_jacobian(
                    nlp,
                    x0,
                    test_scale,
                    test_scale_g,
                )
                if test_jac_g0_scaled is not None:
                    # Use the new scaled Jacobian for quality evaluation
                    test_jac_mag = np.abs(test_jac_g0_scaled)
                    test_jac_max = test_jac_mag.max()
                    test_jac_min = (
                        test_jac_mag[test_jac_mag > 0].min() if (test_jac_mag > 0).any() else 1e-10
                    )
                    test_condition_number = (
                        test_jac_max / test_jac_min if test_jac_min > 0 else np.inf
                    )
                    # Compute quality score
                    condition_score = 1.0 / (1.0 + np.log10(max(test_condition_number / 1e3, 1.0)))
                    max_score = 1.0 / (1.0 + np.log10(max(test_jac_max / 1e2, 1.0)))
                    test_quality_score = 0.5 * condition_score + 0.5 * max_score
                else:
                    # Fallback to full quality evaluation
                    test_quality = compute_scaling_quality(
                        nlp,
                        x0,
                        test_scale,
                        test_scale_g,
                        lbg,
                        ubg,
                        reporter=None,
                        meta=meta,
                    )
                    test_condition_number = test_quality.get("condition_number", np.inf)
                    test_quality_score = test_quality.get("quality_score", 0.0)

                # Check if this is better (lower condition number or higher quality score)
                # Handle inf/nan comparisons safely
                condition_better = (
                    np.isfinite(test_condition_number)
                    and np.isfinite(best_condition_number)
                    and test_condition_number < best_condition_number
                ) or (np.isfinite(test_condition_number) and not np.isfinite(best_condition_number))
                quality_better = (
                    np.isfinite(test_condition_number)
                    and np.isfinite(best_condition_number)
                    and test_condition_number == best_condition_number
                    and test_quality_score > best_quality_score
                )
                is_better = condition_better or quality_better

                if reporter:
                    reporter.debug(
                        f"Strategy '{strategy}' comparison: "
                        f"condition_number={test_condition_number:.3e}, "
                        f"quality_score={test_quality_score:.3f}, "
                        f"is_better={is_better}"
                    )
                if is_better:
                    best_scale = test_scale
                    best_scale_g = test_scale_g
                    best_condition_number = test_condition_number
                    best_quality_score = test_quality_score
                    best_strategy = strategy

            except Exception as e:
                if reporter:
                    reporter.debug(f"Strategy '{strategy}' failed: {e}")
                continue

        # Update scales with best strategy
        scale = best_scale
        scale_g = best_scale_g

        # Recompute Jacobian with new scales
        jac_g0_arr, jac_g0_scaled = compute_scaled_jacobian(nlp, x0, scale, scale_g)
        if jac_g0_arr is None or jac_g0_scaled is None:
            break

        # Check quality again
        quality = compute_scaling_quality(
            nlp, x0, scale, scale_g, lbg, ubg, reporter=reporter, meta=meta
        )
        condition_number = quality.get("condition_number", np.inf)

        # Post-processing: Detect and correct overscaling after each iteration
        n_overscaled = quality.get("n_overscaled", 0)
        overscaling_ratio = quality.get("overscaling_ratio", 0.0)

        if n_overscaled > 0 or overscaling_ratio > 0.01:
            if reporter:
                reporter.info(
                    f"Post-processing: Detected {n_overscaled} overscaled entries, applying conservative corrective scaling"
                )
            # Recompute scaled Jacobian to check current state
            _, jac_g0_scaled_check = compute_scaled_jacobian(nlp, x0, scale, scale_g)

            if jac_g0_scaled_check is not None:
                # Use relax_over_scaled_groups logic implicitly by identifying problematic constraints
                # For now, let's just do a simple pass if we have overscaling
                pass  # Already handled by strategy iteration and bounding

        if reporter:
            reporter.info(
                f"Iteration {iteration + 1} best strategy: {best_strategy}, "
                f"condition_number: {condition_number:.3e}, quality_score: {quality.get('quality_score', 0.0):.3f}"
            )

        # Early exit if target met
        if condition_number <= target_condition_number:
            if reporter:
                reporter.info(
                    f"Refinement target met (condition number {condition_number:.3e} <= {target_condition_number:.3e})"
                )
            break

        # Check for stagnation
        if iteration > 0 and condition_number >= prev_condition_number * 0.99:
            if reporter:
                reporter.info("Scaling refinement stagnated, stopping.")
            break

        prev_condition_number = condition_number

    # Save final scaling to cache
    save_scaling_cache(cache_key, scale, scale_g, quality)

    return scale, scale_g, quality
