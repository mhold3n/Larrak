"""
Surrogate Validation Metrics: Boundary-aware and safety metrics.

Provides validation metrics beyond RMSE that are relevant for CEM deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np


@dataclass
class SurrogateValidationMetrics:
    """
    Comprehensive metrics for surrogate validation.
    
    In CEM context, low RMSE can be useless if the NN:
    - Violates constraints
    - Misclassifies feasibility
    - Fails near boundaries
    """
    # Standard metrics
    rmse: float
    mae: float
    r2: float
    
    # Boundary-aware metrics
    boundary_rmse: float    # Error near constraint boundaries
    interior_rmse: float    # Error in interior of feasible region
    worst_case_error: float  # Maximum error in feasible region
    
    # Safety metrics (for binary feasibility predictions)
    false_accept_rate: float  # Predicted feasible but actually violates
    false_reject_rate: float  # Predicted violation but actually feasible
    
    # Uncertainty calibration
    coverage_95: float       # Fraction of true values within 95% CI
    sharpness: float         # Mean prediction interval width
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'boundary_rmse': self.boundary_rmse,
            'interior_rmse': self.interior_rmse,
            'worst_case_error': self.worst_case_error,
            'false_accept_rate': self.false_accept_rate,
            'false_reject_rate': self.false_reject_rate,
            'coverage_95': self.coverage_95,
            'sharpness': self.sharpness,
        }


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def compute_boundary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    margins: np.ndarray,
    boundary_threshold: float = 0.1
) -> Tuple[float, float]:
    """
    Compute error conditioned on being near constraint boundaries.
    
    Args:
        y_true: True values (n_samples, n_outputs)
        y_pred: Predicted values (n_samples, n_outputs)
        margins: Constraint margin values (n_samples,) or (n_samples, n_constraints)
        boundary_threshold: Relative margin threshold for "near boundary"
        
    Returns:
        boundary_rmse: RMSE for samples near boundaries
        interior_rmse: RMSE for samples in interior
    """
    if margins.ndim > 1:
        # Take minimum margin across constraints
        min_margins = np.min(margins, axis=1)
    else:
        min_margins = margins
    
    # Normalize margins (assuming positive is feasible)
    if np.max(np.abs(min_margins)) > 0:
        normalized_margins = np.abs(min_margins) / np.max(np.abs(min_margins))
    else:
        normalized_margins = np.zeros_like(min_margins)
    
    boundary_mask = normalized_margins < boundary_threshold
    interior_mask = ~boundary_mask
    
    errors = (y_true - y_pred) ** 2
    
    if boundary_mask.sum() > 0:
        boundary_rmse = np.sqrt(errors[boundary_mask].mean())
    else:
        boundary_rmse = 0.0
    
    if interior_mask.sum() > 0:
        interior_rmse = np.sqrt(errors[interior_mask].mean())
    else:
        interior_rmse = 0.0
    
    return boundary_rmse, interior_rmse


def compute_safety_metrics(
    pred_feasible: np.ndarray,
    actual_feasible: np.ndarray
) -> Tuple[float, float]:
    """
    Compute false accept/reject rates for feasibility classification.
    
    Args:
        pred_feasible: Predicted feasibility (boolean array)
        actual_feasible: Actual feasibility (boolean array)
        
    Returns:
        false_accept_rate: Rate of predicting feasible when actually infeasible
        false_reject_rate: Rate of predicting infeasible when actually feasible
    """
    pred_feasible = np.asarray(pred_feasible, dtype=bool)
    actual_feasible = np.asarray(actual_feasible, dtype=bool)
    
    # False accept: predicted feasible but actually infeasible
    actually_infeasible = ~actual_feasible
    if actually_infeasible.sum() > 0:
        false_accept = (pred_feasible & actually_infeasible).sum() / actually_infeasible.sum()
    else:
        false_accept = 0.0
    
    # False reject: predicted infeasible but actually feasible
    if actual_feasible.sum() > 0:
        false_reject = ((~pred_feasible) & actual_feasible).sum() / actual_feasible.sum()
    else:
        false_reject = 0.0
    
    return float(false_accept), float(false_reject)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    coverage_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute uncertainty calibration metrics.
    
    Args:
        y_true: True values
        y_mean: Predicted mean
        y_std: Predicted standard deviation (uncertainty)
        coverage_level: Target coverage (e.g., 0.95)
        
    Returns:
        coverage: Fraction of true values within confidence interval
        sharpness: Mean interval width
    """
    from scipy import stats
    
    # Z-score for desired coverage
    z = stats.norm.ppf((1 + coverage_level) / 2)
    
    # Confidence interval
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std
    
    # Coverage: fraction within interval
    within = (y_true >= lower) & (y_true <= upper)
    coverage = within.mean()
    
    # Sharpness: mean interval width
    sharpness = (upper - lower).mean()
    
    return float(coverage), float(sharpness)


def evaluate_surrogate(
    model: 'torch.nn.Module',
    X: np.ndarray,
    y_true: np.ndarray,
    margins: Optional[np.ndarray] = None,
    feasibility: Optional[np.ndarray] = None,
    is_ensemble: bool = True
) -> SurrogateValidationMetrics:
    """
    Comprehensive evaluation of a surrogate model.
    
    Args:
        model: Trained surrogate (EnsembleSurrogate or single model)
        X: Input features
        y_true: True output values
        margins: Optional constraint margins for boundary analysis
        feasibility: Optional actual feasibility labels
        is_ensemble: Whether model returns (mean, std)
        
    Returns:
        SurrogateValidationMetrics with all computed metrics
    """
    import torch
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        output = model(X_tensor)
        
        if is_ensemble and isinstance(output, tuple):
            y_mean = output[0].numpy()
            y_std = output[1].numpy()
        else:
            y_mean = output.numpy() if torch.is_tensor(output) else output
            y_std = np.zeros_like(y_mean)
    
    # Standard metrics
    errors = y_true - y_mean
    rmse = np.sqrt((errors ** 2).mean())
    mae = np.abs(errors).mean()
    r2 = compute_r2(y_true.flatten(), y_mean.flatten())
    
    # Boundary metrics
    if margins is not None:
        boundary_rmse, interior_rmse = compute_boundary_metrics(y_true, y_mean, margins)
    else:
        boundary_rmse = interior_rmse = rmse
    
    worst_case = np.max(np.abs(errors))
    
    # Safety metrics
    if feasibility is not None:
        # Use uncertainty threshold as proxy for feasibility prediction
        pred_feasible = y_std.max(axis=-1) < np.median(y_std)
        far, frr = compute_safety_metrics(pred_feasible, feasibility)
    else:
        far = frr = 0.0
    
    # Calibration
    if y_std.max() > 0:
        coverage, sharpness = compute_calibration_metrics(
            y_true.flatten(), y_mean.flatten(), y_std.flatten()
        )
    else:
        coverage = sharpness = 0.0
    
    return SurrogateValidationMetrics(
        rmse=float(rmse),
        mae=float(mae),
        r2=float(r2),
        boundary_rmse=float(boundary_rmse),
        interior_rmse=float(interior_rmse),
        worst_case_error=float(worst_case),
        false_accept_rate=float(far),
        false_reject_rate=float(frr),
        coverage_95=float(coverage),
        sharpness=float(sharpness),
    )
