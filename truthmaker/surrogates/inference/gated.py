"""
Gated Surrogate Inference: Production wrapper with CEM gating and fallback.

Provides safe deployment with:
- CEM regime validation
- Uncertainty threshold gating
- Fallback to real evaluator
- Logging for retraining
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, Union
import numpy as np
import torch

from .ensemble import EnsembleSurrogate
from .collector import CEMDataCollector
from truthmaker.cem import CEMClient, ValidationReport, OperatingRegime


@dataclass
class SurrogatePrediction:
    """Result of surrogate prediction with metadata."""
    value: np.ndarray
    uncertainty: np.ndarray
    used_surrogate: bool
    fallback_reason: Optional[str] = None
    regime_id: int = 0
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if prediction has low uncertainty."""
        return self.used_surrogate and self.uncertainty.max() < 0.1


class GatedSurrogateInference:
    """
    Production wrapper with CEM gating and fallback.
    
    Deployment flow:
    1. CEM checks regime validity
    2. NN predicts metrics + uncertainty
    3. If uncertainty/conditions fail → call fallback evaluator
    4. Log prediction for retraining
    
    Usage:
        inference = GatedSurrogateInference(
            surrogate=trained_ensemble,
            cem=cem_client,
            fallback_evaluator=simple_cycle_adapter.evaluate,
            uncertainty_threshold=0.1
        )
        
        result = inference.predict(x)
        if result.used_surrogate:
            # Fast prediction used
        else:
            # Fallback was called, reason in result.fallback_reason
    """
    
    def __init__(
        self,
        surrogate: EnsembleSurrogate,
        cem: CEMClient,
        fallback_evaluator: Callable[[np.ndarray], np.ndarray],
        uncertainty_threshold: float = 0.1,
        valid_regimes: Optional[set] = None,
        collector: Optional[CEMDataCollector] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            surrogate: Trained ensemble surrogate
            cem: CEM client for validation
            fallback_evaluator: Function to call when surrogate is rejected
            uncertainty_threshold: Max acceptable uncertainty (std)
            valid_regimes: Set of valid OperatingRegime values for surrogate
            collector: Optional data collector for logging
            device: Torch device
        """
        self.surrogate = surrogate.to(device)
        self.cem = cem
        self.fallback = fallback_evaluator
        self.uncertainty_threshold = uncertainty_threshold
        self.valid_regimes = valid_regimes or {
            OperatingRegime.CRUISE.value,
            OperatingRegime.FULL_LOAD.value
        }
        self.collector = collector
        self.device = device
        
        # Statistics
        self.n_surrogate_calls = 0
        self.n_fallback_calls = 0
    
    def predict(
        self,
        x: Union[np.ndarray, Dict[str, float]],
        rpm: Optional[float] = None,
        boost_bar: Optional[float] = None,
        motion_profile: Optional[np.ndarray] = None
    ) -> SurrogatePrediction:
        """
        Make gated prediction.
        
        Args:
            x: Input features (array or dict)
            rpm: Engine speed for regime classification
            boost_bar: Boost pressure for regime classification
            motion_profile: Optional motion profile for CEM validation
            
        Returns:
            SurrogatePrediction with value, uncertainty, and metadata
        """
        # Convert dict to array if needed
        if isinstance(x, dict):
            x_array = np.array(list(x.values()), dtype=np.float32)
            rpm = rpm or x.get('rpm')
            boost_bar = boost_bar or x.get('boost_bar', x.get('p_int_bar', 1.0))
        else:
            x_array = np.asarray(x, dtype=np.float32)
        
        # Classify regime
        if rpm is not None and boost_bar is not None:
            regime = self.cem.classify_regime(rpm, boost_bar)
        else:
            regime = OperatingRegime.UNKNOWN
        
        # Gate 1: Regime check
        if regime.value not in self.valid_regimes:
            return self._call_fallback(
                x_array, 
                f"Invalid regime: {regime.name}", 
                regime.value
            )
        
        # Gate 2: CEM motion validation (if profile provided)
        if motion_profile is not None:
            report = self.cem.validate_motion(motion_profile)
            if not report.is_valid:
                return self._call_fallback(
                    x_array,
                    f"CEM validation failed: {[v.code.name for v in report.violations]}",
                    regime.value
                )
        
        # Gate 3: Surrogate prediction with uncertainty
        self.surrogate.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x_array, dtype=torch.float32, device=self.device)
            if x_tensor.dim() == 1:
                x_tensor = x_tensor.unsqueeze(0)
            
            mean, std = self.surrogate(x_tensor)
            mean = mean.cpu().numpy().squeeze()
            std = std.cpu().numpy().squeeze()
        
        # Gate 4: Uncertainty threshold
        if np.max(std) > self.uncertainty_threshold:
            result = self._call_fallback(
                x_array,
                f"High uncertainty: max_std={np.max(std):.4f} > {self.uncertainty_threshold}",
                regime.value
            )
            # Log this for retraining - high uncertainty should trigger data collection
            if self.collector:
                self._log_for_retraining(x_array, result.value, regime.value)
            return result
        
        # Success: use surrogate prediction
        self.n_surrogate_calls += 1
        
        return SurrogatePrediction(
            value=mean,
            uncertainty=std,
            used_surrogate=True,
            fallback_reason=None,
            regime_id=regime.value
        )
    
    def _call_fallback(
        self, 
        x: np.ndarray, 
        reason: str,
        regime_id: int
    ) -> SurrogatePrediction:
        """Call fallback evaluator and return result."""
        self.n_fallback_calls += 1
        
        try:
            value = np.asarray(self.fallback(x), dtype=np.float32)
        except Exception as e:
            # If fallback also fails, return NaN with error
            value = np.full(self.surrogate.models[0].net[-1].out_features, np.nan)
            reason = f"{reason} | Fallback failed: {e}"
        
        return SurrogatePrediction(
            value=value,
            uncertainty=np.zeros_like(value),
            used_surrogate=False,
            fallback_reason=reason,
            regime_id=regime_id
        )
    
    def _log_for_retraining(
        self,
        x: np.ndarray,
        y: np.ndarray,
        regime_id: int
    ) -> None:
        """Log high-uncertainty predictions for retraining."""
        if self.collector is None:
            return
        
        # Create minimal metadata
        metadata = {
            'margins': {},
            'constraint_codes': [],
            'is_valid': True,
            'regime_id': regime_id,
            'cem_version': self.cem.cem_version,
        }
        
        self.collector.log_evaluation(
            x_inputs={f'x_{i}': float(v) for i, v in enumerate(x)},
            y_low={f'y_{i}': float(v) for i, v in enumerate(y)},
            cem_metadata=metadata
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return inference statistics."""
        total = self.n_surrogate_calls + self.n_fallback_calls
        return {
            'total_calls': total,
            'surrogate_calls': self.n_surrogate_calls,
            'fallback_calls': self.n_fallback_calls,
            'surrogate_rate': self.n_surrogate_calls / max(total, 1),
        }
    
    def reset_statistics(self) -> None:
        """Reset call counters."""
        self.n_surrogate_calls = 0
        self.n_fallback_calls = 0
