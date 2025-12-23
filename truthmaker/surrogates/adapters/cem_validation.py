"""
CEM Validation Surrogate Adapter.

Connects EnsembleSurrogate models to CEM validation pipeline. Provides
fast surrogate-based validation with uncertainty-aware rejection, falling
back to physics when uncertainty is too high.

Usage:
    from truthmaker.surrogates.adapters.cem_validation import CEMValidationAdapter

    adapter = CEMValidationAdapter.from_pretrained("thermo_validator")
    result = adapter.validate(x_profile, rpm=3000)

    if result.use_physics:
        # Surrogate uncertain, need full physics check
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)

# Try to import PyTorch and ensemble
try:
    import torch

    from truthmaker.surrogates.models.ensemble import EnsembleSurrogate

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.debug("PyTorch not available, surrogate adapters disabled")


class ValidationMode(Enum):
    """Validation execution mode."""

    SURROGATE = "surrogate"  # Fast ML-based validation
    PHYSICS = "physics"  # Full physics simulation
    HYBRID = "hybrid"  # Surrogate with physics fallback


@dataclass
class SurrogateValidationResult:
    """Result from surrogate-based validation."""

    is_valid: bool
    confidence: float  # 0-1, based on ensemble agreement
    predictions: dict[str, float] = field(default_factory=dict)
    use_physics: bool = False  # True if uncertainty too high
    uncertainty: float = 0.0
    mode_used: ValidationMode = ValidationMode.SURROGATE


class CEMValidationAdapter:
    """
    Adapter connecting EnsembleSurrogate to CEM validation.

    Provides:
    - Fast surrogate-based constraint checking
    - Uncertainty-aware rejection (fallback to physics)
    - Caching for repeated validation calls
    """

    # Default paths for pretrained models
    MODELS_DIR = Path(__file__).parent.parent / "models" / "model_artifacts"

    def __init__(
        self,
        surrogate: "EnsembleSurrogate | None" = None,
        uncertainty_threshold: float = 0.1,
        validation_mode: ValidationMode = ValidationMode.HYBRID,
    ):
        """
        Initialize adapter.

        Args:
            surrogate: Pretrained EnsembleSurrogate (optional)
            uncertainty_threshold: Reject surrogate if std > threshold
            validation_mode: Which validation approach to use
        """
        self.surrogate = surrogate
        self.uncertainty_threshold = uncertainty_threshold
        self.validation_mode = validation_mode
        self._cache: dict[int, SurrogateValidationResult] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        **kwargs,
    ) -> "CEMValidationAdapter":
        """
        Load adapter with pretrained surrogate.

        Args:
            model_name: Name of pretrained model (e.g., "thermo_validator")
            **kwargs: Additional arguments to pass to adapter

        Returns:
            Configured adapter with loaded model
        """
        if not TORCH_AVAILABLE:
            log.warning("PyTorch not available, returning adapter without surrogate")
            return cls(surrogate=None, **kwargs)

        model_path = cls.MODELS_DIR / f"{model_name}.pt"
        if not model_path.exists():
            log.warning(f"Pretrained model not found: {model_path}")
            return cls(surrogate=None, **kwargs)

        try:
            surrogate = EnsembleSurrogate.load(str(model_path))
            log.info(f"Loaded pretrained surrogate: {model_name}")
            return cls(surrogate=surrogate, **kwargs)
        except Exception as e:
            log.error(f"Failed to load surrogate: {e}")
            return cls(surrogate=None, **kwargs)

    def validate(
        self,
        x_profile: np.ndarray,
        rpm: float = 3000.0,
        p_intake_bar: float = 1.5,
        use_cache: bool = True,
    ) -> SurrogateValidationResult:
        """
        Validate motion profile using surrogate.

        Args:
            x_profile: Piston position trajectory [m]
            rpm: Engine speed [rev/min]
            p_intake_bar: Intake pressure [bar]
            use_cache: Whether to cache results

        Returns:
            SurrogateValidationResult with validity and confidence
        """
        # Check cache
        cache_key = hash((x_profile.tobytes(), rpm, p_intake_bar))
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # No surrogate - fallback to physics mode
        if self.surrogate is None or not TORCH_AVAILABLE:
            result = SurrogateValidationResult(
                is_valid=True,  # Assume valid, physics will check
                confidence=0.0,
                use_physics=True,
                mode_used=ValidationMode.PHYSICS,
            )
            return result

        # Prepare input features
        features = self._extract_features(x_profile, rpm, p_intake_bar)

        # Run surrogate inference
        with torch.no_grad():
            x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            mean, std = self.surrogate.forward(x_tensor)

        mean_np = mean.numpy()[0]
        std_np = std.numpy()[0]

        # Check uncertainty threshold
        max_uncertainty = float(std_np.max())
        use_physics = max_uncertainty > self.uncertainty_threshold

        # Interpret predictions
        # Assume output: [p_max, T_max, efficiency]
        predictions = {
            "p_max_predicted": float(mean_np[0]) if len(mean_np) > 0 else 0.0,
            "T_max_predicted": float(mean_np[1]) if len(mean_np) > 1 else 0.0,
            "efficiency_predicted": float(mean_np[2]) if len(mean_np) > 2 else 0.0,
        }

        # Simple validity check (could be more sophisticated)
        is_valid = True
        if predictions.get("p_max_predicted", 0) > 200e5:  # 200 bar
            is_valid = False
        if predictions.get("T_max_predicted", 0) > 2500:  # 2500 K
            is_valid = False

        confidence = 1.0 - min(1.0, max_uncertainty / self.uncertainty_threshold)

        result = SurrogateValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            predictions=predictions,
            use_physics=use_physics,
            uncertainty=max_uncertainty,
            mode_used=ValidationMode.HYBRID if use_physics else ValidationMode.SURROGATE,
        )

        if use_cache:
            self._cache[cache_key] = result

        return result

    def _extract_features(
        self,
        x_profile: np.ndarray,
        rpm: float,
        p_intake_bar: float,
    ) -> np.ndarray:
        """
        Extract input features for surrogate from raw data.

        Computes statistics that capture the motion profile shape.
        """
        # Basic statistics
        x_mean = np.mean(x_profile)
        x_std = np.std(x_profile)
        x_min = np.min(x_profile)
        x_max = np.max(x_profile)
        stroke = x_max - x_min

        # Velocity and acceleration proxies
        dx = np.diff(x_profile)
        v_max = np.max(np.abs(dx)) * rpm / 60 * len(x_profile)  # Approximate

        # Combine into feature vector
        # Input to standard thermo surrogate: [rpm, p_intake, stroke, v_max, ...]
        features = np.array(
            [
                rpm / 5000.0,  # Normalized RPM
                p_intake_bar / 3.0,  # Normalized pressure
                stroke / 0.2,  # Normalized stroke
            ]
        )

        return features

    def clear_cache(self) -> None:
        """Clear the validation cache."""
        self._cache.clear()


class ThermoSurrogateAdapter:
    """
    Adapter for thermodynamic property prediction.

    Wraps surrogate that predicts: [p_max, T_max, eta_thermal]
    from operating conditions and motion profile.
    """

    def __init__(self, surrogate: "EnsembleSurrogate | None" = None):
        self.surrogate = surrogate

    def predict(
        self,
        rpm: float,
        fuel_mass_kg: float,
        cr: float,
        p_intake_bar: float = 1.5,
    ) -> tuple[dict[str, float], float]:
        """
        Predict thermodynamic outputs.

        Args:
            rpm: Engine speed [rev/min]
            fuel_mass_kg: Fuel mass per cycle [kg]
            cr: Compression ratio
            p_intake_bar: Intake pressure [bar]

        Returns:
            Tuple of (predictions dict, uncertainty)
        """
        if self.surrogate is None or not TORCH_AVAILABLE:
            # Return default estimates
            return {
                "p_max": 80e5,  # 80 bar
                "T_max": 1800.0,  # 1800 K
                "eta_thermal": 0.42,
            }, 1.0

        features = np.array(
            [
                rpm / 5000.0,
                fuel_mass_kg / 1e-4,  # Normalized
                cr / 20.0,
                p_intake_bar / 3.0,
            ]
        )

        with torch.no_grad():
            x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            mean, std = self.surrogate.forward(x_tensor)

        predictions = {
            "p_max": float(mean[0, 0]) * 1e7,  # Rescale
            "T_max": float(mean[0, 1]) * 1000,
            "eta_thermal": float(mean[0, 2]),
        }

        return predictions, float(std.max())


__all__ = [
    "CEMValidationAdapter",
    "SurrogateValidationResult",
    "ThermoSurrogateAdapter",
    "ValidationMode",
    "TORCH_AVAILABLE",
]
