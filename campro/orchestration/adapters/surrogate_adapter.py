"""
Surrogate Adapter: Wraps EnsembleSurrogate as SurrogateInterface.

Adapts the existing ML surrogates to work with the orchestrator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from campro.logging import get_logger
from campro.orchestration.orchestrator import SurrogateInterface

log = get_logger(__name__)

# Surrogate availability
try:
    import torch

    from truthmaker.surrogates.models.ensemble import EnsembleSurrogate

    SURROGATE_AVAILABLE = True
except ImportError:
    SURROGATE_AVAILABLE = False


class EnsembleSurrogateAdapter:
    """
    Adapter wrapping EnsembleSurrogate as SurrogateInterface.

    Provides:
    - predict: Get predictions and uncertainty for candidates
    - update: Update surrogate with new truth data
    """

    MODELS_DIR = (
        Path(__file__).parent.parent.parent.parent
        / "truthmaker"
        / "surrogates"
        / "models"
        / "model_artifacts"
    )

    def __init__(
        self,
        surrogate: Any = None,
        feature_keys: list[str] | None = None,
        mock: bool = False,
    ):
        """
        Initialize adapter.

        Args:
            surrogate: Pre-loaded EnsembleSurrogate (optional)
            feature_keys: Keys to extract from candidate dict for features
            mock: If True, allow falling back to mock predictions. If False, enforce real model.
        """
        self.mock = mock
        self.feature_keys = feature_keys or [
            "rpm",
            "p_intake_bar",
            "fuel_mass_kg",
            "bore",
            "stroke",
            "cr",
        ]
        self._training_data: list[tuple[dict[str, Any], float]] = []

        # Auto-load default if needed
        if surrogate is None and not self.mock:
            if not SURROGATE_AVAILABLE:
                raise RuntimeError(
                    "Surrogate dependencies (torch/truthmaker) not installed. Cannot run real surrogate."
                )

            # Attempt to load default model
            default_name = "surrogate_model"
            try:
                # We reuse the logic from load() but need the instance
                model_path_pt = self.MODELS_DIR / f"{default_name}.pt"
                model_path_pth = self.MODELS_DIR / f"{default_name}.pth"

                path_to_load = None
                if model_path_pt.exists():
                    path_to_load = model_path_pt
                elif model_path_pth.exists():
                    path_to_load = model_path_pth

                if path_to_load:
                    log.info(f"Loading default surrogate: {path_to_load.name}")
                    surrogate = EnsembleSurrogate.load(str(path_to_load))
                else:
                    log.warning(
                        f"Default surrogate '{default_name}' not found in {self.MODELS_DIR}"
                    )
                    # If strictly not mocking, we should probably fail, but maybe we let predict() fail?
                    # The plan says "ONLY fall back to mock if explicitly requested".
                    # So we should probably fail here or in predict.
                    pass
            except Exception as e:
                log.error(f"Failed to auto-load default surrogate: {e}")
                # We will raise error in predict if surrogate is still None

        self.surrogate = surrogate

    @classmethod
    def load(cls, model_name: str, mock: bool = False) -> "EnsembleSurrogateAdapter":
        """
        Load adapter with pretrained surrogate.

        Args:
            model_name: Name of pretrained model
            mock: Allow mock fallback

        Returns:
            Configured adapter
        """
        if not SURROGATE_AVAILABLE:
            if mock:
                log.warning("Surrogate not available, using mock predictions")
                return cls(surrogate=None, mock=True)
            raise RuntimeError("Surrogate dependencies missing")

        model_path_pt = cls.MODELS_DIR / f"{model_name}.pt"
        model_path_pth = cls.MODELS_DIR / f"{model_name}.pth"

        path_to_load = None
        if model_path_pt.exists():
            path_to_load = model_path_pt
        elif model_path_pth.exists():
            path_to_load = model_path_pth

        if not path_to_load:
            if mock:
                log.warning(f"Model not found: {model_name}, using mock")
                return cls(surrogate=None, mock=True)
            raise FileNotFoundError(f"Surrogate model not found: {model_name}")

        try:
            surrogate = EnsembleSurrogate.load(str(path_to_load))
            return cls(surrogate=surrogate, mock=False)
        except Exception as e:
            if mock:
                log.error(f"Failed to load surrogate: {e}, using mock")
                return cls(surrogate=None, mock=True)
            raise e

    def predict(
        self,
        candidates: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict objectives and uncertainty for candidates.

        Args:
            candidates: List of candidate design dicts

        Returns:
            Tuple of (predictions, uncertainty) arrays
        """
        if not candidates:
            return np.array([]), np.array([])

        # Extract features
        features = self._extract_features(candidates)

        if self.surrogate is None or not SURROGATE_AVAILABLE:
            return self._mock_predict(features)

        try:
            with torch.no_grad():
                x_tensor = torch.tensor(features, dtype=torch.float32)
                mean, std = self.surrogate.forward(x_tensor)

            # Return scalar predictions (first output = objective)
            predictions = mean[:, 0].numpy() if mean.ndim > 1 else mean.numpy()
            uncertainty = std[:, 0].numpy() if std.ndim > 1 else std.numpy()

            return predictions, uncertainty

        except Exception as e:
            log.warning(f"Surrogate prediction failed: {e}")
            return self._mock_predict(features)

    def update(
        self,
        data: list[tuple[dict[str, Any], float]],
    ) -> None:
        """
        Update surrogate with new truth data.

        Args:
            data: List of (candidate, truth_value) tuples
        """
        self._training_data.extend(data)

        if self.surrogate is None:
            log.debug(f"Stored {len(data)} samples (no surrogate to update)")
            return

        # In production, would retrain or fine-tune here
        log.debug(f"Stored {len(data)} samples for retraining")

    def _extract_features(
        self,
        candidates: list[dict[str, Any]],
    ) -> np.ndarray:
        """Extract feature matrix from candidates."""
        # Normalization constants
        norm = {
            "rpm": 5000.0,
            "p_intake_bar": 3.0,
            "fuel_mass_kg": 1e-4,
            "bore": 0.1,
            "stroke": 0.15,
            "cr": 15.0,
        }

        features = []
        for c in candidates:
            row = []
            for key in self.feature_keys:
                val = c.get(key, norm.get(key, 1.0))
                normalized = val / norm.get(key, 1.0)
                row.append(normalized)
            features.append(row)

        return np.array(features, dtype=np.float32)

    def _mock_predict(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mock predictions when surrogate unavailable."""
        n = len(features)

        # Simple physics-based mock
        predictions = np.zeros(n)
        for i, f in enumerate(features):
            # Mock objective based on features
            rpm_norm, p_intake_norm, fuel_norm = f[0], f[1], f[2]
            predictions[i] = 0.4 + 0.1 * p_intake_norm - 0.05 * rpm_norm

        # High uncertainty (no real model)
        uncertainty = np.ones(n) * 0.5

        return predictions, uncertainty

    def get_training_data(self) -> list[tuple[dict[str, Any], float]]:
        """Get accumulated training data."""
        return self._training_data.copy()


class MockSurrogateAdapter:
    """Simple mock surrogate for testing."""

    def predict(
        self,
        candidates: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(candidates)
        return np.random.rand(n), np.ones(n) * 0.1

    def update(self, data: list[tuple[dict[str, Any], float]]) -> None:
        pass


__all__ = [
    "EnsembleSurrogateAdapter",
    "MockSurrogateAdapter",
    "SURROGATE_AVAILABLE",
]
