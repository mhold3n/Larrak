"""
Unit tests for CEM surrogate adapters.
"""

import numpy as np
import pytest

from truthmaker.surrogates.adapters.cem_validation import (
    TORCH_AVAILABLE,
    CEMValidationAdapter,
    SurrogateValidationResult,
    ThermoSurrogateAdapter,
    ValidationMode,
)


class TestCEMValidationAdapter:
    """Tests for CEMValidationAdapter."""

    def test_create_adapter_without_surrogate(self):
        """Can create adapter without pretrained model."""
        adapter = CEMValidationAdapter()
        assert adapter.surrogate is None
        assert adapter.validation_mode == ValidationMode.HYBRID

    def test_validate_without_surrogate(self):
        """Validation works without surrogate (physics fallback)."""
        adapter = CEMValidationAdapter()
        x_profile = np.sin(np.linspace(0, 2 * np.pi, 100)) * 0.05 + 0.1

        result = adapter.validate(x_profile)

        assert isinstance(result, SurrogateValidationResult)
        assert result.use_physics is True
        assert result.mode_used == ValidationMode.PHYSICS

    def test_validate_returns_proper_result(self):
        """Validation returns all expected fields."""
        adapter = CEMValidationAdapter()
        x_profile = np.linspace(0.02, 0.18, 50)

        result = adapter.validate(x_profile, rpm=3000, p_intake_bar=2.0)

        assert hasattr(result, "is_valid")
        assert hasattr(result, "confidence")
        assert hasattr(result, "predictions")
        assert hasattr(result, "uncertainty")

    def test_cache_works(self):
        """Caching prevents redundant computation (when surrogate exists)."""
        adapter = CEMValidationAdapter()
        x_profile = np.ones(50) * 0.1

        result1 = adapter.validate(x_profile, use_cache=True)
        result2 = adapter.validate(x_profile, use_cache=True)

        # Without surrogate, physics fallback returns fresh objects
        # Just check they have same values
        assert result1.is_valid == result2.is_valid
        assert result1.mode_used == result2.mode_used

    def test_clear_cache(self):
        """Cache can be cleared."""
        adapter = CEMValidationAdapter()

        # Add something to cache manually to test clear
        adapter._cache[12345] = SurrogateValidationResult(is_valid=True, confidence=1.0)
        assert len(adapter._cache) > 0

        adapter.clear_cache()
        assert len(adapter._cache) == 0

    def test_from_pretrained_missing_model(self):
        """from_pretrained handles missing models gracefully."""
        adapter = CEMValidationAdapter.from_pretrained("nonexistent_model")
        assert adapter.surrogate is None


class TestThermoSurrogateAdapter:
    """Tests for ThermoSurrogateAdapter."""

    def test_create_adapter(self):
        """Can create thermo adapter."""
        adapter = ThermoSurrogateAdapter()
        assert adapter.surrogate is None

    def test_predict_without_surrogate(self):
        """Prediction returns defaults without surrogate."""
        adapter = ThermoSurrogateAdapter()

        predictions, uncertainty = adapter.predict(
            rpm=3000,
            fuel_mass_kg=5e-5,
            cr=15.0,
        )

        assert "p_max" in predictions
        assert "T_max" in predictions
        assert "eta_thermal" in predictions
        assert uncertainty == 1.0  # Max uncertainty


class TestValidationMode:
    """Tests for ValidationMode enum."""

    def test_mode_values(self):
        """ValidationMode has expected values."""
        assert ValidationMode.SURROGATE.value == "surrogate"
        assert ValidationMode.PHYSICS.value == "physics"
        assert ValidationMode.HYBRID.value == "hybrid"


class TestSurrogateValidationResult:
    """Tests for SurrogateValidationResult dataclass."""

    def test_result_defaults(self):
        """Result has expected defaults."""
        result = SurrogateValidationResult(is_valid=True, confidence=0.9)

        assert result.use_physics is False
        assert result.uncertainty == 0.0
        assert result.mode_used == ValidationMode.SURROGATE
        assert result.predictions == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
