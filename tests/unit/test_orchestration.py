"""
Unit tests for orchestration module.
"""

import numpy as np
import pytest

from campro.orchestration import (
    BudgetAllocation,
    BudgetManager,
    EvaluationCache,
    OrchestrationConfig,
    TrustRegion,
    TrustRegionConfig,
)


class TestBudgetManager:
    """Tests for BudgetManager."""

    def test_create_budget(self):
        """Can create budget manager."""
        budget = BudgetManager(total_sim_calls=100)
        assert budget.remaining() == 90  # 10% reserved for validation

    def test_budget_exhaustion(self):
        """Budget tracks exhaustion correctly."""
        budget = BudgetManager(total_sim_calls=10, reserve_for_validation=0.0)

        assert not budget.exhausted()

        # Consume budget manually
        budget.state.used = budget.state.total

        assert budget.exhausted()

    def test_select_returns_indices(self):
        """Select returns valid indices."""
        budget = BudgetManager(total_sim_calls=100)

        candidates = list(range(20))
        predictions = np.random.rand(20)
        uncertainty = np.random.rand(20)

        selected = budget.select(candidates, predictions, uncertainty, batch_size=5)

        # May return fewer due to deduplication across strategies
        assert len(selected) <= 5
        assert len(selected) >= 1
        assert all(0 <= idx < 20 for idx in selected)

    def test_allocation_sums_to_one(self):
        """Allocation fractions are normalized."""
        alloc = BudgetAllocation(
            best_fraction=0.5,
            uncertain_fraction=0.5,
            disagreement_fraction=0.5,
            random_fraction=0.5,
        )

        total = (
            alloc.best_fraction
            + alloc.uncertain_fraction
            + alloc.disagreement_fraction
            + alloc.random_fraction
        )
        assert np.isclose(total, 1.0)


class TestTrustRegion:
    """Tests for TrustRegion."""

    def test_create_trust_region(self):
        """Can create trust region."""
        tr = TrustRegion()
        assert tr.radius == 0.1  # Default initial radius

    def test_bound_step(self):
        """Step bounding works."""
        tr = TrustRegion()

        proposed = np.array([1.0, 1.0, 1.0])
        uncertainty = np.array([0.0, 0.0, 0.0])  # Low uncertainty

        bounded = tr.bound_step(proposed, uncertainty)

        # Should be bounded to trust radius
        assert np.all(np.abs(bounded) <= tr.config.max_radius + 1e-6)

    def test_high_uncertainty_shrinks_step(self):
        """High uncertainty results in smaller steps."""
        tr = TrustRegion()

        proposed = np.array([0.5, 0.5, 0.5])

        low_unc = np.array([0.01, 0.01, 0.01])
        high_unc = np.array([1.0, 1.0, 1.0])

        step_low = tr.bound_step(proposed, low_unc)
        step_high = tr.bound_step(proposed, high_unc)

        # High uncertainty should give smaller steps
        assert np.linalg.norm(step_high) < np.linalg.norm(step_low)

    def test_update_expands_on_good_agreement(self):
        """Trust region expands when predictions agree with truth."""
        tr = TrustRegion()
        initial_radius = tr.radius

        tr.update(
            predicted_improvement=1.0,
            actual_improvement=1.05,  # 5% error - good
            uncertainty_at_step=0.1,
        )

        assert tr.radius > initial_radius

    def test_update_shrinks_on_bad_agreement(self):
        """Trust region shrinks when predictions disagree."""
        tr = TrustRegion()
        initial_radius = tr.radius

        tr.update(
            predicted_improvement=1.0,
            actual_improvement=0.5,  # 50% error - bad
            uncertainty_at_step=0.1,
        )

        assert tr.radius < initial_radius


class TestEvaluationCache:
    """Tests for EvaluationCache."""

    def test_create_cache(self):
        """Can create cache."""
        cache = EvaluationCache()
        assert cache.max_size == 10000

    def test_put_and_get(self):
        """Can store and retrieve values."""
        cache = EvaluationCache()

        params = {"x": 1.0, "y": 2.0}
        result = 42.0

        cache.put(params, result)

        retrieved = cache.get(params)
        assert retrieved == result

    def test_cache_miss(self):
        """Returns None on cache miss."""
        cache = EvaluationCache()

        result = cache.get({"x": 999.0})
        assert result is None

    def test_get_or_compute(self):
        """get_or_compute works correctly."""
        cache = EvaluationCache()

        call_count = 0

        def compute_fn(params):
            nonlocal call_count
            call_count += 1
            return params["x"] * 2

        params = {"x": 5.0}

        # First call - computes
        result1, was_cached1 = cache.get_or_compute(params, compute_fn)
        assert result1 == 10.0
        assert was_cached1 is False
        assert call_count == 1

        # Second call - cached
        result2, was_cached2 = cache.get_or_compute(params, compute_fn)
        assert result2 == 10.0
        assert was_cached2 is True
        assert call_count == 1  # Not called again

    def test_numpy_array_hashing(self):
        """Can hash numpy arrays in params."""
        cache = EvaluationCache()

        params = {"arr": np.array([1.0, 2.0, 3.0])}
        cache.put(params, 100.0)

        # Same array values should hit cache
        params2 = {"arr": np.array([1.0, 2.0, 3.0])}
        result = cache.get(params2)

        assert result == 100.0

    def test_lru_eviction(self):
        """LRU eviction works."""
        cache = EvaluationCache(max_size=3)

        cache.put({"x": 1}, 1)
        cache.put({"x": 2}, 2)
        cache.put({"x": 3}, 3)
        cache.put({"x": 4}, 4)  # Should evict x=1

        assert cache.get({"x": 1}) is None
        assert cache.get({"x": 4}) == 4


class TestOrchestrationConfig:
    """Tests for OrchestrationConfig."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = OrchestrationConfig()

        assert config.total_sim_budget == 1000
        assert config.batch_size == 50
        assert config.max_iterations == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
