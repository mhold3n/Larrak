"""
Unit tests for CEM rule framework.

Tests rule definitions, registry, and category filtering.
"""

import pytest

from truthmaker.cem.rules import (
    MaxContactStressRule,
    MaxCrownTemperatureRule,
    MinFilmThicknessRule,
    NOxEmissionRule,
    RuleBase,
    RuleCategory,
    RuleResult,
    RuleSeverity,
)
from truthmaker.cem.rules.registry import (
    RULE_REGISTRY,
    clear_registry,
    get_active_rules,
    get_rules_by_category,
    list_rules,
    register_rule,
)


class TestRuleCategory:
    """Tests for RuleCategory enum."""

    def test_all_categories_defined(self):
        """All expected categories exist."""
        assert RuleCategory.THERMODYNAMIC.value == "thermo"
        assert RuleCategory.MECHANICAL.value == "mech"
        assert RuleCategory.TRIBOLOGICAL.value == "tribo"
        assert RuleCategory.MANUFACTURING.value == "mfg"
        assert RuleCategory.EMISSIONS.value == "emissions"
        assert RuleCategory.ACOUSTIC.value == "acoustic"


class TestRuleSeverity:
    """Tests for RuleSeverity enum."""

    def test_severity_ordering(self):
        """Severities have correct relative values."""
        assert RuleSeverity.INFO.value < RuleSeverity.WARN.value
        assert RuleSeverity.WARN.value < RuleSeverity.ERROR.value
        assert RuleSeverity.ERROR.value < RuleSeverity.FATAL.value


class TestRuleResult:
    """Tests for RuleResult dataclass."""

    def test_passed_result(self):
        """Create a passing rule result."""
        result = RuleResult(
            rule_name="test_rule",
            passed=True,
            margin=10.0,
            message="All good",
        )
        assert result.passed is True
        assert result.margin == 10.0

    def test_failed_result(self):
        """Create a failing rule result."""
        result = RuleResult(
            rule_name="test_rule",
            passed=False,
            margin=-5.0,
            severity=RuleSeverity.ERROR,
        )
        assert result.passed is False
        assert result.margin < 0


class TestRuleStubs:
    """Tests for rule stub implementations."""

    def test_max_crown_temp_not_implemented(self):
        """MaxCrownTemperatureRule raises NotImplementedError."""
        rule = MaxCrownTemperatureRule(limit_k=600.0)
        assert rule.category == RuleCategory.THERMODYNAMIC
        assert rule.name == "max_crown_temperature"
        with pytest.raises(NotImplementedError):
            rule.evaluate({})

    def test_max_contact_stress_not_implemented(self):
        """MaxContactStressRule raises NotImplementedError."""
        rule = MaxContactStressRule(limit_mpa=1200.0)
        assert rule.category == RuleCategory.MECHANICAL
        with pytest.raises(NotImplementedError):
            rule.evaluate({})

    def test_min_film_thickness_not_implemented(self):
        """MinFilmThicknessRule raises NotImplementedError."""
        rule = MinFilmThicknessRule(min_lambda=2.0)
        assert rule.category == RuleCategory.TRIBOLOGICAL
        with pytest.raises(NotImplementedError):
            rule.evaluate({})

    def test_nox_emission_not_implemented(self):
        """NOxEmissionRule raises NotImplementedError."""
        rule = NOxEmissionRule(limit_g_kwh=0.5)
        assert rule.category == RuleCategory.EMISSIONS
        with pytest.raises(NotImplementedError):
            rule.evaluate({})


class TestRuleRegistry:
    """Tests for the rule registry system."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        clear_registry()

    def test_register_rule_decorator(self):
        """Register a rule using decorator."""

        @register_rule("test_rule")
        class TestRule(RuleBase):
            category = RuleCategory.THERMODYNAMIC
            name = "test"

            def evaluate(self, context):
                return RuleResult(rule_name="test", passed=True)

        assert "test_rule" in RULE_REGISTRY
        assert RULE_REGISTRY["test_rule"] is TestRule

    def test_list_rules(self):
        """List registered rule names."""

        @register_rule("rule_a")
        class RuleA(RuleBase):
            category = RuleCategory.MECHANICAL
            name = "a"

            def evaluate(self, context):
                return RuleResult(rule_name="a", passed=True)

        @register_rule("rule_b")
        class RuleB(RuleBase):
            category = RuleCategory.MECHANICAL
            name = "b"

            def evaluate(self, context):
                return RuleResult(rule_name="b", passed=True)

        names = list_rules()
        assert "rule_a" in names
        assert "rule_b" in names

    def test_get_rules_by_category(self):
        """Filter rules by category."""

        @register_rule("thermo_rule")
        class ThermoRule(RuleBase):
            category = RuleCategory.THERMODYNAMIC
            name = "thermo"

            def evaluate(self, context):
                return RuleResult(rule_name="thermo", passed=True)

        @register_rule("mech_rule")
        class MechRule(RuleBase):
            category = RuleCategory.MECHANICAL
            name = "mech"

            def evaluate(self, context):
                return RuleResult(rule_name="mech", passed=True)

        thermo_rules = get_rules_by_category(RuleCategory.THERMODYNAMIC)
        assert len(thermo_rules) == 1
        assert thermo_rules[0].category == RuleCategory.THERMODYNAMIC

    def test_duplicate_registration_raises(self):
        """Registering same name twice raises error."""

        @register_rule("dup_rule")
        class Rule1(RuleBase):
            category = RuleCategory.ACOUSTIC
            name = "dup"

            def evaluate(self, context):
                return RuleResult(rule_name="dup", passed=True)

        with pytest.raises(ValueError, match="already registered"):

            @register_rule("dup_rule")
            class Rule2(RuleBase):
                category = RuleCategory.ACOUSTIC
                name = "dup2"

                def evaluate(self, context):
                    return RuleResult(rule_name="dup2", passed=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# ADAPTIVE RULE TESTS
# =============================================================================


class TestAdaptiveRuleBase:
    """Tests for the adaptive rule framework."""

    def test_initial_state(self):
        """Adaptive rule initializes with correct state."""
        from truthmaker.cem.rules.adaptation import MaxCrownTemperatureAdaptive

        rule = MaxCrownTemperatureAdaptive(limit_k=600.0)
        assert rule.limit == 600.0
        assert rule.initial_limit == 600.0
        assert rule._state.n_observations == 0

    def test_evaluate_within_limit(self):
        """Evaluate passes when value is within limit."""
        from truthmaker.cem.rules.adaptation import MaxCrownTemperatureAdaptive

        rule = MaxCrownTemperatureAdaptive(limit_k=600.0)
        result = rule.evaluate({"T_crown_max": 500.0})
        assert result.passed is True
        assert result.margin > 0

    def test_evaluate_exceeds_limit(self):
        """Evaluate fails when value exceeds limit."""
        from truthmaker.cem.rules.adaptation import MaxCrownTemperatureAdaptive

        rule = MaxCrownTemperatureAdaptive(limit_k=600.0)
        result = rule.evaluate({"T_crown_max": 650.0})
        assert result.passed is False
        assert result.margin < 0

    def test_adapt_tightens_with_headroom(self):
        """Rule tightens limit when consistent safety headroom."""
        from truthmaker.cem.rules.adaptation import MaxCrownTemperatureAdaptive

        rule = MaxCrownTemperatureAdaptive(limit_k=600.0, min_observations=5, learning_rate=0.1)

        # Simulate HiFi results with 25% margin (well below 600K limit)
        for _ in range(10):
            delta = rule.adapt({"T_crown_max": 450.0}, predicted=460.0)

        # Should have tightened (reduced limit)
        assert rule.limit < 600.0
        assert rule._state.total_delta < 0

    def test_adapt_relaxes_when_tight(self):
        """Rule relaxes limit when consistently too tight."""
        from truthmaker.cem.rules.adaptation import MaxCrownTemperatureAdaptive

        rule = MaxCrownTemperatureAdaptive(limit_k=600.0, min_observations=5, learning_rate=0.1)

        # Simulate HiFi results with only 3% margin
        for _ in range(10):
            delta = rule.adapt({"T_crown_max": 582.0}, predicted=580.0)

        # Should have relaxed (increased limit)
        assert rule.limit > 600.0
        assert rule._state.total_delta > 0

    def test_no_adapt_before_min_observations(self):
        """Rule does not adapt before minimum observations."""
        from truthmaker.cem.rules.adaptation import MaxCrownTemperatureAdaptive

        rule = MaxCrownTemperatureAdaptive(limit_k=600.0, min_observations=10)

        # Only 5 observations
        for _ in range(5):
            delta = rule.adapt({"T_crown_max": 450.0}, predicted=460.0)

        # Should not have adapted yet
        assert rule.limit == 600.0
        assert rule._state.total_delta == 0

    def test_state_persistence(self):
        """Adaptive state survives save/load cycle."""
        from truthmaker.cem.rules.adaptation import MaxCrownTemperatureAdaptive

        rule = MaxCrownTemperatureAdaptive(limit_k=600.0, min_observations=5)

        # Train the rule
        for _ in range(10):
            rule.adapt({"T_crown_max": 450.0}, predicted=460.0)

        # Get state and create new rule
        state = rule.get_state()
        new_rule = MaxCrownTemperatureAdaptive(limit_k=600.0)
        new_rule.load_state(state)

        # State should match
        assert new_rule.limit == rule.limit
        assert len(new_rule._state.margin_history) == len(rule._state.margin_history)
        assert new_rule._state.n_observations == rule._state.n_observations

    def test_reset_to_initial(self):
        """Rule can be reset to initial state."""
        from truthmaker.cem.rules.adaptation import MaxCrownTemperatureAdaptive

        rule = MaxCrownTemperatureAdaptive(limit_k=600.0, min_observations=5)

        # Train then reset
        for _ in range(10):
            rule.adapt({"T_crown_max": 450.0}, predicted=460.0)

        rule.reset_to_initial()

        assert rule.limit == 600.0
        assert rule._state.n_observations == 0
        assert len(rule._state.margin_history) == 0


class TestAdaptiveRuleState:
    """Tests for AdaptiveRuleState serialization."""

    def test_to_dict(self):
        """State serializes to dict correctly."""
        from datetime import datetime

        from truthmaker.cem.rules.adaptation import AdaptiveRuleState

        state = AdaptiveRuleState(
            limit=550.0,
            initial_limit=600.0,
            n_observations=15,
            margin_history=[0.1, 0.15, 0.2],
            last_adapted=datetime.now(),
            total_delta=-50.0,
        )

        data = state.to_dict()
        assert data["limit"] == 550.0
        assert data["initial_limit"] == 600.0
        assert data["n_observations"] == 15
        assert data["total_delta"] == -50.0

    def test_from_dict(self):
        """State deserializes from dict correctly."""
        from truthmaker.cem.rules.adaptation import AdaptiveRuleState

        data = {
            "limit": 550.0,
            "initial_limit": 600.0,
            "n_observations": 15,
            "margin_history": [0.1, 0.15, 0.2],
            "last_adapted": "2024-12-26T18:00:00",
            "total_delta": -50.0,
            "regime_states": {},
        }

        state = AdaptiveRuleState.from_dict(data)
        assert state.limit == 550.0
        assert state.n_observations == 15


class TestAdaptationReport:
    """Tests for AdaptationReport."""

    def test_any_adapted_false(self):
        """Report shows no adaptation when empty."""
        from truthmaker.cem.rules.adaptation import AdaptationReport

        report = AdaptationReport(adapted_rules=[])
        assert report.any_adapted is False
        assert report.total_rules_adapted == 0

    def test_any_adapted_true(self):
        """Report shows adaptation when rules changed."""
        from truthmaker.cem.rules.adaptation import AdaptationReport

        report = AdaptationReport(adapted_rules=[("rule_a", -5.0), ("rule_b", 3.0)])
        assert report.any_adapted is True
        assert report.total_rules_adapted == 2

    def test_summary(self):
        """Report generates readable summary."""
        from truthmaker.cem.rules.adaptation import AdaptationReport

        report = AdaptationReport(adapted_rules=[("max_temp", -5.0)])
        summary = report.summary()
        assert "max_temp" in summary
        assert "tightened" in summary
