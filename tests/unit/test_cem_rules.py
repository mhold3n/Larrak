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
