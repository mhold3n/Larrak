"""
CEM Rule Registry: Dynamic registration and lookup of validation rules.

This module provides a plugin-style registry for validation rules. Rules can
be registered at module load time using decorators, then queried by category
during optimization.

Usage:
    from truthmaker.cem.rules.registry import register_rule, get_active_rules

    @register_rule("my_custom_rule")
    class MyCustomRule(RuleBase):
        ...

    # Get all thermodynamic rules
    thermo_rules = get_active_rules([RuleCategory.THERMODYNAMIC])
"""

from __future__ import annotations

from typing import Callable, Type

from . import RuleBase, RuleCategory

# Global registry mapping rule names to rule classes
RULE_REGISTRY: dict[str, Type[RuleBase]] = {}

# Track which rules are enabled by default
DEFAULT_ENABLED: set[str] = set()


def register_rule(
    name: str, enabled_by_default: bool = True
) -> Callable[[Type[RuleBase]], Type[RuleBase]]:
    """
    Decorator to register a rule class in the global registry.

    Args:
        name: Unique identifier for this rule
        enabled_by_default: Whether rule is active without explicit enable

    Returns:
        Decorator function

    Example:
        @register_rule("max_pressure", enabled_by_default=True)
        class MaxPressureRule(RuleBase):
            ...
    """

    def decorator(cls: Type[RuleBase]) -> Type[RuleBase]:
        if name in RULE_REGISTRY:
            raise ValueError(f"Rule '{name}' already registered")
        RULE_REGISTRY[name] = cls
        if enabled_by_default:
            DEFAULT_ENABLED.add(name)
        return cls

    return decorator


def get_rule(name: str) -> Type[RuleBase] | None:
    """Get a rule class by name."""
    return RULE_REGISTRY.get(name)


def list_rules() -> list[str]:
    """List all registered rule names."""
    return list(RULE_REGISTRY.keys())


def get_rules_by_category(category: RuleCategory) -> list[Type[RuleBase]]:
    """Get all rule classes in a given category."""
    return [
        cls
        for cls in RULE_REGISTRY.values()
        if hasattr(cls, "category") and cls.category == category
    ]


def get_active_rules(
    categories: list[RuleCategory] | None = None, include_disabled: bool = False
) -> list[RuleBase]:
    """
    Get instantiated rules filtered by category.

    Args:
        categories: Filter to these categories (None = all)
        include_disabled: Include rules not enabled by default

    Returns:
        List of instantiated rule objects
    """
    rules = []
    for name, cls in RULE_REGISTRY.items():
        # Skip disabled rules unless requested
        if not include_disabled and name not in DEFAULT_ENABLED:
            continue

        # Filter by category if specified
        if categories is not None:
            if not hasattr(cls, "category") or cls.category not in categories:
                continue

        # Instantiate with defaults
        try:
            rules.append(cls())
        except TypeError:
            # Rule requires arguments, skip for now
            pass

    return rules


def enable_rule(name: str) -> None:
    """Enable a rule by name."""
    if name not in RULE_REGISTRY:
        raise ValueError(f"Unknown rule: {name}")
    DEFAULT_ENABLED.add(name)


def disable_rule(name: str) -> None:
    """Disable a rule by name."""
    DEFAULT_ENABLED.discard(name)


def clear_registry() -> None:
    """Clear all registered rules (mainly for testing)."""
    RULE_REGISTRY.clear()
    DEFAULT_ENABLED.clear()


__all__ = [
    "RULE_REGISTRY",
    "DEFAULT_ENABLED",
    "register_rule",
    "get_rule",
    "list_rules",
    "get_rules_by_category",
    "get_active_rules",
    "enable_rule",
    "disable_rule",
    "clear_registry",
]
