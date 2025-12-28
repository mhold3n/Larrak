"""
CEM State Store: Persistence layer for adaptive rule parameters.

Persists learned CEM constraint limits between optimization runs,
enabling incremental learning across sessions.

Supports JSON file-based storage with optional Weaviate integration
for provenance tracking displayed in the orchestrator dashboard.

Usage:
    from truthmaker.cem.state_store import CEMStateStore

    store = CEMStateStore("cem_state.json")

    # Save rule state
    store.save_rule_state("max_crown_temperature", rule.get_state())

    # Load on next run
    state = store.load_rule_state("max_crown_temperature")
    if state:
        rule.load_state(state)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .rules.adaptation import AdaptiveRuleState

log = logging.getLogger(__name__)


class CEMStateStore:
    """
    JSON-backed store for adaptive CEM parameters.

    State structure:
    {
        "version": 1,
        "last_modified": "2024-12-26T18:45:00",
        "rules": {
            "max_crown_temperature_adaptive": { ... state dict ... },
            "max_contact_stress_adaptive": { ... state dict ... }
        },
        "adaptation_history": [
            {"timestamp": "...", "rule": "...", "delta": 0.5, ...},
            ...
        ]
    }
    """

    VERSION = 1
    MAX_HISTORY_ENTRIES = 500  # Rolling window for history

    def __init__(self, path: str | Path = "cem_state.json"):
        """
        Initialize state store.

        Args:
            path: Path to JSON state file (created if not exists)
        """
        self.path = Path(path)
        self._data: dict[str, Any] = self._load_or_init()

    def _load_or_init(self) -> dict[str, Any]:
        """Load existing state or initialize empty."""
        if self.path.exists():
            try:
                with open(self.path) as f:
                    data = json.load(f)
                log.info(f"Loaded CEM state from {self.path}")
                return data
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"Failed to load CEM state, reinitializing: {e}")

        return {
            "version": self.VERSION,
            "last_modified": datetime.now().isoformat(),
            "rules": {},
            "adaptation_history": [],
        }

    def _save(self) -> None:
        """Persist current state to disk."""
        self._data["last_modified"] = datetime.now().isoformat()

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def save_rule_state(self, rule_name: str, state: "AdaptiveRuleState") -> None:
        """
        Persist a rule's learned state.

        Args:
            rule_name: Unique rule identifier
            state: AdaptiveRuleState to persist
        """
        self._data["rules"][rule_name] = state.to_dict()
        self._save()
        log.debug(f"Saved state for {rule_name}")

    def load_rule_state(self, rule_name: str) -> Optional["AdaptiveRuleState"]:
        """
        Load a rule's previously learned state.

        Args:
            rule_name: Unique rule identifier

        Returns:
            AdaptiveRuleState if found, None otherwise
        """
        from .rules.adaptation import AdaptiveRuleState

        state_dict = self._data["rules"].get(rule_name)
        if state_dict is None:
            return None

        return AdaptiveRuleState.from_dict(state_dict)

    def save_all_states(self, rules: dict[str, "AdaptiveRuleState"]) -> None:
        """
        Batch save all rule states.

        Args:
            rules: Dict mapping rule_name to AdaptiveRuleState
        """
        for name, state in rules.items():
            self._data["rules"][name] = state.to_dict()
        self._save()
        log.info(f"Saved {len(rules)} rule states")

    def load_all_states(self) -> dict[str, "AdaptiveRuleState"]:
        """
        Load all persisted rule states.

        Returns:
            Dict mapping rule_name to AdaptiveRuleState
        """
        from .rules.adaptation import AdaptiveRuleState

        return {
            name: AdaptiveRuleState.from_dict(data) for name, data in self._data["rules"].items()
        }

    def log_adaptation(
        self,
        rule_name: str,
        rule_category: str,
        limit_before: float,
        limit_after: float,
        delta: float,
        direction: str,
        trigger_margin: float,
        n_observations: int,
        run_id: Optional[str] = None,
        regime_id: int = 0,
    ) -> None:
        """
        Log an adaptation event to history.

        This is also used to sync with Weaviate for dashboard display.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "rule_name": rule_name,
            "rule_category": rule_category,
            "limit_before": limit_before,
            "limit_after": limit_after,
            "delta": delta,
            "direction": direction,
            "trigger_margin": trigger_margin,
            "n_observations": n_observations,
            "run_id": run_id,
            "regime_id": regime_id,
        }

        self._data["adaptation_history"].append(entry)

        # Trim history to rolling window
        if len(self._data["adaptation_history"]) > self.MAX_HISTORY_ENTRIES:
            self._data["adaptation_history"] = self._data["adaptation_history"][
                -self.MAX_HISTORY_ENTRIES :
            ]

        self._save()
        log.info(f"Logged adaptation: {rule_name} {direction} by {abs(delta):.4f}")

        # Sync to Weaviate if available
        self._sync_to_weaviate(entry)

    def _sync_to_weaviate(self, entry: dict[str, Any]) -> None:
        """
        Sync adaptation event to Weaviate for dashboard tracking.

        Silently skips if Weaviate is not available.
        """
        try:
            from provenance.db import ProvenanceDB

            db = ProvenanceDB()
            if not db.is_connected():
                return

            db.insert_cem_adaptation(
                rule_name=entry["rule_name"],
                rule_category=entry["rule_category"],
                limit_before=entry["limit_before"],
                limit_after=entry["limit_after"],
                delta=entry["delta"],
                direction=entry["direction"],
                trigger_margin=entry["trigger_margin"],
                n_observations=entry["n_observations"],
                run_id=entry.get("run_id"),
                regime_id=entry.get("regime_id", 0),
            )
        except ImportError:
            pass  # Weaviate not available
        except Exception as e:
            log.debug(f"Weaviate sync failed (non-fatal): {e}")

    def get_adaptation_history(self, rule_name: Optional[str] = None) -> list[dict]:
        """
        Return audit log of adaptations.

        Args:
            rule_name: Filter to specific rule (None = all)

        Returns:
            List of adaptation event dicts
        """
        history = self._data["adaptation_history"]
        if rule_name:
            history = [e for e in history if e["rule_name"] == rule_name]
        return history

    def get_statistics(self) -> dict[str, Any]:
        """Get summary statistics for all tracked rules."""
        stats = {}
        for name, state_dict in self._data["rules"].items():
            initial = state_dict.get("initial_limit", 0)
            current = state_dict.get("limit", 0)
            stats[name] = {
                "initial_limit": initial,
                "current_limit": current,
                "total_delta": state_dict.get("total_delta", 0),
                "n_observations": state_dict.get("n_observations", 0),
                "change_pct": ((current - initial) / initial * 100) if initial else 0,
            }
        return stats

    def reset_rule(self, rule_name: str) -> bool:
        """
        Remove a rule's persisted state (next load will use initial).

        Returns:
            True if rule was found and removed
        """
        if rule_name in self._data["rules"]:
            del self._data["rules"][rule_name]
            self._save()
            log.info(f"Reset persisted state for {rule_name}")
            return True
        return False

    def reset_all(self) -> None:
        """Clear all persisted state (fresh start)."""
        self._data["rules"].clear()
        self._data["adaptation_history"].clear()
        self._save()
        log.warning("Reset all CEM state")


__all__ = ["CEMStateStore"]
