"""
Trust Region Controller: Bounds solver steps based on surrogate uncertainty.

Prevents the solver from chasing surrogate hallucinations by restricting
step sizes in regions where uncertainty is high.

Usage:
    from campro.orchestration import TrustRegion

    tr = TrustRegion()
    step = tr.bound_step(proposed_step, uncertainty)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrustRegionConfig:
    """Configuration for trust region behavior."""

    # Base trust radius (fraction of variable range)
    initial_radius: float = 0.1

    # Minimum and maximum radius
    min_radius: float = 0.01
    max_radius: float = 0.5

    # Radius update factors
    expand_factor: float = 1.5  # Expand when prediction is good
    shrink_factor: float = 0.5  # Shrink when prediction is bad

    # Uncertainty scaling
    uncertainty_weight: float = 1.0  # How much uncertainty affects radius

    # Agreement threshold (predicted vs actual)
    good_agreement_threshold: float = 0.1  # < 10% error = good
    bad_agreement_threshold: float = 0.3  # > 30% error = bad


class TrustRegion:
    """
    Trust region controller for surrogate-based optimization.

    Key principles:
    - Restrict step sizes in high-uncertainty regions
    - Expand trust when surrogate agrees with truth
    - Shrink trust when surrogate disagrees
    - Conservative constraint tightening
    """

    def __init__(
        self,
        config: TrustRegionConfig | None = None,
        n_vars: int | None = None,
    ):
        """
        Initialize trust region.

        Args:
            config: Trust region configuration
            n_vars: Number of optimization variables (for per-variable radii)
        """
        self.config = config or TrustRegionConfig()
        self.n_vars = n_vars

        # Current trust radius (scalar or per-variable)
        if n_vars is not None:
            self._radius = np.ones(n_vars) * self.config.initial_radius
        else:
            self._radius = self.config.initial_radius

        # History for diagnostics
        self._history: list[dict[str, Any]] = []

    @property
    def radius(self) -> float | np.ndarray:
        """Get current trust radius."""
        return self._radius

    def bound_step(
        self,
        proposed_step: np.ndarray,
        uncertainty: np.ndarray | float,
        variable_scales: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Bound a proposed step to trust region.

        Args:
            proposed_step: Proposed change in variables
            uncertainty: Surrogate uncertainty (per-variable or scalar)
            variable_scales: Optional scale factors for variables

        Returns:
            Bounded step that respects trust region
        """
        if variable_scales is None:
            variable_scales = np.ones_like(proposed_step)

        # Compute effective radius (shrink in high-uncertainty regions)
        uncertainty = np.atleast_1d(uncertainty)
        if len(uncertainty) == 1:
            uncertainty = np.ones_like(proposed_step) * uncertainty[0]

        # Uncertainty factor: high uncertainty = smaller trust
        uncertainty_factor = 1.0 / (1.0 + self.config.uncertainty_weight * uncertainty)

        # Effective radius
        if isinstance(self._radius, np.ndarray):
            effective_radius = self._radius * uncertainty_factor * variable_scales
        else:
            effective_radius = self._radius * uncertainty_factor * variable_scales

        # Bound step
        step_magnitude = np.abs(proposed_step)
        scale = np.minimum(1.0, effective_radius / np.maximum(step_magnitude, 1e-10))
        bounded_step = proposed_step * scale

        # Log if significant clipping occurred
        if np.any(scale < 0.5):
            n_clipped = np.sum(scale < 0.5)
            log.debug(f"Trust region clipped {n_clipped} variables by >50%")

        return bounded_step

    def update(
        self,
        predicted_improvement: float,
        actual_improvement: float,
        uncertainty_at_step: float | np.ndarray,
    ) -> None:
        """
        Update trust region based on prediction accuracy.

        Args:
            predicted_improvement: Surrogate-predicted improvement
            actual_improvement: Actual improvement from truth evaluation
            uncertainty_at_step: Uncertainty at the step location
        """
        if predicted_improvement == 0:
            agreement = 0.0 if actual_improvement == 0 else 1.0
        else:
            agreement = abs(actual_improvement - predicted_improvement) / abs(predicted_improvement)

        # Record history
        self._history.append(
            {
                "predicted": predicted_improvement,
                "actual": actual_improvement,
                "agreement": agreement,
                "radius_before": float(np.mean(self._radius))
                if isinstance(self._radius, np.ndarray)
                else self._radius,
            }
        )

        # Update radius based on agreement
        if agreement < self.config.good_agreement_threshold:
            # Good agreement: expand trust
            self._expand()
            log.debug(f"Trust region expanded (agreement={agreement:.3f})")
        elif agreement > self.config.bad_agreement_threshold:
            # Bad agreement: shrink trust
            self._shrink()
            log.debug(f"Trust region shrunk (agreement={agreement:.3f})")
        # Otherwise: keep same

    def _expand(self) -> None:
        """Expand trust radius."""
        if isinstance(self._radius, np.ndarray):
            self._radius = np.minimum(
                self._radius * self.config.expand_factor, self.config.max_radius
            )
        else:
            self._radius = min(self._radius * self.config.expand_factor, self.config.max_radius)

    def _shrink(self) -> None:
        """Shrink trust radius."""
        if isinstance(self._radius, np.ndarray):
            self._radius = np.maximum(
                self._radius * self.config.shrink_factor, self.config.min_radius
            )
        else:
            self._radius = max(self._radius * self.config.shrink_factor, self.config.min_radius)

    def conservative_bounds(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        uncertainty: np.ndarray,
        safety_margin: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute conservative constraint bounds based on uncertainty.

        Tightens bounds in high-uncertainty regions to avoid constraint
        violations that the surrogate might miss.

        Args:
            lb: Lower bounds
            ub: Upper bounds
            uncertainty: Surrogate uncertainty
            safety_margin: Base safety margin fraction

        Returns:
            Tuple of (conservative_lb, conservative_ub)
        """
        # Tighten proportionally to uncertainty
        margin = safety_margin * (1.0 + uncertainty)
        range_size = ub - lb

        conservative_lb = lb + margin * range_size
        conservative_ub = ub - margin * range_size

        # Ensure still valid
        mid = (lb + ub) / 2
        conservative_lb = np.minimum(conservative_lb, mid - 1e-6)
        conservative_ub = np.maximum(conservative_ub, mid + 1e-6)

        return conservative_lb, conservative_ub

    def get_statistics(self) -> dict[str, Any]:
        """Get trust region statistics."""
        if not self._history:
            return {"n_updates": 0}

        agreements = [h["agreement"] for h in self._history]
        return {
            "n_updates": len(self._history),
            "mean_agreement": np.mean(agreements),
            "current_radius": float(np.mean(self._radius))
            if isinstance(self._radius, np.ndarray)
            else self._radius,
        }


__all__ = [
    "TrustRegion",
    "TrustRegionConfig",
]
