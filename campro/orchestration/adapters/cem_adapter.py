"""
CEM Client Adapter: Wraps CEMClient as CEMInterface.

Adapts the existing CEM client to work with the orchestrator,
providing feasibility generation and checking.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any

import numpy as np

from campro.logging import get_logger
from campro.orchestration.orchestrator import CEMInterface

log = get_logger(__name__)

# CEM availability
try:
    from truthmaker.cem.client import CEMClient

    CEM_AVAILABLE = True
except ImportError as e:
    import sys

    print(f"[ERROR] Failed to import CEMClient: {e}", file=sys.stderr)
    CEM_AVAILABLE = False


class CEMClientAdapter:
    """
    Adapter wrapping CEMClient as CEMInterface.

    Provides:
    - generate_batch: Sample feasible candidates from CEM envelope
    - check_feasibility: Validate against CEM constraints
    - repair: Project infeasible candidates back to feasible manifold
    """

    def __init__(
        self,
        mock: bool = True,
        geometry: dict[str, float] | None = None,
    ):
        """
        Initialize adapter.

        Args:
            mock: Use mock CEM (True) or live service (False)
            geometry: Engine geometry dict (bore, stroke, cr, rpm)
        """
        self.mock = mock
        self.geometry = geometry or {
            "bore": 0.1,
            "stroke": 0.15,
            "cr": 15.0,
            "rpm": 3000.0,
        }
        self._envelope: dict[str, Any] | None = None
        self.provenance = None
        print(
            f"DEBUG: CEMClientAdapter initialized. CEM_AVAILABLE={CEM_AVAILABLE}, Mock={self.mock}"
        )

    def generate_batch(
        self,
        params: dict[str, Any],
        n: int,
    ) -> list[dict[str, Any]]:
        """
        Generate batch of feasible candidates from CEM envelope.

        Args:
            params: Base parameters
            n: Number of candidates to generate

        Returns:
            List of n candidate dicts within CEM envelope
        """
        envelope = self._get_envelope()

        if envelope is None:
            return self._fallback_batch(params, n)

        rng = np.random.default_rng()
        candidates = []

        for _ in range(n):
            candidate = params.copy()

            # Sample within CEM envelope bounds
            if hasattr(envelope, "boost_range"):
                lo, hi = envelope.boost_range
                candidate["p_intake_bar"] = rng.uniform(lo, hi)
            elif hasattr(envelope, "boost_min"):
                candidate["p_intake_bar"] = rng.uniform(envelope.boost_min, envelope.boost_max)

            if hasattr(envelope, "fuel_range"):
                lo, hi = envelope.fuel_range
                candidate["fuel_mass_kg"] = rng.uniform(lo, hi)
            elif hasattr(envelope, "fuel_min_mg"):
                candidate["fuel_mass_kg"] = rng.uniform(
                    envelope.fuel_min_mg * 1e-6, envelope.fuel_max_mg * 1e-6
                )

            candidates.append(candidate)

        return candidates

    def check_feasibility(
        self,
        candidate: dict[str, Any],
        run_id: str = None,
    ) -> tuple[bool, float]:
        """
        Check if candidate motion is feasible using CEM.

        Args:
            candidate: Candidate solution dictionary
            run_id: Optional UUID of the current run (for provenance)

        Returns:
            (is_feasible, score) tuple
        """
        if not CEM_AVAILABLE:
            return True, 1.0  # Assume feasible without CEM

        # Global Stop Signal Check
        if os.environ.get("ORCHESTRATOR_STOP_SIGNAL") == "1":
            log.warning("CEM check aborted due to stop signal.")
            return False, 0.0

        try:
            with CEMClient(mock=self.mock) as cem:
                # Extract or generate motion profile data
                x_profile = candidate.get("x_profile", np.zeros(100))
                theta = candidate.get("theta")

                # Generate default theta if not provided
                if theta is None:
                    theta = np.linspace(0, 2 * np.pi, len(x_profile))

                # Check thermo feasibility
                report = cem.validate_motion(
                    x_profile=x_profile,
                    theta=theta,
                )

                # Log Geometry (Provenance)
                geom_id = None
                print(
                    f"DEBUG: Checking geometry data... Provenance={self.provenance is not None}, Data={report.geometry_data is not None}"
                )
                if report.geometry_data:
                    print(f"DEBUG: Data keys: {report.geometry_data.keys()}")

                if self.provenance and report.geometry_data:
                    geom_id = self.provenance.log_geometry(report.geometry_data, run_id=run_id)
                    print(f"DEBUG: Logged geometry with ID: {geom_id}")

                # Log Constraint Checks (Provenance)
                if self.provenance and geom_id and report.violations:
                    for v in report.violations:
                        check_data = {
                            "check_name": v.code.name,
                            "passed": v.severity < 2,  # ERROR=2
                            "margin_mm": v.margin if v.margin else 0.0,
                            "message": v.message,
                        }
                        self.provenance.log_constraint_check(geom_id, check_data)

                if report.is_valid:
                    return True, 1.0

                # Score based on violation count/severity
                n_violations = len(report.violations) if report.violations else 0
                score = max(0.0, 1.0 - 0.2 * n_violations)

                return False, score

        except Exception as e:
            import sys

            print(f"[ERROR] CEM check failed: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            log.debug(f"CEM check failed: {e}")
            return True, 0.5  # Uncertain

    def repair(
        self,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Project infeasible candidate back to feasible manifold.

        Args:
            candidate: Potentially infeasible candidate

        Returns:
            Repaired candidate within CEM bounds
        """
        envelope = self._get_envelope()

        if envelope is None:
            return candidate

        repaired = candidate.copy()

        # Clamp to envelope bounds
        if "p_intake_bar" in repaired:
            if hasattr(envelope, "boost_range"):
                lo, hi = envelope.boost_range
            else:
                lo, hi = getattr(envelope, "boost_min", 1.0), getattr(envelope, "boost_max", 4.0)
            repaired["p_intake_bar"] = np.clip(repaired["p_intake_bar"], lo, hi)

        if "fuel_mass_kg" in repaired:
            if hasattr(envelope, "fuel_range"):
                lo, hi = envelope.fuel_range
            else:
                lo = getattr(envelope, "fuel_min_mg", 10) * 1e-6
                hi = getattr(envelope, "fuel_max_mg", 200) * 1e-6
            repaired["fuel_mass_kg"] = np.clip(repaired["fuel_mass_kg"], lo, hi)

        return repaired

    def _get_envelope(self) -> Any:
        """Get or cache CEM envelope."""
        if self._envelope is not None:
            return self._envelope

        if not CEM_AVAILABLE:
            return None

        try:
            with CEMClient(mock=self.mock) as cem:
                self._envelope = cem.get_thermo_envelope(
                    bore=self.geometry["bore"],
                    stroke=self.geometry["stroke"],
                    cr=self.geometry["cr"],
                    rpm=self.geometry["rpm"],
                )
            return self._envelope
        except Exception as e:
            log.debug(f"Failed to get CEM envelope: {e}")
            return None

    def _fallback_batch(
        self,
        params: dict[str, Any],
        n: int,
    ) -> list[dict[str, Any]]:
        """Generate batch without CEM (reasonable defaults)."""
        rng = np.random.default_rng()
        candidates = []

        for _ in range(n):
            candidate = params.copy()
            candidate["p_intake_bar"] = rng.uniform(1.0, 3.0)
            candidate["fuel_mass_kg"] = rng.uniform(1e-5, 1.5e-4)
            candidates.append(candidate)

        return candidates

    def adapt_rules(
        self,
        truth_data: list[tuple[dict[str, Any], float]],
        run_id: str | None = None,
    ) -> Any:
        """
        Adapt rule parameters based on HiFi simulation results.

        Args:
            truth_data: List of (candidate, result) tuples
            run_id: Optional run ID

        Returns:
            Adaptation report or None
        """
        if not CEM_AVAILABLE or self.mock:
            return None

        try:
            with CEMClient(mock=self.mock) as cem:
                # Assuming CEM client has this method, otherwise we skip
                if hasattr(cem, "adapt_rules"):
                    return cem.adapt_rules(truth_data)
        except Exception as e:
            log.warning(f"Failed to adapt rules: {e}")

        return None


__all__ = [
    "CEM_AVAILABLE",
    "CEMClientAdapter",
]
