import logging
from typing import Any

import numpy as np

from campro.optimization.initialization.geometry_solvers import (
    FourierProfileGenerator,
    KinematicOptimizer,
)
from campro.optimization.initialization.thermo_solvers import (
    CasAdiCFDShooter,
    RungeKuttaShooter,
)

log = logging.getLogger(__name__)


class InitializationManager:
    """
    Orchestrates the Ensemble Initialization process.
    Runs multiple geometry and thermodynamic solvers to find the best initial guess.
    """

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.geometry_solvers = [KinematicOptimizer(), FourierProfileGenerator()]

        # Get n_cells from flow config or default
        n_cells = params.get("flow", {}).get("n_cells", 10)

        self.thermo_solvers = [
            RungeKuttaShooter(),
            CasAdiCFDShooter(n_cells=n_cells),
            # SpectralThermoSolver() # Disabled until implemented
        ]

    def solve(self) -> dict[str, Any]:
        """
        Execute the ensemble initialization strategy.
        Returns the best candidate trajectory.
        """
        candidates = []

        # 1. Generate Geometry Candidates
        geom_candidates = []
        for geom_solver in self.geometry_solvers:
            try:
                log.info(f"Running geometry solver: {geom_solver.__class__.__name__}")
                # params need to be flattened or extracted?
                # Solvers expect specific keys like 'stroke', 'cycle_time'
                # Let's extract them from P.planet_ring or P.geometry or P.combustion
                inputs = self._extract_inputs()

                res = geom_solver.solve(inputs)
                if res["success"]:
                    geom_candidates.append(res)
                else:
                    log.warning(
                        f"Geometry solver {geom_solver.__class__.__name__} failed."
                    )
            except Exception as e:
                log.error(
                    f"Error in geometry solver {geom_solver.__class__.__name__}: {e}"
                )

        if not geom_candidates:
            log.error("No valid geometry candidates generated!")
            return {"success": False}

        # 2. Solve Thermodynamics for each Geometry
        for geom in geom_candidates:
            for thermo_solver in self.thermo_solvers:
                try:
                    log.info(
                        f"Running thermo solver {thermo_solver.__class__.__name__} on geometry {geom['method']}"
                    )
                    thermo_res = thermo_solver.solve(geom, self.params)

                    if thermo_res["success"]:
                        # Combine results
                        candidate = {
                            "geometry": geom,
                            "thermo": thermo_res,
                            "score": self._score_candidate(geom, thermo_res),
                        }
                        candidates.append(candidate)

                except Exception as e:
                    log.error(
                        f"Error in thermo solver {thermo_solver.__class__.__name__}: {e}"
                    )

        # 3. Rank and Select
        if not candidates:
            log.error("No valid initialization candidates found!")
            return {"success": False}

        # Sort by score (lower is better)
        candidates.sort(key=lambda x: x["score"])
        best = candidates[0]

        log.info(
            f"Selected best candidate: Geo={best['geometry']['method']}, Thermo={best['thermo']['method']}, Score={best['score']:.4e}"
        )

        return {"success": True, "best_candidate": best, "all_candidates": candidates}

    def _extract_inputs(self) -> dict[str, Any]:
        """Helper to extract flat inputs from nested params dict."""
        P = self.params
        inputs = {}

        # Priority: planet_ring > geometry > combustion (for cycle_time)
        pr = P.get("planet_ring", {})
        geom = P.get("geometry", {})
        comb = P.get("combustion", {})

        inputs["stroke"] = pr.get("stroke", geom.get("stroke", 0.1))
        inputs["cycle_time"] = comb.get("cycle_time_s", 0.02)

        # Velocity bounds?
        bounds = P.get("bounds", {})
        inputs["v_max"] = bounds.get("vL_max", 50.0)  # Symmetric

        return inputs

    def _score_candidate(self, geom: dict, thermo: dict) -> float:
        """
        Score a candidate based on:
        - Geometric smoothness (Jerk) -> From geom
        - Thermodynamic closure (Residuals) -> From thermo (implicit in success, maybe check precision?)
        - Constraints violations?

        Lower is better.
        """
        # Feature 1: Max Jerk (Smoothness)
        # Calculate jerk if available, or approx
        acc = geom.get("a", np.zeros_like(geom["t"]))
        jerk_metric = np.mean(np.diff(acc) ** 2) if len(acc) > 1 else 0.0

        # Feature 2: Peak Pressure (Feasibility)
        p_max_cand = np.max(thermo["p"])
        p_target = 200e5  # 200 bar
        p_penalty = max(0, p_max_cand - p_target)

        # Weighted score
        score = jerk_metric + 1e-3 * p_penalty
        return score
