"""
Optimization Pipeline Module.

This module provides the `OptimizationPipeline` class, which orchestrates the
full optimization lifecycle: Build -> Solve -> Post-Process.

It integrates with the provenance system to emit telemetry events (module_start/end)
ensuring detailed tracking on the Orchestrator Dashboard.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from campro.logging import get_logger
from campro.optimization.builder import OptimizationBuilder
from campro.optimization.core.solution import Solution
from campro.optimization.initialization.manager import InitializationManager
from campro.optimization.postprocessing import OptimizationPostProcessor
from campro.optimization.solvers.ipopt_options import create_ipopt_options
from campro.optimization.solvers.ipopt_solver import IPOPTSolver
from campro.utils import format_duration
from campro.utils.structured_reporter import StructuredReporter

# Import scaling functions for solve phase
# These must be imported here to be available during run()
try:
    from campro.diagnostics.scaling import (
        build_scaled_nlp,
        scale_bounds,
        scale_value,
        unscale_value,
    )
except ImportError:
    # Diagnostic scaling module might be missing or circular
    # If missing, we can't run the pipeline as intended with scaling.
    # We assume it exists as it was in driver.py
    pass

# Telemetry
try:
    from provenance.execution_events import module_end, module_start
except ImportError:
    # Fallback if provenance is not available
    def module_start(*args, **kwargs):
        pass

    def module_end(*args, **kwargs):
        pass


log = get_logger(__name__)


class CombinedDenseOutput:
    """Helper to combine geometry and thermo dense outputs for NLP."""

    def __init__(self, geom_res: dict[str, Any], sol_thermo: Any, cycle_time: float) -> None:
        self.geom_res = geom_res
        self.sol_thermo = sol_thermo
        self.cycle_time = cycle_time

        if "spline" in geom_res:
            self.x_func = lambda t: geom_res["spline"](t)
            self.v_func = lambda t: geom_res["spline"](t, nu=1)
        else:
            self.x_func = interp1d(
                geom_res["t"], geom_res["x"], kind="cubic", fill_value="extrapolate"
            )
            self.v_func = interp1d(
                geom_res["t"], geom_res["v"], kind="cubic", fill_value="extrapolate"
            )

    def sol(self, t: Any) -> np.ndarray:
        t_arr = np.asarray(t)
        t_mod = t_arr % self.cycle_time

        x_val = self.x_func(t_mod)
        v_val = self.v_func(t_mod)
        therm = self.sol_thermo.sol(t_mod)

        if t_arr.ndim == 0:
            return np.hstack([x_val, v_val, therm])
        else:
            return np.vstack([x_val, v_val, therm])


class OptimizationPipeline:
    """
    Orchestrates the optimization process.

    Phases:
    0. Initialization: Ensemble Init / Phase 3 Guess.
    1. Builder: Construct NLP, Bounds, Scaling.
    2. Solver: Execute IPOPT.
    3. PostProcessor: Generate Artifacts (Profiles, Mechanical Results).
    """

    def __init__(self, params: dict[str, Any]):
        """
        Initialize the pipeline.

        Args:
            params: Optimization parameters.
        """
        self.params = params
        self.reporter = StructuredReporter(
            context="PIPELINE",
            logger=log,
            stream_out=None,
            stream_err=None,
            debug_env="FREE_PISTON_DEBUG",
        )
        self.builder = OptimizationBuilder(params, reporter=self.reporter)
        self.post_processor = OptimizationPostProcessor(reporter=self.reporter)

    def _run_ensemble_initialization(self) -> dict[str, Any] | None:
        """Run Ensemble Initialization Suite."""
        try:
            pr_cfg = self.params.get("planet_ring", {})
            use_load_model = pr_cfg.get("use_load_model", False)
            problem_type = self.params.get("problem_type", "default")
            is_phase3_mechanical = "load_profile" in self.params

            if (
                "planet_ring" in self.params
                and not use_load_model
                and problem_type != "kinematic"
                and not is_phase3_mechanical
            ):
                self.reporter.info("Running Ensemble Initialization Suite...")
                module_start("ENSEMBLE_INIT", step="ensemble_search")

                init_manager = InitializationManager(self.params)
                init_result = init_manager.solve()

                if init_result["success"]:
                    best_cand = init_result["best_candidate"]
                    self.reporter.info(
                        f"Ensemble Init Converged: {best_cand['geometry']['method']} + {best_cand['thermo']['method']}"
                    )

                    sol_thermo = best_cand["thermo"]["sol"]
                    geom_res = best_cand["geometry"]
                    cycle_time = geom_res["t"][-1]

                    num_intervals = int(self.params.get("num", {}).get("K", 10))

                    sol = CombinedDenseOutput(geom_res, sol_thermo, cycle_time)
                    t_eval = np.linspace(0, cycle_time, num_intervals + 1)
                    y_eval = sol.sol(t_eval)

                    initial_trajectory = {
                        "t": t_eval.tolist(),
                        "xL": y_eval[0].tolist(),
                        "vL": y_eval[1].tolist(),
                        "xR": (-y_eval[0]).tolist(),
                        "vR": (-y_eval[1]).tolist(),
                        "_sol": sol,
                    }

                    # Unpack Thermo
                    n_rows = y_eval.shape[0]
                    if n_rows == 4:
                        initial_trajectory["rho"] = y_eval[2].tolist()
                        initial_trajectory["T"] = y_eval[3].tolist()
                    elif n_rows > 4:
                        n_cells = (n_rows - 2) // 3
                        for i in range(n_cells):
                            initial_trajectory[f"rho_{i}"] = y_eval[2 + i].tolist()
                            initial_trajectory[f"u_{i}"] = y_eval[2 + n_cells + i].tolist()
                            initial_trajectory[f"E_{i}"] = y_eval[2 + 2 * n_cells + i].tolist()
                        initial_trajectory["rho"] = y_eval[2].tolist()  # Fallback
                        Cv = 718.0
                        initial_trajectory["T"] = (y_eval[2 + 2 * n_cells] / Cv).tolist()

                    module_end("ENSEMBLE_INIT", status="SUCCESS")
                    return initial_trajectory
                else:
                    self.reporter.warning("Ensemble Initialization failed.")
                    module_end("ENSEMBLE_INIT", status="FAILED")
        except Exception as e:
            self.reporter.warning(f"Ensemble Initialization error: {e}")
            module_end("ENSEMBLE_INIT", status="CRASHED", error=str(e))

        return None

    def _generate_phase3_guess(self) -> dict[str, Any] | None:
        """Generate guess for Phase 3 Mechanical."""
        if "load_profile" in self.params:
            self.reporter.info("Phase 3 Detected. Generating linear guess.")
            pr_cfg = self.params.get("planet_ring", {})
            mean_ratio = float(pr_cfg.get("mean_ratio", 2.0))
            target_delta_psi = 2.0 * np.pi * (mean_ratio - 1.0)
            num_intervals = int(self.params.get("num", {}).get("K", 10))
            psi_guess = np.linspace(0, target_delta_psi, num_intervals + 1).tolist()
            return {"psi": psi_guess}
        return None

    def run(self, initial_trajectory: dict[str, Any] | None = None) -> Solution:
        """
        Run the full optimization pipeline.

        Args:
            initial_trajectory: Optional initial guess.

        Returns:
            Solution object.
        """
        self.reporter.info("Starting Optimization Pipeline...")

        optimization_result: dict[str, Any] = {
            "success": False,
            "f_opt": float("inf"),
            "iterations": 0,
            "status": -1,
            "message": "Initialized",
        }
        nlp, meta = None, None
        x_opt_unscaled = None

        try:
            # --- PHASE 0: INITIALIZATION ---
            if initial_trajectory is None:
                initial_trajectory = self._run_ensemble_initialization()

            if initial_trajectory is None:
                initial_trajectory = self._generate_phase3_guess()

            # --- PHASE 1: BUILD ---
            module_start("NLP_BUILD", step="build_nlp")
            try:
                nlp, meta = self.builder.build_nlp(initial_trajectory)

                # Setup Bounds
                x0, lbx, ubx, lbg, ubg, _ = self.builder.setup_bounds(nlp, meta)

                # Compute Scaling
                scale, scale_g, scale_f = self.builder.compute_scaling(
                    nlp, meta, x0, lbx, ubx, lbg, ubg
                )

                module_end("NLP_BUILD", status="SUCCESS")

            except Exception as e:
                module_end("NLP_BUILD", status="FAILED", error=str(e))
                raise e

            # --- PHASE 2: SOLVE ---
            module_start("IPOPT_SOLVE", step="solve_nlp")
            try:
                # Apply Scaling
                self.reporter.info("Applying scaling to NLP...")
                nlp_scaled = build_scaled_nlp(
                    nlp, scale, constraint_scale=scale_g, objective_scale=scale_f
                )
                x0_scaled = scale_value(x0, scale)
                lbx_scaled, ubx_scaled = scale_bounds((lbx, ubx), scale)
                lbg_scaled, ubg_scaled = scale_bounds((lbg, ubg), scale_g)

                # Create Solver
                ipopt_opts_dict = self.params.get("solver", {}).get("ipopt", {})

                # Diagnostic mode checks
                if os.getenv("FREE_PISTON_DIAGNOSTIC_MODE", "0") == "1":
                    ipopt_opts_dict.setdefault("ipopt.derivative_test", "first-order")
                    ipopt_opts_dict.setdefault("ipopt.print_level", 12)

                ipopt_options = create_ipopt_options(ipopt_opts_dict, self.params)
                solver_wrapper = IPOPTSolver(ipopt_options)
                solver_wrapper._meta_for_diagnostics = meta  # For callbacks

                # Solve
                self.reporter.info("Executing IPOPT Solver...")
                solve_start = time.time()
                result = solver_wrapper.solve(
                    nlp_scaled, x0_scaled, lbx_scaled, ubx_scaled, lbg_scaled, ubg_scaled
                )
                solve_duration = time.time() - solve_start
                self.reporter.info(f"Solver finished in {format_duration(solve_duration)}")

                # Process Result
                if result.success:
                    self.reporter.info(f"Optimization Converged! f_opt={result.f_opt:.6e}")
                else:
                    self.reporter.warning(f"Optimization Failed: {result.message}")

                # Unscale Solution
                if result.x_opt is not None:
                    x_opt_unscaled = unscale_value(result.x_opt, scale)
                    f_opt_unscaled = result.f_opt / scale_f
                else:
                    x_opt_unscaled = None
                    f_opt_unscaled = float("inf")

                # Populate Optimization Result Dict
                optimization_result.update(
                    {
                        "success": result.success,
                        "x_opt": x_opt_unscaled,
                        "f_opt": f_opt_unscaled if result.success else result.f_opt,
                        "iterations": result.iterations,
                        "cpu_time": result.cpu_time,
                        "message": result.message,
                        "status": result.status,
                        "kkt_error": result.kkt_error,
                        "feasibility_error": result.feasibility_error,
                        "primal_inf": result.primal_inf,
                        "dual_inf": result.dual_inf,
                        "lam_g": result.lambda_opt,
                        "lam_x": getattr(result, "lam_x", None),
                        "complementarity": result.complementarity,
                        "constraint_violation": result.constraint_violation,
                    }
                )

                module_end("IPOPT_SOLVE", status="SUCCESS" if result.success else "FAILED")

            except Exception as e:
                module_end("IPOPT_SOLVE", status="CRASHED", error=str(e))
                raise e

            # --- PHASE 3: POST-PROCESSING ---
            if x_opt_unscaled is not None and meta is not None:
                # Cycle Time Extraction
                if "variable_groups" in meta:
                    cycle_time_indices = meta["variable_groups"].get("cycle_time", [])
                    if cycle_time_indices and len(cycle_time_indices) > 0:
                        t_cycle_idx = cycle_time_indices[0]
                        if 0 <= t_cycle_idx < len(x_opt_unscaled):
                            t_cycle_val = float(x_opt_unscaled[t_cycle_idx])
                            optimization_result["T_cycle"] = t_cycle_val

                # Conjugate Profiles
                module_start("POST_PROF", step="generate_profiles")
                self.post_processor.generate_conjugate_profiles(
                    x_opt_unscaled, meta, optimization_result
                )
                module_end("POST_PROF", status="COMPLETED")

                # Mechanical Results
                module_start("POST_MECH", step="process_mechanical")
                self.post_processor.process_mechanical_results(
                    x_opt_unscaled, self.params, meta, optimization_result
                )
                module_end("POST_MECH", status="COMPLETED")

                # Checkpoint Save
                run_dir = self.params.get("run_dir")
                if run_dir:
                    try:
                        from campro.optimization.io.save import save_json

                        save_json(
                            {"meta": meta, "opt": optimization_result},
                            run_dir,
                            filename="checkpoint.json",
                        )
                    except Exception:
                        pass

        except Exception as e:
            self.reporter.error(f"Pipeline crashed: {e}")
            optimization_result["success"] = False
            optimization_result["message"] = f"Pipeline Crashed: {str(e)}"
            optimization_result["error"] = str(e)

        # Construct Final Solution
        final_meta = {
            "meta": meta,
            "optimization": optimization_result,
        }
        final_data = {
            "x": optimization_result.get("x_opt"),
            "residual_sample": None,
            "nlp": nlp,
            "lam_g": optimization_result.get("lam_g"),
            "lam_x": optimization_result.get("lam_x"),
            "initial_trajectory": initial_trajectory,
        }

        return Solution(meta=final_meta, data=final_data)
