"""
CasADi unified optimization flow manager.

This module orchestrates Phase 1 optimization using CasADi Opti stack
with warm-starting capabilities and integration with the existing
unified optimization framework.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from campro.logging import get_logger
from campro.optimization.base import OptimizationResult, OptimizationStatus
from campro.optimization.casadi_motion_optimizer import (
    CasADiMotionOptimizer,
    CasADiMotionProblem,
)
from campro.optimization.warmstart_manager import WarmStartManager
from campro.physics.thermal_efficiency_simple import (
    SimplifiedThermalModel,
    ThermalEfficiencyConfig,
)

log = get_logger(__name__)


@dataclass
class CasADiOptimizationSettings:
    """Settings for CasADi unified optimization flow."""

    # Warm-starting settings
    enable_warmstart: bool = True
    max_history: int = 50
    tolerance: float = 0.1
    storage_path: Optional[str] = None

    # Collocation settings
    n_segments: int = 50
    poly_order: int = 3
    collocation_method: str = "legendre"

    # Thermal efficiency settings
    efficiency_target: float = 0.55
    heat_transfer_coeff: float = 0.1
    friction_coeff: float = 0.01

    # Solver settings
    solver_options: Dict[str, Any] = None

    def __post_init__(self):
        if self.solver_options is None:
            self.solver_options = {
                "ipopt.linear_solver": "ma57",
                "ipopt.max_iter": 1000,
                "ipopt.tol": 1e-6,
                "ipopt.print_level": 0,
                "ipopt.warm_start_init_point": "yes",
            }


class CasADiUnifiedFlow:
    """
    Unified optimization flow manager for CasADi-based Phase 1 optimization.

    This class orchestrates the complete optimization flow including:
    - Problem setup and validation
    - Warm-starting from previous solutions
    - CasADi Opti stack optimization
    - Thermal efficiency evaluation
    - Solution storage for future warm-starts
    """

    def __init__(self, settings: Optional[CasADiOptimizationSettings] = None):
        """
        Initialize CasADi unified flow.

        Parameters
        ----------
        settings : Optional[CasADiOptimizationSettings]
            Optimization settings
        """
        self.settings = settings or CasADiOptimizationSettings()

        # Initialize components
        self.motion_optimizer = CasADiMotionOptimizer(
            n_segments=self.settings.n_segments,
            poly_order=self.settings.poly_order,
            collocation_method=self.settings.collocation_method,
        )

        self.warmstart_mgr = WarmStartManager(
            max_history=self.settings.max_history,
            tolerance=self.settings.tolerance,
            storage_path=self.settings.storage_path,
        )

        self.thermal_model = SimplifiedThermalModel(
            config=ThermalEfficiencyConfig(
                efficiency_target=self.settings.efficiency_target,
                heat_transfer_coeff=self.settings.heat_transfer_coeff,
                friction_coeff=self.settings.friction_coeff,
            ),
        )

        log.info(
            f"Initialized CasADiUnifiedFlow: "
            f"warmstart={self.settings.enable_warmstart}, "
            f"n_segments={self.settings.n_segments}, "
            f"efficiency_target={self.settings.efficiency_target}",
        )

    def optimize_phase1(
        self, constraints: Dict[str, Any], targets: Dict[str, Any], **kwargs,
    ) -> OptimizationResult:
        """
        Optimize Phase 1 motion law with thermal efficiency.

        Parameters
        ----------
        constraints : Dict[str, Any]
            Optimization constraints
        targets : Dict[str, Any]
            Optimization targets
        **kwargs
            Additional optimization parameters

        Returns
        -------
        OptimizationResult
            Optimization results
        """
        start_time = time.time()

        try:
            # Convert constraints and targets to CasADi problem
            problem = self._create_problem_from_constraints(constraints, targets)

            # Get initial guess if warm-starting is enabled
            initial_guess = None
            if self.settings.enable_warmstart:
                initial_guess = self.warmstart_mgr.get_initial_guess(problem.to_dict())
                if initial_guess:
                    log.info("Using warm-start initial guess")
                else:
                    log.info(
                        "No suitable warm-start found, using default initial guess",
                    )

            # Solve optimization problem
            result = self.motion_optimizer.solve(problem, initial_guess)

            # Evaluate thermal efficiency if successful
            if result.successful:
                efficiency_metrics = self.thermal_model.evaluate_efficiency(
                    result.variables["position"],
                    result.variables["velocity"],
                    result.variables["acceleration"],
                )

                # Add efficiency metrics to result
                result.metadata.update(efficiency_metrics)

                # Store solution for future warm-starts
                if self.settings.enable_warmstart:
                    self._store_solution(problem, result)

                log.info(
                    f"Phase 1 optimization completed: "
                    f"efficiency={efficiency_metrics['total_efficiency']:.3f}, "
                    f"target={self.settings.efficiency_target:.3f}",
                )

            return result

        except Exception as e:
            log.error(f"Phase 1 optimization failed: {e}")
            return OptimizationResult(
                status=OptimizationStatus.FAILED,
                successful=False,
                objective_value=float("inf"),
                solve_time=time.time() - start_time,
                variables={},
                metadata={"error": str(e)},
            )

    def _create_problem_from_constraints(
        self, constraints: Dict[str, Any], targets: Dict[str, Any],
    ) -> CasADiMotionProblem:
        """Create CasADi problem from constraints and targets."""
        # Extract problem parameters
        problem_params = {
            "stroke": constraints.get("stroke", 0.1),
            "cycle_time": constraints.get("cycle_time", 0.0385),
            "upstroke_percent": constraints.get("upstroke_percent", 50.0),
            "max_velocity": constraints.get("max_velocity", 5.0),
            "max_acceleration": constraints.get("max_acceleration", 500.0),
            "max_jerk": constraints.get("max_jerk", 50000.0),
            "compression_ratio_limits": constraints.get(
                "compression_ratio_limits", (20.0, 70.0),
            ),
        }

        # Extract objective weights
        weights = targets.get("weights", {})
        if not weights:
            weights = {
                "jerk": 1.0,
                "thermal_efficiency": 0.1,
                "smoothness": 0.01,
            }

        # Create problem
        problem = CasADiMotionProblem(
            stroke=problem_params["stroke"],
            cycle_time=problem_params["cycle_time"],
            upstroke_percent=problem_params["upstroke_percent"],
            max_velocity=problem_params["max_velocity"],
            max_acceleration=problem_params["max_acceleration"],
            max_jerk=problem_params["max_jerk"],
            compression_ratio_limits=problem_params["compression_ratio_limits"],
            objectives=targets.get(
                "objectives", ["minimize_jerk", "maximize_thermal_efficiency"],
            ),
            weights=weights,
            n_segments=self.settings.n_segments,
            poly_order=self.settings.poly_order,
            collocation_method=self.settings.collocation_method,
            solver_options=self.settings.solver_options,
            thermal_efficiency_target=self.settings.efficiency_target,
        )

        return problem

    def _store_solution(
        self, problem: CasADiMotionProblem, result: OptimizationResult,
    ) -> None:
        """Store solution for future warm-starts."""
        if not result.successful:
            return

        # Prepare solution data
        solution_data = {
            "position": result.variables["position"],
            "velocity": result.variables["velocity"],
            "acceleration": result.variables["acceleration"],
            "jerk": result.variables["jerk"],
        }

        # Prepare metadata
        metadata = {
            "solve_time": result.solve_time,
            "objective_value": result.objective_value,
            "n_segments": self.settings.n_segments,
            "timestamp": time.time(),
        }

        # Store in warm-start manager
        self.warmstart_mgr.store_solution(problem.to_dict(), solution_data, metadata)

    def get_warmstart_stats(self) -> Dict[str, Any]:
        """Get warm-start statistics."""
        return self.warmstart_mgr.get_history_stats()

    def clear_warmstart_history(self) -> None:
        """Clear warm-start history."""
        self.warmstart_mgr.clear_history()
        log.info("Cleared warm-start history")

    def update_settings(self, **kwargs) -> None:
        """Update optimization settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
                log.debug(f"Updated setting {key} = {value}")
            else:
                log.warning(f"Unknown setting: {key}")

    def setup_multiple_shooting_fallback(self) -> None:
        """
        Setup multiple shooting fallback for stiff cases.

        This is a stub for future implementation of multiple shooting
        as a fallback when direct collocation fails.
        """
        log.info("Multiple shooting fallback not yet implemented")
        # TODO: Implement multiple shooting fallback

    def optimize_with_fallback(
        self, constraints: Dict[str, Any], targets: Dict[str, Any], **kwargs,
    ) -> OptimizationResult:
        """
        Optimize with fallback to multiple shooting if direct collocation fails.

        Parameters
        ----------
        constraints : Dict[str, Any]
            Optimization constraints
        targets : Dict[str, Any]
            Optimization targets
        **kwargs
            Additional optimization parameters

        Returns
        -------
        OptimizationResult
            Optimization results
        """
        # Try direct collocation first
        result = self.optimize_phase1(constraints, targets, **kwargs)

        # If failed, try fallback (when implemented)
        if not result.successful:
            log.warning("Direct collocation failed, attempting fallback")
            # TODO: Implement multiple shooting fallback
            # result = self._optimize_with_multiple_shooting(constraints, targets, **kwargs)

        return result

    def benchmark_optimization(self, problem_specs: list) -> Dict[str, Any]:
        """
        Benchmark optimization performance across multiple problem specifications.

        Parameters
        ----------
        problem_specs : list
            List of problem specifications to benchmark

        Returns
        -------
        Dict[str, Any]
            Benchmark results
        """
        results = []

        for i, spec in enumerate(problem_specs):
            log.info(f"Benchmarking problem {i + 1}/{len(problem_specs)}")

            start_time = time.time()
            result = self.optimize_phase1(spec["constraints"], spec["targets"])
            solve_time = time.time() - start_time

            results.append(
                {
                    "problem_id": i,
                    "successful": result.successful,
                    "solve_time": solve_time,
                    "objective_value": result.objective_value,
                    "n_iterations": result.metadata.get("n_iterations", 0),
                    "efficiency": result.metadata.get("total_efficiency", 0.0),
                },
            )

        # Compute statistics
        successful_results = [r for r in results if r["successful"]]

        benchmark_stats = {
            "total_problems": len(problem_specs),
            "successful_problems": len(successful_results),
            "success_rate": len(successful_results) / len(problem_specs),
            "avg_solve_time": np.mean([r["solve_time"] for r in successful_results])
            if successful_results
            else 0,
            "min_solve_time": np.min([r["solve_time"] for r in successful_results])
            if successful_results
            else 0,
            "max_solve_time": np.max([r["solve_time"] for r in successful_results])
            if successful_results
            else 0,
            "avg_efficiency": np.mean([r["efficiency"] for r in successful_results])
            if successful_results
            else 0,
            "results": results,
        }

        log.info(
            f"Benchmark completed: {benchmark_stats['success_rate']:.1%} success rate, "
            f"avg solve time: {benchmark_stats['avg_solve_time']:.3f}s",
        )

        return benchmark_stats
