"""
Motion Law Optimization Library

This module provides a clean, reusable interface for OP engine motion law optimization.
It encapsulates the complex optimization pipeline into a simple, configurable API
that can be used as part of larger optimization routines.

Key Features:
- Clean separation of concerns
- Multiple solver backends (IPOPT, etc.)
- Flexible problem configuration
- Comprehensive result handling
- Built-in validation and post-processing
- Extensible architecture for custom objectives and constraints

Usage:
    from campro.freepiston.opt.optimization_lib import MotionLawOptimizer
    
    optimizer = MotionLawOptimizer(config)
    result = optimizer.optimize()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np

from campro.freepiston.opt.driver import (
    solve_cycle,
    solve_cycle_adaptive,
    solve_cycle_robust,
)
from campro.freepiston.opt.solution import Solution
from campro.freepiston.validation.physics_validator import PhysicsValidator
from campro.freepiston.validation.solution_validator import SolutionValidator
from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for motion law optimization."""

    # Problem setup
    geometry: Dict[str, float] = field(default_factory=dict)
    thermodynamics: Dict[str, float] = field(default_factory=dict)
    bounds: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

    # Optimization parameters
    num: Dict[str, int] = field(default_factory=lambda: {"K": 20, "C": 3})
    solver: Dict[str, Any] = field(default_factory=dict)
    objective: Dict[str, Any] = field(default_factory=lambda: {"method": "indicated_work"})

    # Model configuration
    model_type: str = "0d"  # "0d" or "1d"
    use_1d_gas: bool = False
    n_cells: int = 50

    # Validation and post-processing
    validation: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)

    # Advanced options
    warm_start: Optional[Dict[str, Any]] = None
    refinement_strategy: str = "adaptive"
    max_refinements: int = 3


@dataclass
class OptimizationResult:
    """Result of motion law optimization."""

    # Core results
    success: bool
    solution: Optional[Solution] = None
    objective_value: float = float("inf")
    iterations: int = 0
    cpu_time: float = 0.0

    # Convergence metrics
    kkt_error: float = float("inf")
    feasibility_error: float = float("inf")
    message: str = ""
    status: int = -1

    # Validation results
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    physics_validation: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    config: Optional[OptimizationConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class SolverBackend(Protocol):
    """Protocol for optimization solver backends."""

    def solve(self, problem: Dict[str, Any]) -> OptimizationResult:
        """Solve the optimization problem."""
        ...


class IPOPTBackend:
    """IPOPT solver backend."""

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        self.options = options or {}

    def solve(self, problem: Dict[str, Any]) -> OptimizationResult:
        """Solve using IPOPT."""
        start_time = time.time()

        try:
            # Use the existing driver function
            result = solve_cycle(problem)

            cpu_time = time.time() - start_time

            return OptimizationResult(
                success=result.meta.get("optimization", {}).get("success", False),
                solution=result,
                objective_value=result.meta.get("optimization", {}).get("f_opt", float("inf")),
                iterations=result.meta.get("optimization", {}).get("iterations", 0),
                cpu_time=cpu_time,
                kkt_error=result.meta.get("optimization", {}).get("kkt_error", float("inf")),
                feasibility_error=result.meta.get("optimization", {}).get("feasibility_error", float("inf")),
                message=result.meta.get("optimization", {}).get("message", ""),
                status=result.meta.get("optimization", {}).get("status", -1),
            )

        except Exception as e:
            log.error(f"IPOPT solve failed: {e}")
            return OptimizationResult(
                success=False,
                errors=[str(e)],
                cpu_time=time.time() - start_time,
            )


class RobustIPOPTBackend:
    """Robust IPOPT solver backend with conservative settings."""

    def __init__(self, options: Optional[Dict[str, Any]] = None):
        self.options = options or {}

    def solve(self, problem: Dict[str, Any]) -> OptimizationResult:
        """Solve using robust IPOPT settings."""
        start_time = time.time()

        try:
            result = solve_cycle_robust(problem)
            cpu_time = time.time() - start_time

            return OptimizationResult(
                success=result.meta.get("optimization", {}).get("success", False),
                solution=result,
                objective_value=result.meta.get("optimization", {}).get("f_opt", float("inf")),
                iterations=result.meta.get("optimization", {}).get("iterations", 0),
                cpu_time=cpu_time,
                kkt_error=result.meta.get("optimization", {}).get("kkt_error", float("inf")),
                feasibility_error=result.meta.get("optimization", {}).get("feasibility_error", float("inf")),
                message=result.meta.get("optimization", {}).get("message", ""),
                status=result.meta.get("optimization", {}).get("status", -1),
            )

        except Exception as e:
            log.error(f"Robust IPOPT solve failed: {e}")
            return OptimizationResult(
                success=False,
                errors=[str(e)],
                cpu_time=time.time() - start_time,
            )


class AdaptiveBackend:
    """Adaptive solver backend with refinement strategy."""

    def __init__(self, max_refinements: int = 3, options: Optional[Dict[str, Any]] = None):
        self.max_refinements = max_refinements
        self.options = options or {}

    def solve(self, problem: Dict[str, Any]) -> OptimizationResult:
        """Solve using adaptive refinement strategy."""
        start_time = time.time()

        try:
            result = solve_cycle_adaptive(problem, max_refinements=self.max_refinements)
            cpu_time = time.time() - start_time

            return OptimizationResult(
                success=result.meta.get("optimization", {}).get("success", False),
                solution=result,
                objective_value=result.meta.get("optimization", {}).get("f_opt", float("inf")),
                iterations=result.meta.get("optimization", {}).get("iterations", 0),
                cpu_time=cpu_time,
                kkt_error=result.meta.get("optimization", {}).get("kkt_error", float("inf")),
                feasibility_error=result.meta.get("optimization", {}).get("feasibility_error", float("inf")),
                message=result.meta.get("optimization", {}).get("message", ""),
                status=result.meta.get("optimization", {}).get("status", -1),
            )

        except Exception as e:
            log.error(f"Adaptive solve failed: {e}")
            return OptimizationResult(
                success=False,
                errors=[str(e)],
                cpu_time=time.time() - start_time,
            )


class ProblemBuilder:
    """Builder for optimization problems."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._problem: Dict[str, Any] = {}

    def build(self) -> Dict[str, Any]:
        """Build the complete optimization problem."""
        self._problem = {
            "geometry": self.config.geometry,
            "thermodynamics": self.config.thermodynamics,
            "bounds": self.config.bounds,
            "constraints": self.config.constraints,
            "num": self.config.num,
            "solver": self.config.solver,
            "obj": self.config.objective,
            "model_type": self.config.model_type,
            "flow": {
                "use_1d_gas": self.config.use_1d_gas,
                "mesh_cells": self.config.n_cells,
            },
            "validation": self.config.validation,
            "output": self.config.output,
        }

        # Add warm start if provided
        if self.config.warm_start:
            self._problem["warm_start"] = self.config.warm_start

        return self._problem

    def with_geometry(self, geometry: Dict[str, float]) -> ProblemBuilder:
        """Set geometry parameters."""
        self.config.geometry.update(geometry)
        return self

    def with_thermodynamics(self, thermo: Dict[str, float]) -> ProblemBuilder:
        """Set thermodynamics parameters."""
        self.config.thermodynamics.update(thermo)
        return self

    def with_bounds(self, bounds: Dict[str, float]) -> ProblemBuilder:
        """Set variable bounds."""
        self.config.bounds.update(bounds)
        return self

    def with_constraints(self, constraints: Dict[str, Any]) -> ProblemBuilder:
        """Set constraints."""
        self.config.constraints.update(constraints)
        return self

    def with_objective(self, objective: Dict[str, Any]) -> ProblemBuilder:
        """Set objective function."""
        self.config.objective.update(objective)
        return self

    def with_solver_options(self, options: Dict[str, Any]) -> ProblemBuilder:
        """Set solver options."""
        self.config.solver.update(options)
        return self

    def with_1d_model(self, n_cells: int = 50) -> ProblemBuilder:
        """Enable 1D gas model."""
        self.config.model_type = "1d"
        self.config.use_1d_gas = True
        self.config.n_cells = n_cells
        return self

    def with_0d_model(self) -> ProblemBuilder:
        """Enable 0D gas model."""
        self.config.model_type = "0d"
        self.config.use_1d_gas = False
        return self


class ResultProcessor:
    """Process and validate optimization results."""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.solution_validator = SolutionValidator()
        self.physics_validator = PhysicsValidator()

    def process(self, result: OptimizationResult) -> OptimizationResult:
        """Process and validate the optimization result."""
        if not result.success or result.solution is None:
            return result

        # Validate solution
        validation_metrics = self._validate_solution(result.solution)
        result.validation_metrics = validation_metrics

        # Validate physics
        physics_validation = self._validate_physics(result.solution)
        result.physics_validation = physics_validation

        # Compute performance metrics
        performance_metrics = self._compute_performance_metrics(result.solution)
        result.performance_metrics = performance_metrics

        # Add warnings and errors
        result.warnings.extend(validation_metrics.get("warnings", []))
        result.errors.extend(validation_metrics.get("errors", []))
        result.warnings.extend(physics_validation.get("warnings", []))
        result.errors.extend(physics_validation.get("errors", []))

        return result

    def _validate_solution(self, solution: Solution) -> Dict[str, Any]:
        """Validate the optimization solution."""
        try:
            return self.solution_validator.validate(solution)
        except Exception as e:
            log.warning(f"Solution validation failed: {e}")
            return {"warnings": [f"Solution validation failed: {e}"]}

    def _validate_physics(self, solution: Solution) -> Dict[str, Any]:
        """Validate physics constraints."""
        try:
            return self.physics_validator.validate(solution)
        except Exception as e:
            log.warning(f"Physics validation failed: {e}")
            return {"warnings": [f"Physics validation failed: {e}"]}

    def _compute_performance_metrics(self, solution: Solution) -> Dict[str, Any]:
        """Compute performance metrics from solution."""
        metrics = {}

        try:
            # Extract state variables from solution
            if hasattr(solution, "data") and "states" in solution.data:
                states = solution.data["states"]

                # Compute basic metrics
                if "pressure" in states:
                    pressures = states["pressure"]
                    metrics["max_pressure"] = float(np.max(pressures))
                    metrics["min_pressure"] = float(np.min(pressures))
                    metrics["mean_pressure"] = float(np.mean(pressures))

                if "temperature" in states:
                    temperatures = states["temperature"]
                    metrics["max_temperature"] = float(np.max(temperatures))
                    metrics["min_temperature"] = float(np.min(temperatures))
                    metrics["mean_temperature"] = float(np.mean(temperatures))

                if "x_L" in states and "x_R" in states:
                    x_L = states["x_L"]
                    x_R = states["x_R"]
                    gaps = [x_R[i] - x_L[i] for i in range(len(x_L))]
                    metrics["min_piston_gap"] = float(np.min(gaps))
                    metrics["max_piston_gap"] = float(np.max(gaps))
                    metrics["mean_piston_gap"] = float(np.mean(gaps))

        except Exception as e:
            log.warning(f"Performance metrics computation failed: {e}")
            metrics["error"] = str(e)

        return metrics


class MotionLawOptimizer:
    """Main optimization class for motion law optimization."""

    def __init__(self, config: Union[OptimizationConfig, Dict[str, Any]],
                 solver_backend: Optional[SolverBackend] = None):
        """
        Initialize the motion law optimizer.
        
        Args:
            config: Optimization configuration
            solver_backend: Solver backend to use (default: IPOPT)
        """
        if isinstance(config, dict):
            self.config = self._dict_to_config(config)
        else:
            self.config = config

        self.solver_backend = solver_backend or IPOPTBackend()
        self.problem_builder = ProblemBuilder(self.config)
        self.result_processor = ResultProcessor(self.config)

        log.info("MotionLawOptimizer initialized")

    def optimize(self) -> OptimizationResult:
        """
        Run the optimization.
        
        Returns:
            OptimizationResult with complete optimization results
        """
        log.info("Starting motion law optimization...")

        try:
            # Build problem
            problem = self.problem_builder.build()

            # Solve optimization
            result = self.solver_backend.solve(problem)
            result.config = self.config

            # Process results
            result = self.result_processor.process(result)

            # Log results
            if result.success:
                log.info(f"Optimization successful: {result.message}")
                log.info(f"Objective value: {result.objective_value:.6e}")
                log.info(f"Iterations: {result.iterations}, CPU time: {result.cpu_time:.2f}s")
            else:
                log.warning(f"Optimization failed: {result.message}")
                if result.errors:
                    log.error(f"Errors: {result.errors}")

            return result

        except Exception as e:
            log.error(f"Optimization failed with exception: {e}")
            return OptimizationResult(
                success=False,
                errors=[str(e)],
                config=self.config,
            )

    def optimize_with_validation(self, validate: bool = True) -> OptimizationResult:
        """
        Run optimization with optional validation.
        
        Args:
            validate: Whether to perform validation
            
        Returns:
            OptimizationResult with validation results
        """
        result = self.optimize()

        if validate and result.success:
            result = self.result_processor.process(result)

        return result

    def get_problem_builder(self) -> ProblemBuilder:
        """Get the problem builder for configuration."""
        return self.problem_builder

    def set_solver_backend(self, backend: SolverBackend) -> None:
        """Set the solver backend."""
        self.solver_backend = backend

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> OptimizationConfig:
        """Convert dictionary to OptimizationConfig."""
        return OptimizationConfig(
            geometry=config_dict.get("geometry", {}),
            thermodynamics=config_dict.get("thermodynamics", {}),
            bounds=config_dict.get("bounds", {}),
            constraints=config_dict.get("constraints", {}),
            num=config_dict.get("num", {"K": 20, "C": 3}),
            solver=config_dict.get("solver", {}),
            objective=config_dict.get("objective", {"method": "indicated_work"}),
            model_type=config_dict.get("model_type", "0d"),
            use_1d_gas=config_dict.get("use_1d_gas", False),
            n_cells=config_dict.get("n_cells", 50),
            validation=config_dict.get("validation", {}),
            output=config_dict.get("output", {}),
            warm_start=config_dict.get("warm_start"),
            refinement_strategy=config_dict.get("refinement_strategy", "adaptive"),
            max_refinements=config_dict.get("max_refinements", 3),
        )


# Convenience functions for common use cases

def create_standard_optimizer(config: Union[OptimizationConfig, Dict[str, Any]]) -> MotionLawOptimizer:
    """Create a standard optimizer with IPOPT backend."""
    return MotionLawOptimizer(config, IPOPTBackend())


def create_robust_optimizer(config: Union[OptimizationConfig, Dict[str, Any]]) -> MotionLawOptimizer:
    """Create a robust optimizer with conservative IPOPT settings."""
    return MotionLawOptimizer(config, RobustIPOPTBackend())


def create_adaptive_optimizer(config: Union[OptimizationConfig, Dict[str, Any]],
                             max_refinements: int = 3) -> MotionLawOptimizer:
    """Create an adaptive optimizer with refinement strategy."""
    return MotionLawOptimizer(config, AdaptiveBackend(max_refinements))


def quick_optimize(config: Union[OptimizationConfig, Dict[str, Any]],
                  backend: str = "standard") -> OptimizationResult:
    """
    Quick optimization with minimal setup.
    
    Args:
        config: Optimization configuration
        backend: Solver backend ("standard", "robust", "adaptive")
        
    Returns:
        OptimizationResult
    """
    if backend == "robust":
        optimizer = create_robust_optimizer(config)
    elif backend == "adaptive":
        optimizer = create_adaptive_optimizer(config)
    else:
        optimizer = create_standard_optimizer(config)

    return optimizer.optimize()
