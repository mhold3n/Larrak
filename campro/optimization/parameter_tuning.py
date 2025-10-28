"""Dynamic parameter tuning based on problem characteristics."""
from __future__ import annotations

from dataclasses import dataclass

from campro.logging import get_logger
from campro.optimization.solver_selection import AnalysisHistory, ProblemCharacteristics

log = get_logger(__name__)


@dataclass
class SolverParameters:
    """Solver parameters for optimization."""

    max_iter: int
    tol: float
    linear_solver: str
    mu_strategy: str
    print_level: int
    hessian_approximation: str


class DynamicParameterTuner:
    """Tune solver parameters based on problem characteristics."""

    def __init__(self):
        self.default_params = SolverParameters(
            max_iter=1000,
            tol=1e-6,
            linear_solver="ma27",
            mu_strategy="adaptive",
            print_level=3,
            hessian_approximation="limited-memory",
        )

    def tune_parameters(
        self,
        phase: str,
        problem_chars: ProblemCharacteristics,
        analysis_history: AnalysisHistory | None,
    ) -> SolverParameters:
        """Tune parameters for optimal performance."""

        # Start with base parameters
        params = SolverParameters(
            max_iter=self.default_params.max_iter,
            tol=self.default_params.tol,
            linear_solver=self.default_params.linear_solver,
            mu_strategy=self.default_params.mu_strategy,
            print_level=self.default_params.print_level,
            hessian_approximation=self.default_params.hessian_approximation,
        )

        # Phase-specific tuning
        if phase == "primary":
            # Thermal efficiency optimization - typically well-behaved
            if problem_chars.expected_iterations > 2000:
                params.max_iter = 3000
                params.tol = 1e-7
            elif problem_chars.expected_iterations < 500:
                params.max_iter = 500
                params.tol = 1e-5

        elif phase == "secondary":
            # Litvin optimization - can be challenging
            if problem_chars.n_variables > 200:
                params.max_iter = 2000
                params.tol = 1e-7
            if problem_chars.has_convergence_issues:
                params.mu_strategy = "monotone"
                params.max_iter = 3000

        elif phase == "tertiary":
            # Crank center optimization - typically smaller problems
            if analysis_history and analysis_history.avg_grade in ["medium", "high"]:
                params.max_iter = 5000
                params.mu_strategy = "monotone"
                params.tol = 1e-8
            elif problem_chars.n_variables < 50:
                params.max_iter = 500
                params.tol = 1e-5

        # Problem size-based adjustments
        if problem_chars.n_variables > 1000:
            # Large problems - more conservative settings
            params.max_iter = min(params.max_iter * 2, 10000)
            params.tol = max(params.tol * 0.1, 1e-8)
        elif problem_chars.n_variables < 20:
            # Small problems - can be more aggressive
            params.max_iter = min(params.max_iter, 1000)
            params.tol = max(params.tol, 1e-5)

        # Convergence issues adjustments
        if problem_chars.has_convergence_issues:
            params.mu_strategy = "monotone"
            params.max_iter = min(params.max_iter * 1.5, 10000)
            params.tol = max(params.tol * 0.1, 1e-8)

        # Linear solver ratio adjustments
        if problem_chars.linear_solver_ratio > 0.5:
            # Linear solver is dominating - use more conservative settings
            params.max_iter = min(params.max_iter * 1.2, 8000)
            params.tol = max(params.tol * 0.5, 1e-7)

        log.debug(
            f"Tuned parameters for {phase} phase: max_iter={params.max_iter}, "
            f"tol={params.tol}, mu_strategy={params.mu_strategy}",
        )

        return params

    def get_default_parameters(self) -> SolverParameters:
        """Get default solver parameters."""
        return SolverParameters(
            max_iter=self.default_params.max_iter,
            tol=self.default_params.tol,
            linear_solver=self.default_params.linear_solver,
            mu_strategy=self.default_params.mu_strategy,
            print_level=self.default_params.print_level,
            hessian_approximation=self.default_params.hessian_approximation,
        )

    def create_ipopt_options(self, params: SolverParameters) -> dict:
        """Create Ipopt options dictionary from SolverParameters."""
        # Ensure MA27 is always used
        if params.linear_solver not in ["ma27", "ma57"]:
            log.warning(f"Invalid linear solver '{params.linear_solver}', forcing MA27")
            params.linear_solver = "ma27"

        return {
            "max_iter": params.max_iter,
            "tol": params.tol,
            "linear_solver": params.linear_solver,
            "mu_strategy": params.mu_strategy,
            "print_level": params.print_level,
            "hessian_approximation": params.hessian_approximation,
        }

    def create_casadi_options(self, params: SolverParameters) -> dict:
        """Create CasADi options dictionary from SolverParameters."""
        # Ensure MA27 is always used
        if params.linear_solver not in ["ma27", "ma57"]:
            log.warning(f"Invalid linear solver '{params.linear_solver}', forcing MA27")
            params.linear_solver = "ma27"

        return {
            "ipopt.max_iter": params.max_iter,
            "ipopt.tol": params.tol,
            # Note: linear_solver is set by the IPOPT factory
            "ipopt.mu_strategy": params.mu_strategy,
            "ipopt.print_level": params.print_level,
            "ipopt.hessian_approximation": params.hessian_approximation,
        }

    def estimate_problem_characteristics(
        self, n_variables: int, n_constraints: int, phase: str,
    ) -> ProblemCharacteristics:
        """Estimate problem characteristics for parameter tuning."""

        # Estimate expected iterations based on problem size and phase
        if phase == "primary":
            # Thermal efficiency - typically 500-2000 iterations
            expected_iterations = min(max(n_variables * 10, 500), 2000)
        elif phase == "secondary":
            # Litvin - can be challenging, 1000-3000 iterations
            expected_iterations = min(max(n_variables * 15, 1000), 3000)
        elif phase == "tertiary":
            # Crank center - typically smaller, 200-1000 iterations
            expected_iterations = min(max(n_variables * 20, 200), 1000)
        else:
            expected_iterations = n_variables * 10

        # Estimate linear solver ratio based on problem characteristics
        if n_variables > 500:
            linear_solver_ratio = 0.4  # Large problems tend to have higher LS ratio
        elif n_variables > 100:
            linear_solver_ratio = 0.3
        else:
            linear_solver_ratio = 0.2

        # Estimate convergence issues based on problem size
        has_convergence_issues = n_variables > 300 or n_constraints > 200

        return ProblemCharacteristics(
            n_variables=n_variables,
            n_constraints=n_constraints,
            problem_type=phase,
            expected_iterations=expected_iterations,
            linear_solver_ratio=linear_solver_ratio,
            has_convergence_issues=has_convergence_issues,
        )

    def tune_for_phase(
        self,
        phase: str,
        n_variables: int,
        n_constraints: int,
        analysis_history: AnalysisHistory | None = None,
    ) -> SolverParameters:
        """Convenience method to tune parameters for a specific phase."""

        problem_chars = self.estimate_problem_characteristics(
            n_variables, n_constraints, phase,
        )
        return self.tune_parameters(phase, problem_chars, analysis_history)

    def get_tuning_summary(
        self,
        phase: str,
        problem_chars: ProblemCharacteristics,
        params: SolverParameters,
    ) -> dict:
        """Get summary of parameter tuning decisions."""
        return {
            "phase": phase,
            "problem_characteristics": {
                "n_variables": problem_chars.n_variables,
                "n_constraints": problem_chars.n_constraints,
                "expected_iterations": problem_chars.expected_iterations,
                "linear_solver_ratio": problem_chars.linear_solver_ratio,
                "has_convergence_issues": problem_chars.has_convergence_issues,
            },
            "tuned_parameters": {
                "max_iter": params.max_iter,
                "tol": params.tol,
                "linear_solver": params.linear_solver,
                "mu_strategy": params.mu_strategy,
                "print_level": params.print_level,
                "hessian_approximation": params.hessian_approximation,
            },
            "tuning_rationale": self._get_tuning_rationale(
                phase, problem_chars, params,
            ),
        }

    def _get_tuning_rationale(
        self,
        phase: str,
        problem_chars: ProblemCharacteristics,
        params: SolverParameters,
    ) -> list[str]:
        """Get rationale for parameter tuning decisions."""
        rationale = []

        if params.max_iter > self.default_params.max_iter:
            rationale.append(
                f"Increased max_iter to {params.max_iter} for {phase} phase",
            )

        if params.tol < self.default_params.tol:
            rationale.append(f"Decreased tolerance to {params.tol} for better accuracy")

        if params.mu_strategy != self.default_params.mu_strategy:
            rationale.append(
                f"Changed mu_strategy to {params.mu_strategy} for convergence issues",
            )

        if problem_chars.n_variables > 500:
            rationale.append("Large problem size - using conservative settings")

        if problem_chars.has_convergence_issues:
            rationale.append("Convergence issues detected - using monotone strategy")

        if problem_chars.linear_solver_ratio > 0.5:
            rationale.append("High linear solver ratio - using conservative settings")

        return rationale
