from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np

# Force iteration tracing unless explicitly overridden later in the process.
os.environ["FREE_PISTON_IPOPT_TRACE"] = "1"

_FALSEY = {"0", "false", "no", "off"}
NLPSOL_OUTPUT_NAMES: tuple[str, ...] = tuple(ca.nlpsol_out())


def _env_flag(name: str, default: bool = False) -> bool:
    """Read boolean flag from environment."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in _FALSEY

from campro.constants import IPOPT_LOG_DIR
from campro.logging import get_logger
from campro.utils.structured_reporter import StructuredReporter

log = get_logger(__name__)


@dataclass
class IPOPTOptions:
    """IPOPT solver options."""

    # Basic solver options
    max_iter: int = 3000
    max_cpu_time: float = 3600.0  # 1 hour
    tol: float = 1e-8
    acceptable_tol: float = 1e-6
    acceptable_iter: int = 15

    # Linear solver options (unused when option_file_name is provided)
    linear_solver: str = "ma57"  # Prefer MA57 for better performance; falls back to MA27 if unavailable
    linear_solver_options: dict[str, Any] = None

    # Barrier parameter options
    mu_strategy: str = "monotone"  # "monotone", "adaptive" - monotone provides better control
    mu_init: float = 1e-2  # Reduced from 0.1 for better initial convergence
    mu_max: float = 1e3  # Reduced from 1e5 to prevent getting stuck at maximum
    mu_min: float = 1e-11
    mu_linear_decrease_factor: float = 0.2  # Factor for linear decrease in monotone mode
    barrier_tol_factor: float = 10.0  # Factor for barrier complementarity tolerance

    # Line search options
    line_search_method: str = "filter"  # "filter", "cg-penalty", "cg-penalty-equality"

    # Convergence options
    dual_inf_tol: float = 1e-4
    compl_inf_tol: float = 1e-4
    constr_viol_tol: float = 1e-4

    # Output options
    print_level: int = 8  # 0=silent, 5=normal (iteration summary), 8+ includes detailed constraint residuals and convergence info
    print_frequency_iter: int = 1
    print_frequency_time: float = 5.0
    output_file: str | None = None
    print_timing_statistics: bool = False

    # Warm start options
    warm_start_init_point: str = "no"  # "no", "yes"
    warm_start_bound_push: float = 1e-6
    warm_start_mult_bound_push: float = 1e-6

    # Advanced options
    hessian_approximation: str = "limited-memory"  # "exact", "limited-memory"
    limited_memory_max_history: int = 6
    limited_memory_update_type: str = "bfgs"  # "bfgs", "sr1", "bfgs-powell"
    
    # Restoration phase options (for finding feasible initial point)
    bound_relax_factor: float = 0.0  # 0.0 = no relaxation, >0 = relax bounds to find feasible point
    expect_infeasible_problem: str = "no"  # "no" = disabled (scaling should make problems more feasible)
    soft_resto_pderror_reduction_factor: float = 0.9999  # Restoration phase tolerance
    required_infeasibility_reduction: float = 0.9  # Required reduction in infeasibility for restoration

    def __post_init__(self):
        if self.linear_solver_options is None:
            self.linear_solver_options = {}

    # Analysis options (optional)
    enable_analysis: bool = False


@dataclass
class IPOPTResult:
    """Result of IPOPT optimization."""

    success: bool
    x_opt: np.ndarray
    f_opt: float
    g_opt: np.ndarray
    lambda_opt: np.ndarray
    iterations: int
    cpu_time: float
    message: str
    status: int

    # Convergence information
    primal_inf: float
    dual_inf: float
    complementarity: float
    constraint_violation: float

    # Solution quality
    kkt_error: float
    feasibility_error: float


class IPOPTIterationCallback(ca.Callback):
    """Stream IPOPT iterates through CasADi's iteration_callback hook."""

    def __init__(
        self,
        reporter: StructuredReporter,
        n_vars: int,
        n_constraints: int,
        n_params: int,
        step: int = 1,
    ) -> None:
        self._reporter = reporter
        self.callback_step = max(1, int(step))
        self._n_vars = int(n_vars)
        self._n_constraints = int(n_constraints)
        self._n_params = int(n_params)
        self._lbx: np.ndarray | None = None
        self._ubx: np.ndarray | None = None
        self._lbg: np.ndarray | None = None
        self._ubg: np.ndarray | None = None
        self._prev_x: np.ndarray | None = None
        self._iteration = 0
        self._reported_failure = False
        self._names = NLPSOL_OUTPUT_NAMES
        sparsity_lookup = {
            "x": ca.Sparsity.dense(self._n_vars, 1),
            "f": ca.Sparsity.dense(1, 1),
            "g": ca.Sparsity.dense(self._n_constraints, 1),
            "lam_x": ca.Sparsity.dense(self._n_vars, 1),
            "lam_g": ca.Sparsity.dense(self._n_constraints, 1),
            "lam_p": ca.Sparsity.dense(self._n_params, 1),
        }
        self._sparsity_lookup = sparsity_lookup
        ca.Callback.__init__(self)
        self.construct("ipopt_iteration_callback", {"enable_fd": False})

    def get_n_in(self) -> int:  # noqa: D401
        return len(self._names)

    def get_n_out(self) -> int:  # noqa: D401
        return 1

    def get_sparsity_in(self, idx: int) -> ca.Sparsity:
        name = self._names[idx]
        return self._sparsity_lookup.get(name, ca.Sparsity.dense(0, 1))

    def get_sparsity_out(self, idx: int) -> ca.Sparsity:  # noqa: D401
        return ca.Sparsity.dense(1, 1)

    def update_bounds(
        self,
        lbx: np.ndarray | None,
        ubx: np.ndarray | None,
        lbg: np.ndarray | None,
        ubg: np.ndarray | None,
    ) -> None:
        """Attach the current bounds so violations can be measured."""
        self._lbx = self._flatten_or_none(lbx)
        self._ubx = self._flatten_or_none(ubx)
        self._lbg = self._flatten_or_none(lbg)
        self._ubg = self._flatten_or_none(ubg)

    def eval(self, args: list[Any]) -> list[int]:
        self._iteration += 1
        if (self._iteration - 1) % self.callback_step != 0:
            return [0]

        try:
            data = {
                name: self._flatten_or_none(arg)
                for name, arg in zip(self._names, args, strict=False)
            }
            x = data.get("x")
            g = data.get("g")
            lam_g = data.get("lam_g")
            obj_val = float(data.get("f")[0]) if data.get("f") is not None and data.get("f").size else float("nan")

            step_inf = self._compute_step_norm(x)
            primal_violation = self._compute_violation(g, self._lbg, self._ubg)
            bound_violation = self._compute_violation(x, self._lbx, self._ubx)
            lambda_inf = (
                float(np.max(np.abs(lam_g))) if lam_g is not None and lam_g.size else 0.0
            )

            # Use info level for iteration output so it's always visible
            self._reporter.info(
                "[iter] "
                f"k={self._iteration:04d} "
                f"obj={obj_val: .3e} "
                f"step_inf={step_inf: .2e} "
                f"g_violation={primal_violation: .2e} "
                f"bound_violation={bound_violation: .2e} "
                f"|lam_g|_inf={lambda_inf: .2e}"
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            if not self._reported_failure:
                self._reporter.debug(f"Iteration diagnostics unavailable: {exc}")
                self._reported_failure = True

        return [0]

    def _compute_step_norm(self, x: np.ndarray | None) -> float:
        if x is None or x.size == 0:
            return 0.0
        if self._prev_x is None or self._prev_x.size != x.size:
            self._prev_x = x.copy()
            return 0.0
        delta = np.abs(x - self._prev_x)
        self._prev_x = x.copy()
        return float(delta.max()) if delta.size else 0.0

    @staticmethod
    def _flatten_or_none(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
        except Exception:
            return None
        return arr

    @staticmethod
    def _compute_violation(
        values: np.ndarray | None,
        lower: np.ndarray | None,
        upper: np.ndarray | None,
    ) -> float:
        if values is None or values.size == 0:
            return 0.0
        vec = values
        if lower is None or upper is None:
            return float(np.max(np.abs(vec))) if vec.size else 0.0
        lb_violation = np.maximum(0.0, lower - vec)
        ub_violation = np.maximum(0.0, vec - upper)
        violation = np.maximum(lb_violation, ub_violation)
        return float(np.max(violation)) if violation.size else 0.0


class IPOPTSolver:
    """
    IPOPT solver wrapper for large-scale nonlinear optimization.

    This class provides a Python interface to the IPOPT solver for solving
    the collocation-based NLP problems in the OP engine optimization.
    """

    def __init__(self, options: IPOPTOptions | None = None):
        """
        Initialize IPOPT solver.

        Args:
            options: IPOPT solver options
        """
        self.options = options or IPOPTOptions()
        self._check_ipopt_availability()

    def _check_ipopt_availability(self) -> None:
        """Check if IPOPT is available."""
        try:
            import casadi as ca

            # Prefer a direct instantiation probe to handle builds without nlpsol_plugins
            try:
                from campro.optimization.ipopt_factory import create_ipopt_solver

                nlp = {"x": ca.SX.sym("x"), "f": 0, "g": ca.SX([])}
                _ = create_ipopt_solver("probe", nlp, linear_solver="ma27")
                self.ipopt_available = True
                return
            except Exception:
                # Fallback to plugin list if available
                if hasattr(ca, "nlpsol_plugins"):
                    try:
                        plugins = ca.nlpsol_plugins()
                        self.ipopt_available = "ipopt" in plugins
                        if not self.ipopt_available:
                            log.warning(
                                "IPOPT not available in CasADi plugins. Using alternative solver.",
                            )
                        return
                    except Exception:
                        pass
                log.warning("IPOPT not available in this CasADi build.")
                self.ipopt_available = False
        except ImportError:
            log.warning("CasADi not available. IPOPT integration disabled.")
            self.ipopt_available = False

    def solve(
        self,
        nlp: Any,  # CasADi NLP object
        x0: np.ndarray | None = None,
        lbx: np.ndarray | None = None,
        ubx: np.ndarray | None = None,
        lbg: np.ndarray | None = None,
        ubg: np.ndarray | None = None,
        p: np.ndarray | None = None,
    ) -> IPOPTResult:
        """
        Solve NLP using IPOPT.

        Args:
            nlp: CasADi NLP object
            x0: Initial guess for variables
            lbx: Lower bounds on variables
            ubx: Upper bounds on variables
            lbg: Lower bounds on constraints
            ubg: Upper bounds on constraints
            p: Parameters

        Returns:
            IPOPT result
        """
        if not self.ipopt_available:
            raise RuntimeError(
                "IPOPT solver is not available in this environment. "
                "Ensure CasADi is built with IPOPT support or install the IPOPT binaries.",
            )

        start_time = time.time()
        solver_create_start = time.time()
        reporter = StructuredReporter(
            context="IPOPT",
            logger=None,
            stream_out=sys.stderr,
            stream_err=sys.stderr,
            debug_env="IPOPT_DEBUG",
            force_debug=True,
        )

        try:
            # Infer problem dimensions up-front so diagnostic hooks can be configured
            n_vars, n_constraints, n_params = self._infer_dimensions(nlp)

            # Iteration streaming: enabled by default for verbose output (every iteration)
            # Can be disabled via FREE_PISTON_IPOPT_TRACE=false or controlled via FREE_PISTON_IPOPT_TRACE_STEP
            iteration_callback: IPOPTIterationCallback | None = None
            trace_enabled = _env_flag("FREE_PISTON_IPOPT_TRACE", default=True)  # Default to True for verbose output
            if trace_enabled:
                step_raw = os.environ.get("FREE_PISTON_IPOPT_TRACE_STEP", "1")  # Default to every iteration
                try:
                    callback_step = max(1, int(step_raw))
                except Exception:
                    callback_step = 1
                iteration_callback = IPOPTIterationCallback(
                    reporter=reporter,
                    n_vars=n_vars,
                    n_constraints=n_constraints,
                    n_params=n_params,
                    step=callback_step,
                )
                reporter.debug(
                    f"IPOPT iteration streaming enabled every {callback_step} iteration(s)"
                )

            # Create IPOPT solver
            solver = self._create_solver(nlp, iteration_callback=iteration_callback)
            solver_create_elapsed = time.time() - solver_create_start
            reporter.info(f"Solver created in {solver_create_elapsed:.3f}s")

            reporter.info(
                f"Beginning solve: max_iter={self.options.max_iter}, tol={self.options.tol:.2e}, "
                f"print_level={self.options.print_level}, n_vars={n_vars}, n_constraints={n_constraints}"
            )

            # Set initial guess and bounds
            if x0 is None:
                x0 = np.zeros(n_vars)
            if lbx is None:
                lbx = -np.inf * np.ones_like(x0)
            if ubx is None:
                ubx = np.inf * np.ones_like(x0)
            if lbg is None:
                lbg = -np.inf * np.ones(n_constraints)
            if ubg is None:
                ubg = np.inf * np.ones(n_constraints)
            if p is None:
                p = np.array([])

            # Try warm-start persistence if enabled
            warm_kwargs: dict[str, Any] = {}
            try:
                if str(self.options.warm_start_init_point).lower() == "yes":
                    from campro.diagnostics.warmstart import (
                        load_warmstart,
                        save_warmstart,
                    )

                    n_x = int(np.asarray(x0).size)
                    n_g = (
                        int(np.asarray(lbg).size)
                        if lbg is not None
                        else (int(np.asarray(ubg).size) if ubg is not None else 0)
                    )
                    warm_kwargs = load_warmstart(n_x, n_g)
                    if warm_kwargs:
                        reporter.info("Using warm start from previous solution")
            except Exception:
                warm_kwargs = {}

            # Log initial guess characteristics
            if x0 is not None and len(x0) > 0:
                reporter.info(
                    f"Initial guess: range=[{x0.min():.3e}, {x0.max():.3e}], "
                    f"mean={x0.mean():.3e}, std={x0.std():.3e}"
                )

            # Solve
            reporter.info("Calling IPOPT solver...")
            reporter.info(
                f"Note: Solving large problem (n_vars={n_vars}, n_constraints={n_constraints}). "
                f"This may take several minutes. IPOPT is running now (print_level={self.options.print_level})."
            )
            reporter.info(
                f"IPOPT output will appear below. If no progress is visible, "
                f"consider increasing print_level (current={self.options.print_level}) for more verbose output."
            )
            if iteration_callback is not None:
                iteration_callback.update_bounds(lbx, ubx, lbg, ubg)
            solve_call_start = time.time()
            # Flush all buffers before blocking call
            sys.stderr.flush()
            sys.stdout.flush()
            try:
                result = solver(
                    x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p, **warm_kwargs,
                )
                solve_call_elapsed = time.time() - solve_call_start
                # Flush after return
                sys.stderr.flush()
                sys.stdout.flush()
                reporter.info(f"Solver call returned after {solve_call_elapsed:.3f}s")
            except Exception as solve_exc:
                solve_call_elapsed = time.time() - solve_call_start
                sys.stderr.flush()
                sys.stdout.flush()
                reporter.error(
                    f"ERROR: Solver call raised exception after {solve_call_elapsed:.3f}s: {solve_exc}"
                )
                raise

            iter_count = int(result.get("iter_count", 0))
            status_flag = int(result.get("return_status", -1))
            elapsed = time.time() - start_time
            reporter.info(
                f"Completed solve in {elapsed:.3f}s (solver call: {solve_call_elapsed:.3f}s): "
                f"iter={iter_count}, status={status_flag}"
            )

            # Extract solution
            x_opt = result["x"].full().flatten()
            f_opt = float(result["f"])
            g_opt = result["g"].full().flatten()
            lambda_opt = result["lam_g"].full().flatten()

            # Get solver statistics
            stats = solver.stats()
            iterations = stats["iter_count"]
            cpu_time = time.time() - start_time

            # Check convergence
            success = stats["success"]
            status = stats["return_status"]

            # Compute solution quality metrics
            primal_inf = stats.get("primal_inf", 0.0)
            dual_inf = stats.get("dual_inf", 0.0)
            complementarity = stats.get("complementarity", 0.0)
            constraint_violation = stats.get("constraint_violation", 0.0)
            
            # Log detailed convergence information
            reporter.info(
                "Solve statistics: "
                f"success={success}, iterations={iterations}, cpu_time={cpu_time:.3f}s, "
                f"primal_inf={primal_inf:.2e}, dual_inf={dual_inf:.2e}, "
                f"complementarity={complementarity:.2e}, constraint_violation={constraint_violation:.2e}"
            )
            reporter.debug(f"Problem size: n_vars={n_vars}, n_constraints={n_constraints}")

            # Compute KKT error
            kkt_error = self._compute_kkt_error(nlp, x_opt, lambda_opt, p)
            feasibility_error = self._compute_feasibility_error(g_opt, lbg, ubg)

            message = self._get_status_message(status)

            out = IPOPTResult(
                success=success,
                x_opt=x_opt,
                f_opt=f_opt,
                g_opt=g_opt,
                lambda_opt=lambda_opt,
                iterations=iterations,
                cpu_time=cpu_time,
                message=message,
                status=status,
                primal_inf=primal_inf,
                dual_inf=dual_inf,
                complementarity=complementarity,
                constraint_violation=constraint_violation,
                kkt_error=kkt_error,
                feasibility_error=feasibility_error,
            )

            # Save warm-start for subsequent runs
            try:
                if str(self.options.warm_start_init_point).lower() == "yes":
                    from campro.diagnostics.warmstart import save_warmstart

                    # Multipliers for constraints are available as lam_g; lam_x may be present too
                    lam_g = result.get("lam_g", None)
                    lam_x = result.get("lam_x", None)
                    lam_g_arr = None if lam_g is None else lam_g.full().flatten()
                    lam_x_arr = None if lam_x is None else lam_x.full().flatten()
                    save_warmstart(x_opt, lam_g=lam_g_arr, lam_x=lam_x_arr)
            except Exception:
                pass

            return out

        except Exception as e:
            elapsed = time.time() - start_time
            reporter.error(f"ERROR: IPOPT solve failed after {elapsed:.3f}s: {e!s}")
            log.error(f"IPOPT solve failed: {e!s}", exc_info=True)
            return self._create_error_result(str(e), elapsed)

    def _create_solver(
        self,
        nlp: Any,
        iteration_callback: IPOPTIterationCallback | None = None,
    ) -> Any:
        """Create IPOPT solver with options."""

        # Convert options to CasADi format
        opts = self._convert_options()

        if iteration_callback is not None:
            opts["iteration_callback"] = iteration_callback
            opts["iteration_callback_step"] = max(1, iteration_callback.callback_step)

        # Use centralized factory with explicit linear solver
        from campro.optimization.ipopt_factory import create_ipopt_solver

        solver = create_ipopt_solver(
            "solver",
            nlp,
            opts,
            linear_solver=self.options.linear_solver,
        )

        return solver

    def _convert_options(self) -> dict[str, Any]:
        """Convert IPOPTOptions to CasADi format."""
        opts = {}

        # Basic options
        opts["ipopt.max_iter"] = self.options.max_iter
        opts["ipopt.max_cpu_time"] = self.options.max_cpu_time
        opts["ipopt.tol"] = self.options.tol
        opts["ipopt.acceptable_tol"] = self.options.acceptable_tol
        opts["ipopt.acceptable_iter"] = self.options.acceptable_iter

        # Note: linear_solver is set by the IPOPT factory
        # HSL library path is also set by the factory

        # Barrier parameter
        opts["ipopt.mu_strategy"] = self.options.mu_strategy
        opts["ipopt.mu_init"] = self.options.mu_init
        opts["ipopt.mu_max"] = self.options.mu_max
        opts["ipopt.mu_min"] = self.options.mu_min
        if hasattr(self.options, "mu_linear_decrease_factor"):
            opts["ipopt.mu_linear_decrease_factor"] = self.options.mu_linear_decrease_factor
        if hasattr(self.options, "barrier_tol_factor"):
            opts["ipopt.barrier_tol_factor"] = self.options.barrier_tol_factor

        # Line search
        opts["ipopt.line_search_method"] = self.options.line_search_method

        # Convergence
        opts["ipopt.dual_inf_tol"] = self.options.dual_inf_tol
        opts["ipopt.compl_inf_tol"] = self.options.compl_inf_tol
        opts["ipopt.constr_viol_tol"] = self.options.constr_viol_tol

        # Output
        opts["ipopt.print_level"] = self.options.print_level
        opts["ipopt.print_frequency_iter"] = self.options.print_frequency_iter
        opts["ipopt.print_frequency_time"] = self.options.print_frequency_time

        # Handle analysis output options
        if self.options.output_file:
            opts["ipopt.output_file"] = self.options.output_file
        elif getattr(self.options, "enable_analysis", False):
            # Route detailed Ipopt output to a timestamped file for analysis
            log_dir = Path(IPOPT_LOG_DIR)
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = str(log_dir / f"ipopt_{ts}.log")
            opts["ipopt.output_file"] = out_file

        if self.options.print_timing_statistics:
            opts["ipopt.print_timing_statistics"] = "yes"

        # Warm start
        opts["ipopt.warm_start_init_point"] = self.options.warm_start_init_point
        opts["ipopt.warm_start_bound_push"] = self.options.warm_start_bound_push
        opts["ipopt.warm_start_mult_bound_push"] = (
            self.options.warm_start_mult_bound_push
        )

        # Advanced options
        opts["ipopt.hessian_approximation"] = self.options.hessian_approximation
        opts["ipopt.limited_memory_max_history"] = (
            self.options.limited_memory_max_history
        )
        opts["ipopt.limited_memory_update_type"] = (
            self.options.limited_memory_update_type
        )
        
        # Restoration phase options (help find feasible initial point)
        opts["ipopt.bound_relax_factor"] = self.options.bound_relax_factor
        opts["ipopt.expect_infeasible_problem"] = self.options.expect_infeasible_problem
        opts["ipopt.soft_resto_pderror_reduction_factor"] = (
            self.options.soft_resto_pderror_reduction_factor
        )
        if hasattr(self.options, "required_infeasibility_reduction"):
            opts["ipopt.required_infeasibility_reduction"] = (
                self.options.required_infeasibility_reduction
            )

        # Add linear solver options if provided
        solver_specific_options = self.options.linear_solver_options or {}
        for key, value in solver_specific_options.items():
            opts[f"ipopt.{key}"] = value

        # Log configuration for debugging
        log.debug(
            f"IPOPT options configured: linear_solver={self.options.linear_solver}, "
            f"max_iter={self.options.max_iter}, tol={self.options.tol}",
        )
        log.debug(f"Full IPOPT options dict: {opts}")

        return opts

    def _infer_dimensions(self, nlp: Any) -> tuple[int, int, int]:
        """Return (n_vars, n_constraints, n_params) for the NLP."""
        try:
            if isinstance(nlp, dict):
                n_vars = int(nlp["x"].size1())
                n_constraints = int(nlp["g"].size1())
                n_params = int(nlp.get("p", ca.SX()).size1()) if "p" in nlp else 0
                return n_vars, n_constraints, n_params

            x_index = ca.nlpsol_in().index("x0")
            p_index = ca.nlpsol_in().index("p")
            n_vars = int(nlp.size1_in(x_index))
            n_constraints = int(nlp.size1_out(0))
            n_params = int(nlp.size1_in(p_index))
            return n_vars, n_constraints, n_params
        except Exception:
            return 0, 0, 0

    def _compute_kkt_error(
        self,
        nlp: Any,
        x: np.ndarray,
        lambda_opt: np.ndarray,
        p: np.ndarray,
    ) -> float:
        """Compute KKT error for solution quality assessment."""
        try:
            # Evaluate gradient of objective
            grad_f = nlp.grad_f(x, p)["grad_f_x"].full().flatten()

            # Evaluate Jacobian of constraints
            jac_g = nlp.jac_g(x, p)["jac_g_x"].full()

            # Compute KKT error
            kkt_residual = grad_f - jac_g.T @ lambda_opt
            kkt_error = np.linalg.norm(kkt_residual)

            return kkt_error

        except Exception:
            return float("inf")

    def _compute_feasibility_error(
        self,
        g: np.ndarray,
        lbg: np.ndarray,
        ubg: np.ndarray,
    ) -> float:
        """Compute feasibility error."""
        # Compute constraint violations
        violations = np.maximum(0, g - ubg) + np.maximum(0, lbg - g)
        feasibility_error = np.linalg.norm(violations)

        return feasibility_error

    def _get_status_message(self, status: int) -> str:
        """Get human-readable status message."""
        status_messages = {
            0: "Solve succeeded",
            1: "Solved to acceptable level",
            2: "Infeasible problem detected",
            3: "Search direction becomes too small",
            4: "Diverging iterates",
            5: "User requested stop",
            6: "Feasible point found",
            -1: "Maximum number of iterations exceeded",
            -2: "Restoration failed",
            -3: "Error in step computation",
            -4: "Maximum CPU time exceeded",
            -10: "Not enough degrees of freedom",
            -11: "Invalid problem definition",
            -12: "Invalid option",
            -13: "Invalid number detected",
            -100: "Unrecoverable exception",
            -101: "Non-IPOPT exception thrown",
            -102: "Insufficient memory",
            -199: "Internal error",
        }

        return status_messages.get(status, f"Unknown status: {status}")

    def _create_error_result(self, error_message: str, cpu_time: float) -> IPOPTResult:
        """Create error result."""
        return IPOPTResult(
            success=False,
            x_opt=np.array([]),
            f_opt=float("inf"),
            g_opt=np.array([]),
            lambda_opt=np.array([]),
            iterations=0,
            cpu_time=cpu_time,
            message=f"Solve failed: {error_message}",
            status=-100,
            primal_inf=float("inf"),
            dual_inf=float("inf"),
            complementarity=float("inf"),
            constraint_violation=float("inf"),
            kkt_error=float("inf"),
            feasibility_error=float("inf"),
        )


def solve_with_ipopt(
    nlp: Any,
    options: IPOPTOptions | None = None,
    x0: np.ndarray | None = None,
    lbx: np.ndarray | None = None,
    ubx: np.ndarray | None = None,
    lbg: np.ndarray | None = None,
    ubg: np.ndarray | None = None,
    p: np.ndarray | None = None,
) -> IPOPTResult:
    """
    Convenience function to solve NLP with IPOPT.

    Args:
        nlp: CasADi NLP object
        options: IPOPT solver options
        x0: Initial guess for variables
        lbx: Lower bounds on variables
        ubx: Upper bounds on variables
        lbg: Lower bounds on constraints
        ubg: Upper bounds on constraints
        p: Parameters

    Returns:
        IPOPT result
    """
    solver = IPOPTSolver(options)
    return solver.solve(nlp, x0, lbx, ubx, lbg, ubg, p)


def create_ipopt_options_from_dict(options_dict: dict[str, Any]) -> IPOPTOptions:
    """
    Create IPOPTOptions from dictionary.

    Args:
        options_dict: Dictionary of options

    Returns:
        IPOPTOptions object
    """
    options = IPOPTOptions()

    # Update options from dictionary
    for key, value in options_dict.items():
        if hasattr(options, key):
            setattr(options, key, value)
        else:
            log.warning(f"Unknown IPOPT option: {key}")

    return options


def get_default_ipopt_options() -> IPOPTOptions:
    """Get default IPOPT options for OP engine optimization."""
    options = IPOPTOptions()

    # Optimize for OP engine problems
    options.max_iter = 5000
    options.tol = 1e-6
    options.acceptable_tol = 1e-4
    # Linear solver is configured via ipopt.opt
    options.mu_strategy = "adaptive"  # Use adaptive for better convergence with improved scaling
    options.mu_init = 1e-1  # Increased from 1e-2 for better initial stability
    options.mu_max = 1e4  # Increased from 1e3 to allow more barrier parameter growth
    options.line_search_method = "filter"
    options.print_level = 8  # Verbose output: includes detailed iteration info and convergence diagnostics
    options.hessian_approximation = "limited-memory"
    # Disable restoration phase (scaling should make problems more feasible)
    options.expect_infeasible_problem = "no"
    options.bound_relax_factor = 0.0  # No relaxation needed with proper scaling

    return options


def get_robust_ipopt_options() -> IPOPTOptions:
    """Get robust IPOPT options for difficult problems."""
    options = IPOPTOptions()

    # More conservative settings
    options.max_iter = 10000
    options.tol = 1e-5
    options.acceptable_tol = 1e-3
    # Linear solver is configured via ipopt.opt
    options.mu_strategy = "adaptive"  # Use adaptive for better convergence with improved scaling
    options.mu_init = 1e-1  # Increased from 1e-2 for better initial stability
    options.mu_max = 1e4  # Increased from 1e3 to allow more barrier parameter growth
    options.line_search_method = "cg-penalty"
    options.print_level = 8  # Verbose output: includes detailed iteration info and convergence diagnostics
    options.hessian_approximation = "exact"
    # Disable restoration phase (scaling should make problems more feasible)
    options.expect_infeasible_problem = "no"
    options.bound_relax_factor = 0.0  # No relaxation needed with proper scaling

    return options
