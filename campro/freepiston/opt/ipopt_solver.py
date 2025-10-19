from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from campro.constants import HSLLIB_PATH, IPOPT_LOG_DIR, IPOPT_OPT_PATH
from campro.logging import get_logger

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
    linear_solver: str = "ma27"  # kept for backward compatibility
    linear_solver_options: Dict[str, Any] = None

    # Barrier parameter options
    mu_strategy: str = "adaptive"  # "monotone", "adaptive"
    mu_init: float = 0.1
    mu_max: float = 1e5
    mu_min: float = 1e-11

    # Line search options
    line_search_method: str = "filter"  # "filter", "cg-penalty", "cg-penalty-equality"

    # Convergence options
    dual_inf_tol: float = 1e-4
    compl_inf_tol: float = 1e-4
    constr_viol_tol: float = 1e-4

    # Output options
    print_level: int = 5  # 0=silent, 5=normal, 12=verbose
    print_frequency_iter: int = 1
    print_frequency_time: float = 5.0
    output_file: Optional[str] = None
    print_timing_statistics: bool = False

    # Warm start options
    warm_start_init_point: str = "no"  # "no", "yes"
    warm_start_bound_push: float = 1e-6
    warm_start_mult_bound_push: float = 1e-6

    # Advanced options
    hessian_approximation: str = "limited-memory"  # "exact", "limited-memory"
    limited_memory_max_history: int = 6
    limited_memory_update_type: str = "bfgs"  # "bfgs", "sr1", "bfgs-powell"

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


class IPOPTSolver:
    """
    IPOPT solver wrapper for large-scale nonlinear optimization.
    
    This class provides a Python interface to the IPOPT solver for solving
    the collocation-based NLP problems in the OP engine optimization.
    """

    def __init__(self, options: Optional[IPOPTOptions] = None):
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
                _ = create_ipopt_solver("probe", nlp, force_linear_solver=True)
                self.ipopt_available = True
                return
            except Exception:
                # Fallback to plugin list if available
                if hasattr(ca, "nlpsol_plugins"):
                    try:
                        plugins = ca.nlpsol_plugins()
                        self.ipopt_available = "ipopt" in plugins
                        if not self.ipopt_available:
                            log.warning("IPOPT not available in CasADi plugins. Using alternative solver.")
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
        x0: Optional[np.ndarray] = None,
        lbx: Optional[np.ndarray] = None,
        ubx: Optional[np.ndarray] = None,
        lbg: Optional[np.ndarray] = None,
        ubg: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
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
            return self._fallback_solve(nlp, x0, lbx, ubx, lbg, ubg, p)

        start_time = time.time()

        try:
            # Create IPOPT solver
            solver = self._create_solver(nlp)

            # Set initial guess and bounds
            if x0 is None:
                x0 = np.zeros(nlp.size1_in(0))
            if lbx is None:
                lbx = -np.inf * np.ones_like(x0)
            if ubx is None:
                ubx = np.inf * np.ones_like(x0)
            if lbg is None:
                lbg = -np.inf * np.ones(nlp.size1_out(0))
            if ubg is None:
                ubg = np.inf * np.ones(nlp.size1_out(0))
            if p is None:
                p = np.array([])

            # Solve
            result = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p)

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

            # Compute KKT error
            kkt_error = self._compute_kkt_error(nlp, x_opt, lambda_opt, p)
            feasibility_error = self._compute_feasibility_error(g_opt, lbg, ubg)

            message = self._get_status_message(status)

            return IPOPTResult(
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

        except Exception as e:
            log.error(f"IPOPT solve failed: {e!s}")
            return self._create_error_result(str(e), time.time() - start_time)

    def _create_solver(self, nlp: Any) -> Any:
        """Create IPOPT solver with options."""
        import casadi as ca

        # Convert options to CasADi format
        opts = self._convert_options()

        # Ensure MA27 is always used - fail hard if not available
        if opts.get("ipopt.linear_solver") != "ma27":
            raise RuntimeError(
                f"Linear solver must be 'ma27', got '{opts.get('ipopt.linear_solver')}'. "
                "MUMPS fallback is not allowed for optimal performance."
            )

        # Create solver directly (temporarily bypass factory to debug)
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        return solver

    def _convert_options(self) -> Dict[str, Any]:
        """Convert IPOPTOptions to CasADi format."""
        opts = {}

        # Basic options
        opts["ipopt.max_iter"] = self.options.max_iter
        opts["ipopt.max_cpu_time"] = self.options.max_cpu_time
        opts["ipopt.tol"] = self.options.tol
        opts["ipopt.acceptable_tol"] = self.options.acceptable_tol
        opts["ipopt.acceptable_iter"] = self.options.acceptable_iter

        # Linear solver and HSL library (set once at creation time)
        opts["ipopt.linear_solver"] = self.options.linear_solver
        opts["ipopt.hsllib"] = HSLLIB_PATH

        # Barrier parameter
        opts["ipopt.mu_strategy"] = self.options.mu_strategy
        opts["ipopt.mu_init"] = self.options.mu_init
        opts["ipopt.mu_max"] = self.options.mu_max
        opts["ipopt.mu_min"] = self.options.mu_min

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
        opts["ipopt.warm_start_mult_bound_push"] = self.options.warm_start_mult_bound_push

        # Advanced options
        opts["ipopt.hessian_approximation"] = self.options.hessian_approximation
        opts["ipopt.limited_memory_max_history"] = self.options.limited_memory_max_history
        opts["ipopt.limited_memory_update_type"] = self.options.limited_memory_update_type

        # Add linear solver options if provided
        for key, value in self.options.linear_solver_options.items():
            opts[f"ipopt.{key}"] = value

        # Log configuration for debugging
        log.debug(f"IPOPT options configured: linear_solver={self.options.linear_solver}, "
                 f"max_iter={self.options.max_iter}, tol={self.options.tol}")
        log.debug(f"Full IPOPT options dict: {opts}")

        return opts

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

    def _fallback_solve(
        self,
        nlp: Any,
        x0: Optional[np.ndarray] = None,
        lbx: Optional[np.ndarray] = None,
        ubx: Optional[np.ndarray] = None,
        lbg: Optional[np.ndarray] = None,
        ubg: Optional[np.ndarray] = None,
        p: Optional[np.ndarray] = None,
    ) -> IPOPTResult:
        """Fallback solver when IPOPT is not available."""
        log.warning("IPOPT not available, using fallback solver")

        # Simple gradient descent fallback
        if x0 is None:
            x0 = np.zeros(10)  # Default size

        # Run simple optimization
        x_opt = x0.copy()
        f_opt = 0.0
        g_opt = np.zeros(10)
        lambda_opt = np.zeros(10)

        return IPOPTResult(
            success=False,
            x_opt=x_opt,
            f_opt=f_opt,
            g_opt=g_opt,
            lambda_opt=lambda_opt,
            iterations=0,
            cpu_time=0.0,
            message="IPOPT not available, using fallback solver",
            status=-1,
            primal_inf=float("inf"),
            dual_inf=float("inf"),
            complementarity=float("inf"),
            constraint_violation=float("inf"),
            kkt_error=float("inf"),
            feasibility_error=float("inf"),
        )

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
    options: Optional[IPOPTOptions] = None,
    x0: Optional[np.ndarray] = None,
    lbx: Optional[np.ndarray] = None,
    ubx: Optional[np.ndarray] = None,
    lbg: Optional[np.ndarray] = None,
    ubg: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
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


def create_ipopt_options_from_dict(options_dict: Dict[str, Any]) -> IPOPTOptions:
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
    options.mu_strategy = "adaptive"
    options.line_search_method = "filter"
    options.print_level = 3  # Reduced output
    options.hessian_approximation = "limited-memory"

    return options


def get_robust_ipopt_options() -> IPOPTOptions:
    """Get robust IPOPT options for difficult problems."""
    options = IPOPTOptions()

    # More conservative settings
    options.max_iter = 10000
    options.tol = 1e-5
    options.acceptable_tol = 1e-3
    # Linear solver is configured via ipopt.opt
    options.mu_strategy = "monotone"
    options.line_search_method = "cg-penalty"
    options.print_level = 5
    options.hessian_approximation = "exact"

    return options
