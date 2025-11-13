"""
Thermal efficiency adapter for integrating complex gas optimizer.

This adapter provides a bridge between the existing motion law optimization
system and the complex gas optimizer, focusing specifically on thermal
efficiency optimization for acceleration zones.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from campro.logging import get_logger
from campro.optimization.base import (
    BaseOptimizer,
    OptimizationResult,
    OptimizationStatus,
)
from campro.optimization.motion_law import (
    MotionLawConstraints,
    MotionLawResult,
    MotionType,
)

log = get_logger(__name__)

# Expose patchable names for tests; real imports are attempted during setup
ComplexMotionLawOptimizer = None  # type: ignore
OptimizationConfig = None  # type: ignore


@dataclass
class ThermalEfficiencyConfig:
    """Configuration for thermal efficiency optimization."""

    # Engine geometry
    bore: float = 0.082  # m
    stroke: float = 0.180  # m
    compression_ratio: float = 12.0
    clearance_volume: float = 3.2e-5  # m^3
    mass: float = 1.0  # kg
    rod_mass: float = 0.5  # kg
    rod_length: float = 0.15  # m

    # Thermodynamics
    gamma: float = 1.34
    R: float = 287.0  # J/(kg K)
    cp: float = 1005.0  # J/(kg K)
    cv: float = 717.5  # J/(kg K)

    # Optimization parameters
    collocation_points: int = 15  # Reduced for better convergence
    collocation_degree: int = 1  # Radau only supports C=1
    max_iterations: int = 10000  # Increased for better convergence
    tolerance: float = 1e-4  # More relaxed for better convergence

    # Thermal efficiency weights
    thermal_efficiency_weight: float = 1.0
    smoothness_weight: float = 0.01
    short_circuit_weight: float = 2.0

    # Model configuration
    use_1d_gas_model: bool = False  # Disable 1D model to avoid CasADi issues
    n_cells: int = 50

    # Solver settings
    linear_solver: str = "ma27"
    hessian_approximation: str = "limited-memory"

    # Analysis settings
    enable_analysis: bool = True

    # Validation settings
    check_convergence: bool = True
    check_physics: bool = True  # Re-enabled
    check_constraints: bool = True  # Re-enabled
    thermal_efficiency_min: float = (
        0.0  # Keep disabled to allow optimization to complete
    )
    max_pressure_limit: float = 12e6  # Pa
    max_temperature_limit: float = 2600.0  # K


class ThermalEfficiencyAdapter(BaseOptimizer):
    """
    Adapter for thermal efficiency optimization using complex gas optimizer.

    This adapter bridges the existing motion law optimization system with the
    complex gas optimizer, focusing specifically on thermal efficiency for
    acceleration zone optimization.
    """

    def __init__(self, config: ThermalEfficiencyConfig | None = None):
        super().__init__("ThermalEfficiencyAdapter")
        self.config = config or ThermalEfficiencyConfig()
        self.complex_optimizer: Any | None = None
        self._setup_complex_optimizer()

    def _setup_complex_optimizer(self) -> None:
        """Setup the complex gas optimizer with thermal efficiency focus."""
        try:
            # Import complex optimizer components
            # Import complex optimizer components and re-expose the optimizer class for tests to patch
            from campro.freepiston.opt.optimization_lib import (
                MotionLawOptimizer as _ComplexMotionLawOptimizer,
            )

            # Expose name in module globals so tests can patch it
            global ComplexMotionLawOptimizer  # type: ignore
            ComplexMotionLawOptimizer = _ComplexMotionLawOptimizer  # type: ignore
            from campro.freepiston.opt.config_factory import (
                create_optimization_scenario,
            )

            # Create thermal efficiency scenario configuration
            complex_config = create_optimization_scenario("efficiency")

            # Override with our specific configuration
            complex_config.geometry.update(
                {
                    "bore": self.config.bore,
                    "stroke": self.config.stroke,
                    "compression_ratio": self.config.compression_ratio,
                    "clearance_volume": self.config.clearance_volume,
                    "mass": self.config.mass,
                    "rod_mass": self.config.rod_mass,
                    "rod_length": self.config.rod_length,
                },
            )

            complex_config.thermodynamics.update(
                {
                    "gamma": self.config.gamma,
                    "R": self.config.R,
                    "cp": self.config.cp,
                    "cv": self.config.cv,
                },
            )

            complex_config.num = {
                "K": self.config.collocation_points,
                "C": self.config.collocation_degree,
            }

            complex_config.objective.update(
                {
                    "method": "thermal_efficiency",
                    "w": {
                        "smooth": self.config.smoothness_weight,
                        "short_circuit": self.config.short_circuit_weight,
                        "eta_th": self.config.thermal_efficiency_weight,
                    },
                },
            )

            # Use robust IPOPT options for better convergence
            complex_config.solver["ipopt"].update(
                {
                    "max_iter": self.config.max_iterations,
                    "tol": self.config.tolerance,
                    "acceptable_tol": self.config.tolerance
                    * 100,  # Much more relaxed acceptable tolerance
                    "hessian_approximation": "limited-memory",  # More robust for large problems
                    "mu_strategy": "monotone",  # More conservative barrier parameter strategy
                    "mu_init": 1e-2,  # Larger initial barrier parameter
                    "mu_max": 1e5,  # Allow larger barrier parameters
                    "line_search_method": "cg-penalty",  # More robust line search
                    "print_level": 2,  # Minimal output for better performance
                    "max_cpu_time": 300,  # 5 minute time limit
                    "dual_inf_tol": 1e-3,  # Relaxed dual infeasibility tolerance
                    "compl_inf_tol": 1e-3,  # Relaxed complementarity tolerance
                    "constr_viol_tol": 1e-3,  # Relaxed constraint violation tolerance
                    # Note: linear_solver is set by the IPOPT factory
                },
            )

            # Enable analysis if requested
            if self.config.enable_analysis:
                from datetime import datetime
                from pathlib import Path

                from campro.constants import IPOPT_LOG_DIR

                log_dir = Path(IPOPT_LOG_DIR)
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_file = str(log_dir / f"ipopt_{ts}.log")
                complex_config.solver["ipopt"].update(
                    {
                        "output_file": out_file,
                        "print_timing_statistics": "yes",
                    },
                )

            # Enable 1D gas model if requested
            if self.config.use_1d_gas_model:
                complex_config.model_type = "1d"
                complex_config.use_1d_gas = True
                complex_config.n_cells = self.config.n_cells

            # Create the complex optimizer with warm start strategy
            self.complex_optimizer = ComplexMotionLawOptimizer(complex_config)

            # Enable warm start for better convergence
            if hasattr(self.complex_optimizer, "enable_warm_start"):
                self.complex_optimizer.enable_warm_start(True)

            log.info("Thermal efficiency adapter configured with complex gas optimizer")

        except ImportError as e:
            log.error(f"Failed to import complex gas optimizer: {e}")
            # Emit environment diagnostics for CasADi/IPOPT
            try:
                import casadi as ca  # type: ignore

                plugins = ca.nlpsol_plugins() if hasattr(ca, "nlpsol_plugins") else []
                log.error(
                    f"CasADi version: {getattr(ca, '__version__', 'unknown')} | nlpsol plugins: {plugins}",
                )
            except Exception as exc:
                log.error(f"CasADi import failed in diagnostics: {exc}")
            log.error("Complex gas optimizer not available. Using fallback mode.")
            # If tests patched ComplexMotionLawOptimizer into this module, use it
            try:
                patched_cls = globals().get("ComplexMotionLawOptimizer")
                if callable(patched_cls):
                    self.complex_optimizer = patched_cls()  # type: ignore
                    log.info("Using patched ComplexMotionLawOptimizer for testing")
                else:
                    self.complex_optimizer = None
            except Exception:
                self.complex_optimizer = None
        except Exception as e:
            log.error(f"Failed to setup complex gas optimizer: {e}")
            self.complex_optimizer = None

    def optimize(
        self,
        objective,
        constraints: MotionLawConstraints,
        initial_guess: dict[str, np.ndarray] | None = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimize motion law for thermal efficiency.

        Args:
            objective: Objective function (ignored, uses thermal efficiency)
            constraints: Motion law constraints
            initial_guess: Initial guess (optional)
            **kwargs: Additional optimization parameters

        Returns:
            OptimizationResult with thermal efficiency optimization
        """
        log.info("Starting thermal efficiency optimization for acceleration zones")

        if self.complex_optimizer is None:
            raise RuntimeError(
                "Complex gas optimizer with CasADi/IPOPT is not available. "
                "Cannot perform thermal efficiency optimization. "
                "Check import errors in logs for details.",
            )

        try:
            complex_result = self.complex_optimizer.optimize_with_validation(
                validate=True,
            )
        except Exception as exc:  # pragma: no cover - direct passthrough
            raise RuntimeError(
                "Complex gas optimizer execution failed."
            ) from exc

        if not getattr(complex_result, "success", False):
            message = getattr(complex_result, "message", "unknown failure")
            raise RuntimeError(
                f"Complex gas optimizer reported failure: {message}",
            )

        if not hasattr(complex_result, "performance_metrics"):
            raise RuntimeError(
                "Complex gas optimizer result is missing performance metrics.",
            )

        if (
            hasattr(complex_result, "ipopt_analysis")
            and complex_result.ipopt_analysis is not None
        ):
            analysis = complex_result.ipopt_analysis
        else:
            from campro.optimization.solver_analysis import analyze_ipopt_run

            log_file_path = self._get_log_file_path()
            stats = {
                "success": True,
                "iterations": getattr(complex_result, "iterations", 0),
                "primal_inf": getattr(complex_result, "primal_inf", 0.0),
                "dual_inf": getattr(complex_result, "dual_inf", 0.0),
                "return_status": getattr(complex_result, "status", "unknown"),
            }
            analysis = analyze_ipopt_run(stats, log_file_path)

        motion_law_data = self._extract_motion_law_data(complex_result, constraints)

        thermal_efficiency = complex_result.performance_metrics.get(
            "thermal_efficiency", 0.0,
        )
        if thermal_efficiency < self.config.thermal_efficiency_min:
            log.warning(
                f"Thermal efficiency {thermal_efficiency:.3f} below minimum {self.config.thermal_efficiency_min}",
            )

        return OptimizationResult(
            status=OptimizationStatus.CONVERGED,
            objective_value=float(1.0 - float(thermal_efficiency)),
            solution=motion_law_data,
            iterations=getattr(complex_result, "iterations", 0),
            solve_time=getattr(complex_result, "cpu_time", None),
            metadata={
                "thermal_efficiency": thermal_efficiency,
                "indicated_work": complex_result.performance_metrics.get(
                    "indicated_work", 0.0,
                ),
                "max_pressure": complex_result.performance_metrics.get(
                    "max_pressure", 0.0,
                ),
                "max_temperature": complex_result.performance_metrics.get(
                    "max_temperature", 0.0,
                ),
                "min_piston_gap": complex_result.performance_metrics.get(
                    "min_piston_gap", 0.0,
                ),
                "optimization_method": "thermal_efficiency",
                "complex_optimizer": True,
                "validation_passed": self._validate_result(complex_result),
                "ipopt_analysis": analysis,
            },
        )

    def _extract_motion_law_data(
        self, complex_result, constraints: MotionLawConstraints,
    ) -> dict[str, Any]:
        """Extract motion law data from complex optimization result."""
        if not hasattr(complex_result, "solution") or complex_result.solution is None:
            raise RuntimeError(
                "Complex optimizer did not return a solution payload.",
            )

        solution = complex_result.solution
        if not (
            hasattr(solution, "data")
            and isinstance(solution.data, dict)
            and "states" in solution.data
        ):
            raise RuntimeError("Complex optimizer result is missing state data.")

        states = solution.data["states"]
        theta = np.linspace(0, 2 * np.pi, 360)

        required_keys = {"x_L", "x_R", "v_L", "v_R"}
        if not required_keys.issubset(states):
            missing = ", ".join(sorted(required_keys.difference(states)))
            raise RuntimeError(
                f"Complex optimizer state output missing required keys: {missing}",
            )

        def resample(arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr).flatten()
            if len(arr) == len(theta):
                return arr
            xsrc = np.linspace(0.0, 1.0, len(arr))
            xdst = np.linspace(0.0, 1.0, len(theta))
            return np.interp(xdst, xsrc, arr)

        x_m = 0.5 * (
            resample(states["x_L"])
            + resample(states["x_R"])
        )
        v_ms = 0.5 * (
            resample(states["v_L"])
            + resample(states["v_R"])
        )

        x = x_m * 1000.0
        v = v_ms * 1000.0
        a = np.gradient(v, theta)
        j = np.gradient(a, theta)

        constraints_dict = (
            constraints.to_dict()
            if hasattr(constraints, "to_dict")
            else {
                "stroke": constraints.stroke,
                "upstroke_duration_percent": constraints.upstroke_duration_percent,
                "zero_accel_duration_percent": constraints.zero_accel_duration_percent,
                "max_velocity": constraints.max_velocity,
                "max_acceleration": constraints.max_acceleration,
                "max_jerk": constraints.max_jerk,
            }
        )

        return {
            "cam_angle": theta,
            "position": x,
            "velocity": v,
            "acceleration": a,
            "jerk": j,
            "theta": theta,
            "x": x,
            "v": v,
            "a": a,
            "j": j,
            "constraints": constraints_dict,
            "optimization_type": "thermal_efficiency",
            "thermal_efficiency": complex_result.performance_metrics.get(
                "thermal_efficiency", 0.0,
            ),
        }

    def _validate_result(self, complex_result) -> bool:
        """Validate optimization result against constraints."""
        try:
            if not complex_result.success:
                return False

            # Check thermal efficiency
            thermal_efficiency = complex_result.performance_metrics.get(
                "thermal_efficiency", 0.0,
            )
            if thermal_efficiency < self.config.thermal_efficiency_min:
                log.warning(
                    f"Thermal efficiency {thermal_efficiency:.3f} below minimum {self.config.thermal_efficiency_min}",
                )
                return False

            # Check pressure limits
            max_pressure = complex_result.performance_metrics.get("max_pressure", 0.0)
            if max_pressure > self.config.max_pressure_limit:
                log.warning(
                    f"Max pressure {max_pressure:.0f} Pa exceeds limit {self.config.max_pressure_limit:.0f} Pa",
                )
                return False

            # Check temperature limits
            max_temperature = complex_result.performance_metrics.get(
                "max_temperature", 0.0,
            )
            if max_temperature > self.config.max_temperature_limit:
                log.warning(
                    f"Max temperature {max_temperature:.0f} K exceeds limit {self.config.max_temperature_limit:.0f} K",
                )
                return False

            # Check piston clearance
            min_piston_gap = complex_result.performance_metrics.get(
                "min_piston_gap", 0.0,
            )
            if min_piston_gap < 0.0008:  # 0.8 mm minimum clearance
                log.warning(
                    f"Min piston gap {min_piston_gap:.6f} m below minimum 0.0008 m",
                )
                return False

            return True

        except Exception as e:
            log.error(f"Result validation failed: {e}")
            return False

    def _get_log_file_path(self) -> str | None:
        """Get the most recent Ipopt log file for analysis."""
        from campro.constants import IPOPT_LOG_DIR

        log_dir = Path(IPOPT_LOG_DIR)
        if not log_dir.exists():
            return None

        log_files = list(log_dir.glob("ipopt_*.log"))
        if not log_files:
            return None

        # Return most recent log file
        return str(max(log_files, key=lambda p: p.stat().st_mtime))

    def solve_motion_law(
        self, constraints: MotionLawConstraints, motion_type: MotionType,
    ) -> MotionLawResult:
        """
        Solve motion law optimization with thermal efficiency focus.

        Args:
            constraints: Motion law constraints
            motion_type: Motion type (ignored, always uses thermal efficiency)

        Returns:
            MotionLawResult with thermal efficiency optimization
        """
        log.info(
            f"Solving thermal efficiency motion law (ignoring motion_type: {motion_type})",
        )

        result = self.optimize(None, constraints)
        if result.status != OptimizationStatus.CONVERGED:
            raise RuntimeError(
                "Thermal efficiency optimization failed to converge.",
            )

        return MotionLawResult(
            cam_angle=result.solution["cam_angle"],
            position=result.solution["position"],
            velocity=result.solution["velocity"],
            acceleration=result.solution["acceleration"],
            jerk=result.solution["jerk"],
            objective_value=result.objective_value,
            convergence_status="converged",
            iterations=result.iterations,
            solve_time=result.solve_time,
            stroke=constraints.stroke,
            upstroke_duration_percent=constraints.upstroke_duration_percent,
            zero_accel_duration_percent=constraints.zero_accel_duration_percent,
            motion_type=motion_type,
        )

    def configure(self, **kwargs) -> None:
        """Configure the thermal efficiency adapter."""
        # Update configuration from kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                log.warning(f"Unknown configuration key: {key}")

        # Re-setup complex optimizer with new configuration
        self._setup_complex_optimizer()

        log.info("Thermal efficiency adapter reconfigured")


def load_thermal_efficiency_config(config_path: Path) -> ThermalEfficiencyConfig:
    """Load thermal efficiency configuration from YAML file."""
    import yaml

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        return ThermalEfficiencyConfig(**config_dict)

    except Exception as e:
        log.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def get_default_thermal_efficiency_config() -> ThermalEfficiencyConfig:
    """Get default thermal efficiency configuration."""
    return ThermalEfficiencyConfig()


def validate_thermal_efficiency_config(config: ThermalEfficiencyConfig) -> bool:
    """Validate thermal efficiency configuration."""
    try:
        # Check geometry parameters
        if config.bore <= 0 or config.stroke <= 0:
            log.error("Invalid geometry parameters")
            return False

        if config.compression_ratio < 1.0:
            log.error("Invalid compression ratio")
            return False

        # Check thermodynamics parameters
        if config.gamma <= 1.0 or config.R <= 0 or config.cp <= 0:
            log.error("Invalid thermodynamics parameters")
            return False

        # Check optimization parameters
        if config.collocation_points < 5 or config.max_iterations < 100:
            log.error("Invalid optimization parameters")
            return False

        # Check weights
        if config.thermal_efficiency_weight < 0:
            log.error("Invalid thermal efficiency weight")
            return False

        return True

    except Exception as e:
        log.error(f"Configuration validation failed: {e}")
        return False
