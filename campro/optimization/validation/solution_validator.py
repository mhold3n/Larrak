"""Solution validation for OP engine optimization."""

from __future__ import annotations

import math
from typing import Any

from campro.logging import get_logger
from campro.optimization.core.solution import Solution

log = get_logger(__name__)


class SolutionValidator:
    """Comprehensive solution validator for OP engine optimization."""

    def __init__(self, config: dict[str, Any]):
        """Initialize solution validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.geometry = config.get("geometry", {})
        self.thermo = config.get("thermodynamics", {})
        self.bounds = config.get("bounds", {})

        # Default validation parameters
        self.default_limits = {
            "pressure_min": 1e3,  # 0.01 bar
            "pressure_max": 1e7,  # 100 bar
            "temperature_min": 200.0,  # K
            "temperature_max": 2000.0,  # K
            "piston_gap_min": 0.0008,  # 0.8 mm
            "velocity_max": 50.0,  # m/s
            "acceleration_max": 1000.0,  # m/s^2
            "scavenging_efficiency_min": 0.0,
            "scavenging_efficiency_max": 1.0,
            "trapping_efficiency_min": 0.0,
            "trapping_efficiency_max": 1.0,
        }

        # Override with config limits if available
        if "limits" in config:
            self.default_limits.update(config["limits"])

    def validate_solution(self, solution: Solution) -> dict[str, Any]:
        """Validate optimization solution.

        Args:
            solution: Optimization solution

        Returns:
            Validation results dictionary
        """
        log.info("Starting solution validation...")
        log.debug(
            f"Solution structure: meta keys={list(solution.meta.keys())}, data keys={list(solution.data.keys())}",
        )

        validation_results = {
            "success": False,
            "convergence": False,
            "physical_constraints": False,
            "scavenging_constraints": False,
            "performance_constraints": False,
            "warnings": [],
            "errors": [],
            "metrics": {},
            "violations": [],
        }

        try:
            # Check basic solution properties
            self._validate_basic_properties(solution, validation_results)

            # Validate convergence
            self._validate_convergence(solution, validation_results)

            # Validate physical constraints
            self._validate_physical_constraints(solution, validation_results)

            # Validate scavenging constraints
            self._validate_scavenging_constraints(solution, validation_results)

            # Validate performance constraints
            self._validate_performance_constraints(solution, validation_results)

            # Calculate validation metrics
            self._calculate_validation_metrics(solution, validation_results)

            # Determine overall success
            validation_results["success"] = (
                validation_results["convergence"]
                and validation_results["physical_constraints"]
                and validation_results["scavenging_constraints"]
                and validation_results["performance_constraints"]
                and len(validation_results["errors"]) == 0
            )

            if validation_results["success"]:
                log.info("Solution validation successful")
            else:
                log.warning("Solution validation failed")
                if validation_results["errors"]:
                    log.error(f"Validation errors: {validation_results['errors']}")
                if validation_results["warnings"]:
                    log.warning(
                        f"Validation warnings: {validation_results['warnings']}",
                    )

        except Exception as e:
            log.error(f"Solution validation failed with exception: {e}")
            validation_results["errors"].append(f"Validation exception: {e!s}")

        return validation_results

    def _validate_basic_properties(
        self,
        solution: Solution,
        results: dict[str, Any],
    ) -> None:
        """Validate basic solution properties."""
        # Check if solution has the expected structure
        if not hasattr(solution, "data") or not hasattr(solution, "meta"):
            results["errors"].append(
                "Solution missing required data or meta attributes",
            )
            return

        # For now, skip detailed state/control validation since the current Solution
        # structure doesn't include states and controls in the expected format
        # This is a placeholder for future implementation when the Solution structure
        # is updated to include states and controls
        results["warnings"].append(
            "Detailed state/control validation not implemented for current Solution structure",
        )

        # Basic validation of available data
        if not solution.data:
            results["warnings"].append("Solution data is empty")

        if not solution.meta:
            results["warnings"].append("Solution meta is empty")

        # Check if optimization was successful
        optimization_info = solution.meta.get("optimization", {})
        if not optimization_info.get("success", False):
            results["warnings"].append(
                f"Optimization did not converge: {optimization_info.get('message', 'Unknown error')}",
            )

    def _validate_convergence(
        self,
        solution: Solution,
        results: dict[str, Any],
    ) -> None:
        """Validate solution convergence."""
        # Check both property and metadata for success
        success = getattr(solution, "success", False)
        if not success:
            # Check IPOPT status codes in metadata
            opt_meta = solution.meta.get("optimization", {})
            status = opt_meta.get("status", -1)
            # Status 0 = Optimal, Status 1 = "Solved To Acceptable Level"
            if status in [0, 1]:
                success = True
                if status == 1:
                    results["warnings"].append(
                        "Converged to acceptable level (not optimal)",
                    )
                    log.info("Solution converged to acceptable level")
                else:
                    log.info("Solution converged optimally")
            else:
                results["warnings"].append(
                    f"Solution did not converge (status: {status})",
                )
                log.warning(f"Solution did not converge (status: {status})")

        results["convergence"] = success

        # Check iteration count (use property with fallback)
        iterations = getattr(solution, "iterations", 0)
        if iterations == 0:
            # Fallback to metadata
            iterations = solution.meta.get("optimization", {}).get("iterations", 0)

        max_iter = self.config.get("optimization", {}).get("max_iterations", 1000)
        if iterations >= max_iter:
            results["warnings"].append(f"Maximum iterations reached: {iterations}")

        # Check objective function value (use property with fallback)
        objective_value = getattr(solution, "objective_value", float("inf"))
        if objective_value == float("inf"):
            # Fallback to metadata
            objective_value = solution.meta.get("optimization", {}).get(
                "f_opt",
                float("inf"),
            )

        if math.isnan(objective_value):
            results["errors"].append("Objective function value is NaN")
        elif math.isinf(objective_value):
            results["warnings"].append("Objective function value is infinite")

    def _validate_physical_constraints(
        self,
        solution: Solution,
        results: dict[str, Any],
    ) -> None:
        """Validate physical constraints."""
        if not hasattr(solution, "states"):
            results["warnings"].append(
                "Physical constraints validation not implemented for current Solution structure",
            )
            return

        states = solution.states
        violations = []

        # Pressure constraints
        if "pressure" in states:
            pressures = states["pressure"]
            for i, p in enumerate(pressures):
                if p < self.default_limits["pressure_min"]:
                    violations.append(f"Low pressure at step {i}: {p:.0f} Pa")
                elif p > self.default_limits["pressure_max"]:
                    violations.append(f"High pressure at step {i}: {p:.0f} Pa")

        # Temperature constraints
        if "temperature" in states:
            temperatures = states["temperature"]
            for i, T in enumerate(temperatures):
                if self.default_limits["temperature_min"] > T:
                    violations.append(f"Low temperature at step {i}: {T:.0f} K")
                elif self.default_limits["temperature_max"] < T:
                    violations.append(f"High temperature at step {i}: {T:.0f} K")

        # Piston clearance constraints
        if "x_L" in states and "x_R" in states:
            x_L = states["x_L"]
            x_R = states["x_R"]
            for i, (x_l, x_r) in enumerate(zip(x_L, x_R)):
                gap = x_r - x_l
                if gap < self.default_limits["piston_gap_min"]:
                    violations.append(
                        f"Piston clearance violation at step {i}: {gap:.6f} m",
                    )

        # Velocity constraints
        if "v_L" in states and "v_R" in states:
            v_L = states["v_L"]
            v_R = states["v_R"]
            for i, (v_l, v_r) in enumerate(zip(v_L, v_R)):
                if abs(v_l) > self.default_limits["velocity_max"]:
                    violations.append(
                        f"High left piston velocity at step {i}: {v_l:.1f} m/s",
                    )
                if abs(v_r) > self.default_limits["velocity_max"]:
                    violations.append(
                        f"High right piston velocity at step {i}: {v_r:.1f} m/s",
                    )

        # Acceleration constraints (if available)
        if "a_L" in states and "a_R" in states:
            a_L = states["a_L"]
            a_R = states["a_R"]
            for i, (a_l, a_r) in enumerate(zip(a_L, a_R)):
                if abs(a_l) > self.default_limits["acceleration_max"]:
                    violations.append(
                        f"High left piston acceleration at step {i}: {a_l:.1f} m/s²",
                    )
                if abs(a_r) > self.default_limits["acceleration_max"]:
                    violations.append(
                        f"High right piston acceleration at step {i}: {a_r:.1f} m/s²",
                    )

        if violations:
            results["violations"].extend(violations)
            results["physical_constraints"] = False
            log.warning(f"Physical constraint violations: {len(violations)}")
        else:
            results["physical_constraints"] = True
            log.info("Physical constraints satisfied")

    def _validate_scavenging_constraints(
        self,
        solution: Solution,
        results: dict[str, Any],
    ) -> None:
        """Validate scavenging constraints."""
        # Check if scavenging state is available
        if hasattr(solution, "scavenging_state"):
            scav_state = solution.scavenging_state

            violations = []

            # Scavenging efficiency constraints
            if hasattr(scav_state, "eta_scavenging"):
                eta_scav = scav_state.eta_scavenging
                if eta_scav < self.default_limits["scavenging_efficiency_min"]:
                    violations.append(f"Low scavenging efficiency: {eta_scav:.3f}")
                elif eta_scav > self.default_limits["scavenging_efficiency_max"]:
                    violations.append(f"High scavenging efficiency: {eta_scav:.3f}")

            # Trapping efficiency constraints
            if hasattr(scav_state, "eta_trapping"):
                eta_trap = scav_state.eta_trapping
                if eta_trap < self.default_limits["trapping_efficiency_min"]:
                    violations.append(f"Low trapping efficiency: {eta_trap:.3f}")
                elif eta_trap > self.default_limits["trapping_efficiency_max"]:
                    violations.append(f"High trapping efficiency: {eta_trap:.3f}")

            # Short-circuit loss constraints
            if hasattr(scav_state, "eta_short_circuit"):
                eta_sc = scav_state.eta_short_circuit
                if eta_sc > 0.5:  # More than 50% short-circuit loss
                    violations.append(f"High short-circuit loss: {eta_sc:.3f}")

            if violations:
                results["violations"].extend(violations)
                results["scavenging_constraints"] = False
                log.warning(f"Scavenging constraint violations: {len(violations)}")
            else:
                results["scavenging_constraints"] = True
                log.info("Scavenging constraints satisfied")
        else:
            results["scavenging_constraints"] = True  # No scavenging data to validate
            log.info("No scavenging data available for validation")

    def _validate_performance_constraints(
        self,
        solution: Solution,
        results: dict[str, Any],
    ) -> None:
        """Validate performance constraints."""
        # Check if performance metrics are available (use property with fallback)
        perf_metrics = getattr(solution, "performance_metrics", {})
        if not perf_metrics:
            # Fallback to direct dict access
            perf_metrics = solution.meta.get("performance_metrics", {})

        if perf_metrics:
            violations = []

            # Check specific performance constraints
            if "power_output" in perf_metrics:
                power = perf_metrics["power_output"]
                if power < 0:
                    violations.append(f"Negative power output: {power:.1f} W")

            if "efficiency" in perf_metrics:
                efficiency = perf_metrics["efficiency"]
                if efficiency < 0 or efficiency > 1:
                    violations.append(f"Invalid efficiency: {efficiency:.3f}")

            if violations:
                results["violations"].extend(violations)
                results["performance_constraints"] = False
                log.warning(f"Performance constraint violations: {len(violations)}")
            else:
                results["performance_constraints"] = True
                log.info("Performance constraints satisfied")
        else:
            results["performance_constraints"] = True  # No performance data to validate
            log.info("No performance data available for validation")

    def _calculate_validation_metrics(
        self,
        solution: Solution,
        results: dict[str, Any],
    ) -> None:
        """Calculate validation metrics."""
        metrics = {}
        if not hasattr(solution, "states"):
            results["warnings"].append(
                "Validation metrics calculation not implemented for current Solution structure",
            )
            return

        states = solution.states

        # Pressure metrics
        if "pressure" in states:
            pressures = states["pressure"]
            metrics["pressure_max"] = max(pressures)
            metrics["pressure_min"] = min(pressures)
            metrics["pressure_avg"] = sum(pressures) / len(pressures)
            metrics["pressure_std"] = math.sqrt(
                sum((p - metrics["pressure_avg"]) ** 2 for p in pressures) / len(pressures),
            )

        # Temperature metrics
        if "temperature" in states:
            temperatures = states["temperature"]
            metrics["temperature_max"] = max(temperatures)
            metrics["temperature_min"] = min(temperatures)
            metrics["temperature_avg"] = sum(temperatures) / len(temperatures)
            metrics["temperature_std"] = math.sqrt(
                sum((T - metrics["temperature_avg"]) ** 2 for T in temperatures)
                / len(temperatures),
            )

        # Piston gap metrics
        if "x_L" in states and "x_R" in states:
            x_L = states["x_L"]
            x_R = states["x_R"]
            gaps = [x_r - x_l for x_l, x_r in zip(x_L, x_R)]
            metrics["piston_gap_max"] = max(gaps)
            metrics["piston_gap_min"] = min(gaps)
            metrics["piston_gap_avg"] = sum(gaps) / len(gaps)
            metrics["piston_gap_std"] = math.sqrt(
                sum((g - metrics["piston_gap_avg"]) ** 2 for g in gaps) / len(gaps),
            )

        # Velocity metrics
        if "v_L" in states and "v_R" in states:
            v_L = states["v_L"]
            v_R = states["v_R"]
            v_abs = [abs(v_l) for v_l in v_L] + [abs(v_r) for v_r in v_R]
            metrics["velocity_max"] = max(v_abs)
            metrics["velocity_avg"] = sum(v_abs) / len(v_abs)

        # Scavenging metrics
        if hasattr(solution, "scavenging_state"):
            scav_state = solution.scavenging_state
            if hasattr(scav_state, "eta_scavenging"):
                metrics["scavenging_efficiency"] = scav_state.eta_scavenging
            if hasattr(scav_state, "eta_trapping"):
                metrics["trapping_efficiency"] = scav_state.eta_trapping
            if hasattr(scav_state, "eta_blowdown"):
                metrics["blowdown_efficiency"] = scav_state.eta_blowdown
            if hasattr(scav_state, "eta_short_circuit"):
                metrics["short_circuit_loss"] = scav_state.eta_short_circuit

        results["metrics"] = metrics
        log.info(f"Validation metrics calculated: {len(metrics)} metrics")

    def generate_validation_report(self, validation_results: dict[str, Any]) -> str:
        """Generate detailed validation report.

        Args:
            validation_results: Validation results dictionary

        Returns:
            Validation report text
        """
        report = []
        report.append("=" * 80)
        report.append("SOLUTION VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall status
        status = "✅ PASSED" if validation_results["success"] else "❌ FAILED"
        report.append(f"Overall Status: {status}")
        report.append("")

        # Individual constraint status
        report.append("CONSTRAINT STATUS:")
        report.append("-" * 40)
        report.append(
            f"Convergence: {'✅' if validation_results['convergence'] else '❌'}",
        )
        report.append(
            f"Physical Constraints: {'✅' if validation_results['physical_constraints'] else '❌'}",
        )
        report.append(
            f"Scavenging Constraints: {'✅' if validation_results['scavenging_constraints'] else '❌'}",
        )
        report.append(
            f"Performance Constraints: {'✅' if validation_results['performance_constraints'] else '❌'}",
        )
        report.append("")

        # Metrics
        if validation_results["metrics"]:
            report.append("VALIDATION METRICS:")
            report.append("-" * 40)
            metrics = validation_results["metrics"]
            for key, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"{key}: {value:.6f}")
                else:
                    report.append(f"{key}: {value}")
            report.append("")

        # Violations
        if validation_results["violations"]:
            report.append("CONSTRAINT VIOLATIONS:")
            report.append("-" * 40)
            for violation in validation_results["violations"]:
                report.append(f"❌ {violation}")
            report.append("")

        # Warnings
        if validation_results["warnings"]:
            report.append("WARNINGS:")
            report.append("-" * 40)
            for warning in validation_results["warnings"]:
                report.append(f"⚠️  {warning}")
            report.append("")

        # Errors
        if validation_results["errors"]:
            report.append("ERRORS:")
            report.append("-" * 40)
            for error in validation_results["errors"]:
                report.append(f"❌ {error}")
            report.append("")

        report.append("=" * 80)
        report.append("END OF VALIDATION REPORT")
        report.append("=" * 80)

        return "\n".join(report)
