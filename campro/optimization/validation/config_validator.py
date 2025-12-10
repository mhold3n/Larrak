"""Configuration validation for OP engine optimization."""

from __future__ import annotations

from typing import Any

from campro.logging import get_logger

log = get_logger(__name__)


class ConfigValidator:
    """Comprehensive configuration validator for OP engine optimization."""

    def __init__(self):
        """Initialize configuration validator."""
        self.required_sections = [
            "geometry",
            "thermodynamics",
            "optimization",
            "bounds",
        ]

        self.required_geometry = [
            "bore",
            "stroke",
            "compression_ratio",
            "mass",
        ]

        self.required_thermo = [
            "gamma",
            "R",
            "cp",
        ]

        self.required_optimization = [
            "method",
            "tolerance",
            "max_iterations",
        ]

        self.required_bounds = [
            "p_min",
            "p_max",
            "T_min",
            "T_max",
            "gap_min",
            "v_max",
        ]

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Validation results dictionary
        """
        log.info("Starting configuration validation...")

        validation_results = {
            "success": False,
            "errors": [],
            "warnings": [],
            "missing_sections": [],
            "missing_parameters": [],
            "invalid_values": [],
            "recommendations": [],
        }

        try:
            # Check required sections
            self._validate_sections(config, validation_results)

            # Validate geometry parameters
            if "geometry" in config:
                self._validate_geometry(config["geometry"], validation_results)

            # Validate thermodynamics parameters
            if "thermodynamics" in config:
                self._validate_thermodynamics(
                    config["thermodynamics"], validation_results,
                )

            # Validate optimization parameters
            if "optimization" in config:
                self._validate_optimization(config["optimization"], validation_results)

            # Validate bounds parameters
            if "bounds" in config:
                self._validate_bounds(config["bounds"], validation_results)

            # Validate parameter relationships
            self._validate_parameter_relationships(config, validation_results)

            # Generate recommendations
            self._generate_recommendations(config, validation_results)

            # Determine overall success
            validation_results["success"] = (
                len(validation_results["errors"]) == 0
                and len(validation_results["missing_sections"]) == 0
                and len(validation_results["missing_parameters"]) == 0
            )

            if validation_results["success"]:
                log.info("Configuration validation successful")
            else:
                log.warning("Configuration validation failed")
                if validation_results["errors"]:
                    log.error(f"Configuration errors: {validation_results['errors']}")
                if validation_results["warnings"]:
                    log.warning(
                        f"Configuration warnings: {validation_results['warnings']}",
                    )

        except Exception as e:
            log.error(f"Configuration validation failed with exception: {e}")
            validation_results["errors"].append(f"Validation exception: {e!s}")

        return validation_results

    def _validate_sections(
        self, config: dict[str, Any], results: dict[str, Any],
    ) -> None:
        """Validate required configuration sections."""
        for section in self.required_sections:
            if section not in config:
                results["missing_sections"].append(section)
                results["errors"].append(f"Missing required section: {section}")
            elif not isinstance(config[section], dict):
                results["errors"].append(f"Section '{section}' must be a dictionary")

    def _validate_geometry(
        self, geometry: dict[str, Any], results: dict[str, Any],
    ) -> None:
        """Validate geometry parameters."""
        for param in self.required_geometry:
            if param not in geometry:
                results["missing_parameters"].append(f"geometry.{param}")
                results["errors"].append(
                    f"Missing required geometry parameter: {param}",
                )
            else:
                value = geometry[param]
                if not isinstance(value, (int, float)):
                    results["errors"].append(
                        f"Geometry parameter '{param}' must be numeric",
                    )
                elif value <= 0:
                    results["errors"].append(
                        f"Geometry parameter '{param}' must be positive",
                    )

        # Validate specific geometry constraints
        if "bore" in geometry and "stroke" in geometry:
            bore = geometry["bore"]
            stroke = geometry["stroke"]
            if bore <= 0 or stroke <= 0:
                results["errors"].append("Bore and stroke must be positive")
            elif bore > stroke * 2:
                results["warnings"].append("Bore is unusually large compared to stroke")

        if "compression_ratio" in geometry:
            cr = geometry["compression_ratio"]
            if cr < 1.0:
                results["errors"].append("Compression ratio must be >= 1.0")
            elif cr > 20.0:
                results["warnings"].append("Compression ratio is unusually high")

        if "mass" in geometry:
            mass = geometry["mass"]
            if mass <= 0:
                results["errors"].append("Piston mass must be positive")
            elif mass > 100.0:  # kg
                results["warnings"].append("Piston mass is unusually high")

    def _validate_thermodynamics(
        self, thermo: dict[str, Any], results: dict[str, Any],
    ) -> None:
        """Validate thermodynamics parameters."""
        for param in self.required_thermo:
            if param not in thermo:
                results["missing_parameters"].append(f"thermodynamics.{param}")
                results["errors"].append(
                    f"Missing required thermodynamics parameter: {param}",
                )
            else:
                value = thermo[param]
                if not isinstance(value, (int, float)):
                    results["errors"].append(
                        f"Thermodynamics parameter '{param}' must be numeric",
                    )
                elif value <= 0:
                    results["errors"].append(
                        f"Thermodynamics parameter '{param}' must be positive",
                    )

        # Validate specific thermodynamics constraints
        if "gamma" in thermo:
            gamma = thermo["gamma"]
            if gamma <= 1.0:
                results["errors"].append("Heat capacity ratio (gamma) must be > 1.0")
            elif gamma > 2.0:
                results["warnings"].append(
                    "Heat capacity ratio (gamma) is unusually high",
                )

        if "R" in thermo:
            R = thermo["R"]
            if R < 100.0 or R > 1000.0:  # J/(kg·K)
                results["warnings"].append("Gas constant (R) is outside typical range")

        if "cp" in thermo:
            cp = thermo["cp"]
            if cp < 500.0 or cp > 2000.0:  # J/(kg·K)
                results["warnings"].append(
                    "Specific heat capacity (cp) is outside typical range",
                )

    def _validate_optimization(
        self, opt: dict[str, Any], results: dict[str, Any],
    ) -> None:
        """Validate optimization parameters."""
        for param in self.required_optimization:
            if param not in opt:
                results["missing_parameters"].append(f"optimization.{param}")
                results["errors"].append(
                    f"Missing required optimization parameter: {param}",
                )
            else:
                value = opt[param]
                if param == "method":
                    if not isinstance(value, str):
                        results["errors"].append("Optimization method must be a string")
                    elif value not in ["hllc", "enhanced_hllc", "roe"]:
                        results["warnings"].append(
                            f"Unknown optimization method: {value}",
                        )
                elif param in ["tolerance", "max_iterations"]:
                    if not isinstance(value, (int, float)):
                        results["errors"].append(
                            f"Optimization parameter '{param}' must be numeric",
                        )
                    elif value <= 0:
                        results["errors"].append(
                            f"Optimization parameter '{param}' must be positive",
                        )

        # Validate specific optimization constraints
        if "tolerance" in opt:
            tolerance = opt["tolerance"]
            if tolerance < 1e-12:
                results["warnings"].append(
                    "Tolerance is very small, may cause numerical issues",
                )
            elif tolerance > 1e-3:
                results["warnings"].append(
                    "Tolerance is large, may affect solution accuracy",
                )

        if "max_iterations" in opt:
            max_iter = opt["max_iterations"]
            if max_iter < 100:
                results["warnings"].append(
                    "Maximum iterations is low, may not converge",
                )
            elif max_iter > 10000:
                results["warnings"].append(
                    "Maximum iterations is high, may take long time",
                )

    def _validate_bounds(self, bounds: dict[str, Any], results: dict[str, Any]) -> None:
        """Validate bounds parameters."""
        for param in self.required_bounds:
            if param not in bounds:
                results["missing_parameters"].append(f"bounds.{param}")
                results["errors"].append(f"Missing required bounds parameter: {param}")
            else:
                value = bounds[param]
                if not isinstance(value, (int, float)):
                    results["errors"].append(
                        f"Bounds parameter '{param}' must be numeric",
                    )
                elif value <= 0:
                    results["errors"].append(
                        f"Bounds parameter '{param}' must be positive",
                    )

        # Validate bounds relationships
        if "p_min" in bounds and "p_max" in bounds:
            p_min = bounds["p_min"]
            p_max = bounds["p_max"]
            if p_min >= p_max:
                results["errors"].append("Pressure minimum must be less than maximum")
            elif p_max / p_min > 1000:
                results["warnings"].append("Pressure range is very large")

        if "T_min" in bounds and "T_max" in bounds:
            T_min = bounds["T_min"]
            T_max = bounds["T_max"]
            if T_min >= T_max:
                results["errors"].append(
                    "Temperature minimum must be less than maximum",
                )
            elif T_max / T_min > 10:
                results["warnings"].append("Temperature range is very large")

        if "gap_min" in bounds:
            gap_min = bounds["gap_min"]
            if gap_min < 0.0001:  # 0.1 mm
                results["warnings"].append("Minimum piston gap is very small")
            elif gap_min > 0.01:  # 10 mm
                results["warnings"].append("Minimum piston gap is large")

        if "v_max" in bounds:
            v_max = bounds["v_max"]
            if v_max < 10.0:  # m/s
                results["warnings"].append("Maximum velocity is low")
            elif v_max > 100.0:  # m/s
                results["warnings"].append("Maximum velocity is high")

    def _validate_parameter_relationships(
        self, config: dict[str, Any], results: dict[str, Any],
    ) -> None:
        """Validate relationships between parameters."""
        geometry = config.get("geometry", {})
        thermo = config.get("thermodynamics", {})
        bounds = config.get("bounds", {})

        # Check if pressure bounds are reasonable for the geometry
        if "bore" in geometry and "stroke" in geometry and "p_max" in bounds:
            bore = geometry["bore"]
            stroke = geometry["stroke"]
            p_max = bounds["p_max"]

            # Estimate maximum force
            area = 3.14159 * (bore / 2) ** 2
            max_force = p_max * area

            if max_force > 1e6:  # 1 MN
                results["warnings"].append(
                    "Maximum pressure may result in very high forces",
                )

        # Check if temperature bounds are reasonable for the thermodynamics
        if "gamma" in thermo and "T_max" in bounds:
            gamma = thermo["gamma"]
            T_max = bounds["T_max"]

            # Check if temperature is reasonable for the gas
            if gamma < 1.3 and T_max > 1500:
                results["warnings"].append("High temperature for low gamma gas")
            elif gamma > 1.4 and T_max < 500:
                results["warnings"].append("Low temperature for high gamma gas")

    def _generate_recommendations(
        self, config: dict[str, Any], results: dict[str, Any],
    ) -> None:
        """Generate configuration recommendations."""
        geometry = config.get("geometry", {})
        thermo = config.get("thermodynamics", {})
        opt = config.get("optimization", {})
        bounds = config.get("bounds", {})

        # Geometry recommendations
        if "bore" in geometry and "stroke" in geometry:
            bore = geometry["bore"]
            stroke = geometry["stroke"]
            ratio = bore / stroke
            if ratio < 0.5:
                results["recommendations"].append(
                    "Consider increasing bore/stroke ratio for better performance",
                )
            elif ratio > 2.0:
                results["recommendations"].append(
                    "Consider decreasing bore/stroke ratio for better efficiency",
                )

        # Thermodynamics recommendations
        if "gamma" in thermo:
            gamma = thermo["gamma"]
            if gamma < 1.3:
                results["recommendations"].append(
                    "Consider using a gas with higher gamma for better efficiency",
                )
            elif gamma > 1.4:
                results["recommendations"].append(
                    "Consider using a gas with lower gamma for better performance",
                )

        # Optimization recommendations
        if "method" in opt:
            method = opt["method"]
            if method == "hllc":
                results["recommendations"].append(
                    "Consider using 'enhanced_hllc' for better robustness",
                )

        # Bounds recommendations
        if "p_max" in bounds:
            p_max = bounds["p_max"]
            if p_max < 1e6:  # 10 bar
                results["recommendations"].append(
                    "Consider increasing maximum pressure for better performance",
                )
            elif p_max > 1e7:  # 100 bar
                results["recommendations"].append(
                    "Consider decreasing maximum pressure for safety",
                )

    def generate_validation_report(self, validation_results: dict[str, Any]) -> str:
        """Generate detailed validation report.

        Args:
            validation_results: Validation results dictionary

        Returns:
            Validation report text
        """
        report = []
        report.append("=" * 80)
        report.append("CONFIGURATION VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall status
        status = "✅ PASSED" if validation_results["success"] else "❌ FAILED"
        report.append(f"Overall Status: {status}")
        report.append("")

        # Missing sections
        if validation_results["missing_sections"]:
            report.append("MISSING SECTIONS:")
            report.append("-" * 40)
            for section in validation_results["missing_sections"]:
                report.append(f"❌ {section}")
            report.append("")

        # Missing parameters
        if validation_results["missing_parameters"]:
            report.append("MISSING PARAMETERS:")
            report.append("-" * 40)
            for param in validation_results["missing_parameters"]:
                report.append(f"❌ {param}")
            report.append("")

        # Invalid values
        if validation_results["invalid_values"]:
            report.append("INVALID VALUES:")
            report.append("-" * 40)
            for value in validation_results["invalid_values"]:
                report.append(f"❌ {value}")
            report.append("")

        # Errors
        if validation_results["errors"]:
            report.append("ERRORS:")
            report.append("-" * 40)
            for error in validation_results["errors"]:
                report.append(f"❌ {error}")
            report.append("")

        # Warnings
        if validation_results["warnings"]:
            report.append("WARNINGS:")
            report.append("-" * 40)
            for warning in validation_results["warnings"]:
                report.append(f"⚠️  {warning}")
            report.append("")

        # Recommendations
        if validation_results["recommendations"]:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 40)
            for rec in validation_results["recommendations"]:
                report.append(f"💡 {rec}")
            report.append("")

        report.append("=" * 80)
        report.append("END OF VALIDATION REPORT")
        report.append("=" * 80)

        return "\n".join(report)
