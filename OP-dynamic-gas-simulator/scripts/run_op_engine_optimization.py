#!/usr/bin/env python3
"""
Complete end-to-end integration script for OP engine motion-law optimization.

This script provides a comprehensive interface for running the complete
OP engine optimization pipeline, including:
- Configuration validation and setup
- 1D gas-structure coupled simulation
- Collocation NLP optimization
- Scavenging efficiency analysis
- Solution validation and reporting
- Visualization and post-processing

Usage:
    python run_op_engine_optimization.py [--config CONFIG_FILE] [--output OUTPUT_DIR] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from campro.freepiston.io.load import load_cfg
from campro.freepiston.io.save import save_solution
from campro.freepiston.opt.driver import solve_cycle
from campro.freepiston.opt.nlp import build_collocation_nlp_with_1d_coupling
from campro.freepiston.opt.solution import Solution
from campro.freepiston.zerod.cv import ScavengingState
from campro.logging import get_logger

log = get_logger(__name__)


class OPEngineOptimizationPipeline:
    """Complete OP engine optimization pipeline."""

    def __init__(self, config_path: Path, output_dir: Path, verbose: bool = False):
        """Initialize the optimization pipeline.

        Args:
            config_path: Path to configuration file
            output_dir: Output directory for results
            verbose: Enable verbose logging
        """
        self.config_path = config_path
        self.output_dir = output_dir
        self.verbose = verbose

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Load and validate configuration
        self.config = self._load_and_validate_config()

        # Initialize pipeline state
        self.solution: Optional[Solution] = None
        self.scavenging_state: Optional[ScavengingState] = None
        self.optimization_metrics: Dict[str, Any] = {}

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO

        # Create log file
        log_file = self.output_dir / "optimization.log"

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )

        log.info("OP Engine Optimization Pipeline initialized")
        log.info(f"Output directory: {self.output_dir}")
        log.info(f"Configuration file: {self.config_path}")

    def _load_and_validate_config(self) -> Dict[str, Any]:
        """Load and validate configuration file."""
        log.info("Loading configuration...")

        try:
            config = load_cfg(self.config_path)
        except Exception as e:
            log.error(f"Failed to load configuration: {e}")
            raise

        # Validate required configuration sections
        required_sections = ["geometry", "thermodynamics", "optimization"]
        for section in required_sections:
            if section not in config:
                log.error(f"Missing required configuration section: {section}")
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate geometry parameters
        geometry = config["geometry"]
        required_geometry = ["bore", "stroke", "compression_ratio", "mass"]
        for param in required_geometry:
            if param not in geometry:
                log.error(f"Missing required geometry parameter: {param}")
                raise ValueError(f"Missing required geometry parameter: {param}")

        # Validate thermodynamics parameters
        thermo = config["thermodynamics"]
        required_thermo = ["gamma", "R", "cp"]
        for param in required_thermo:
            if param not in thermo:
                log.error(f"Missing required thermodynamics parameter: {param}")
                raise ValueError(f"Missing required thermodynamics parameter: {param}")

        # Validate optimization parameters
        opt = config["optimization"]
        required_opt = ["method", "tolerance", "max_iterations"]
        for param in required_opt:
            if param not in opt:
                log.error(f"Missing required optimization parameter: {param}")
                raise ValueError(f"Missing required optimization parameter: {param}")

        log.info("Configuration validation successful")
        return config

    def _validate_solution(self, solution: Solution) -> Dict[str, Any]:
        """Validate optimization solution.

        Args:
            solution: Optimization solution

        Returns:
            Validation metrics
        """
        log.info("Validating solution...")

        validation_metrics = {
            "success": False,
            "convergence": False,
            "physical_constraints": False,
            "scavenging_metrics": {},
            "performance_metrics": {},
            "warnings": [],
            "errors": [],
        }

        try:
            # Check convergence
            if hasattr(solution, "success") and solution.success:
                validation_metrics["convergence"] = True
                log.info("Solution converged successfully")
            else:
                validation_metrics["warnings"].append("Solution did not converge")
                log.warning("Solution did not converge")

            # Check physical constraints
            if hasattr(solution, "states"):
                states = solution.states

                # Check pressure bounds
                if "pressure" in states:
                    pressures = states["pressure"]
                    max_pressure = max(pressures)
                    min_pressure = min(pressures)

                    if max_pressure > 1e7:  # 100 bar
                        validation_metrics["warnings"].append(
                            f"High pressure detected: {max_pressure:.0f} Pa",
                        )

                    if min_pressure < 1e3:  # 0.01 bar
                        validation_metrics["warnings"].append(
                            f"Low pressure detected: {min_pressure:.0f} Pa",
                        )

                # Check temperature bounds
                if "temperature" in states:
                    temperatures = states["temperature"]
                    max_temp = max(temperatures)
                    min_temp = min(temperatures)

                    if max_temp > 2000.0:  # K
                        validation_metrics["warnings"].append(
                            f"High temperature detected: {max_temp:.0f} K",
                        )

                    if min_temp < 200.0:  # K
                        validation_metrics["warnings"].append(
                            f"Low temperature detected: {min_temp:.0f} K",
                        )

                # Check piston clearance
                if "x_L" in states and "x_R" in states:
                    x_L = states["x_L"]
                    x_R = states["x_R"]
                    gaps = [x_R[i] - x_L[i] for i in range(len(x_L))]
                    min_gap = min(gaps)

                    if min_gap < 0.0008:  # 0.8 mm
                        validation_metrics["errors"].append(
                            f"Piston clearance violation: {min_gap:.6f} m",
                        )
                    else:
                        validation_metrics["physical_constraints"] = True

                validation_metrics["performance_metrics"] = {
                    "max_pressure": max_pressure if "pressure" in states else 0.0,
                    "max_temperature": max_temp if "temperature" in states else 0.0,
                    "min_piston_gap": min_gap
                    if "x_L" in states and "x_R" in states
                    else 0.0,
                }

            # Check scavenging metrics
            if self.scavenging_state:
                validation_metrics["scavenging_metrics"] = {
                    "scavenging_efficiency": self.scavenging_state.eta_scavenging,
                    "trapping_efficiency": self.scavenging_state.eta_trapping,
                    "blowdown_efficiency": self.scavenging_state.eta_blowdown,
                    "short_circuit_loss": self.scavenging_state.eta_short_circuit,
                }

            # Overall success
            validation_metrics["success"] = (
                validation_metrics["convergence"]
                and validation_metrics["physical_constraints"]
                and len(validation_metrics["errors"]) == 0
            )

            if validation_metrics["success"]:
                log.info("Solution validation successful")
            else:
                log.warning("Solution validation failed")
                if validation_metrics["errors"]:
                    log.error(f"Validation errors: {validation_metrics['errors']}")
                if validation_metrics["warnings"]:
                    log.warning(
                        f"Validation warnings: {validation_metrics['warnings']}",
                    )

        except Exception as e:
            log.error(f"Solution validation failed: {e}")
            validation_metrics["errors"].append(str(e))

        return validation_metrics

    def _generate_report(self, validation_metrics: Dict[str, Any]) -> str:
        """Generate optimization report.

        Args:
            validation_metrics: Validation metrics

        Returns:
            Report text
        """
        report = []
        report.append("=" * 80)
        report.append("OP ENGINE OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Configuration summary
        report.append("CONFIGURATION SUMMARY:")
        report.append("-" * 40)
        report.append(f"Configuration file: {self.config_path}")
        report.append(f"Output directory: {self.output_dir}")
        report.append(f"Optimization method: {self.config['optimization']['method']}")
        report.append(f"Tolerance: {self.config['optimization']['tolerance']}")
        report.append(
            f"Max iterations: {self.config['optimization']['max_iterations']}",
        )
        report.append("")

        # Solution validation
        report.append("SOLUTION VALIDATION:")
        report.append("-" * 40)
        report.append(f"Success: {'✓' if validation_metrics['success'] else '✗'}")
        report.append(
            f"Convergence: {'✓' if validation_metrics['convergence'] else '✗'}",
        )
        report.append(
            f"Physical constraints: {'✓' if validation_metrics['physical_constraints'] else '✗'}",
        )
        report.append("")

        # Performance metrics
        if validation_metrics["performance_metrics"]:
            report.append("PERFORMANCE METRICS:")
            report.append("-" * 40)
            perf = validation_metrics["performance_metrics"]
            report.append(f"Maximum pressure: {perf.get('max_pressure', 0.0):.0f} Pa")
            report.append(
                f"Maximum temperature: {perf.get('max_temperature', 0.0):.0f} K",
            )
            report.append(
                f"Minimum piston gap: {perf.get('min_piston_gap', 0.0):.6f} m",
            )
            report.append("")

        # Scavenging metrics
        if validation_metrics["scavenging_metrics"]:
            report.append("SCAVENGING METRICS:")
            report.append("-" * 40)
            scav = validation_metrics["scavenging_metrics"]
            report.append(
                f"Scavenging efficiency: {scav.get('scavenging_efficiency', 0.0):.3f}",
            )
            report.append(
                f"Trapping efficiency: {scav.get('trapping_efficiency', 0.0):.3f}",
            )
            report.append(
                f"Blowdown efficiency: {scav.get('blowdown_efficiency', 0.0):.3f}",
            )
            report.append(
                f"Short-circuit loss: {scav.get('short_circuit_loss', 0.0):.3f}",
            )
            report.append("")

        # Warnings and errors
        if validation_metrics["warnings"]:
            report.append("WARNINGS:")
            report.append("-" * 40)
            for warning in validation_metrics["warnings"]:
                report.append(f"⚠️  {warning}")
            report.append("")

        if validation_metrics["errors"]:
            report.append("ERRORS:")
            report.append("-" * 40)
            for error in validation_metrics["errors"]:
                report.append(f"❌ {error}")
            report.append("")

        # Optimization metrics
        if self.optimization_metrics:
            report.append("OPTIMIZATION METRICS:")
            report.append("-" * 40)
            for key, value in self.optimization_metrics.items():
                report.append(f"{key}: {value}")
            report.append("")

        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        return "\n".join(report)

    def _save_results(self, validation_metrics: Dict[str, Any]) -> None:
        """Save optimization results.

        Args:
            validation_metrics: Validation metrics
        """
        log.info("Saving results...")

        # Save solution
        if self.solution:
            solution_file = self.output_dir / "solution.pkl"
            save_solution(self.solution, solution_file)
            log.info(f"Solution saved to {solution_file}")

        # Save validation metrics
        validation_file = self.output_dir / "validation_metrics.json"
        with open(validation_file, "w") as f:
            json.dump(validation_metrics, f, indent=2)
        log.info(f"Validation metrics saved to {validation_file}")

        # Save optimization metrics
        if self.optimization_metrics:
            opt_file = self.output_dir / "optimization_metrics.json"
            with open(opt_file, "w") as f:
                json.dump(self.optimization_metrics, f, indent=2)
            log.info(f"Optimization metrics saved to {opt_file}")

        # Save configuration
        config_file = self.output_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        log.info(f"Configuration saved to {config_file}")

        # Generate and save report
        report = self._generate_report(validation_metrics)
        report_file = self.output_dir / "optimization_report.txt"
        with open(report_file, "w") as f:
            f.write(report)
        log.info(f"Report saved to {report_file}")

        # Print report to console
        print("\n" + report)

    def run_optimization(self) -> bool:
        """Run the complete optimization pipeline.

        Returns:
            True if optimization was successful, False otherwise
        """
        log.info("Starting OP engine optimization pipeline...")

        try:
            # Step 1: Setup optimization problem
            log.info("Setting up optimization problem...")
            start_time = time.time()

            # Build NLP with 1D coupling
            nlp, nlp_params = build_collocation_nlp_with_1d_coupling(
                self.config,
                use_1d_gas=True,  # Enable 1D gas model
                n_cells=50,  # Number of 1D cells
            )

            setup_time = time.time() - start_time
            self.optimization_metrics["setup_time"] = setup_time
            log.info(
                f"Optimization problem setup completed in {setup_time:.2f} seconds",
            )

            # Step 2: Run optimization
            log.info("Running optimization...")
            opt_start_time = time.time()

            # Set run directory in config
            self.config["run_dir"] = str(self.output_dir)

            # Solve the optimization problem
            self.solution = solve_cycle(self.config)

            opt_time = time.time() - opt_start_time
            self.optimization_metrics["optimization_time"] = opt_time
            log.info(f"Optimization completed in {opt_time:.2f} seconds")

            # Step 3: Validate solution
            log.info("Validating solution...")
            validation_metrics = self._validate_solution(self.solution)

            # Step 4: Save results
            self._save_results(validation_metrics)

            # Step 5: Generate summary
            total_time = time.time() - start_time
            self.optimization_metrics["total_time"] = total_time

            log.info(f"Optimization pipeline completed in {total_time:.2f} seconds")

            return validation_metrics["success"]

        except Exception as e:
            log.error(f"Optimization pipeline failed: {e}")
            return False


def main() -> None:
    """Main entry point for the optimization script."""
    parser = argparse.ArgumentParser(
        description="OP Engine Motion-Law Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default configuration
    python run_op_engine_optimization.py
    
    # Run with custom configuration
    python run_op_engine_optimization.py --config my_config.yaml --output results/
    
    # Run with verbose logging
    python run_op_engine_optimization.py --verbose
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("cfg/defaults.yaml"),
        help="Configuration file path (default: cfg/defaults.yaml)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("runs/op_optimization"),
        help="Output directory for results (default: runs/op_optimization)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Validate input arguments
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Run optimization pipeline
    pipeline = OPEngineOptimizationPipeline(
        config_path=args.config,
        output_dir=args.output,
        verbose=args.verbose,
    )

    success = pipeline.run_optimization()

    if success:
        print("\n✅ Optimization completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Optimization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
