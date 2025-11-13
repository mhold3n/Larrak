"""
Configuration Factory for Motion Law Optimization

This module provides factory functions and presets for creating optimization
configurations for different scenarios and use cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from campro.freepiston.opt.optimization_lib import OptimizationConfig
from campro.logging import get_logger

log = get_logger(__name__)


class ConfigFactory:
    """Factory for creating optimization configurations."""

    @staticmethod
    def create_default_config() -> OptimizationConfig:
        """Create a default optimization configuration."""
        return OptimizationConfig(
            geometry={
                "bore": 0.1,  # m
                "stroke": 0.1,  # m
                "compression_ratio": 10.0,
                "mass": 1.0,  # kg
                "rod_mass": 0.5,  # kg
                "rod_length": 0.15,  # m
                "rod_cg_offset": 0.075,  # m
                "clearance_volume": 1e-4,  # m^3
                "clearance_min": 0.001,  # m
                "friction_coefficient": 0.1,
                "damping_coefficient": 100.0,  # Ns/m
                "ring_count": 3,
                "ring_tension": 100.0,  # N
                "ring_width": 0.002,  # m
                "ring_friction_coefficient": 0.1,
                "T_wall": 400.0,  # K
                "p_intake": 1e5,  # Pa
                "T_intake": 300.0,  # K
                "p_exhaust": 1e5,  # Pa
            },
            thermodynamics={
                "gamma": 1.4,
                "R": 287.0,  # J/(kg K)
                "cp": 1005.0,  # J/(kg K)
                "cv": 717.5,  # J/(kg K)
            },
            bounds={
                "xL_min": -0.1,
                "xL_max": 0.1,
                "xR_min": 0.0,
                "xR_max": 0.2,
                "vL_min": -50.0,
                "vL_max": 50.0,
                "vR_min": -50.0,
                "vR_max": 50.0,
                "rho_min": 0.1,
                "rho_max": 10.0,
                "T_min": 200.0,
                "T_max": 2000.0,
                "p_min": 0.01,  # MPa (normalized from 1e4 Pa to match variable scaling reference)
                "p_max": 10.0,  # MPa (normalized from 1e7 Pa to match variable scaling reference)
                "Ain_max": 0.01,
                "Aex_max": 0.01,
                "Q_comb_max": 10.0,  # kJ (normalized from 10000.0 J to bring energy terms to O(1-10) range)
                "dA_dt_max": 0.02,
                "a_max": 1000.0,
                "x_gap_min": 0.0008,
            },
            constraints={
                "short_circuit_max": 0.1,
                "scavenging_min": 0.8,
                "trapping_min": 0.9,
            },
            num={"K": 20, "C": 1},
            solver={
                "ipopt": {
                    "max_iter": 5000,
                    "tol": 1e-6,
                    # creation-time options ensure HSL is initialized before reading file
                    "hessian_approximation": "limited-memory",
                    # Note: linear_solver is set by the IPOPT factory
                },
            },
            objective={
                "method": "indicated_work",
                "w": {
                    "smooth": 0.01,
                    "short_circuit": 1.0,
                    "eta_th": 0.1,
                },
            },
            combustion={
                "use_integrated_model": False,
                "fuel_type": "diesel",
                "afr": 18.0,
                "cycle_time_s": 0.02,
                "initial_temperature_K": 900.0,
                "initial_pressure_Pa": 1e5,
                "fuel_mass_kg": 5e-4,
                "target_mfb": 0.99,
                "m_wiebe": 2.0,
                "k_turb": 0.3,
                "c_burn": 3.0,
                "turbulence_exponent": 0.7,
                "min_flame_speed": 0.2,
                "ignition_initial_s": 0.005,
                "ignition_bounds_s": (0.001, 0.015),
                "w_ca50": 0.0,
                "ca50_target_deg": 0.0,
                "w_ca_duration": 0.0,
                "ca_duration_target_deg": 0.0,
                "ca_softening": 200.0,
            },
            validation={
                "check_convergence": True,
                "check_physics": True,
                "check_constraints": True,
            },
            output={
                "save_solution": True,
                "save_metrics": True,
                "generate_report": True,
            },
        )

    @staticmethod
    def create_high_performance_config() -> OptimizationConfig:
        """Create configuration for high-performance optimization."""
        config = ConfigFactory.create_default_config()

        # Higher resolution
        config.num = {"K": 50, "C": 1}  # Radau only supports C=1

        # More aggressive solver settings
        config.solver["ipopt"].update(
            {
                "max_iter": 5000,
                "tol": 1e-8,
                "hessian_approximation": "exact",
                # Note: linear_solver is set by the IPOPT factory
            },
        )

        # Stricter constraints
        config.constraints.update(
            {
                "short_circuit_max": 0.05,
                "scavenging_min": 0.9,
                "trapping_min": 0.95,
            },
        )

        return config

    @staticmethod
    def create_quick_test_config() -> OptimizationConfig:
        """Create configuration for quick testing."""
        config = ConfigFactory.create_default_config()

        # Lower resolution for speed
        config.num = {"K": 10, "C": 1}

        # Relaxed solver settings
        config.solver["ipopt"].update(
            {
                "max_iter": 500,
                "tol": 1e-4,
                # Note: linear_solver is set by the IPOPT factory
            },
        )

        # Relaxed constraints
        config.constraints.update(
            {
                "short_circuit_max": 0.2,
                "scavenging_min": 0.7,
                "trapping_min": 0.8,
            },
        )

        return config

    @staticmethod
    def create_1d_config(n_cells: int = 50) -> OptimizationConfig:
        """Create configuration for 1D gas model optimization."""
        config = ConfigFactory.create_default_config()

        config.model_type = "1d"
        config.use_1d_gas = True
        config.n_cells = n_cells

        # Adjust resolution for 1D model
        config.num = {"K": 30, "C": 1}

        # More conservative solver settings for 1D
        config.solver["ipopt"].update(
            {
                "max_iter": 2000,
                "tol": 1e-6,
                # Note: linear_solver is set by the IPOPT factory
            },
        )

        return config

    @staticmethod
    def create_robust_config() -> OptimizationConfig:
        """Create configuration for robust optimization."""
        config = ConfigFactory.create_default_config()

        # Conservative solver settings
        config.solver["ipopt"].update(
            {
                "max_iter": 10000,
                "tol": 1e-4,
                "hessian_approximation": "limited-memory",
                "mu_strategy": "adaptive",
                "mu_init": 1e-3,
                # Note: linear_solver is set by the IPOPT factory
            },
        )

        # Enable adaptive refinement
        config.refinement_strategy = "adaptive"
        config.max_refinements = 3

        return config

    @staticmethod
    def create_custom_config(**kwargs) -> OptimizationConfig:
        """Create custom configuration with overrides."""
        config = ConfigFactory.create_default_config()

        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                log.warning(f"Unknown configuration key: {key}")

        return config

    @staticmethod
    def from_yaml(file_path: str | Path) -> OptimizationConfig:
        """Load configuration from YAML file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path) as f:
                config_dict = yaml.safe_load(f)

            return ConfigFactory.from_dict(config_dict)

        except Exception as e:
            log.error(f"Failed to load configuration from {file_path}: {e}")
            raise

    @staticmethod
    def from_dict(config_dict: dict[str, Any]) -> OptimizationConfig:
        """Create configuration from dictionary."""
        return OptimizationConfig(
            geometry=config_dict.get("geometry", {}),
            thermodynamics=config_dict.get("thermodynamics", {}),
            bounds=config_dict.get("bounds", {}),
            constraints=config_dict.get("constraints", {}),
            num=config_dict.get("num", {"K": 20, "C": 1}),  # Radau only supports C=1
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

    @staticmethod
    def to_yaml(config: OptimizationConfig, file_path: str | Path) -> None:
        """Save configuration to YAML file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "geometry": config.geometry,
            "thermodynamics": config.thermodynamics,
            "bounds": config.bounds,
            "constraints": config.constraints,
            "num": config.num,
            "solver": config.solver,
            "objective": config.objective,
            "model_type": config.model_type,
            "use_1d_gas": config.use_1d_gas,
            "n_cells": config.n_cells,
            "validation": config.validation,
            "output": config.output,
            "warm_start": config.warm_start,
            "refinement_strategy": config.refinement_strategy,
            "max_refinements": config.max_refinements,
        }

        try:
            with open(file_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            log.info(f"Configuration saved to {file_path}")

        except Exception as e:
            log.error(f"Failed to save configuration to {file_path}: {e}")
            raise


# Preset configurations for common scenarios


def get_preset_config(preset_name: str) -> OptimizationConfig:
    """
    Get a preset configuration by name.

    Args:
        preset_name: Name of the preset ("default", "high_performance",
                    "quick_test", "1d", "robust")

    Returns:
        OptimizationConfig
    """
    presets = {
        "default": ConfigFactory.create_default_config,
        "high_performance": ConfigFactory.create_high_performance_config,
        "quick_test": ConfigFactory.create_quick_test_config,
        "1d": ConfigFactory.create_1d_config,
        "robust": ConfigFactory.create_robust_config,
    }

    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    return presets[preset_name]()


def create_engine_config(engine_type: str, **kwargs) -> OptimizationConfig:
    """
    Create configuration for specific engine types.

    Args:
        engine_type: Type of engine ("opposed_piston", "free_piston", "conventional")
        **kwargs: Additional configuration overrides

    Returns:
        OptimizationConfig
    """
    base_config = ConfigFactory.create_default_config()

    if engine_type == "opposed_piston":
        # Opposed piston specific settings
        base_config.geometry.update(
            {
                "bore": 0.12,
                "stroke": 0.08,
                "compression_ratio": 12.0,
                "mass": 1.2,
            },
        )
        base_config.bounds.update(
            {
                "xL_min": -0.05,
                "xL_max": 0.05,
                "xR_min": 0.05,
                "xR_max": 0.15,
            },
        )

    elif engine_type == "free_piston":
        # Free piston specific settings
        base_config.geometry.update(
            {
                "bore": 0.08,
                "stroke": 0.06,
                "compression_ratio": 15.0,
                "mass": 0.8,
            },
        )
        base_config.bounds.update(
            {
                "xL_min": -0.03,
                "xL_max": 0.03,
                "xR_min": 0.03,
                "xR_max": 0.09,
            },
        )

    elif engine_type == "conventional":
        # Conventional engine settings
        base_config.geometry.update(
            {
                "bore": 0.1,
                "stroke": 0.1,
                "compression_ratio": 10.0,
                "mass": 1.0,
            },
        )
        base_config.bounds.update(
            {
                "xL_min": -0.1,
                "xL_max": 0.1,
                "xR_min": 0.0,
                "xR_max": 0.2,
            },
        )

    else:
        raise ValueError(f"Unknown engine type: {engine_type}")

    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)

    return base_config


def create_optimization_scenario(scenario: str, **kwargs) -> OptimizationConfig:
    """
    Create configuration for specific optimization scenarios.

    Args:
        scenario: Optimization scenario ("efficiency", "power", "emissions", "durability")
        **kwargs: Additional configuration overrides

    Returns:
        OptimizationConfig
    """
    base_config = ConfigFactory.create_default_config()

    if scenario == "efficiency":
        # Focus on thermal efficiency
        base_config.objective = {
            "method": "thermal_efficiency",
            "w": {
                "smooth": 0.01,
                "short_circuit": 2.0,
                "eta_th": 1.0,
            },
        }
        base_config.constraints.update(
            {
                "short_circuit_max": 0.05,
                "scavenging_min": 0.9,
            },
        )

    elif scenario == "power":
        # Focus on power output
        base_config.objective = {
            "method": "indicated_work",
            "w": {
                "smooth": 0.005,
                "short_circuit": 0.5,
                "eta_th": 0.1,
            },
        }
        base_config.bounds.update(
            {
                "Q_comb_max": 20.0,  # kJ (normalized from 20000.0 J)
                "p_max": 20.0,  # MPa (normalized from 2e7 Pa)
            },
        )

    elif scenario == "emissions":
        # Focus on emissions reduction
        base_config.objective = {
            "method": "scavenging",
            "w": {
                "smooth": 0.02,
                "short_circuit": 3.0,
                "eta_th": 0.05,
            },
        }
        base_config.constraints.update(
            {
                "short_circuit_max": 0.02,
                "scavenging_min": 0.95,
                "trapping_min": 0.98,
            },
        )

    elif scenario == "durability":
        # Focus on mechanical durability
        base_config.objective = {
            "method": "smoothness",
            "w": {
                "smooth": 1.0,
                "short_circuit": 0.1,
                "eta_th": 0.01,
            },
        }
        base_config.bounds.update(
            {
                "v_max": 30.0,
                "a_max": 500.0,
                "dA_dt_max": 0.01,
            },
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Apply any additional overrides
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)

    return base_config
