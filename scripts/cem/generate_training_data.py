#!/usr/bin/env python3
"""
Generate Training Data for CEM Surrogates.

Runs a Design of Experiments (DOE) sweep across operating conditions
and motion profiles to generate training data for the validation surrogates.

Usage:
    python scripts/cem/generate_training_data.py --output data/surrogate_training
    python scripts/cem/generate_training_data.py --n-samples 1000 --parallel 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from campro.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainingSample:
    """Single training data point."""

    # Inputs
    rpm: float
    p_intake_bar: float
    fuel_mass_kg: float
    bore: float
    stroke: float
    cr: float

    # Motion profile features
    x_amplitude: float
    x_phase: float
    motion_type: str  # "sine", "optimal", "random"

    # Outputs (from physics simulation)
    p_max_bar: float
    T_max_K: float
    eta_thermal: float
    is_valid: bool

    # Metadata
    sample_id: int = 0


@dataclass
class DOEConfig:
    """Configuration for Design of Experiments sweep."""

    # Operating condition ranges
    rpm_range: tuple[float, float] = (1000.0, 6000.0)
    p_intake_range: tuple[float, float] = (1.0, 4.0)  # bar
    fuel_range: tuple[float, float] = (1e-5, 2e-4)  # kg

    # Geometry ranges
    bore_range: tuple[float, float] = (0.08, 0.15)  # m
    stroke_range: tuple[float, float] = (0.08, 0.20)  # m
    cr_range: tuple[float, float] = (10.0, 20.0)

    # Motion profile parameters
    amplitude_range: tuple[float, float] = (0.8, 1.2)  # Fraction of stroke
    phase_range: tuple[float, float] = (-0.2, 0.2)  # rad

    # Sampling
    n_samples: int = 1000
    seed: int = 42


def generate_sample(
    sample_id: int,
    config: DOEConfig,
    rng: np.random.Generator,
) -> TrainingSample:
    """
    Generate a single training sample.

    Samples operating conditions, generates motion profile,
    runs physics simulation (or mock), and records outputs.
    """
    # Sample operating conditions uniformly
    rpm = rng.uniform(*config.rpm_range)
    p_intake = rng.uniform(*config.p_intake_range)
    fuel_mass = rng.uniform(*config.fuel_range)

    # Sample geometry
    bore = rng.uniform(*config.bore_range)
    stroke = rng.uniform(*config.stroke_range)
    cr = rng.uniform(*config.cr_range)

    # Sample motion profile parameters
    amplitude = rng.uniform(*config.amplitude_range)
    phase = rng.uniform(*config.phase_range)
    motion_type = rng.choice(["sine", "optimal", "random"])

    # Generate mock physics outputs
    # In production, this would call the actual physics solver
    p_max = _mock_physics_p_max(rpm, p_intake, cr, amplitude)
    T_max = _mock_physics_T_max(rpm, p_intake, fuel_mass, cr)
    eta = _mock_physics_efficiency(rpm, p_intake, fuel_mass, stroke, cr)

    # Validity check
    is_valid = p_max < 200 and T_max < 2500 and eta > 0.30

    return TrainingSample(
        sample_id=sample_id,
        rpm=rpm,
        p_intake_bar=p_intake,
        fuel_mass_kg=fuel_mass,
        bore=bore,
        stroke=stroke,
        cr=cr,
        x_amplitude=amplitude,
        x_phase=phase,
        motion_type=motion_type,
        p_max_bar=p_max,
        T_max_K=T_max,
        eta_thermal=eta,
        is_valid=is_valid,
    )


def _mock_physics_p_max(rpm: float, p_intake: float, cr: float, amplitude: float) -> float:
    """Mock physics for peak pressure."""
    # Simple polytropic compression model
    base_p = p_intake * (cr**1.3)  # bar
    rpm_factor = 1.0 + 0.1 * (rpm / 3000 - 1.0)
    amp_factor = amplitude**0.5
    noise = np.random.normal(0, 2)
    return max(0, base_p * rpm_factor * amp_factor + noise)


def _mock_physics_T_max(rpm: float, p_intake: float, fuel_mass: float, cr: float) -> float:
    """Mock physics for peak temperature."""
    # Simple adiabatic compression + combustion model
    T_amb = 300.0  # K
    T_comp = T_amb * (cr**0.3)  # Compression
    fuel_factor = 1.0 + 15000 * fuel_mass  # Combustion heat release
    noise = np.random.normal(0, 50)
    return max(300, T_comp * fuel_factor + noise)


def _mock_physics_efficiency(
    rpm: float,
    p_intake: float,
    fuel_mass: float,
    stroke: float,
    cr: float,
) -> float:
    """Mock physics for brake thermal efficiency."""
    # Otto cycle efficiency with losses
    gamma = 1.3
    eta_otto = 1 - (1 / cr) ** (gamma - 1)

    # Friction and pumping losses
    fmep_factor = 0.05 + 0.02 * (rpm / 3000)
    eta_mech = 1 - fmep_factor

    # Combustion efficiency
    phi = fuel_mass / 5e-5  # Approximate equivalence ratio
    eta_comb = 0.95 if 0.8 < phi < 1.2 else 0.80

    eta = eta_otto * eta_mech * eta_comb
    noise = np.random.normal(0, 0.02)
    return max(0.1, min(0.6, eta + noise))


def generate_batch(
    start_id: int,
    count: int,
    config: DOEConfig,
    seed: int,
) -> list[TrainingSample]:
    """Generate a batch of samples (for parallel execution)."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(count):
        sample = generate_sample(start_id + i, config, rng)
        samples.append(sample)
    return samples


def main(args: argparse.Namespace) -> int:
    """Main entry point."""
    log.info(f"Generating {args.n_samples} training samples")

    config = DOEConfig(
        n_samples=args.n_samples,
        seed=args.seed,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[TrainingSample] = []

    if args.parallel > 1:
        # Parallel generation
        batch_size = max(1, args.n_samples // args.parallel)
        futures = []

        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            for i in range(args.parallel):
                start = i * batch_size
                count = min(batch_size, args.n_samples - start)
                if count <= 0:
                    continue
                seed = config.seed + i
                futures.append(executor.submit(generate_batch, start, count, config, seed))

            for future in futures:
                all_samples.extend(future.result())
    else:
        # Sequential generation
        rng = np.random.default_rng(config.seed)
        for i in range(args.n_samples):
            sample = generate_sample(i, config, rng)
            all_samples.append(sample)
            if (i + 1) % 100 == 0:
                log.info(f"Generated {i + 1}/{args.n_samples} samples")

    # Save as JSON
    json_path = output_dir / "training_data.json"
    with open(json_path, "w") as f:
        json.dump([asdict(s) for s in all_samples], f, indent=2)
    log.info(f"Saved {len(all_samples)} samples to {json_path}")

    # Save as numpy arrays for training
    X = np.array(
        [
            [
                s.rpm / 5000,
                s.p_intake_bar / 3,
                s.fuel_mass_kg / 1e-4,
                s.bore / 0.1,
                s.stroke / 0.15,
                s.cr / 15,
            ]
            for s in all_samples
        ]
    )
    y = np.array([[s.p_max_bar / 100, s.T_max_K / 2000, s.eta_thermal] for s in all_samples])

    np.save(output_dir / "X_train.npy", X)
    np.save(output_dir / "y_train.npy", y)
    log.info(f"Saved numpy arrays: X{X.shape}, y{y.shape}")

    # Summary statistics
    valid_count = sum(1 for s in all_samples if s.is_valid)
    log.info(
        f"Valid samples: {valid_count}/{len(all_samples)} ({100 * valid_count / len(all_samples):.1f}%)"
    )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CEM surrogate training data")
    parser.add_argument(
        "--output", type=str, default="data/surrogate_training", help="Output directory"
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    sys.exit(main(args))
