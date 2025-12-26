"""Design of Experiments Runner for High-Fidelity Simulations.

Generates training data for surrogates by sweeping parameter ranges
and executing solvers in parallel.

Usage:
    # Dry run (no solver execution)
    python doe_runner.py --n-samples 10 --dry-run

    # Full run with thermal adapter
    python doe_runner.py --adapter thermal --n-samples 50 --output data/thermal_doe.parquet
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from pyDOE2 import lhs

    PYDOE_AVAILABLE = True
except ImportError:
    PYDOE_AVAILABLE = False
    lhs = None

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from Simulations.hifi import (
    CombustionCFDAdapter,
    ConjugateHTAdapter,
    PortFlowCFDAdapter,
    StructuralFEAAdapter,
)
from Simulations.hifi.example_inputs import create_simulation_input

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ParameterRange:
    """Definition of a DOE parameter with bounds."""

    name: str
    min_val: float
    max_val: float
    unit: str = ""

    def sample(self, fraction: float) -> float:
        """Convert [0,1] fraction to actual value."""
        return self.min_val + fraction * (self.max_val - self.min_val)


@dataclass
class DOEConfig:
    """Configuration for DOE sweep."""

    n_samples: int = 50
    n_workers: int = 4
    output_dir: str = "data/doe_runs"
    checkpoint_interval: int = 10
    seed: int = 42


# Default parameter ranges for engine simulations
DEFAULT_RANGES = [
    ParameterRange("bore_mm", 70, 100, "mm"),
    ParameterRange("stroke_mm", 75, 110, "mm"),
    ParameterRange("compression_ratio", 10, 16, ""),
    ParameterRange("rpm", 1000, 7000, "rpm"),
    ParameterRange("load_fraction", 0.2, 1.0, ""),
]


def generate_lhs_samples(
    ranges: list[ParameterRange], n_samples: int, seed: int = 42
) -> np.ndarray:
    """
    Generate Latin Hypercube Samples for parameter ranges.

    Returns:
        Array of shape (n_samples, n_params) with actual parameter values
    """
    if not PYDOE_AVAILABLE:
        # Fallback to random sampling
        logger.warning("pyDOE2 not available, using random sampling")
        np.random.seed(seed)
        unit_samples = np.random.rand(n_samples, len(ranges))
    else:
        np.random.seed(seed)
        unit_samples = lhs(len(ranges), samples=n_samples, criterion="maximin")

    # Convert to actual values
    samples = np.zeros_like(unit_samples)
    for i, param in enumerate(ranges):
        samples[:, i] = [param.sample(u) for u in unit_samples[:, i]]

    return samples


def run_single_case(
    case_id: int, params: dict[str, float], adapter_type: str, dry_run: bool = False
) -> dict[str, Any]:
    """
    Execute a single simulation case.

    Args:
        case_id: Unique case identifier
        params: Parameter values from DOE
        adapter_type: One of "structural", "combustion", "thermal", "port"
        dry_run: If True, return mock results

    Returns:
        Dict with inputs and outputs
    """
    result = {
        "case_id": case_id,
        "timestamp": datetime.now().isoformat(),
        "params": params,
        "success": False,
        "outputs": {},
        "error": None,
    }

    try:
        # Create simulation input
        sim_input = create_simulation_input(
            run_id=f"doe_case_{case_id:05d}",
            bore_mm=params["bore_mm"],
            stroke_mm=params["stroke_mm"],
            rpm=params["rpm"],
            load_fraction=params["load_fraction"],
            compression_ratio=params["compression_ratio"],
        )

        if dry_run:
            # Return mock outputs for testing
            result["success"] = True
            result["outputs"] = {
                "T_crown_max": 450 + 100 * params["load_fraction"] + 0.01 * params["rpm"],
                "p_max": 60 + 20 * params["load_fraction"] * (params["compression_ratio"] / 12),
                "von_mises_max": 100 + 50 * params["load_fraction"],
            }
            return result

        # Select adapter
        adapters = {
            "structural": StructuralFEAAdapter,
            "combustion": CombustionCFDAdapter,
            "thermal": ConjugateHTAdapter,
            "port": PortFlowCFDAdapter,
        }

        adapter_cls = adapters.get(adapter_type)
        if not adapter_cls:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        adapter = adapter_cls()
        adapter.load_input(sim_input)
        output = adapter.solve_steady_state()

        result["success"] = output.success
        result["outputs"] = {
            "T_crown_max": output.T_crown_max,
            "T_liner_max": output.T_liner_max,
            "max_von_mises": output.max_von_mises,
            **output.calibration_params,
        }

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Case {case_id} failed: {e}")

    return result


def run_doe(
    config: DOEConfig,
    ranges: list[ParameterRange],
    adapter_type: str = "thermal",
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Execute full DOE sweep.

    Args:
        config: DOE configuration
        ranges: Parameter ranges to sweep
        adapter_type: Solver adapter to use
        dry_run: If True, use mock results

    Returns:
        List of result dictionaries
    """
    # Generate samples
    samples = generate_lhs_samples(ranges, config.n_samples, config.seed)
    param_names = [r.name for r in ranges]

    logger.info(f"Generated {config.n_samples} LHS samples for {len(ranges)} parameters")

    # Prepare cases
    cases = []
    for i, sample in enumerate(samples):
        params = dict(zip(param_names, sample))
        cases.append((i, params))

    results = []
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run cases (parallel for real runs, sequential for dry run)
    if dry_run or config.n_workers == 1:
        for case_id, params in cases:
            result = run_single_case(case_id, params, adapter_type, dry_run)
            results.append(result)

            if (len(results) % config.checkpoint_interval) == 0:
                logger.info(f"Completed {len(results)}/{config.n_samples} cases")
                _save_checkpoint(results, output_dir / "checkpoint.json")
    else:
        with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
            futures = {
                executor.submit(run_single_case, case_id, params, adapter_type, dry_run): case_id
                for case_id, params in cases
            }

            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Case {case_id} exception: {e}")
                    results.append(
                        {
                            "case_id": case_id,
                            "success": False,
                            "error": str(e),
                        }
                    )

                if (len(results) % config.checkpoint_interval) == 0:
                    logger.info(f"Completed {len(results)}/{config.n_samples} cases")
                    _save_checkpoint(results, output_dir / "checkpoint.json")

    return results


def _save_checkpoint(results: list[dict], path: Path):
    """Save intermediate results."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def save_results(results: list[dict], output_path: str):
    """
    Save DOE results to file.

    Supports JSON and Parquet (if pandas available).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        try:
            import pandas as pd

            # Flatten nested dicts
            flat_results = []
            for r in results:
                flat = {
                    "case_id": r["case_id"],
                    "success": r["success"],
                    "error": r.get("error"),
                    "timestamp": r.get("timestamp"),
                }
                flat.update({f"param_{k}": v for k, v in r.get("params", {}).items()})
                flat.update({f"output_{k}": v for k, v in r.get("outputs", {}).items()})
                flat_results.append(flat)

            df = pd.DataFrame(flat_results)
            df.to_parquet(output_path)
            logger.info(f"Saved {len(results)} results to {output_path}")
        except ImportError:
            logger.warning("pandas not available, saving as JSON")
            output_path = output_path.with_suffix(".json")
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
    else:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    logger.info(f"Saved {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run DOE for HiFi simulations")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of LHS samples")
    parser.add_argument("--n-workers", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--adapter",
        type=str,
        default="thermal",
        choices=["structural", "combustion", "thermal", "port"],
    )
    parser.add_argument("--output", type=str, default="data/doe_results.parquet")
    parser.add_argument("--dry-run", action="store_true", help="Use mock results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    config = DOEConfig(
        n_samples=args.n_samples,
        n_workers=args.n_workers,
        seed=args.seed,
    )

    logger.info(
        f"Starting DOE: {args.n_samples} samples, adapter={args.adapter}, dry_run={args.dry_run}"
    )

    results = run_doe(
        config=config,
        ranges=DEFAULT_RANGES,
        adapter_type=args.adapter,
        dry_run=args.dry_run,
    )

    save_results(results, args.output)

    # Summary
    n_success = sum(1 for r in results if r.get("success", False))
    logger.info(f"DOE complete: {n_success}/{len(results)} successful")

    return 0 if n_success == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
