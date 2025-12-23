"""
Golden Matrix Runner.
Executes a test function over a combinatorial grid of parameters.
"""

import itertools
import json
import os
import concurrent.futures
import traceback
from typing import Callable, Any, Dict, List


def run_matrix(
    matrix_name: str,
    axes: Dict[str, List[Any]],
    test_func: Callable[[Dict[str, Any]], Dict[str, Any]],
    output_dir: str,
    workers: int = 1,
) -> Dict[str, Any]:
    """
    Run a test matrix.

    Args:
        matrix_name: Name of the matrix (e.g. "rpm_x_stoic").
        axes: Dictionary mapping parameter names to lists of values.
        test_func: Function that takes a param dict and returns a result dict (to be saved as JSON).
        output_dir: Base directory to save goldens.
        workers: Number of parallel workers (default 1, serial).

    Returns:
        Summary dictionary with stats.
    """
    # 1. Generate Combinations
    param_names = list(axes.keys())
    param_values = list(axes.values())

    combinations = list(itertools.product(*param_values))
    total_cases = len(combinations)

    matrix_dir = os.path.join(output_dir, matrix_name)
    os.makedirs(matrix_dir, exist_ok=True)

    print(f"Running Matrix: {matrix_name}")
    print(f"  Axes: {', '.join([f'{k}({len(v)})' for k, v in axes.items()])}")
    print(f"  Total Cases: {total_cases}")

    results = []

    # Helper for single run
    def _run_single(params_tuple):
        params = dict(zip(param_names, params_tuple))

        # Construct filename: param_val_param_val.json
        # Clean values for filename
        slug_parts = []
        for k, v in params.items():
            val_str = str(v).replace(".", "p")  # 0.5 -> 0p5
            slug_parts.append(f"{k}_{val_str}")

        filename = "_".join(slug_parts) + ".json"
        filepath = os.path.join(matrix_dir, filename)

        try:
            # Execute Test
            output_data = test_func(params)

            # Save Golden
            # Add metadata about inputs
            final_data = {"inputs": params, "matrix": matrix_name, "output": output_data}

            with open(filepath, "w") as f:
                json.dump(final_data, f, indent=2)

            return {"status": "success", "file": filename}

        except Exception as e:
            print(f"FAILED case {params}: {e}")
            traceback.print_exc()
            return {"status": "failed", "params": params, "error": str(e)}

    # Execute
    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_run_single, combinations))
    else:
        for combo in combinations:
            results.append(_run_single(combo))

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = total_cases - success

    print(f"Matrix {matrix_name} Complete.")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")

    return {
        "matrix": matrix_name,
        "total": total_cases,
        "success": success,
        "failed": failed,
        "results": results,
    }
