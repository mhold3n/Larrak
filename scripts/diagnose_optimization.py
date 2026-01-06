#!/usr/bin/env python3
"""Diagnostic script to analyze optimization convergence behavior.

This script runs optimizations with different configurations and measures:
- Time per iteration
- Convergence criteria triggering
- Simulation call counts
- CEM validation success rates

Usage:
    python scripts/diagnose_optimization.py
    python scripts/diagnose_optimization.py --full-physics --verbose
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_docker_logs(run_id: str, tail: int = 100) -> str:
    """Get recent docker logs for a specific run."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), "larrak-outline-api"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        logs = result.stdout + result.stderr
        # Filter to this run
        lines = logs.split("\n")
        run_lines = []
        in_run = False
        for line in lines:
            if f"Run ID: {run_id}" in line:
                in_run = True
            if in_run:
                run_lines.append(line)
            if "Orchestration finished" in line and in_run:
                break
        return "\n".join(run_lines)
    except Exception as e:
        return f"Error getting logs: {e}"


def run_timed_optimization(config: dict, label: str, verbose: bool = False) -> dict:
    """Run optimization with timing and analysis."""
    print(f"\n{'=' * 70}")
    print(f"Test: {label}")
    print(f"{'=' * 70}")

    api_url = "http://localhost:5001"

    start_time = time.time()

    try:
        print("📡 Sending API request...")
        response = requests.post(f"{api_url}/api/start", json=config, timeout=5)
        response.raise_for_status()
        result = response.json()
        run_id = result.get("run_id")

        print(f"✅ Optimization started")
        print(f"   Run ID: {run_id}")
        print(f"   Monitoring progress via docker logs...")
        print()

        # Poll docker logs for completion
        completed = False
        iteration_times = []
        last_iter = 0
        last_log_hash = ""
        iter_start = time.time()
        poll_count = 0

        for _ in range(600):  # 10 minute timeout
            time.sleep(1)
            poll_count += 1

            # Get logs
            logs = get_docker_logs(run_id, tail=200)
            log_hash = hash(logs)

            # Only process if logs changed
            if log_hash != last_log_hash:
                last_log_hash = log_hash

                # Check for completion
                if "Orchestration finished" in logs:
                    print("   ✅ Orchestration completed!")
                    completed = True
                    break

                # Track iterations
                iters = re.findall(r"Iter (\d+):", logs)
                if iters:
                    current_iter = int(iters[-1])
                    if current_iter > last_iter:
                        iter_time = time.time() - iter_start
                        iteration_times.append(iter_time)

                        # Extract best value for this iteration
                        best_match = re.search(rf"Iter {current_iter}: New best = ([\d.]+)", logs)
                        best_val = f" → {best_match.group(1)}" if best_match else ""

                        print(f"   🔄 Iter {current_iter} completed in {iter_time:.2f}s{best_val}")
                        last_iter = current_iter
                        iter_start = time.time()

                # Check for convergence message
                if "Converged after" in logs and "Converged after" not in str(last_log_hash):
                    conv_match = re.search(r"Converged after (\d+) iterations", logs)
                    if conv_match:
                        print(
                            f"   ⏹️  Converged after {conv_match.group(1)} iterations without improvement"
                        )

                # Show validation phase
                if "validation" in logs.lower() and "validation" not in str(last_log_hash):
                    val_match = re.search(r"Final validation: ([\d.]+)", logs)
                    if val_match:
                        print(f"   🔍 Final validation: {val_match.group(1)}")

            # Periodic progress indicator
            if poll_count % 10 == 0 and not completed:
                elapsed = time.time() - start_time
                print(f"   ⏱️  Waiting... ({elapsed:.0f}s elapsed, {last_iter} iterations so far)")

        if not completed:
            print("   ⚠️  Timeout reached - optimization may still be running")

        end_time = time.time()
        total_time = end_time - start_time

        # Extract final results from logs
        logs_full = get_docker_logs(run_id, tail=500)

        best_match = re.search(r"Best: ([\d.]+)", logs_full)
        iter_matches = re.findall(r"Iter (\d+):", logs_full)
        budget_match = re.search(r"Budget: (\d+) sim calls", logs_full)
        conv_match = re.search(r"Converged after (\d+) iterations", logs_full)

        best_value = float(best_match.group(1)) if best_match else None
        iters_used = int(iter_matches[-1]) if iter_matches else last_iter
        budget_used = int(budget_match.group(1)) if budget_match else None
        conv_patience = int(conv_match.group(1)) if conv_match else None

        # Count CEM errors
        cem_errors = len(re.findall(r"\[ERROR\] CEM check failed", logs_full))

        results = {
            "run_id": run_id,
            "completed": completed,
            "total_time_s": total_time,
            "iterations": iters_used,
            "budget": budget_used,
            "best_objective": best_value,
            "avg_iter_time_s": sum(iteration_times) / len(iteration_times)
            if iteration_times
            else None,
            "iteration_times": iteration_times,
            "convergence_patience": conv_patience,
            "cem_errors": cem_errors,
        }

        print(f"\n📊 Summary:")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")
        print(f"   🔁 Iterations: {iters_used}")
        print(f"   💰 Budget: {budget_used} sim calls")
        print(f"   🎯 Best Objective: {best_value:.6f}" if best_value else "   Best: N/A")
        print(
            f"   ⚡ Avg Iter Time: {results['avg_iter_time_s']:.2f}s"
            if results["avg_iter_time_s"]
            else ""
        )
        print(f"   🛑 Convergence Patience: {conv_patience}" if conv_patience else "")
        print(f"   ❌ CEM Errors: {cem_errors}")
        print(f"   {'✅ Completed' if completed else '⚠️ Timeout/Incomplete'}")

        if verbose and cem_errors > 0:
            print(f"\n⚠️  Warning: {cem_errors} CEM feasibility check failures detected")

        return results

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed - is the dashboard running at {api_url}?")
        return {"error": "Connection failed"}
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out")
        return {"error": "Timeout"}
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
        return {"error": str(e)}
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


def main():
    """Run diagnostic tests."""
    parser = argparse.ArgumentParser(description="Diagnose optimization convergence")
    parser.add_argument("--full-physics", action="store_true", help="Test with 1D physics")
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs")
    parser.add_argument("--quick", action="store_true", help="Run only baseline test")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("OPTIMIZATION PIPELINE DIAGNOSTICS")
    print("=" * 70)
    print("\n🔍 This script will test multiple optimization configurations")
    print("   and measure convergence behavior, timing, and iteration counts.\n")

    # Test configurations
    tests = [
        {
            "label": "Baseline (0D physics, 50 budget, 10 max_iter)",
            "config": {
                "optimization": {"max_iterations": 10, "batch_size": 5},
                "budget": {"total_sim_calls": 50, "active_budget": 45, "validation_budget": 5},
                "simulation": {"use_full_physics": False},
            },
        },
    ]

    if not args.quick:
        tests.extend(
            [
                {
                    "label": "Extended iterations (0D physics, 50 budget, 50 max_iter)",
                    "config": {
                        "optimization": {"max_iterations": 50, "batch_size": 5},
                        "budget": {
                            "total_sim_calls": 50,
                            "active_budget": 45,
                            "validation_budget": 5,
                        },
                        "simulation": {"use_full_physics": False},
                    },
                },
                {
                    "label": "Larger budget (0D physics, 200 budget, 20 max_iter)",
                    "config": {
                        "optimization": {"max_iterations": 20, "batch_size": 10},
                        "budget": {
                            "total_sim_calls": 200,
                            "active_budget": 180,
                            "validation_budget": 20,
                        },
                        "simulation": {"use_full_physics": False},
                    },
                },
            ]
        )

    if args.full_physics:
        tests.append(
            {
                "label": "Full Physics (1D, 100 budget, 10 max_iter) - SLOW",
                "config": {
                    "optimization": {"max_iterations": 10, "batch_size": 5},
                    "budget": {
                        "total_sim_calls": 100,
                        "active_budget": 90,
                        "validation_budget": 10,
                    },
                    "simulation": {"use_full_physics": True},
                },
            }
        )

    results = []
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Starting next test...")
        result = run_timed_optimization(test["config"], test["label"], args.verbose)
        results.append({"test": test["label"], "results": result})

        if i < len(tests):
            print("\n⏸️  Pausing 3 seconds before next test...")
            time.sleep(3)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for item in results:
        r = item["results"]
        if "error" not in r:
            status = "✅" if r["completed"] else "⚠️"
            print(f"\n{status} {item['test']}:")
            print(
                f"   Time: {r['total_time_s']:.1f}s | Iters: {r['iterations']} | Best: {r.get('best_objective', 'N/A'):.6f if r.get('best_objective') else 'N/A'}"
            )
            if r.get("cem_errors", 0) > 0:
                print(f"   ⚠️  CEM Errors: {r['cem_errors']}")

    print("\n" + "=" * 70)
    print("✅ Diagnostics complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
