#!/usr/bin/env python3
"""Simplified headless optimization trigger for quick testing.

Usage:
    python scripts/test_dashboard_headless.py
    python scripts/test_dashboard_headless.py --budget 100 --full-physics
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def monitor_run(run_id: str, timeout_s: int = 300):
    """Monitor a run via docker logs until completion."""
    print(f"\n📊 Monitoring Run {run_id}")
    print("=" * 60)

    start = time.time()
    last_iter = 0

    while (time.time() - start) < timeout_s:
        # Get recent logs
        result = subprocess.run(
            ["docker", "logs", "--tail", "500", "larrak-outline-api"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        logs = result.stdout + result.stderr

        # Find our run's section
        lines = logs.split("\n")
        in_run = False
        run_lines = []

        for line in lines:
            if f"Run ID: {run_id}" in line:
                in_run = True
            elif "Run ID:" in line and in_run:
                # Hit next run, stop
                break

            if in_run:
                run_lines.append(line)

        full_log = "\n".join(run_lines)

        # Check completion
        if "Orchestration finished" in full_log:
            best_match = re.search(r"Best: ([\d.]+)", full_log)
            best = best_match.group(1) if best_match else "N/A"
            print(f"\n✅ Complete! Best: {best}")
            print(f"⏱️  Total time: {time.time() - start:.1f}s")
            return True

        # Show iterations
        iters = re.findall(r"Iter (\d+): New best = ([\d.]+)", full_log)
        for iter_num, best_val in iters:
            iter_num = int(iter_num)
            if iter_num > last_iter:
                print(f"   🔄 Iter {iter_num}: {best_val}")
                last_iter = iter_num

        # Show convergence
        if "Converged after" in full_log:
            conv = re.search(r"Converged after (\d+) iterations", full_log)
            if conv:
                print(f"   ⏹️  Converged ({conv.group(1)} iterations w/o improvement)")

        time.sleep(2)

    print(f"   ⚠️  Timeout after {timeout_s}s")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--full-physics", action="store_true")
    parser.add_argument("--no-monitor", action="store_true", help="Just trigger, don't monitor")

    args = parser.parse_args()

    config = {
        "optimization": {"max_iterations": args.iterations, "batch_size": 5},
        "budget": {
            "total_sim_calls": args.budget,
            "active_budget": int(args.budget * 0.9),
            "validation_budget": int(args.budget * 0.1),
        },
        "simulation": {"use_full_physics": args.full_physics},
    }

    print("\n" + "=" * 60)
    print("🚀 HEADLESS OPTIMIZATION TEST")
    print("=" * 60)
    print(
        f"Budget: {args.budget} | Max Iter: {args.iterations} | Full Physics: {args.full_physics}"
    )

    # Trigger
    response = requests.post("http://localhost:5001/api/start", json=config, timeout=5)
    response.raise_for_status()
    run_id = response.json()["run_id"]

    print(f"\n✅ Started: Run ID {run_id}")

    if args.no_monitor:
        print("\n💡 Monitor with: docker logs -f larrak-outline-api")
        return

    # Monitor
    monitor_run(run_id, timeout_s=600 if args.full_physics else 120)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
