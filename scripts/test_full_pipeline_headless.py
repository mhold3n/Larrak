#!/usr/bin/env python3
"""
Comprehensive headless pipeline test script.

Tests full optimization pipeline in two stages:
1. CEM-driven surrogate training + dashboard optimization
2. Gear profile optimization (motion law → conjugate synthesis)

Usage:
    python scripts/test_full_pipeline_headless.py --full
    python scripts/test_full_pipeline_headless.py --surrogate-only
    python scripts/test_full_pipeline_headless.py --gear-only
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Terminal colors
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_stage(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_success(msg):
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")


def print_error(msg):
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


def print_info(msg):
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")


def check_pilot_data():
    """Check for pilot DOE data required for surrogate training."""
    data_dir = PROJECT_ROOT / "data" / "pilot_doe"
    if not data_dir.exists():
        print_error(f"Pilot DOE directory not found: {data_dir}")
        return None

    results_files = sorted(data_dir.glob("pilot_results_*.json"))
    if not results_files:
        print_error("No pilot results files found")
        return None

    latest = results_files[-1]
    print_success(f"Found pilot data: {latest.name}")
    return latest


def train_surrogates():
    """Train structural and thermal surrogates from pilot DOE data."""
    print_stage("STAGE 1A: Training Surrogates")

    # Check for pilot data
    pilot_data = check_pilot_data()
    if not pilot_data:
        print_error("Cannot train surrogates without pilot data")
        return False

    # Run surrogate training script
    train_script = PROJECT_ROOT / "scripts" / "train_surrogates.py"
    print_info(f"Running: {train_script}")

    try:
        result = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )

        print(result.stdout)
        if result.returncode != 0:
            print_error("Surrogate training failed")
            print(result.stderr)
            return False

        # Verify models were created
        models_dir = PROJECT_ROOT / "models" / "hifi"
        structural = models_dir / "structural_surrogate.pt"
        thermal = models_dir / "thermal_surrogate.pt"

        if structural.exists() and thermal.exists():
            print_success("Surrogates trained and saved")
            print_info(f"  - {structural}")
            print_info(f"  - {thermal}")
            return True
        else:
            print_error("Surrogate model files not found after training")
            return False

    except subprocess.TimeoutExpired:
        print_error("Surrogate training timed out")
        return False
    except Exception as e:
        print_error(f"Error during surrogate training: {e}")
        return False


def run_dashboard_optimization():
    """Run dashboard optimization with trained surrogates."""
    print_stage("STAGE 1B: Dashboard Optimization with CEM")

    config = {
        "optimization": {"max_iterations": 20, "batch_size": 5},
        "budget": {
            "total_sim_calls": 100,
            "active_budget": 90,
            "validation_budget": 10,
        },
        "simulation": {"use_full_physics": False},  # Fast 0D for initial test
    }

    print_info("Triggering optimization via API...")
    try:
        response = requests.post("http://localhost:5001/api/start", json=config, timeout=5)
        response.raise_for_status()
        run_id = response.json()["run_id"]
        print_success(f"Optimization started: Run ID {run_id}")

        # Monitor completion - wait longer for optimization to finish
        print_info("Waiting for optimization to complete (this may take 60-90 seconds)...")
        time.sleep(90)  # Optimization typically takes 60-90s

        # Check logs for completion
        result = subprocess.run(
            ["docker", "logs", "--tail", "200", "larrak-outline-api"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        logs = result.stdout + result.stderr

        # Check if this specific run completed
        run_section = logs.split(f"Run ID: {run_id}")[-1] if f"Run ID: {run_id}" in logs else ""

        if "Orchestration finished" in run_section or "Best:" in run_section:
            print_success("Optimization completed")

            # Check for CEM errors
            if "CEM check failed" in run_section or "TypeError" in run_section:
                print_error("CEM validation errors detected")
            else:
                print_success("No CEM errors!")

            return True
        else:
            print_error("Optimization may not have completed in 90 seconds")
            print_info("Check Docker logs manually: docker logs larrak-outline-api")
            return False

    except requests.exceptions.RequestException as e:
        print_error(f"API request failed: {e}")
        print_info("Is the dashboard running? (docker-compose up)")
        return False
    except Exception as e:
        print_error(f"Error during optimization: {e}")
        return False


def run_gear_optimization():
    """Run Phase 3 conjugate gear profile optimization."""
    print_stage("STAGE 2: Gear Profile Optimization")

    # Check for motion law input
    motion_law_file = (
        PROJECT_ROOT
        / "tests"
        / "goldens"
        / "phase4"
        / "valve_strategy"
        / "valve_strategy_results.json"
    )
    if not motion_law_file.exists():
        print_error(f"Motion law input not found: {motion_law_file}")
        return False

    print_success(f"Found motion law data: {motion_law_file.name}")

    # Run conjugate optimization
    gear_script = PROJECT_ROOT / "scripts" / "phase3" / "run_conjugate_optimization.py"
    print_info(f"Running: {gear_script}")

    try:
        result = subprocess.run(
            [sys.executable, str(gear_script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

        print(result.stdout)
        if result.returncode != 0:
            print_error("Gear optimization failed")
            print(result.stderr)
            return False

        # Verify outputs
        output_dir = PROJECT_ROOT / "output"
        shapes_html = output_dir / "conjugate_shapes.html"
        radii_html = output_dir / "conjugate_radii.html"

        if shapes_html.exists() and radii_html.exists():
            print_success("Gear profiles generated")
            print_info(f"  - {shapes_html}")
            print_info(f"  - {radii_html}")
            return True
        else:
            print_error("Expected output files not found")
            return False

    except subprocess.TimeoutExpired:
        print_error("Gear optimization timed out")
        return False
    except Exception as e:
        print_error(f"Error during gear optimization: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive pipeline test: CEM+Surrogates and Gear Optimization"
    )
    parser.add_argument(
        "--surrogate-only", action="store_true", help="Only test CEM + surrogate training pipeline"
    )
    parser.add_argument(
        "--gear-only", action="store_true", help="Only test gear profile optimization"
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full end-to-end pipeline (default)"
    )

    args = parser.parse_args()

    # Default to full if no flags set
    if not (args.surrogate_only or args.gear_only):
        args.full = True

    print_stage("COMPREHENSIVE PIPELINE TEST")
    print_info(
        f"Mode: {'Full Pipeline' if args.full else 'Surrogate Only' if args.surrogate_only else 'Gear Only'}"
    )

    success = True

    # Stage 1: CEM + Surrogates
    if args.full or args.surrogate_only:
        surrogate_success = train_surrogates()
        if not surrogate_success:
            print_error("Stage 1A failed: Surrogate training")
            success = False
        else:
            opt_success = run_dashboard_optimization()
            if not opt_success:
                print_error("Stage 1B failed: Dashboard optimization")
                success = False

    # Stage 2: Gear Optimization
    if args.full or args.gear_only:
        gear_success = run_gear_optimization()
        if not gear_success:
            print_error("Stage 2 failed: Gear optimization")
            success = False

    # Final summary
    print_stage("PIPELINE TEST SUMMARY")
    if success:
        print_success("All stages completed successfully!")
        return 0
    else:
        print_error("Some stages failed - check logs above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
