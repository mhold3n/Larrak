#!/usr/bin/env python3
"""
Cam motion law examples demonstrating the simplified constraint system.

This script shows how to use the new cam-specific constraints for intuitive
cam follower motion law design.
"""

# Add parent directory to path for imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from campro.logging import get_logger
from CamPro_OptimalMotion import solve_cam_motion_law

log = get_logger(__name__)


def example_basic_cam_motion():
    """Example: Basic cam motion law with simple constraints."""
    print("\n=== Basic Cam Motion Law ===")

    # Simple cam parameters
    stroke = 20.0  # 20mm stroke
    upstroke_duration = 60.0  # 60% of cycle for upstroke
    cycle_time = 1.0  # 1 second cycle

    print(f"Stroke: {stroke} mm")
    print(f"Upstroke duration: {upstroke_duration}% of cycle")
    print(f"Cycle time: {cycle_time} s")

    # Solve using convenience function
    solution = solve_cam_motion_law(
        stroke=stroke,
        upstroke_duration_percent=upstroke_duration,
        motion_type="minimum_jerk",
        cycle_time=cycle_time,
    )

    # Results
    max_velocity = np.max(np.abs(solution["velocity"]))
    max_acceleration = np.max(np.abs(solution["acceleration"]))
    max_jerk = np.max(np.abs(solution["control"]))

    print("\nResults:")
    print(f"Max velocity: {max_velocity:.3f} mm/s")
    print(f"Max acceleration: {max_acceleration:.3f} mm/s²")
    print(f"Max jerk: {max_jerk:.3f} mm/s³")

    return solution


def example_cam_with_velocity_limit():
    """Example: Cam motion with velocity constraint."""
    print("\n=== Cam Motion with Velocity Limit ===")

    # Cam parameters with velocity constraint
    stroke = 25.0  # 25mm stroke
    upstroke_duration = 50.0  # 50% of cycle for upstroke
    max_velocity = 100.0  # 100 mm/s max velocity
    cycle_time = 0.5  # 0.5 second cycle (faster)

    print(f"Stroke: {stroke} mm")
    print(f"Upstroke duration: {upstroke_duration}% of cycle")
    print(f"Max velocity: {max_velocity} mm/s")
    print(f"Cycle time: {cycle_time} s")

    # Solve with velocity constraint
    solution = solve_cam_motion_law(
        stroke=stroke,
        upstroke_duration_percent=upstroke_duration,
        motion_type="minimum_jerk",
        cycle_time=cycle_time,
        max_velocity=max_velocity,
    )

    # Results
    actual_max_velocity = np.max(np.abs(solution["velocity"]))
    max_acceleration = np.max(np.abs(solution["acceleration"]))

    print("\nResults:")
    print(f"Actual max velocity: {actual_max_velocity:.3f} mm/s")
    print(f"Max acceleration: {max_acceleration:.3f} mm/s²")
    print(f"Velocity constraint satisfied: {actual_max_velocity <= max_velocity + 1e-6}")

    return solution


def example_cam_with_zero_acceleration_phase():
    """Example: Cam motion with zero acceleration phase during expansion."""
    print("\n=== Cam Motion with Zero Acceleration Phase ===")

    # Cam parameters with zero acceleration phase
    stroke = 30.0  # 30mm stroke
    upstroke_duration = 70.0  # 70% of cycle for upstroke
    zero_accel_duration = 20.0  # 20% of cycle with zero acceleration
    cycle_time = 1.0  # 1 second cycle

    print(f"Stroke: {stroke} mm")
    print(f"Upstroke duration: {upstroke_duration}% of cycle")
    print(f"Zero acceleration duration: {zero_accel_duration}% of cycle")
    print(f"Cycle time: {cycle_time} s")

    # Solve with zero acceleration phase
    solution = solve_cam_motion_law(
        stroke=stroke,
        upstroke_duration_percent=upstroke_duration,
        motion_type="minimum_jerk",
        cycle_time=cycle_time,
        zero_accel_duration_percent=zero_accel_duration,
    )

    # Results
    max_velocity = np.max(np.abs(solution["velocity"]))
    max_acceleration = np.max(np.abs(solution["acceleration"]))

    print("\nResults:")
    print(f"Max velocity: {max_velocity:.3f} mm/s")
    print(f"Max acceleration: {max_acceleration:.3f} mm/s²")

    return solution


def example_cam_motion_types_comparison():
    """Example: Compare different cam motion law types."""
    print("\n=== Cam Motion Law Types Comparison ===")

    # Common cam parameters
    stroke = 20.0
    upstroke_duration = 60.0
    cycle_time = 1.0
    max_velocity = 50.0
    max_acceleration = 200.0

    motion_types = ["minimum_jerk", "minimum_energy", "minimum_time"]
    solutions = {}

    for motion_type in motion_types:
        print(f"\nSolving {motion_type}...")

        solution = solve_cam_motion_law(
            stroke=stroke,
            upstroke_duration_percent=upstroke_duration,
            motion_type=motion_type,
            cycle_time=cycle_time,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        )

        solutions[motion_type] = solution

        # Calculate metrics
        max_vel = np.max(np.abs(solution["velocity"]))
        max_acc = np.max(np.abs(solution["acceleration"]))
        max_jerk = np.max(np.abs(solution["control"]))

        print(f"  Max velocity: {max_vel:.3f} mm/s")
        print(f"  Max acceleration: {max_acc:.3f} mm/s²")
        print(f"  Max jerk: {max_jerk:.3f} mm/s³")

    return solutions


def example_cam_without_dwell():
    """Example: Cam motion without dwell at TDC or BDC."""
    print("\n=== Cam Motion without Dwell ===")

    # Cam parameters without dwell
    stroke = 15.0
    upstroke_duration = 40.0  # Short upstroke
    cycle_time = 0.8
    max_velocity = 80.0

    print(f"Stroke: {stroke} mm")
    print(f"Upstroke duration: {upstroke_duration}% of cycle")
    print("No dwell at TDC or BDC")
    print(f"Cycle time: {cycle_time} s")

    # Solve without dwell
    solution = solve_cam_motion_law(
        stroke=stroke,
        upstroke_duration_percent=upstroke_duration,
        motion_type="minimum_jerk",
        cycle_time=cycle_time,
        max_velocity=max_velocity,
        dwell_at_tdc=False,
        dwell_at_bdc=False,
    )

    # Results
    initial_velocity = solution["velocity"][0]
    final_velocity = solution["velocity"][-1]
    max_velocity_actual = np.max(np.abs(solution["velocity"]))

    print("\nResults:")
    print(f"Initial velocity: {initial_velocity:.3f} mm/s")
    print(f"Final velocity: {final_velocity:.3f} mm/s")
    print(f"Max velocity: {max_velocity_actual:.3f} mm/s")

    return solution


def plot_cam_comparison(solutions_dict, title="Cam Motion Law Comparison"):
    """Plot comparison of different cam solutions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title)

    colors = ["blue", "red", "green", "orange", "purple"]

    for i, (name, solution) in enumerate(solutions_dict.items()):
        color = colors[i % len(colors)]
        cam_angle = solution["cam_angle"]

        # Position
        axes[0, 0].plot(cam_angle, solution["position"], color=color, label=name, linewidth=2)

        # Velocity
        axes[0, 1].plot(cam_angle, solution["velocity"], color=color, label=name, linewidth=2)

        # Acceleration
        axes[1, 0].plot(cam_angle, solution["acceleration"], color=color, label=name, linewidth=2)

        # Jerk
        axes[1, 1].plot(cam_angle, solution["control"], color=color, label=name, linewidth=2)

    # Format plots
    for ax in axes.flat:
        ax.set_xlabel("Cam Angle [°]")
        ax.grid(True)
        ax.legend()

        # Add TDC/BDC markers
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5, label="TDC")
        ax.axvline(x=180, color="black", linestyle="--", alpha=0.5, label="BDC")

    axes[0, 0].set_title("Follower Position")
    axes[0, 0].set_ylabel("Position [mm]")

    axes[0, 1].set_title("Follower Velocity")
    axes[0, 1].set_ylabel("Velocity [mm/s]")

    axes[1, 0].set_title("Follower Acceleration")
    axes[1, 0].set_ylabel("Acceleration [mm/s²]")

    axes[1, 1].set_title("Follower Jerk")
    axes[1, 1].set_ylabel("Jerk [mm/s³]")

    plt.tight_layout()
    return fig


def plot_single_cam_solution(solution, title="Cam Motion Law Solution"):
    """Plot a single cam solution with detailed annotations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title)

    cam_angle = solution["cam_angle"]

    # Position
    axes[0, 0].plot(cam_angle, solution["position"], "b-", linewidth=2)
    axes[0, 0].set_title("Follower Position")
    axes[0, 0].set_xlabel("Cam Angle [°]")
    axes[0, 0].set_ylabel("Position [mm]")
    axes[0, 0].grid(True)
    axes[0, 0].axvline(x=0, color="r", linestyle="--", alpha=0.7, label="TDC")
    axes[0, 0].axvline(x=180, color="b", linestyle="--", alpha=0.7, label="BDC")
    axes[0, 0].legend()

    # Velocity
    axes[0, 1].plot(cam_angle, solution["velocity"], "g-", linewidth=2)
    axes[0, 1].set_title("Follower Velocity")
    axes[0, 1].set_xlabel("Cam Angle [°]")
    axes[0, 1].set_ylabel("Velocity [mm/s]")
    axes[0, 1].grid(True)
    axes[0, 1].axvline(x=0, color="r", linestyle="--", alpha=0.7)
    axes[0, 1].axvline(x=180, color="b", linestyle="--", alpha=0.7)

    # Acceleration
    axes[1, 0].plot(cam_angle, solution["acceleration"], "r-", linewidth=2)
    axes[1, 0].set_title("Follower Acceleration")
    axes[1, 0].set_xlabel("Cam Angle [°]")
    axes[1, 0].set_ylabel("Acceleration [mm/s²]")
    axes[1, 0].grid(True)
    axes[1, 0].axvline(x=0, color="r", linestyle="--", alpha=0.7)
    axes[1, 0].axvline(x=180, color="b", linestyle="--", alpha=0.7)

    # Jerk
    axes[1, 1].plot(cam_angle, solution["control"], "m-", linewidth=2)
    axes[1, 1].set_title("Follower Jerk")
    axes[1, 1].set_xlabel("Cam Angle [°]")
    axes[1, 1].set_ylabel("Jerk [mm/s³]")
    axes[1, 1].grid(True)
    axes[1, 1].axvline(x=0, color="r", linestyle="--", alpha=0.7)
    axes[1, 1].axvline(x=180, color="b", linestyle="--", alpha=0.7)

    plt.tight_layout()
    return fig


def main():
    """Run all cam examples."""
    print("Larrak Cam Motion Law Examples")
    print("=" * 40)

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Run examples
        basic_sol = example_basic_cam_motion()
        velocity_limited_sol = example_cam_with_velocity_limit()
        zero_accel_sol = example_cam_with_zero_acceleration_phase()
        motion_types_sols = example_cam_motion_types_comparison()
        no_dwell_sol = example_cam_without_dwell()

        # Plot individual solutions
        print("\nGenerating plots...")

        fig1 = plot_single_cam_solution(basic_sol, "Basic Cam Motion Law")
        fig1.savefig(output_dir / "basic_cam_motion.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'basic_cam_motion.png'}")

        fig2 = plot_single_cam_solution(velocity_limited_sol, "Cam Motion with Velocity Limit")
        fig2.savefig(output_dir / "velocity_limited_cam.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'velocity_limited_cam.png'}")

        fig3 = plot_single_cam_solution(zero_accel_sol, "Cam Motion with Zero Acceleration Phase")
        fig3.savefig(output_dir / "zero_accel_cam.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'zero_accel_cam.png'}")

        # Plot comparison
        fig4 = plot_cam_comparison(motion_types_sols, "Cam Motion Law Types Comparison")
        fig4.savefig(output_dir / "cam_motion_types_comparison.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'cam_motion_types_comparison.png'}")

        fig5 = plot_single_cam_solution(no_dwell_sol, "Cam Motion without Dwell")
        fig5.savefig(output_dir / "no_dwell_cam.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'no_dwell_cam.png'}")

        # Show plots
        plt.show()

        print("\nAll cam examples completed successfully!")

    except Exception as e:
        log.error(f"Error running cam examples: {e}")
        raise


if __name__ == "__main__":
    main()





