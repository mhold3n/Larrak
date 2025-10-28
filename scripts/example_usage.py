#!/usr/bin/env python3
"""
Example usage script for Larrak optimal motion law solver.

This script demonstrates various motion law problems and their solutions.
"""

# Add parent directory to path for imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from campro.logging import get_logger
from CamPro_OptimalMotion import (
    CollocationSettings,
    MotionConstraints,
    OptimalMotionSolver,
    solve_minimum_energy_motion,
    solve_minimum_jerk_motion,
    solve_minimum_time_motion,
)

log = get_logger(__name__)


def example_minimum_time():
    """Example: Minimum time motion law."""
    print("\n=== Minimum Time Motion Law ===")

    # Problem parameters
    distance = 20.0
    max_velocity = 8.0
    max_acceleration = 3.0
    max_jerk = 2.0

    print(f"Distance: {distance} m")
    print(f"Max velocity: {max_velocity} m/s")
    print(f"Max acceleration: {max_acceleration} m/s²")
    print(f"Max jerk: {max_jerk} m/s³")

    # Solve
    solution = solve_minimum_time_motion(
        distance=distance,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        max_jerk=max_jerk,
    )

    # Results
    final_time = solution["time"][-1]
    max_vel_achieved = np.max(np.abs(solution["velocity"]))
    max_acc_achieved = np.max(np.abs(solution["acceleration"]))
    max_jerk_achieved = np.max(np.abs(solution["control"]))

    print("\nResults:")
    print(f"Minimum time: {final_time:.3f} s")
    print(f"Max velocity achieved: {max_vel_achieved:.3f} m/s")
    print(f"Max acceleration achieved: {max_acc_achieved:.3f} m/s²")
    print(f"Max jerk achieved: {max_jerk_achieved:.3f} m/s³")

    return solution


def example_minimum_energy():
    """Example: Minimum energy motion law."""
    print("\n=== Minimum Energy Motion Law ===")

    # Problem parameters
    distance = 15.0
    time_horizon = 6.0
    max_velocity = 5.0
    max_acceleration = 2.5

    print(f"Distance: {distance} m")
    print(f"Time horizon: {time_horizon} s")
    print(f"Max velocity: {max_velocity} m/s")
    print(f"Max acceleration: {max_acceleration} m/s²")

    # Solve
    solution = solve_minimum_energy_motion(
        distance=distance,
        time_horizon=time_horizon,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
    )

    # Results
    energy = np.trapz(solution["control"] ** 2, solution["time"])
    max_vel_achieved = np.max(np.abs(solution["velocity"]))
    max_acc_achieved = np.max(np.abs(solution["acceleration"]))

    print("\nResults:")
    print(f"Energy consumed: {energy:.3f} (units²·s)")
    print(f"Max velocity achieved: {max_vel_achieved:.3f} m/s")
    print(f"Max acceleration achieved: {max_acc_achieved:.3f} m/s²")

    return solution


def example_minimum_jerk():
    """Example: Minimum jerk motion law."""
    print("\n=== Minimum Jerk Motion Law ===")

    # Problem parameters
    distance = 12.0
    time_horizon = 5.0
    max_velocity = 4.0
    max_acceleration = 2.0

    print(f"Distance: {distance} m")
    print(f"Time horizon: {time_horizon} s")
    print(f"Max velocity: {max_velocity} m/s")
    print(f"Max acceleration: {max_acceleration} m/s²")

    # Solve
    solution = solve_minimum_jerk_motion(
        distance=distance,
        time_horizon=time_horizon,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
    )

    # Results
    jerk_integral = np.trapz(solution["control"] ** 2, solution["time"])
    max_vel_achieved = np.max(np.abs(solution["velocity"]))
    max_acc_achieved = np.max(np.abs(solution["acceleration"]))
    max_jerk_achieved = np.max(np.abs(solution["control"]))

    print("\nResults:")
    print(f"Jerk integral: {jerk_integral:.3f} (m/s³)²·s")
    print(f"Max velocity achieved: {max_vel_achieved:.3f} m/s")
    print(f"Max acceleration achieved: {max_acc_achieved:.3f} m/s²")
    print(f"Max jerk achieved: {max_jerk_achieved:.3f} m/s³")

    return solution


def example_custom_objective():
    """Example: Custom objective function."""
    print("\n=== Custom Objective Motion Law ===")

    import casadi as ca

    # Custom objective: minimize energy + smoothness penalty
    def custom_objective(t, x, v, a, u):
        return ca.integral(u**2 + 0.1 * v**2)

    # Problem parameters
    distance = 18.0
    time_horizon = 7.0
    max_velocity = 6.0
    max_acceleration = 2.5

    print(f"Distance: {distance} m")
    print(f"Time horizon: {time_horizon} s")
    print(f"Max velocity: {max_velocity} m/s")
    print(f"Max acceleration: {max_acceleration} m/s²")
    print("Objective: minimize energy + 0.1 × velocity²")

    # Setup solver and constraints
    settings = CollocationSettings(degree=3, verbose=False)
    solver = OptimalMotionSolver(settings)

    constraints = MotionConstraints(
        initial_position=0.0,
        initial_velocity=0.0,
        final_position=distance,
        final_velocity=0.0,
        velocity_bounds=(-max_velocity, max_velocity),
        acceleration_bounds=(-max_acceleration, max_acceleration),
    )

    # Solve
    solution = solver.solve_custom_objective(
        objective_function=custom_objective,
        constraints=constraints,
        distance=distance,
        time_horizon=time_horizon,
    )

    # Results
    energy = np.trapz(solution["control"] ** 2, solution["time"])
    smoothness = np.trapz(solution["velocity"] ** 2, solution["time"])
    objective_value = energy + 0.1 * smoothness

    print("\nResults:")
    print(f"Energy term: {energy:.3f}")
    print(f"Smoothness term: {0.1 * smoothness:.3f}")
    print(f"Total objective: {objective_value:.3f}")

    return solution


def example_different_collocation_methods():
    """Example: Compare different collocation methods."""
    print("\n=== Collocation Method Comparison ===")

    # Problem parameters
    distance = 10.0
    time_horizon = 4.0
    max_velocity = 5.0
    max_acceleration = 2.0

    methods = ["legendre", "radau", "lobatto"]
    solutions = {}

    for method in methods:
        print(f"\nSolving with {method} collocation...")

        settings = CollocationSettings(
            degree=3,
            method=method,
            verbose=False,
        )

        solver = OptimalMotionSolver(settings)
        constraints = MotionConstraints(
            initial_position=0.0,
            final_position=distance,
        )

        solution = solver.solve_minimum_energy(
            constraints=constraints,
            distance=distance,
            time_horizon=time_horizon,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        )

        solutions[method] = solution

        energy = np.trapz(solution["control"] ** 2, solution["time"])
        print(f"Energy: {energy:.6f}")

    return solutions


def plot_comparison(solutions_dict, title="Motion Law Comparison"):
    """Plot comparison of different solutions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    colors = ["blue", "red", "green", "orange", "purple"]

    for i, (name, solution) in enumerate(solutions_dict.items()):
        color = colors[i % len(colors)]
        t = solution["time"]

        # Position
        axes[0, 0].plot(t, solution["position"], color=color, label=name, linewidth=2)

        # Velocity
        axes[0, 1].plot(t, solution["velocity"], color=color, label=name, linewidth=2)

        # Acceleration
        axes[1, 0].plot(
            t, solution["acceleration"], color=color, label=name, linewidth=2,
        )

        # Control (Jerk)
        axes[1, 1].plot(t, solution["control"], color=color, label=name, linewidth=2)

    # Format plots
    axes[0, 0].set_title("Position")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Position [m]")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].set_title("Velocity")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Velocity [m/s]")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].set_title("Acceleration")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("Acceleration [m/s²]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].set_title("Control (Jerk)")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Jerk [m/s³]")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    plt.tight_layout()
    return fig


def main():
    """Run all examples."""
    print("Larrak Optimal Motion Law Examples")
    print("=" * 40)

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Run examples
        min_time_sol = example_minimum_time()
        min_energy_sol = example_minimum_energy()
        min_jerk_sol = example_minimum_jerk()
        custom_sol = example_custom_objective()
        collocation_sols = example_different_collocation_methods()

        # Plot comparisons
        print("\nGenerating plots...")

        # Compare different motion law types
        motion_laws = {
            "Minimum Time": min_time_sol,
            "Minimum Energy": min_energy_sol,
            "Minimum Jerk": min_jerk_sol,
            "Custom Objective": custom_sol,
        }

        fig1 = plot_comparison(motion_laws, "Motion Law Type Comparison")
        fig1.savefig(
            output_dir / "motion_law_comparison.png", dpi=300, bbox_inches="tight",
        )
        print(f"Saved: {output_dir / 'motion_law_comparison.png'}")

        # Compare collocation methods
        fig2 = plot_comparison(collocation_sols, "Collocation Method Comparison")
        fig2.savefig(
            output_dir / "collocation_comparison.png", dpi=300, bbox_inches="tight",
        )
        print(f"Saved: {output_dir / 'collocation_comparison.png'}")

        # Show plots
        plt.show()

        print("\nAll examples completed successfully!")

    except Exception as e:
        log.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
