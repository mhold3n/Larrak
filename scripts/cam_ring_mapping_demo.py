"""
Demonstration of cam-ring-linear follower mapping with cascaded optimization.

This script demonstrates how to use the cam-ring mapping framework to create
circular follower (ring) designs based on linear follower motion laws from
primary optimization.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from campro.logging import get_logger
from campro.optimization import SecondaryOptimizer, process_linear_to_ring_follower
from campro.physics import CamRingMapper, CamRingParameters
from campro.storage import OptimizationRegistry

log = get_logger(__name__)


def create_sample_linear_follower_motion():
    """
    Create a sample linear follower motion law for demonstration.

    Returns
    -------
    Dict[str, np.ndarray]
        Sample motion law data
    """
    # Create a realistic cam motion law
    theta = np.linspace(0, 2 * np.pi, 200)

    # Simple harmonic motion with dwell periods
    x_theta = np.zeros_like(theta)

    # Upstroke (0 to π/2)
    upstroke_mask = (theta >= 0) & (theta < np.pi / 2)
    upstroke_theta = theta[upstroke_mask]
    x_theta[upstroke_mask] = 10.0 * (1 - np.cos(2 * upstroke_theta))

    # Dwell at top (π/2 to 3π/2)
    dwell_mask = (theta >= np.pi / 2) & (theta < 3 * np.pi / 2)
    x_theta[dwell_mask] = 20.0

    # Downstroke (3π/2 to 2π)
    downstroke_mask = (theta >= 3 * np.pi / 2) & (theta < 2 * np.pi)
    downstroke_theta = theta[downstroke_mask]
    x_theta[downstroke_mask] = 10.0 * (
        1 + np.cos(2 * (downstroke_theta - 3 * np.pi / 2))
    )

    # Convert to time-based motion (assuming constant cam speed)
    omega = 2.0  # rad/s
    time = theta / omega

    # Compute derivatives
    velocity = np.gradient(x_theta, theta) * omega
    acceleration = np.gradient(velocity, theta) * omega
    jerk = np.gradient(acceleration, theta) * omega

    return {
        "time": time,
        "theta": theta,
        "position": x_theta,
        "velocity": velocity,
        "acceleration": acceleration,
        "control": jerk,
    }


def demo_basic_cam_ring_mapping():
    """Demonstrate basic cam-ring mapping functionality."""
    print("=== Basic Cam-Ring Mapping Demo ===")

    # Create sample motion law
    motion_data = create_sample_linear_follower_motion()
    theta = motion_data["theta"]
    x_theta = motion_data["position"]

    # Create mapper with realistic parameters
    params = CamRingParameters(
        base_radius=15.0,
        follower_roller_radius=2.5,
        ring_roller_radius=1.5,
        contact_type="external",
    )
    mapper = CamRingMapper(params)

    print(f"Linear follower stroke: {np.max(x_theta) - np.min(x_theta):.2f} mm")
    print(f"Cam base radius: {params.base_radius:.2f} mm")
    print(f"Follower roller radius: {params.follower_roller_radius:.2f} mm")

    # Test different ring designs
    ring_designs = [
        {
            "name": "Constant Radius",
            "design_type": "constant",
            "base_radius": 20.0,
        },
        {
            "name": "Linear Variation",
            "design_type": "linear",
            "base_radius": 18.0,
            "slope": 1.0,
        },
        {
            "name": "Sinusoidal Variation",
            "design_type": "sinusoidal",
            "base_radius": 20.0,
            "amplitude": 3.0,
            "frequency": 2.0,
        },
    ]

    results = {}
    for design in ring_designs:
        print(f"\n--- {design['name']} ---")

        # Perform mapping
        result = mapper.map_linear_to_ring_follower(theta, x_theta, design)

        # Validate design
        validation = mapper.validate_design(result)

        print(f"Design valid: {all(validation.values())}")
        print(
            f"Ring radius range: {np.min(result['R_psi']):.2f} - {np.max(result['R_psi']):.2f} mm",
        )
        print(f"Max cam curvature: {np.max(np.abs(result['kappa_c'])):.4f} mm^-1")

        results[design["name"]] = result

    return results


def demo_secondary_optimizer_integration():
    """Demonstrate integration with secondary optimizer."""
    print("\n=== Secondary Optimizer Integration Demo ===")

    # Create registry and store primary result
    registry = OptimizationRegistry()

    # Create sample primary motion law
    motion_data = create_sample_linear_follower_motion()

    # Store as primary optimization result
    registry.store_result(
        "motion_optimizer",
        motion_data,
        {
            "objective_value": 100.0,
            "solve_time": 2.5,
            "constraints": {"max_velocity": 50.0, "max_acceleration": 100.0},
        },
    )

    print("Stored primary optimization result")

    # Create secondary optimizer
    secondary_optimizer = SecondaryOptimizer(
        name="RingDesignOptimizer",
        registry=registry,
    )

    # Define objective function (minimize jerk)
    def jerk_objective(t, x, v, a, u):
        return np.trapz(u**2, t)

    # Define constraints for ring design
    constraints = {
        "cam_parameters": {
            "base_radius": 12.0,
            "follower_roller_radius": 2.0,
            "ring_roller_radius": 1.0,
            "contact_type": "external",
        },
        "ring_design_type": "constant",
        "ring_design_params": {"base_radius": 25.0},
    }

    # Perform secondary optimization
    result = secondary_optimizer.optimize(
        objective=jerk_objective,
        constraints=None,
        primary_optimizer_id="motion_optimizer",
        processing_function=process_linear_to_ring_follower,
        secondary_constraints=constraints,
        secondary_relationships={},
        optimization_targets={},
    )

    print(f"Secondary optimization status: {result.status.name}")
    print(f"Objective value: {result.metadata.get('objective_value', 'N/A')}")

    if result.status.name == "SUCCESS":
        ring_data = result.data
        print(f"Ring radius: {np.mean(ring_data['R_psi']):.2f} mm")
        print(f"Design validation: {ring_data.get('validation', {})}")

    return result


def demo_multi_objective_ring_design():
    """Demonstrate multi-objective ring design optimization."""
    print("\n=== Multi-Objective Ring Design Demo ===")

    # Create sample motion law
    motion_data = create_sample_linear_follower_motion()

    # Test different ring designs and evaluate them
    from campro.optimization.cam_ring_processing import (
        process_multi_objective_ring_design,
    )

    constraints = {
        "cam_parameters": {
            "base_radius": 10.0,
            "follower_roller_radius": 2.0,
            "ring_roller_radius": 1.0,
        },
        "design_alternatives": [
            {"design_type": "constant", "base_radius": 15.0},
            {"design_type": "constant", "base_radius": 20.0},
            {"design_type": "linear", "base_radius": 15.0, "slope": 0.5},
            {
                "design_type": "sinusoidal",
                "base_radius": 18.0,
                "amplitude": 2.0,
                "frequency": 1.0,
            },
        ],
    }

    targets = {
        "weights": {
            "ring_size": 0.4,  # Minimize ring size
            "efficiency": 0.3,  # Maximize efficiency
            "smoothness": 0.2,  # Maximize smoothness
            "stress": 0.1,  # Minimize stress
        },
    }

    result = process_multi_objective_ring_design(
        motion_data,
        constraints,
        {},
        targets,
    )

    print(f"Best design alternative: {result['design_alternative']}")
    print(f"Multi-objective score: {result['multi_objective_score']:.4f}")
    print(
        f"Ring radius range: {np.min(result['R_psi']):.2f} - {np.max(result['R_psi']):.2f} mm",
    )

    return result


def plot_results(results_dict, save_path=None):
    """Plot the cam-ring mapping results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Cam-Ring-Linear Follower Mapping Results", fontsize=16)

    # Plot 1: Linear follower motion law
    ax1 = axes[0, 0]
    motion_data = create_sample_linear_follower_motion()
    ax1.plot(
        motion_data["theta"],
        motion_data["position"],
        "b-",
        linewidth=2,
        label="Position",
    )
    ax1.plot(
        motion_data["theta"],
        motion_data["velocity"],
        "r--",
        linewidth=2,
        label="Velocity",
    )
    ax1.set_xlabel("Cam Angle (rad)")
    ax1.set_ylabel("Displacement (mm)")
    ax1.set_title("Linear Follower Motion Law")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cam curves
    ax2 = axes[0, 1]
    if "Constant Radius" in results_dict:
        result = results_dict["Constant Radius"]
        ax2.plot(
            result["theta"],
            result["cam_curves"]["pitch_radius"],
            "b-",
            label="Pitch Curve",
        )
        ax2.plot(
            result["theta"],
            result["cam_curves"]["profile_radius"],
            "r-",
            label="Profile",
        )
        ax2.plot(
            result["theta"],
            result["cam_curves"]["contact_radius"],
            "g-",
            label="Contact",
        )
        ax2.set_xlabel("Cam Angle (rad)")
        ax2.set_ylabel("Radius (mm)")
        ax2.set_title("Cam Curves")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Ring radius designs
    ax3 = axes[1, 0]
    colors = ["b-", "r-", "g-", "m-"]
    for i, (name, result) in enumerate(results_dict.items()):
        ax3.plot(
            result["psi"],
            result["R_psi"],
            colors[i % len(colors)],
            linewidth=2,
            label=name,
        )
    ax3.set_xlabel("Ring Angle (rad)")
    ax3.set_ylabel("Ring Radius (mm)")
    ax3.set_title("Ring Radius Designs")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cam curvature
    ax4 = axes[1, 1]
    if "Constant Radius" in results_dict:
        result = results_dict["Constant Radius"]
        ax4.plot(result["theta"], result["kappa_c"], "b-", linewidth=2)
        ax4.set_xlabel("Cam Angle (rad)")
        ax4.set_ylabel("Curvature (mm⁻¹)")
        ax4.set_title("Cam Curvature")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    plt.show()


def main():
    """Main demonstration function."""
    print("Cam-Ring-Linear Follower Mapping Demonstration")
    print("=" * 50)

    # Create output directory
    output_dir = Path("plots/cam_ring_mapping")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Demo 1: Basic mapping
        results = demo_basic_cam_ring_mapping()

        # Demo 2: Secondary optimizer integration
        secondary_result = demo_secondary_optimizer_integration()

        # Demo 3: Multi-objective design
        multi_obj_result = demo_multi_objective_ring_design()

        # Plot results
        plot_path = output_dir / "cam_ring_mapping_demo.png"
        plot_results(results, save_path=plot_path)

        print("\n=== Demo Summary ===")
        print("+ Basic cam-ring mapping functionality")
        print("+ Secondary optimizer integration")
        print("+ Multi-objective ring design")
        print("+ Visualization and validation")
        print(f"+ Results saved to: {output_dir}")

    except Exception as e:
        log.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
