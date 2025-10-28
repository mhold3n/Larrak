#!/usr/bin/env python3
"""
Demonstration of the modular architecture for cam-ring systems.

This script showcases the flexibility and adaptability of the new
modular component architecture.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path modification
from campro.config import SystemBuilder  # noqa: E402
from campro.logging import get_logger  # noqa: E402
from campro.physics.geometry import CamCurveComponent, CurvatureComponent  # noqa: E402
from campro.physics.kinematics import MeshingLawComponent  # noqa: E402

log = get_logger(__name__)


def demo_modular_components():
    """Demonstrate individual modular components."""
    print("=" * 60)
    print("MODULAR COMPONENTS DEMONSTRATION")
    print("=" * 60)

    # Create test data
    theta = np.linspace(0, 2 * np.pi, 100)
    x_theta = 5 * np.sin(theta)  # Simple sinusoidal motion

    print(
        f"Test data: {len(theta)} points, theta range: [{theta[0]:.2f}, {theta[-1]:.2f}] rad",
    )

    # 1. Cam Curve Component
    print("\n1. Cam Curve Component")
    print("-" * 30)

    cam_curve = CamCurveComponent(
        parameters={"base_radius": 15.0},
        name="test_cam_curve",
    )

    result = cam_curve.compute({"theta": theta, "x_theta": x_theta})

    if result.is_successful:
        print("+ Cam curves computed successfully")
        print(
            f"  - Pitch radius range: [{np.min(result.outputs['pitch_radius']):.2f}, {np.max(result.outputs['pitch_radius']):.2f}]",
        )
        print(
            f"  - Profile radius range: [{np.min(result.outputs['profile_radius']):.2f}, {np.max(result.outputs['profile_radius']):.2f}]",
        )
        print(f"  - Metadata: {result.metadata}")
    else:
        print(f"- Cam curve computation failed: {result.error_message}")

    # 2. Curvature Component
    print("\n2. Curvature Component")
    print("-" * 30)

    curvature = CurvatureComponent(
        parameters={},
        name="test_curvature",
    )

    # Use the profile radius from cam curve as input
    if result.is_successful:
        curvature_result = curvature.compute(
            {
                "theta": theta,
                "r_theta": result.outputs["profile_radius"],
            },
        )

        if curvature_result.is_successful:
            print("+ Curvature computed successfully")
            print(
                f"  - Curvature range: [{np.min(curvature_result.outputs['kappa']):.3f}, {np.max(curvature_result.outputs['kappa']):.3f}]",
            )
            print(
                f"  - Osculating radius range: [{np.min(curvature_result.outputs['rho'][np.isfinite(curvature_result.outputs['rho'])]):.2f}, {np.max(curvature_result.outputs['rho'][np.isfinite(curvature_result.outputs['rho'])]):.2f}]",
            )
        else:
            print(f"- Curvature computation failed: {curvature_result.error_message}")

    # 3. Meshing Law Component
    print("\n3. Meshing Law Component")
    print("-" * 30)

    meshing_law = MeshingLawComponent(
        parameters={},
        name="test_meshing_law",
    )

    if result.is_successful and curvature_result.is_successful:
        meshing_result = meshing_law.compute(
            {
                "theta": theta,
                "rho_c": curvature_result.outputs["rho"],
                "psi_initial": 0.0,
                "R_psi": 20.0,  # Constant ring radius
            },
        )

        if meshing_result.is_successful:
            print("+ Meshing law solved successfully")
            print(
                f"  - Ring angle range: [{np.min(meshing_result.outputs['psi']):.2f}, {np.max(meshing_result.outputs['psi']):.2f}] rad",
            )
            print(
                f"  - Ring angle range: [{np.min(meshing_result.outputs['psi']) * 180 / np.pi:.1f}, {np.max(meshing_result.outputs['psi']) * 180 / np.pi:.1f}] deg",
            )
        else:
            print(f"- Meshing law solution failed: {meshing_result.error_message}")


def demo_system_builder():
    """Demonstrate the system builder."""
    print("\n" + "=" * 60)
    print("SYSTEM BUILDER DEMONSTRATION")
    print("=" * 60)

    # 1. Manual System Building
    print("\n1. Manual System Building")
    print("-" * 30)

    builder = SystemBuilder("ManualCamRingSystem")

    # Add components manually
    builder.add_component("cam_curves", "cam_curve", {"base_radius": 12.0})
    builder.add_component("curvature", "curvature", {})
    builder.add_component("meshing_law", "meshing_law", {})

    # Connect components
    builder.connect_components("cam_curves", "curvature")
    builder.connect_components("curvature", "meshing_law")

    # Set system parameters
    builder.set_parameters(
        {
            "connecting_rod_length": 30.0,
            "contact_type": "external",
        },
    )

    # Validate configuration
    if builder.validate_configuration():
        print("+ Manual system configuration is valid")
        config = builder.get_configuration()
        print(f"  - System name: {config.name}")
        print(f"  - Components: {list(config.components.keys())}")
        print(f"  - Connections: {len(config.connections)}")
        print(f"  - Parameters: {list(config.parameters.keys())}")
    else:
        print("- Manual system configuration is invalid")

    # 2. Pre-built System Configuration
    print("\n2. Pre-built Cam-Ring System")
    print("-" * 30)

    builder2 = SystemBuilder("PrebuiltCamRingSystem")
    builder2.build_cam_ring_system(base_radius=18.0, connecting_rod_length=35.0)

    if builder2.validate_configuration():
        print("+ Pre-built system configuration is valid")
        config2 = builder2.get_configuration()
        print(f"  - System name: {config2.name}")
        print(f"  - Components: {list(config2.components.keys())}")
        print(f"  - System type: {config2.parameters.get('system_type', 'unknown')}")
        print(f"  - Base radius: {config2.parameters.get('base_radius', 'unknown')}")
    else:
        print("- Pre-built system configuration is invalid")


def demo_adaptability():
    """Demonstrate the adaptability of the architecture."""
    print("\n" + "=" * 60)
    print("ADAPTABILITY DEMONSTRATION")
    print("=" * 60)

    # Show how easy it is to create different system configurations
    configurations = [
        {
            "name": "SmallCamSystem",
            "base_radius": 8.0,
            "connecting_rod_length": 15.0,
        },
        {
            "name": "LargeCamSystem",
            "base_radius": 25.0,
            "connecting_rod_length": 50.0,
        },
        {
            "name": "HighPrecisionSystem",
            "base_radius": 12.0,
            "connecting_rod_length": 20.0,
        },
    ]

    print("\nCreating multiple system configurations:")
    print("-" * 40)

    for config in configurations:
        builder = SystemBuilder(config["name"])
        builder.build_cam_ring_system(
            base_radius=config["base_radius"],
            connecting_rod_length=config["connecting_rod_length"],
        )

        if builder.validate_configuration():
            print(
                f"+ {config['name']}: base_radius={config['base_radius']}, rod_length={config['connecting_rod_length']}",
            )
        else:
            print(f"- {config['name']}: configuration invalid")

    print("\nBenefits of modular architecture:")
    print("-" * 40)
    print("• Easy to create different system configurations")
    print("• Components can be reused across different systems")
    print("• Simple to add new component types")
    print("• Clear separation of concerns")
    print("• Easy to test individual components")
    print("• Flexible parameter management")


def main():
    """Main demonstration function."""
    print("CAM-RING MODULAR ARCHITECTURE DEMONSTRATION")
    print("=" * 60)
    print("This demo showcases the new modular architecture designed")
    print("for maximum adaptability and modification.")
    print()

    try:
        # Run demonstrations
        demo_modular_components()
        demo_system_builder()
        demo_adaptability()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The modular architecture provides:")
        print("• Maximum adaptability for complex optimizations")
        print("• Easy modification and extension")
        print("• Clean separation of concerns")
        print("• Reusable components")
        print("• Flexible system configuration")

    except Exception as e:
        print(f"\n- Demonstration failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
