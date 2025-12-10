import json
from pathlib import Path

import numpy as np


def generate_golden_shapes(output_dir="shapes", num_points=40):
    """
    Generates golden shape JSON files for optimization tracking.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Circle (Constant Ratio)
    # Target: Constant 2.0
    circle_profile = np.full(num_points, 2.0).tolist()

    with open(out_path / "circle.json", "w") as f:
        json.dump(
            {
                "name": "circle",
                "description": "Constant gear ratio of 2.0",
                "ratio_profile": circle_profile,
            },
            f,
            indent=2,
        )
    print(f"Generated {out_path / 'circle.json'}")

    # 2. Sawtooth (Alternating Ratio)
    # Target: Alternates between 1.5 and 2.5
    # For N=40, we want segments. Let's do a simple alternating pattern or blocks?
    # User said "alternating between two inverse ratios... sawtooth profiles".
    # Let's make it a wave or step function for clarity in the optimizer.
    # Collocation points are sequential in time/angle.

    # A simple checkerboard might be too harsh for d(psi)/dt physics if N is small.
    # Let's do a smoother "Sawtooth" or "Square Wave".
    # Square wave: 10 points at 1.5, 10 points at 2.5, etc.

    # Creating a profile that goes 1.5 -> 2.5 -> 1.5 over the cycle
    x = np.linspace(0, 4 * np.pi, num_points)
    # Sine wave centered at 2.0 with amplitude 0.5 => [1.5, 2.5]
    sawtooth_profile = (2.0 + 0.5 * np.sin(x)).tolist()

    # Square wave option (if 'sawtooth' implies sharp edges)
    # points = np.arange(num_points)
    # square_profile = [1.5 if (i // 5) % 2 == 0 else 2.5 for i in points]

    with open(out_path / "sawtooth.json", "w") as f:
        json.dump(
            {
                "name": "sawtooth",
                "description": "Sinusoidal varying ratio between 1.5 and 2.5",
                # "ratio_profile": square_profile
                "ratio_profile": sawtooth_profile,
            },
            f,
            indent=2,
        )
    print(f"Generated {out_path / 'sawtooth.json'}")

    # 3. Sine (Large Amplitude, 2 Cycles)
    # Target: Variation between 1.5 and 2.5 (Amplitude 0.5) to test limits.
    # Frequency: 2 full cycles over the duration (0 to 4pi).
    sine_profile = (2.0 + 0.5 * np.sin(x)).tolist()

    with open(out_path / "sine.json", "w") as f:
        json.dump(
            {
                "name": "sine_large",
                "description": "Large amplitude sinusoidal varying ratio between 1.5 and 2.5, 2 cycles",
                "ratio_profile": sine_profile,
            },
            f,
            indent=2,
        )
    print(f"Generated {out_path / 'sine.json'}")


if __name__ == "__main__":
    generate_golden_shapes()
