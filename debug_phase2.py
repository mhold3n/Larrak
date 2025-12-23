import sys
import os
import numpy as np

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from interpreter.interface import Interpreter


def main():
    interp = Interpreter()

    # Synthetic data
    theta = np.linspace(0, 4 * np.pi, 100)
    x = np.sin(theta)
    v = np.cos(theta)
    motion_data = {"theta": theta.tolist(), "x": x.tolist(), "v": v.tolist()}
    geom = {"stroke": 0.2, "conrod": 0.4}
    opts = {"mean_ratio": 2.0}

    print("Running Interpreter...")
    try:
        res = interp.process(motion_data, geom, options=opts)
        print("Success!", res["status"])
    except Exception as e:
        print("Failed:", e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
