#!/usr/bin/env python3
"""
Run Phase 2 Interpreter.
Reads Phase 1 JSON -> Runs Interpreter -> Writes Phase 2 JSON.
"""

import argparse
import json
import os
import sys

# Add root to path to allow 'interpreter' import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interpreter import Interpreter


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 Interpreter")
    parser.add_argument("--input", required=True, help="Input Cycle JSON (Phase 1)")
    parser.add_argument("--output", required=True, help="Output Shape JSON (Phase 2)")
    parser.add_argument("--stroke", type=float, default=0.2, help="Stroke length (m)")
    parser.add_argument("--conrod", type=float, default=0.4, help="Conrod length (m)")
    parser.add_argument("--mean-ratio", type=float, default=2.0, help="Target Mean Ratio")
    args = parser.parse_args()

    # Load Input
    with open(args.input, "r") as f:
        data = json.load(f)

    # Extract Motion Data
    # Expecting generic format or campro specific?
    # Trying to support generic dict: keys 'x', 'v', 'theta'
    # Or 'trajectory': {'x': ...}

    if "trajectory" in data:
        motion_data = data["trajectory"]
    else:
        # Assume flat
        motion_data = data

    # Verify keys
    for k in ["x", "v"]:
        if k not in motion_data:
            print(f"Error: Missing key '{k}' in input data.")
            sys.exit(1)

    # Run Interpreter
    interp = Interpreter()

    geom = {"stroke": args.stroke, "conrod": args.conrod}
    opts = {"mean_ratio": args.mean_ratio}

    print(f"Running Interpreter...")
    print(f"  Geometry: {geom}")
    print(f"  Target Ratio: {args.mean_ratio}")

    try:
        result = interp.process(motion_data, geom, options=opts)

        # Save
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Success! Output written to {args.output}")
        print(f"  Fitting Error (MSE): {result['meta']['fitting_error']:.6e}")

        # Only save shape-ready file?
        # campro expects "ratio_profile": [...]
        # The output has "profile": {"ratio": ...}
        # Let's create a secondary simplified file or just ensure output format matches campro expectation?
        # The user said "Phase 2 will operate on a global scale which phase 1 and phase 3 can interact with".
        # So a standard Schema is implied.

    except Exception as e:
        print(f"Error during interpretation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
