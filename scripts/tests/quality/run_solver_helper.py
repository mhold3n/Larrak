import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


def main() -> None:
    try:
        from campro.optimization.driver import solve_cycle

        # Define params (same as in test_regression.py)
        params = {
            "problem_type": "kinematic",
            "planet_ring": {
                "stroke": 0.05,
                "outer_diameter": 0.150,
                "theta_bdc": 3.1415,
                "gen_mode": "spline",
                "ratios": (2.0, 2.0),
                "mean_ratio": 2.0,
                "r_drive": 0.025,
                "attachment_radius": 0.0,
            },
            "geometry": {
                "bore": 0.016,
                "stroke": 0.05,
                "mass": 0.04,
            },
            "bounds": {
                "r_planet_min": 0.001,
                "r_planet_max": 0.075,
                "R_ring_min": 0.001,
                "R_ring_max": 0.300,
                "xL_min": -0.1,
                "xL_max": 0.1,
            },
            "constraints": {},
            "num": {
                "K": 50,
                "C": 3,
                "cycle_time": 0.02,
            },
            "solver": {
                "ipopt": {
                    "max_iter": 500,
                    "tol": 1e-4,
                    "print_level": 5,  # Enable output to parse result from logs if crash occurs
                },
            },
            "obj": {"type": "kinematic_tracking", "w": {"ratio_tracking": 1.0}},
            "shape_file": "circle.json",
            "auto_plot": False,
        }

        # Ensure shape file exists in CWD
        if not os.path.exists("circle.json"):
            data = {"ratio_profile": [2.0] * 51, "operating_point": {"cycle_time": 0.02}}
            with open("circle.json", "w") as f:
                json.dump(data, f)

        # Run solver
        # Silence stdout/stderr to keep JSON clean?
        # No, subprocess capture handles mixed output if iterate, but better to print JSON last.

        res_obj = solve_cycle(params)

        # Extract result
        f_val = 0.0
        success = False
        if isinstance(res_obj, dict):
            final = res_obj.get("final", {})
            f_val = float(final.get("objective", 0.0))
            success = True

        result = {"success": success, "f": f_val}

        # Print magic delimiter + JSON
        print("\n__RESULT_JSON__")
        print(json.dumps(result))
        print("__END_RESULT__")

        # Flush
        sys.stdout.flush()

    except Exception as e:
        print(f"\n__ERROR__: {e}")
        sys.exit(1)

    # Force exit to avoid prolonged teardown crash (though crash is acceptable)
    os._exit(0)


if __name__ == "__main__":
    main()
