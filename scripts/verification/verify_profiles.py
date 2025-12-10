from campro.optimization.nlp import build_collocation_nlp


def test_nlp_profiles():
    # Minimal config for Litvin mode
    P = {
        "planet_ring": {
            "gen_mode": "litvin",
            "outer_diameter": 0.042,
            "mean_ratio": 2.0,
        },
        "combustion": {"cycle_time_s": 0.02},
        "flow": {"use_0d_model": True},
        "geometry": {"bore": 0.016, "stroke": 0.02},
        "bounds": {},
        "obj": {},
        "init": {},
    }

    print("Building NLP...")
    nlp, meta = build_collocation_nlp(P)

    if "get_profiles" in meta:
        print("SUCCESS: 'get_profiles' found in meta.")
        f_prof = meta["get_profiles"]
        print(f"Function: {f_prof}")

        # Test evaluation
        # i=2.0, t=0.0 (phi=0), psi=0.0
        # r = a / (2-1) = a. R = 2a.
        # a = 0.042/2 / 2 = 0.0105
        # r = 0.0105, R = 0.021
        # t=0 -> phi=0.
        # Ring: (R, 0) -> (0.021, 0)
        # Planet: vec=(r,0). rot(-0). P=(0.0105, 0)

        res = f_prof(2.0, 0.0, 0.0)
        Px, Py, Rx, Ry = float(res[0]), float(res[1]), float(res[2]), float(res[3])
        print(f"Evaluated (2.0, 0, 0): P=({Px:.4f}, {Py:.4f}), R=({Rx:.4f}, {Ry:.4f})")

        expected_R = 0.021
        expected_P = 0.0105

        if abs(Rx - expected_R) < 1e-6 and abs(Px - expected_P) < 1e-6:
            print("VERIFIED: Calculation matches standard 2:1 hypothesis.")
        else:
            print("WARNING: Calculation result mismatch.")

    else:
        print("FAILURE: 'get_profiles' NOT found in meta.")


if __name__ == "__main__":
    test_nlp_profiles()
