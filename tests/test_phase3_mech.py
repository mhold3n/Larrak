import numpy as np

from campro.optimization.driver import solve_cycle


def test_phase3_mechanical_optimization():
    """
    Test Phase 3 Mechanical Optimization (Geometric/MECH only).
    Verifies that the solver runs without thermodynamics and produces mechanical metrics.
    """

    # 1. Mock Phase 1 Load Profile (F_gas vs Angle)
    # Simple sine wave load
    angles = np.linspace(0, 2 * np.pi, 50)
    f_gas = 1000.0 * np.sin(angles) + 2000.0  # Positive pressure mostly

    load_profile = {"angle": angles.tolist(), "F_gas": f_gas.tolist()}

    # 2. Mock Phase 2 Target Ratio
    # Target ratio 2.0 +/- 0.1
    target_ratio = 2.0 + 0.1 * np.sin(
        np.linspace(0, 4 * np.pi, 21)
    )  # Ratio defined on coarse grid usually?
    # NLP expects list
    target_ratio_list = target_ratio.tolist()

    # 3. Setup Parameters
    P = {
        "problem_type": "mechanical",  # Though we detect via load_profile
        "load_profile": load_profile,
        "target_ratio_profile": target_ratio_list,
        "planet_ring": {"mean_ratio": 2.0, "use_load_model": False},
        "bounds": {
            "r_planet_min": 0.01,
            "r_planet_max": 0.05,
            "R_ring_min": 0.02,
            "R_ring_max": 0.15,
        },
        "num": {"K": 20, "C": 3},
        "solver": {"ipopt": {"ipopt.max_iter": 500, "ipopt.print_level": 5}},
        "weights": {"tracking": 10.0, "efficiency": 0.1, "stress": 0.1},
    }

    # 4. Run Solver
    result = solve_cycle(P)

    # 5. Verify Results
    # solve_cycle returns a Solution object, not a dict directly
    # The optimization result dict is in solution.meta["optimization"]
    opt_res = result.meta["optimization"]

    assert opt_res["success"], f"Optimization failed: {opt_res.get('message')}"

    # Check Mechanical Outputs
    assert "mechanical" in opt_res, "Mechanical results not found in output"
    mech = opt_res["mechanical"]

    expected_keys = [
        "phi",
        "psi",
        "r",
        "R",
        "i_ratio",
        "T_out_ideal",
        "efficiency",
        "mean_efficiency",
    ]
    for k in expected_keys:
        assert k in mech, f"Key {k} missing from mechanical results"

    # Check dimensions
    assert len(mech["r"]) == P["num"]["K"]
    # psi should be K+1?
    # In driver logic we truncated to K intervals for metrics
    assert len(mech["psi"]) == P["num"]["K"]

    # Check Physics Feasibility
    # Efficiency should be <= 1.0
    eff = np.array(mech["efficiency"])
    assert np.all(eff <= 1.0 + 1e-6), "Efficiency > 1.0 detected"

    print(f"Test Passed. Mean Eff: {mech['mean_efficiency']:.2%}")


if __name__ == "__main__":
    test_phase3_mechanical_optimization()
