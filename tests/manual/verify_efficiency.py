import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.goldens.phase1.generate_doe import phase1_test


def test_efficiency():
    print("Running Efficiency Verification Test...")
    params = {"rpm": 1000.0, "p_int_bar": 1.0, "fuel_mass_mg": 20.0}
    
    # 1. Test MA86 (Current)
    print("\n--- Test 1: Linear Solver MA86 ---")
    opts_ma86 = {
        "ipopt": {
            "max_iter": 1000,
            "print_level": 5, # Verbose
            "sb": "no",
            "tol": 1e-6,
            "linear_solver": "ma86",
        },
        "print_time": 1,
    }
    result_86 = phase1_test(params, solver_opts=opts_ma86)
    result_86 = phase1_test(params, solver_opts=opts_ma86)
    print(f"MA86 Status: {result_86.get('status')}")
    print(f"P_max: {result_86.get('p_max_bar'):.2f} bar")
    print(f"Efficiency: {result_86.get('thermal_efficiency')*100:.2f} %")
    print(f"Work: {result_86.get('abs_work_net_j'):.2f} J")
    print(f"Q: {result_86.get('q_total_j'):.2f} J")

    # 2. Test MA57 (Baseline)
    print("\n--- Test 2: Linear Solver MA57 ---")
    opts_ma57 = {
        "ipopt": {
            "max_iter": 1000,
            "print_level": 5, # Verbose
            "sb": "no",
            "tol": 1e-6,
            "linear_solver": "ma57",
        },
        "print_time": 1,
    }
    try:
        # Enable Debug Feasibility to relax constraints and find the culprit
        result_57 = phase1_test(params, solver_opts=opts_ma57, debug_feasibility=True)
        print(f"MA57 Status: {result_57.get('status')}")
        print(f"MA57 Efficiency: {result_57.get('thermal_efficiency')*100:.2f} %")
    except Exception as e:
        print(f"MA57 Failed: {e}")

if __name__ == "__main__":
    test_efficiency()
