"""Verification script for driver.py refactoring."""

import os
import sys

# Add repo root to path
sys.path.append(os.getcwd())

try:
    print("Importing campro.optimization.driver...")
    from campro.optimization import driver

    print("Checking main solve_cycle function...")
    assert hasattr(driver, "solve_cycle")
    assert callable(driver.solve_cycle)

    print("Checking imports from scaling.py...")
    from campro.optimization.nlp import scaling

    assert hasattr(scaling, "analyze_constraint_rank")
    assert callable(scaling.analyze_constraint_rank)

    print("Checking imports from diagnostics.py...")
    from campro.optimization.nlp import diagnostics

    assert hasattr(diagnostics, "diagnose_nan_in_jacobian")
    assert callable(diagnostics.diagnose_nan_in_jacobian)

    print("Checking imports from solve_strategies.py...")
    from campro.optimization import solve_strategies

    assert hasattr(solve_strategies, "solve_cycle_robust")
    assert callable(solve_strategies.solve_cycle_robust)

    print("Checking imports from ipopt_options.py...")
    from campro.optimization.solvers import ipopt_options

    assert hasattr(ipopt_options, "create_ipopt_options")
    assert callable(ipopt_options.create_ipopt_options)

    print("SUCCESS: Refactoring verification passed.")
except Exception as e:
    print(f"FAILURE: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
