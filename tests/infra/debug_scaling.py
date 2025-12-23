import numpy as np
import casadi as ca
from thermo.nlp import build_thermo_nlp
import pandas as pd


def debug_scaling():
    print("Building NLP...")
    nlp, meta = build_thermo_nlp(n_coll=50)
    w0 = meta["w0"]

    # 1. Evaluate Constraints
    print("\nEvaluating Constraints at w0...")
    g_fn = ca.Function("g", [nlp["x"]], [nlp["g"]])
    g_val = g_fn(w0).full().flatten()

    # Find max violation
    g_abs = np.abs(g_val)
    max_idx = np.argmax(g_abs)
    max_val = g_abs[max_idx]
    min_val = np.min(g_abs[g_abs > 0]) if np.any(g_abs > 0) else 0.0

    print(f"Max Constraint |g|: {max_val:.4e} at index {max_idx}")
    print(f"Min Non-Zero g: {min_val:.4e}")
    print(f"Log Range: {np.log10(max_val / min_val) if min_val > 0 else 'Inf'}")

    # Identify large constraints
    large_idxs = np.where(g_abs > 1e3)[0]
    if len(large_idxs) > 0:
        print(f"\nFound {len(large_idxs)} constraints > 1e3:")
        # Try to identify which constraint this is (Collocation vs Boundary)
        # N constraints = n_coll * n_states + boundaries
        n_coll = 50
        n_states = 5  # x, v, m_c, T_c, Y_f
        # The order in collocation builder depends on how they are added.
        # Usually: Defect_state1_k0, Defect_state2_k0...

        for idx in large_idxs[:10]:  # Print first 10
            print(f"  Index {idx}: {g_abs[idx]:.4e}")

    # 2. Variable Scaling
    print("\nVariable Scaling at w0:")
    w0_abs = np.abs(np.array(w0))
    print(f"Max |w|: {np.max(w0_abs):.4e}")
    print(f"Min |w|: {np.min(w0_abs[w0_abs > 0]):.4e}")

    very_small = np.where((w0_abs < 1e-6) & (w0_abs > 0))[0]
    print(f"Variables < 1e-6: {len(very_small)}")
    if len(very_small) > 0:
        print(f"  First 10 small vars indices: {very_small[:10]}")
        print(f"  Values: {w0_abs[very_small[:10]]}")


if __name__ == "__main__":
    debug_scaling()
