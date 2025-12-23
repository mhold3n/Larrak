import numpy as np
import casadi as ca
from thermo.nlp import build_thermo_nlp


def debug_redundancy():
    print("Building NLP (Original Structure)...")
    # Must use original structure (no slacks) to find inherent redundancy
    nlp_dict, meta = build_thermo_nlp(n_coll=50, debug_feasibility=False)
    w0 = meta["w0"]

    print(f"Variables: {meta['n_vars']}")
    print(f"Constraints: {meta['n_constraints']}")

    print("Computing Jacobian...")
    J_fn = ca.Function("J_fn", [nlp_dict["x"]], [ca.jacobian(nlp_dict["g"], nlp_dict["x"])])
    J = J_fn(w0).full()

    m, n = J.shape
    print(f"Jacobian Shape: {m}x{n}")

    print("Computing SVD...")
    U, S, Vh = np.linalg.svd(J)

    # Dynamic tolerance based on spectral radius
    max_s = S[0]
    tol = max(max_s * 1e-10, 1e-8)
    print(f"Max SV: {max_s:.2e}, Tolerance: {tol:.2e}")
    print(f"Smallest 5 SVs: {S[-5:]}")

    rank = np.sum(S > tol)
    # Row deficiency = M - Rank (number of dependent rows)
    row_deficiency = m - rank

    print(f"Rank: {rank}")
    print(f"Row Deficiency (Redundant Constraints): {row_deficiency}")

    if row_deficiency == 0:
        print("No redundant constraints found at w0.")
        return

    print("\n=== Redundancy Analysis ===")
    # Build map idx -> name
    idx_to_group = {}
    for group, indices in meta["constraint_groups"].items():
        for idx in indices:
            idx_to_group[idx] = group

    # Columns of U corresponding to zero singular values span the left null space (row dependencies)
    null_cols_indices = []

    # 1. Columns corresponding to zero SVs
    for k in range(len(S)):
        if S[k] <= tol:
            null_cols_indices.append(k)

    # 2. Structural null space if M > N (remaining columns of U)
    # np.linalg.svd returns U of shape (M, M)
    for k in range(len(S), m):
        null_cols_indices.append(k)

    print(f"Found {len(null_cols_indices)} vectors in left null space.")

    # Aggregate participation of each constraint in the null space
    constraint_participation = np.zeros(m)
    for k in null_cols_indices:
        vec = np.abs(U[:, k])
        constraint_participation += vec

    # Report top groups participating in null space
    group_scores = {}
    for idx, score in enumerate(constraint_participation):
        if score > 1e-4:
            g = idx_to_group.get(idx, "unknown")
            group_scores[g] = group_scores.get(g, 0) + score

    print("\nTop Redundant Constraint Groups (Cumulative Participation):")
    sorted_groups = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
    for g, score in sorted_groups[:20]:
        print(f"{g:<30} : {score:.4f}")

    print("\nDetailed Single Constraint Check (Top 20 contributors):")
    indices = np.argsort(constraint_participation)[::-1]
    for i in indices[:20]:
        if constraint_participation[i] > 1e-4:
            print(f"Row {i:<4} ({idx_to_group.get(i, '?')}) : {constraint_participation[i]:.4f}")


if __name__ == "__main__":
    debug_redundancy()
