from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results(result, output_dirs=None):
    """
    Plot Planet-Ring optimization results.

    Args:
        result: optimization_lib.Solution object returned by solve_cycle
        output_dirs: List of directories to save plots (or single path)
    """
    if output_dirs is None:
        output_dirs = ["plots"]

    if isinstance(output_dirs, (str, Path)):
        output_dirs = [output_dirs]

    # ensure all dirs exist
    for d in output_dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Primary dir for main logic
    primary_dir = Path(output_dirs[0])

    # 1. Extract Data
    # ----------------

    # Shooting (Initial Guess)
    traj_init = result.data.get("initial_trajectory")

    # Collocation (Final Result)
    # We need to extract this from result.data['x'] using the adapter/NLP logic
    # But result.solution.data might already have processed states?
    # Let's check how FreePistonPhase1Adapter extracts it.
    # For now, let's look at what's in 'result.meta["optimization"]["x_opt"]' (raw vector)
    # or better, use the 'iteration_summary' or 'states' if available.

    # Actually, the driver returns result which IS a Solution object.
    # Does it have parsed states in result.data?
    # result.data['x'] is just the raw vector.

    # We need to map the raw vector x_opt back to physical variables.
    # This usually requires the 'meta' dictionary from build_collocation_nlp.
    meta = result.meta.get("meta")
    x_opt = result.data.get("x")

    if x_opt is None or meta is None:
        print("Error: Missing optimization data or metadata.")
        return

    # Helper to extract variable from x_opt
    def get_var(name):
        # 1. Try detailed map (preferred)
        detailed = meta.get("variables_detailed", {})
        indices = detailed.get(name, [])

        # 2. Try without underscores (e.g., x_L -> xL)
        if not indices and "_" in name:
            indices = detailed.get(name.replace("_", ""), [])

        # 3. Try with underscores (e.g., xL -> x_L)
        if not indices and "_" not in name:
            for alt in [name[:1] + "_" + name[1:], name.lower()]:
                indices = detailed.get(alt, [])
                if indices:
                    break

        # 4. Try groups (fallback)
        if not indices:
            indices = meta.get("variable_groups", {}).get(name, [])

        if not indices:
            return None
        return np.array([x_opt[i] for i in indices])

    # Debug: Print available groups
    print(f"Available Variable Groups: {list(meta.get('variable_groups', {}).keys())}")
    print(
        f"Available Detailed Vars: {list(meta.get('variables_detailed', {}).keys())[:10]}..."
    )

    # Extract Collocation States (try both naming conventions)
    theta_p = get_var("psi")  # Planet Angle (Degree Space)
    xL = get_var("xL")
    if xL is None and theta_p is not None:
        # Degree Space: Calculate xL algebraically
        # xL = (R - r) * cos(psi)  (Simplified approx for now)
        r_p = get_var("r_planet")
        R_r = get_var("R_ring")
        if r_p is not None and R_r is not None:
            # Resize controls to match state length if needed
            # Controls: (K,), States: (N_total,)
            # We assume piecewise constant controls for now (step per interval)
            # Typically plot_results gets the full array.

            # Simple heuristic: if len(r_p) < len(theta_p):
            # We usually have K intervals. State has K*(C+1)+1 points?

            def match_shape(ctrl, target_arr):
                if len(ctrl) == len(target_arr):
                    return ctrl
                if len(ctrl) < len(target_arr):
                    # Assume ctrl is per-interval (K). target is (K*(C+1)+1) or similar.
                    # We repeat each control value C+1 times? Or extrapolate?
                    # Let's try simple repetition for each interval.
                    # ratio = len(target_arr) // len(ctrl) # Rough check

                    # Better: use resize/repeat.
                    # If K=40, N=161. 161 / 40 ~ 4. (C=3, +1 for endpoint).
                    # So C+1 = 4. 40*4 = 160. +1 for final point?
                    # Standard collocation: Controls are U_0 ... U_{K-1}.
                    # We apply U_k to all points in interval k.

                    matched = []
                    K_ctrl = len(ctrl)
                    N_target = len(target_arr)

                    # Check if we can deduce C
                    # N = K*(C+1) + 1  => (N-1)/K = C+1
                    c_plus_1 = (N_target - 1) // K_ctrl

                    for k in range(K_ctrl):
                        val = ctrl[k]
                        matched.extend([val] * c_plus_1)

                    # Add final point (reuse last control)
                    while len(matched) < N_target:
                        matched.append(ctrl[-1])

                    return np.array(matched[:N_target])
                return ctrl

            r_p_expanded = match_shape(r_p, theta_p)
            R_r_expanded = match_shape(R_r, theta_p)

            xL = (R_r_expanded - r_p_expanded) * np.cos(theta_p)
            print(
                f"Computed xL from Kinematics. Shapes: r={r_p_expanded.shape}, psi={theta_p.shape}"
            )

    # Legacy fallback
    if xL is None:
        xL = get_var("x_L")

    vL = get_var("vL")
    if vL is None:
        vL = get_var("v_L")

    rho = get_var("rho")
    T = get_var("T")

    # For symmetric piston: xR = -xL, vR = -vL
    xR = (
        -xL
        if xL is not None
        else (get_var("xR") if get_var("xR") is not None else get_var("x_R"))
    )
    vR = (
        -vL
        if vL is not None
        else (get_var("vR") if get_var("vR") is not None else get_var("v_R"))
    )

    # Litvin / Extended States
    psi = get_var("psi")
    i_ratio = get_var("i_ratio")
    # F_cam usually F_cam or F_cam_state
    F_cam = get_var("F_cam") if get_var("F_cam") is not None else get_var("F_cam_state")

    # Time grid
    K = meta.get("K", 20)
    C = meta.get("C", 3)
    stride = C + 1

    cycle_time_indices = meta.get("variables_detailed", {}).get("t_cycle", [])
    if not cycle_time_indices:
        cycle_time_indices = meta.get("variable_groups", {}).get("cycle_time", [])
    if cycle_time_indices:
        t_cycle = x_opt[cycle_time_indices[-1]]  # Use last value (end of cycle)
    else:
        t_cycle = 0.02  # Default fallback

    t_coll = np.linspace(0, t_cycle, K + 1)

    def extract_knots(arr):
        if arr is None:
            return None
        # If array matches full collocation size (K*(C+1) + 1), subsample
        expected_full_size = K * (C + 1) + 1
        if len(arr) == expected_full_size:
            return arr[::stride]
        elif len(arr) == K + 1:
            return arr  # Already at knot level
        return arr

    # Extract Collocation States (Knots)
    xL = extract_knots(xL)
    xR = extract_knots(xR)
    vL = extract_knots(vL)
    vR = extract_knots(vR)

    # Plot 1: Kinematics Comparison
    # -----------------------------
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Shooting (if available)
    if traj_init:
        t_shot = traj_init["t"]
        # Comparison: Right Piston
        ax[0].plot(
            t_shot,
            traj_init["xR"],
            "k--",
            alpha=0.6,
            linewidth=1.5,
            label="Shooting (Rough)",
        )
        # Comparison: Left Piston (same style, just context)
        ax[0].plot(t_shot, traj_init["xL"], "k--", alpha=0.6, linewidth=1.5)

        ax[1].plot(
            t_shot,
            traj_init["vR"],
            "k--",
            alpha=0.6,
            linewidth=1.5,
            label="Shooting (Rough)",
        )
        ax[1].plot(t_shot, traj_init["vL"], "k--", alpha=0.6, linewidth=1.5)

    # Plot Collocation
    if xR is not None and xL is not None:
        # Right Piston = Red, Left Piston = Blue
        # Solid = Position, Dotted = Velocity?
        # Actually standard: Position top, velocity bottom.

        ax[0].plot(
            t_coll,
            xR,
            "r.-",
            linewidth=2,
            markersize=8,
            label="Collocation: Right Piston ($x_R$)",
        )
        ax[0].plot(
            t_coll,
            xL,
            "b.-",
            linewidth=2,
            markersize=8,
            label="Collocation: Left Piston ($x_L$)",
        )

        ax[0].set_ylabel("Position (m)")
        ax[0].legend(loc="best")
        ax[0].grid(True, alpha=0.3)
        ax[0].set_title("Piston Kinematics: Shooting vs Collocation Optimization")

    if vR is not None and vL is not None:
        ax[1].plot(
            t_coll,
            vR,
            "r.-",
            linewidth=2,
            markersize=8,
            label="Collocation: Right Piston ($v_R$)",
        )
        ax[1].plot(
            t_coll,
            vL,
            "b.-",
            linewidth=2,
            markersize=8,
            label="Collocation: Left Piston ($v_L$)",
        )

        ax[1].set_ylabel("Velocity (m/s)")
        ax[1].set_xlabel("Time (s)")
        ax[1].legend(loc="best")
        ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.tight_layout()
    for d in output_dirs:
        # Save sequential version
        # Save overwrite version
        plt.savefig(Path(d) / "kinematics_comparison.png", dpi=150)
        # Also save 'latest' alias
        plt.savefig(Path(d) / "kinematics_comparison_latest.png", dpi=150)
    plt.close()

    # Plot 2: Polar Cam Geometry Check (Conjugate Profiles)
    # --------------------------------
    profiles = result.meta.get("optimization", {}).get("profiles")

    if profiles or xR is not None:
        fig, ax = plt.subplots(figsize=(10, 10))

        if profiles:
            # Use explicit conjugate profiles
            Rx = np.array(profiles["Rx"])
            Ry = np.array(profiles["Ry"])
            Px = np.array(profiles["Px"])
            Py = np.array(profiles["Py"])

            # --- DEBUG: Data Stats ---
            print("\n[DEBUG] Profile Data Stats:")
            print(f"  Rx: Range=[{Rx.min():.4f}, {Rx.max():.4f}], Size={len(Rx)}")
            print(f"  Ry: Range=[{Ry.min():.4f}, {Ry.max():.4f}]")
            print(f"  Px: Range=[{Px.min():.4f}, {Px.max():.4f}]")
            print(f"  Py: Range=[{Py.min():.4f}, {Py.max():.4f}]")

            psi_stats = np.array(profiles.get("psi", []))
            t_stats = np.array(profiles.get("t", []))
            i_stats = np.array(profiles.get("i", []))

            if len(psi_stats) > 0:
                print(
                    f"  Psi: Range=[{psi_stats[0]:.4f}, {psi_stats[-1]:.4f}] rad ({np.degrees(psi_stats[0]):.1f}, {np.degrees(psi_stats[-1]):.1f} deg)"
                )
                print(
                    f"       Delta={psi_stats[-1] - psi_stats[0]:.4f} rad ({np.degrees(psi_stats[-1] - psi_stats[0]):.1f} deg)"
                )
            if len(t_stats) > 0:
                print(f"  Time: Max={t_stats[-1]:.4f}s")
            if len(i_stats) > 0:
                print(f"  Ratio (i): Range=[{i_stats.min():.4f}, {i_stats.max():.4f}]")
            # -------------------------

            # Close the loop for plotting if needed
            if len(Rx) > 0 and (Rx[0] != Rx[-1] or Ry[0] != Ry[-1]):
                Rx = np.append(Rx, Rx[0])
                Ry = np.append(Ry, Ry[0])
                Px = np.append(Px, Px[0])
                Py = np.append(Py, Py[0])

            # --- Transformed Planet for Proper Meshing Visualization ---
            # To show contact, we must transform the Planet Pitch Curve (Body Frame)
            # into the Fixed Frame at a specific instant (e.g. t=0).

            # 1. Get Kinematic Parameters
            if "R" in profiles and "r" in profiles:
                # Direct calculation from controls
                a_est = profiles["R"][0] - profiles["r"][0]
            else:
                # Fallback estimation
                R0 = np.sqrt(Rx[0] ** 2 + Ry[0] ** 2)
                i0 = 2.0  # Default if missing
                if len(i_stats) > 0:
                    i0 = i_stats[0]
                # Estimate a at k=0
                a_est = R0 * (i0 - 1.0) / i0

            # Snapshots to plot (e.g. Start and maybe others)
            # Snapshots to plot (Start, 1/3 way, 2/3 way)
            N = len(Rx)
            snapshots = [0, N // 3, 2 * N // 3] if N > 3 else [0]

            # Colors for snapshots
            colors = ["r", "m", "g"]

            for idx, k in enumerate(snapshots):
                # Parameters at instant k
                if "psi" in profiles:
                    psi_k = profiles["psi"][k]
                    t_k = profiles["t"][k] if "t" in profiles else 0.0
                else:
                    psi_k = 0.0
                    t_k = 0.0

                # Arm Angle phi (assuming omega * t)
                # But we don't have omega easily.
                # Direction of R vector gives phi!
                phi_k = np.arctan2(Ry[k], Rx[k])

                # Planet Center Op
                # Op is along phi vector at distance a
                Ox_k = a_est * np.cos(phi_k)
                Oy_k = a_est * np.sin(phi_k)

                # Rotation: The Planet Body was created by Rot(-psi).
                # To bring it back to "aligned with Op-M vector": Rot(+psi)
                # To bring it to global frame: Add Op?
                # P_body = Rot(-psi) * (M - Op)
                # => Rot(psi) * P_body = M - Op
                # => M_reconstructed = Op + Rot(psi) * P_body

                cos_psi = np.cos(psi_k)
                sin_psi = np.sin(psi_k)

                P_fixed_x = Ox_k + (Px * cos_psi - Py * sin_psi)
                P_fixed_y = Oy_k + (Px * sin_psi + Py * cos_psi)

                color = colors[idx % len(colors)]
                ax.plot(
                    P_fixed_x,
                    P_fixed_y,
                    color=color,
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Planet (t={t_k:.3f}s)",
                )

                # Mark the Center Op
                ax.plot(Ox_k, Oy_k, "ro", markersize=5, label="Planet Center")

                # Mark the expected contact point M (from Ring data)
                ax.plot(Rx[k], Ry[k], "ko", markersize=4, label="Contact Point")

            # ax.plot(Px, Py, "r--", linewidth=1.5, label="Planet Pitch (Body)") # Don't plot body frame, confusing
            # -----------------------------------------------------------

            # Plot reference circle
            theta = np.linspace(0, 2 * np.pi, 100)
            ax.plot(Rx, Ry, "b-", linewidth=2, label="Ring Pitch Curve (Fixed)")
            # ax.plot(Px, Py, "r--", linewidth=2, label="Planet Pitch Curve (Body)")

            # Plot center point
            ax.plot(0, 0, "k+", markersize=10, label="Ring Center")

            ax.axis("equal")
            ax.set_title("Optimized Conjugate Gear Profiles")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        else:
            # Fallback to generic polar plot (previous logic)
            # Re-create polar ax
            plt.close(fig)
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 10))

            theta_coll = (t_coll / t_cycle) * 2 * np.pi

            # Ring Radius (Outer Bound)
            r_ring = np.max(xR)

            # Planet Radius (Assuming 2:1 ratio)
            r_planet = r_ring / 2.0

            # Reference Circles
            theta_grid = np.linspace(0, 2 * np.pi, 200)
            ax.plot(
                theta_grid,
                [r_ring] * 200,
                "k-",
                alpha=0.2,
                label=f"Ref Ring R={r_ring * 1000:.1f}mm",
            )

            # Optimised Ring Shape (R_ring vs Theta)
            # Need to extract R_ring from optimization result
            # We already have theta_coll. We need R_ring at those points.
            R_ring_var = get_var("R_ring")
            if R_ring_var is not None:
                R_ring_plot = extract_knots(R_ring_var)
                # Ensure length matches theta_coll
                if len(R_ring_plot) == len(theta_coll) - 1:
                    R_ring_plot = np.append(R_ring_plot, R_ring_plot[-1])

                if len(R_ring_plot) == len(theta_coll):
                    ax.plot(
                        theta_coll,
                        R_ring_plot,
                        "g-",
                        linewidth=3,
                        label="Optimized Ring Gear Shape",
                    )

            # Nominal Path (Piston Stroke)
            ax.plot(
                theta_coll,
                xR,
                "r--",
                linewidth=1.5,
                alpha=0.7,
                label="Piston Path (Lift)",
            )

            ax.set_theta_zero_location("N")
            # Enforce circular aspect
            # ax.set_aspect("equal") # Not supported in polar, but we can set rmax
            # ax.set_rlim(0, r_ring * 1.1)
        plt.tight_layout()
        for d in output_dirs:
            # Use base name depending on what was plotted
            base = "conjugate_profiles" if profiles else "polar_profile"
            fpath = Path(d) / f"{base}.png"
            plt.savefig(fpath, dpi=150)
            plt.savefig(Path(d) / f"{base}_latest.png", dpi=150)
        plt.close()

    # Plot 3: Litvin Variables (i, psi, F_cam)
    # ----------------------------------------
    if psi is not None or i_ratio is not None or F_cam is not None:
        extract_knots_safe = lambda x: extract_knots(x) if x is not None else None

        psi_k = extract_knots_safe(psi)
        i_ratio_k = extract_knots_safe(i_ratio)
        F_cam_k = extract_knots_safe(F_cam)

        n_subplots = sum(x is not None for x in [psi_k, i_ratio_k, F_cam_k])
        if n_subplots > 0:
            fig, ax = plt.subplots(
                n_subplots, 1, figsize=(12, 4 * n_subplots), sharex=True
            )
            if n_subplots == 1:
                ax = [ax]

            idx = 0
            if i_ratio_k is not None:
                # Handle Controls (Len=K) vs Time (Len=K+1)
                # Controls are usually piecewise constant. Duplicate last value for plotting.
                y_plot = i_ratio_k
                if len(y_plot) == len(t_coll) - 1:
                    y_plot = np.append(y_plot, y_plot[-1])

                ax[idx].plot(t_coll, y_plot, "g.-", linewidth=2, label="Gear Ratio (i)")
                ax[idx].set_ylabel("Inst. Ratio")
                ax[idx].legend(loc="best")
                ax[idx].grid(True, alpha=0.3)
                idx += 1

            if psi_k is not None:
                ax[idx].plot(
                    t_coll,
                    np.degrees(psi_k),
                    "m.-",
                    linewidth=2,
                    label="Planet Angle (psi)",
                )
                ax[idx].set_ylabel("Angle (deg)")
                ax[idx].legend(loc="best")
                ax[idx].grid(True, alpha=0.3)
                idx += 1

            if F_cam_k is not None:
                ax[idx].plot(
                    t_coll, F_cam_k, "c.-", linewidth=2, label="Reaction Force (F_cam)"
                )
                ax[idx].set_ylabel("Force (N)")
                ax[idx].legend(loc="best")
                ax[idx].grid(True, alpha=0.3)
                idx += 1

            ax[-1].set_xlabel("Time (s)")
            ax[0].set_title("Litvin Kinematics & Dynamics")

            plt.tight_layout()
            plt.tight_layout()
            for d in output_dirs:
                fpath = Path(d) / "litvin_variables.png"
                plt.savefig(fpath, dpi=150)
                plt.savefig(Path(d) / "litvin_variables_latest.png", dpi=150)
            plt.close()

    print(f"Plots saved to {output_dirs}")
