#!/usr/bin/env python3
import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from campro.litvin.opt.conjugate_optimizer import ConjugateOptimizer


def run_conjugate_opt():
    print("[DEBUG] Starting conjugate optimization...")
    json_path = os.path.join(
        PROJECT_ROOT, "tests/goldens/phase4/valve_strategy/valve_strategy_results.json"
    )

    print(f"[DEBUG] Loading motion law from: {json_path}")
    with open(json_path, "r") as f:
        results = json.load(f)
    print(f"[DEBUG] Loaded {len(results)} results")

    # Pick 55mg Case for design
    res = next(r for r in results if abs(r["fuel"] - 55.0) < 1.0)
    theta_rad = np.radians(np.array(res["theta_x"]))
    x_phys = np.array(res["x"])

    # Create 2-Cycle Target (1 Carrier Rev = 2 Engine Cycles)
    # Original: x_phys over 360 crank degrees.
    # We want to map TWO of these cycles into ONE Carrier Revolution (360 degrees).
    # New Theta = 0..2pi.

    # Concatenate 2 cycles
    x_2cycle = np.concatenate([x_phys, x_phys])
    # Create new theta grid 0..2pi with double points
    theta_2cycle = np.linspace(0, 2 * np.pi, len(x_2cycle))

    print(f"[DEBUG] 2-cycle profile: {len(theta_2cycle)} points")
    opt = ConjugateOptimizer(n_points=len(x_2cycle))
    print(f"Optimizing Conjugate Pair for 55mg (2-Cycle 2:1 Ratio)...")
    print("[DEBUG] Calling opt.solve() - this may take a while or hang...")

    # Target Psi: 2 Rotations = 4pi
    # Symmetry: 2 Blocks (identital halves)
    sol = opt.solve(theta_2cycle, x_2cycle, target_psi_total=4 * np.pi, symmetry_blocks=2)
    print(f"[DEBUG] opt.solve() returned. Success: {sol.get('success', 'UNKNOWN')}")

    if not sol["success"]:
        print("Optimization Failed! Plotting debug values.")

    # Extract Results
    Rr = sol["Rr"]
    Rp = sol["Rp"]
    C = sol["C"]
    Psi = sol["Psi"]

    # Plotting
    # 1. Radii Profiles (Time Domain)
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=theta_2cycle, y=Rr, name="Ring Radius"))
    fig_r.add_trace(go.Scatter(x=theta_2cycle, y=Rp, name="Planet Radius"))
    fig_r.add_trace(go.Scatter(x=theta_2cycle, y=C, name="Center Distance", line=dict(dash="dash")))
    fig_r.update_layout(
        title="Optimized Gear Radii vs Carrier Angle (2-Cycle 2:1)", yaxis_title="Radius [m]"
    )

    # 2. Polar Pitch Curves (Shapes)
    # Ring Shape (in Carrier Frame? No, Ring Frame)
    # Ring is Output. If Ring Speed is roughly const vs Carrier.
    # Polar plot of Rr(angle).

    # Planet Shape (in Planet Frame)

    # Note: Psi is monotonic increasing? Check.
    psi_wrap = Psi % (2 * np.pi)

    fig_polar = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "polar"}, {"type": "polar"}]],
        subplot_titles=["Ring Pitch Curve", "Planet Pitch Curve"],
    )

    fig_polar.add_trace(
        go.Scatterpolar(
            r=Rr,
            theta=np.degrees(theta_2cycle),  # Approx mapping
            mode="lines",
            name="Ring",
            fill="toself",
        ),
        row=1,
        col=1,
    )

    fig_polar.add_trace(
        go.Scatterpolar(
            r=Rp,
            theta=np.degrees(psi_wrap),  # Mapped to Planet Spin
            mode="lines",
            name="Planet",
            fill="toself",
        ),
        row=1,
        col=2,
    )

    fig_polar.update_layout(title="Conjugate Pitch Shapes", template="plotly_dark")

    # Save
    fig_r.write_html(os.path.join(PROJECT_ROOT, "output/conjugate_radii.html"))
    fig_polar.write_html(os.path.join(PROJECT_ROOT, "output/conjugate_shapes.html"))
    print("Saved optimization plots to dashboard/")


if __name__ == "__main__":
    run_conjugate_opt()
