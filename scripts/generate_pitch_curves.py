import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Setup Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from surrogate.breathing import BreathingKinematics

def generate_and_plot_pitch_curves():
    json_path = os.path.join(PROJECT_ROOT, "tests/goldens/phase4/valve_strategy/valve_strategy_results.json")
    
    with open(json_path, "r") as f:
        results = json.load(f)

    # Sort
    results.sort(key=lambda x: x["fuel"])

    # Figure
    rows = len(results)
    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=[f"{res['fuel']}mg Pitch Curves" for res in results],
        vertical_spacing=0.08
    )

    bk = BreathingKinematics(stroke=0.1)
    
    # Assume Ring Radius (Fixed Stator)
    # Must be larger than Max Center Distance to contain the planet
    # Max C is roughly 0.1m.
    # Let's pick R_ring = 0.15m (150mm)
    R_ring = 0.150 
    
    # Ring Circle (Visual)
    theta_circ = np.linspace(0, 2*np.pi, 200)
    ring_x = R_ring * np.cos(theta_circ)
    ring_y = R_ring * np.sin(theta_circ)

    for i, res in enumerate(results):
        row = i + 1
        fuel = res["fuel"]
        theta_rad = np.radians(np.array(res["theta_x"]))
        x_phys = np.array(res["x"])
        
        # 1. Kinematics
        C_arr, psi_arr, ratio_arr = bk.solve_trajectory(x_phys, theta_rad)
        
        # 2. Derive Planet Pitch Curve (Centrode)
        # Geometry: Internal Gear Pair
        # C = R_ring - R_planet
        # -> R_planet(theta) = R_ring - C(theta)
        
        R_planet_arr = R_ring - C_arr
        
        # Check Validity
        if np.any(R_planet_arr <= 0):
            print(f"Warning: {fuel}mg requires R_planet <= 0 at some points. Increase R_ring.")
            
        # 3. Transform to Planet Frame
        # The Pitch Point M is on the Line of Centers.
        # Length = R_planet_arr
        # Orientation in Global Frame = theta (Carrier Angle)
        # Orientation in Planet Frame?
        # Planet is rotated by psi_total = psi_arr (from kinematics) relative to Vertical?
        # Actually Phase 2c defined: x = C + r_p * cos(psi).
        # Assuming psi is the angle of the planet body relative to Global Vertical.
        # We need the vector to the Contact Point (M) in the Planet's Body Frame.
        # Vector C->M in Global = R_planet * [cos(theta), sin(theta)]
        # Planet Rotation Calc:
        # Rotation Matrix R(-psi) converts Global to Body.
        # Pt_body = R(-psi) * (Pt_global - C_global) ??
        # Center of Planet is at C_global.
        # Vector M - C_global = R_planet * [cos(theta), sin(theta)]
        # So in Planet Frame (aligned with psi):
        # angle_local = theta - psi
        
        # NOTE: psi_arr from BreathingKinematics was derived from x = C + r cos(psi).
        # This psi is "Angle of the Piston Pin Vector" in the Planet Frame?
        # Not necessarily the rotation of the planet itself?
        # Wait, usually Hypocycloid: x = ...
        # If we define psi as the Planet Rotation, then yes.
        
        angle_local = theta_rad - psi_arr
        
        planet_x = R_planet_arr * np.cos(angle_local)
        planet_y = R_planet_arr * np.sin(angle_local)
        
        # 4. Plot
        # Ring
        fig.add_trace(go.Scatter(
            x=ring_x, y=ring_y,
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name="Ring" if i==0 else None
        ), row=row, col=1)
        
        # Planet Pitch Curve
        # Center it at (0.05, 0) just for relative vis?
        # Or plot it "As if it were at TDC"?
        # Let's plot the SHAPE of the planet (In Body Frame).
        # To make it look like it's inside the ring, we need to locate it?
        # No, "Profile" usu means the shape itself.
        # But user wants "Ring, Planet, Sun".
        # If I plot the Ring centered at 0, and the Planet centered at 0, they overlap?
        # A Planet is smaller than the Ring.
        # I should plot the Planet Pitch Curve centered at (0,0) (Its own frame).
        # And maybe a shifted version?
        # Let's plot the Planet Pitch Curve centered at (0,0).
        
        fig.add_trace(go.Scatter(
            x=planet_x, y=planet_y,
            mode='lines',
            line=dict(color='red', width=2),
            name="Planet Pitch" if i==0 else None
        ), row=row, col=1)
        
        # Sun?
        # If Sun is the Carrier Center, it's at (0,0) of the Machine.
        # But we are in Planet Frame?
        # In Planet Frame, the Sun Center orbits the Planet?
        # User wants "Ring, Planet, Sun".
        # Maybe looking at the MECHANISM layout (Kinematic Diagram) at a specific instant (e.g. TDC)?
        # Or the "Profiles" (Shapes)?
        # "Gears are not knots" implies Shape.
        # I will plot the Planet Shape (Centrode).
        
        # Aspect Ratio
        fig.update_yaxes(scaleanchor=f"x{row}", scaleratio=1, row=row, col=1)

    fig.update_layout(
        title="Breathing Gear Pitch Curves (Planet Shape in Body Frame)",
        height=350 * rows,
        template="plotly_white",
        showlegend=True
    )
    
    out_path = os.path.join(PROJECT_ROOT, "dashboard/breathing_pitch_plot.html")
    fig.write_html(out_path)
    print(f"Saved Pitch Plot to {out_path}")

if __name__ == "__main__":
    generate_and_plot_pitch_curves()
