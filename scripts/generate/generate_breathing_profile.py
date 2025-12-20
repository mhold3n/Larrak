import os
import sys
import json
import numpy as np
import csv

# Setup Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from truthmaker.surrogates.adapters.breathing import BreathingKinematics
from truthmaker.surrogates.adapters.breathing_adapter import BreathingAdapter
from campro.litvin.config import PlanetSynthesisConfig
from campro.litvin.planetary_synthesis import synthesize_planet_from_motion

def generate_profiles():
    json_path = os.path.join(PROJECT_ROOT, "tests/goldens/phase4/valve_strategy/valve_strategy_results.json")
    out_dir = os.path.join(PROJECT_ROOT, "output/profiles")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(json_path, "r") as f:
        results = json.load(f)

    # Use the optimized R_p from previous run (approx 0.1m)
    # Or let BreathingKinematics auto-detect.
    bk = BreathingKinematics(stroke=0.1)
    
    print(f"Generating Breathing Gear Profiles...")

    for res in results:
        fuel = res["fuel"]
        theta_deg = np.array(res["theta_x"])
        theta_rad = np.radians(theta_deg)
        x_phys = np.array(res["x"])
        
        try:
            # 1. Kinematics
            C_arr, psi_arr, _ = bk.solve_trajectory(x_phys, theta_rad)
            
            # 2. Adapter
            adapter = BreathingAdapter(theta_rad, C_arr, psi_arr)
            
            # 3. Config
            # Standard constraints approx?
            # R0 ~ 0.1m?
            # Ring Teeth? Planet Teeth?
            # User said TDC/BDC 1:1.
            # Let's assume Ring=60, Planet=30 ? (Ratio 2:1 basic).
            # Or Ring=40, Planet=40 (Ratio 1:1, but impossible for internal).
            # Internal requires Ring > Planet.
            # If Ring=80, Planet=40. Ratio 2.
            # Let's try flexible geometry.
            
            cfg = PlanetSynthesisConfig(
                ring_teeth=80,
                planet_teeth=40,
                pressure_angle_deg=20.0,
                addendum_factor=1.0,
                base_center_radius=adapter.R0, # From Data
                samples_per_rev=360,
                motion=adapter
            )
            
            # 4. Synthesize
            profile = synthesize_planet_from_motion(cfg)
            
            # 5. Save
            fname = f"breathing_profile_{fuel}mg.csv"
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["x", "y"])
                for pt in profile.points:
                    writer.writerow(pt)
            
            print(f"  -> Generated {fname} ({len(profile.points)} pts)")
            
        except Exception as e:
            print(f"  -> Failed {fuel}mg: {e}")

if __name__ == "__main__":
    generate_profiles()
