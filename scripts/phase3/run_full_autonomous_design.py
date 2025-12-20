import subprocess
import sys
import os
import time

def run_step(script_name, description):
    print(f"\n===================================================")
    print(f">>> PIPELINE STEP: {description}")
    print(f">>> Script: {script_name}")
    print(f"===================================================")
    
    cmd = [sys.executable, script_name]
    start_time = time.time()
    ret = subprocess.call(cmd)
    duration = time.time() - start_time
    
    if ret != 0:
        print(f"\n[!!!] STEP FAILED: {script_name} returned exit code {ret}")
        print("Pipeline Aborted.")
        sys.exit(ret)
    else:
        print(f"\n[OK] STEP COMPLETED in {duration:.2f}s")

def main():
    print("###################################################")
    print("# LARRAK AUTONOMOUS DESIGN PIPELINE               #")
    print("# Protocol: Thermo -> Motion -> Gear Kinematics   #")
    print("###################################################")
    
    # 1. Thermo Calibration: Adaptive DOE Loop
    # Goal: Converged Thermodynamic Cycle
    run_step("scripts/run_thermo_calibration.py", "Thermo Calibration: Adaptive Thermodynamic DOE")
    
    # Check Output
    p1_out = "output/thermo/thermo_doe_results.csv"
    if not os.path.exists(p1_out):
        print(f"Error: Thermo Calibration output {p1_out} missing even after success code.")
        sys.exit(1)
        
    # 2. Motion Mapping: Motion Extraction & Interpretation
    # Goal: Best Motion Profile from Thermo Calibration
    run_step("scripts/export_optimal_motion.py", "Motion Mapping: Optimal Motion Extraction")
    
    # Check Output
    # export_optimal_motion saves to dashboard/orchestration/target_motion_profile.csv and copies to dashboard/
    p2_source_out = "output/orchestration/target_motion_profile.csv"
    p2_target_out = "output/final_motion_profile.csv"
    
    if os.path.exists(p2_source_out):
         if not os.path.exists(p2_target_out) or os.path.getmtime(p2_source_out) > os.path.getmtime(p2_target_out):
              print(f"Transferring {p2_source_out} -> {p2_target_out}")
              import shutil
              shutil.copy2(p2_source_out, p2_target_out)
    
    if not os.path.exists(p2_target_out):
        print(f"Error: Motion Mapping output {p2_target_out} missing.")
        sys.exit(1)

    # 3. Gear Kinematics: Adaptive Gear Design Loop
    # Goal: Valid Conjugate Gear Geometry
    run_step("scripts/run_adaptive_gear_design.py", "Gear Kinematics: Adaptive Conjugate Gear Optimization")
    
    p3_out = "output/gear_kinematics_results.csv"
    if not os.path.exists(p3_out):
        print(f"Error: Gear Kinematics output {p3_out} missing.")
        sys.exit(1)
        
    print("\n###################################################")
    print("# PIPELINE SUCCESS: Production Gearset Generated #")
    print(f"# Final Artifact: {p3_out}")
    print("###################################################")

if __name__ == "__main__":
    main()
