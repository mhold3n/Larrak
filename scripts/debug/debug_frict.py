
import sys
import os
sys.path.append(os.getcwd())

from Simulations.structural.friction import FrictionSimulation, FrictionConfig

try:
    sim = FrictionSimulation("test", FrictionConfig())
    print("Success: Instantiated FrictionSimulation")
    sim.setup()
    print("Success: Called output")
except Exception as e:
    print(f"Failed: {e}")
