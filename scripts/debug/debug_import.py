
import sys
import os
# Add current dir to path explicitly
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

print(f"Sys Path: {sys.path}")

try:
    import Simulations.common.simulation
    print("Success: Simulations.common.simulation")
except ImportError as e:
    print(f"Failed common: {e}")
except Exception as e:
    print(f"Error common: {e}")

try:
    import Simulations.structural.friction
    print("Success: Simulations.structural.friction")
except ImportError as e:
    print(f"Failed structural: {e}")
except Exception as e:
    print(f"Error structural: {e}")
