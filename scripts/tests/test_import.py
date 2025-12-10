import sys
from unittest.mock import MagicMock

sys.modules["matplotlib"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()
sys.modules["matplotlib.figure"] = MagicMock()
sys.modules["campro.environment.env_manager"] = MagicMock()
sys.modules["campro.environment.hsl_detector"] = MagicMock()

with open("/tmp/debug_log.txt", "w") as f:
    f.write("Starting import test...\n")

try:

    with open("/tmp/debug_log.txt", "a") as f:
        f.write("Import successful!\n")
except Exception as e:
    with open("/tmp/debug_log.txt", "a") as f:
        f.write(f"Import failed: {e}\n")
