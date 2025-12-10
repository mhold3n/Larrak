import sys
from unittest.mock import MagicMock

print("Imported mock", flush=True)
sys.modules["matplotlib"] = MagicMock()
print("Mocked matplotlib", flush=True)
