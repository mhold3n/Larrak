"""
Pytest configuration for Larrak test suite.

This conftest sets up the test environment, including skipping environment
validation during test collection to avoid segmentation faults caused by
HSL solver validation.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path to allow imports from campro
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "solver: tests that call CasADi/IPOPT")


# Set environment variable to skip validation during test collection
# This prevents segmentation faults when CamPro_OptimalMotion.py is imported
# and tries to validate HSL solvers by creating multiple CasADi solvers
os.environ["CAMPRO_SKIP_VALIDATION"] = "1"

# Also set a flag that can be checked in modules
sys._pytest_mode = True
