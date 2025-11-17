"""
Pytest configuration for Larrak test suite.

This conftest sets up the test environment, including skipping environment
validation during test collection to avoid segmentation faults caused by
HSL solver validation.
"""

import os
import sys

# Set environment variable to skip validation during test collection
# This prevents segmentation faults when CamPro_OptimalMotion.py is imported
# and tries to validate HSL solvers by creating multiple CasADi solvers
os.environ["CAMPRO_SKIP_VALIDATION"] = "1"

# Also set a flag that can be checked in modules
sys._pytest_mode = True
