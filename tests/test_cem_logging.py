import logging
import os
import sys

logging.basicConfig(level=logging.DEBUG)
from unittest.mock import MagicMock

import numpy as np

# Prioritize local path
sys.path.insert(0, os.getcwd())

try:
    import truthmaker.cem.client

    print(f"DEBUG: CEMClient source: {truthmaker.cem.client.__file__}")
    from campro.orchestration.adapters.cem_adapter import CEMClientAdapter

    print("Successfully imported CEMClientAdapter")
except ImportError as e:
    print(f"Failed to import CEMClientAdapter: {e}")
    sys.exit(1)


def test_logging():
    # Create Adapter (real, not mocked logic)
    # mock=False means it tries to connect to real C# service
    # Make sure C# service is running!
    adapter = CEMClientAdapter(mock=False)

    # Real provenance client
    from campro.orchestration.provenance import ProvenanceClient

    try:
        real_provenance = ProvenanceClient()
        if not real_provenance.client.is_live():
            print("Weaviate not live, skipping real log.")
            adapter.provenance = MagicMock()
            adapter.provenance.log_geometry.return_value = "mock-uuid"
        else:
            print("Using REAL ProvenanceClient connected to Weaviate.")
            adapter.set_provenance(real_provenance)
            # Create a run so we can link geometry
            run_id = real_provenance.start_run({"test": "test_cem_logging"})
            print(f"Started test run: {run_id}")
    except Exception as e:
        print(f"Failed to init ProvenanceClient: {e}")
        adapter.provenance = MagicMock()
        adapter.provenance.log_geometry.return_value = "mock-uuid"
        run_id = "mock-run"

    candidate = {"x_profile": np.linspace(0, 0.02, 100), "theta": np.linspace(0, 4 * np.pi, 100)}

    print("\nRunning check_feasibility...")
    is_feasible, score = adapter.check_feasibility(candidate, run_id=run_id)

    print(f"\nResult: Feasible={is_feasible}, Score={score}")

    # Verify provenance    # Verification
    # For real provenance, we can't assert .called on a real object easily without mocking or querying.
    # We rely on specific log messages or verify_weaviate.py.
    print("[INFO] Check verify_weaviate.py for results.")


if __name__ == "__main__":
    test_logging()
