import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

try:
    from campro.orchestration.provenance import ProvenanceClient
    from provenance.schema import create_schema

    # Inspect ProvenanceClient methods
    client = ProvenanceClient(use_provenance=False)
    if not hasattr(client, "log_geometry"):
        print("FAIL: log_geometry method missing from ProvenanceClient")
        sys.exit(1)
    if not hasattr(client, "log_constraint_check"):
        print("FAIL: log_constraint_check method missing from ProvenanceClient")
        sys.exit(1)

    print("SUCCESS: ProvenanceClient has new logging methods.")

    # Inspect Schema structure (static analysis since we can't inspect weaviate state easily without running instance)
    import inspect

    schema_source = inspect.getsource(create_schema)

    if "Geometry" not in schema_source:
        print("FAIL: Geometry collection not found in create_schema source")
        sys.exit(1)

    if "ConstraintResult" not in schema_source:
        print("FAIL: ConstraintResult collection not found in create_schema source")
        sys.exit(1)

    if "generated_by" not in schema_source:
        print("FAIL: generated_by reference missing in Geometry definition")
        sys.exit(1)

    print("SUCCESS: Schema definition contains new collections.")

except ImportError as e:
    print(f"FAIL: ImportError: {e}")
    print("Ensure weaviate-client is installed.")
    sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
