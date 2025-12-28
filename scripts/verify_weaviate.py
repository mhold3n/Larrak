"""Verify Weaviate connection and collection statistics."""

import os

import weaviate


def verify_connection() -> bool:
    """Connect to Weaviate and verify collections exist with data."""
    url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    print(f"Connecting to Weaviate at {url}...")

    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50052)
        print("Connected!")

        # Check meta
        meta = client.get_meta()
        print(f"Weaviate Version: {meta.get('version')}")

        # List Collections (v4 API)
        collections = client.collections.list_all()
        print("\nExisting Collections:")
        # Specific check for Runs
        if "Run" in collections:
            runs = client.collections.get("Run")
            count = runs.aggregate.over_all(total_count=True).total_count
            print(f"\n[VERIFICATION] Total Runs logged: {count}")
        else:
            print("\n[VERIFICATION] Run collection not found!")

        if "Geometry" in collections:
            geoms = client.collections.get("Geometry")
            count = geoms.aggregate.over_all(total_count=True).total_count
            print(f"[VERIFICATION] Total Geometry objects logged: {count}")
        else:
            print("[VERIFICATION] Geometry collection not found!")

        if "ConstraintResult" in collections:
            constraints = client.collections.get("ConstraintResult")
            count = constraints.aggregate.over_all(total_count=True).total_count
            print(f"[VERIFICATION] Total ConstraintResult objects logged: {count}")
        else:
            print("[VERIFICATION] ConstraintResult collection not found!")

        client.close()
        return True
    except (ConnectionError, TimeoutError, ValueError) as e:
        print(f"Connection failed: {e}")
        return False


if __name__ == "__main__":
    verify_connection()
