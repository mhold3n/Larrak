import sqlite3
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from provenance.db import db
from provenance.spec import Artifact, FileRole

DB_PATH = Path("provenance.db")

def migrate():
    if not DB_PATH.exists():
        print(f"No existing database found at {DB_PATH}")
        return

    print(f"Migrating data from {DB_PATH} to Weaviate...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 1. Migrate Runs
    runs = c.execute("SELECT * FROM runs").fetchall()
    print(f"Found {len(runs)} runs.")
    
    for row in runs:
        run_id = row['run_id']
        module_id = row['module_id']
        try:
            args = json.loads(row['args']) if row['args'] else []
            env = json.loads(row['env']) if row['env'] else {}
        except:
            args = []
            env = {}

        # Use db.start_run to create the Run and Module objects
        # Note: start_run sets status to RUNNING and time to NOW.
        # We need a way to insert historical data.
        # Since db.start_run logic is simple, we will use the internal client directly 
        # to respect historical timestamps if we wanted perfection, 
        # but for this MVP migration reused logic is safer.
        # ACTUALLY: We should just use the client to get precise control over ID and timestamps.
        
        # Let's use the Weaviate client directly via the 'db' instance helper
        # to avoid overwriting timestamps.
        if not db._client:
            print("Weaviate client not connected.")
            return

        print(f"Migrating run {run_id}...")
        
        # Create/Get Module
        module_col = db._client.collections.get("Module")
        module_uuid = db._get_or_create_module(module_col, module_id)
        
        # Create Run
        run_col = db._client.collections.get("Run")
        
        import weaviate.util
        run_uuid = weaviate.util.generate_uuid5(run_id)
        
        # Parse timestamps? They are stored as strings or timestamps in SQLite.
        # For simplicity, we might just re-use them as strings if the schema allowed,
        # but our schema expects Date. Weaviate handles ISO strings.
        start_time = row['start_time']
        end_time = row['end_time']
        
        try:
            run_col.data.insert(
                uuid=run_uuid,
                properties={
                    "run_id": run_id,
                    "start_time": start_time, # Weaviate auto-parses ISO-8601
                    "end_time": end_time,
                    "status": row['status'],
                    "args": args,
                    "env": json.dumps(env),
                },
                references={
                    "executed_module": module_uuid
                }
            )
        except Exception as e:
            print(f"Skipping run {run_id} (maybe exists): {e}")

    # 2. Migrate Artifacts
    artifacts = c.execute("SELECT * FROM artifacts").fetchall()
    print(f"Found {len(artifacts)} artifacts.")

    for row in artifacts:
        # Reconstruct Artifact object to use helper? 
        # Or direct insert. Direct insert is better for batching usually, but here 
        # we will just use the register_artifact method which handles linking.
        # Wait, register_artifact generates UUIDs. We want to be consistent.
        
        art = Artifact(
            artifact_id=row['artifact_id'],
            path=row['path'],
            content_hash=row['content_hash'],
            run_id=row['run_id'],
            producer_module_id="unknown", # explicit field in spec, not in DB schema in older version?
            role=FileRole(row['role']),
            size_bytes=0, # not in DB
            creation_time=None, # not in DB
            metadata=json.loads(row['meta']) if row['meta'] else {}
        )
        
        print(f"Migrating artifact {art.path}...")
        db.register_artifact(art)

    print("Migration complete.")
    conn.close()

if __name__ == "__main__":
    migrate()
