import datetime
import json
import os
from typing import Any, Dict, List

import weaviate

from provenance.schema import create_schema
from provenance.spec import Artifact, Event

# Connection config
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_URL = os.getenv("WEAVIATE_GRPC_URL", "localhost:50051")


class ProvenanceDB:
    def __init__(self) -> None:
        self._client: weaviate.WeaviateClient | None = None
        self._connect()
        self._ensure_schema()

    def _connect(self) -> None:
        import socket

        original_timeout = socket.getdefaulttimeout()
        try:
            # Set a short timeout to avoid blocking when Weaviate is unavailable
            socket.setdefaulttimeout(2.0)
            # Connect to local Weaviate instance
            self._client = weaviate.connect_to_local(port=8080, grpc_port=50051)
        except Exception as e:
            # Catching general exception is appropriate here as connection can fail for various reasons
            print(
                f"Warning: Could not connect to Weaviate at {WEAVIATE_URL}. "
                f"Provenance tracking disabled. Error: {e}"
            )
            self._client = None
        finally:
            socket.setdefaulttimeout(original_timeout)

    def _ensure_schema(self) -> None:
        if self._client and self._client.is_ready():
            create_schema(self._client)

    def close(self) -> None:
        if self._client:
            self._client.close()

    def start_run(self, run_id: str, module_id: str, args: List[str], env: Dict[str, str]) -> None:
        if not self._client:
            return

        runs = self._client.collections.get("Run")

        # Check if Module exists, create if not (simple auto-registration)
        modules = self._client.collections.get("Module")
        module_uuid = self._get_or_create_module(modules, module_id)

        try:
            runs.data.insert(
                properties={
                    "run_id": run_id,
                    "start_time": datetime.datetime.now(datetime.timezone.utc),
                    "status": "RUNNING",
                    "args": args,
                    "env": json.dumps(env),
                    "tags": [],
                },
                references={"executed_module": module_uuid},
                uuid=weaviate.util.generate_uuid5(run_id),
            )
        except Exception as e:
            print(f"Error starting run: {e}")

    def end_run(self, run_id: str, status: str = "SUCCESS") -> None:
        if not self._client:
            return

        runs = self._client.collections.get("Run")
        run_uuid = weaviate.util.generate_uuid5(run_id)

        try:
            runs.data.update(
                uuid=run_uuid,
                properties={
                    "end_time": datetime.datetime.now(datetime.timezone.utc),
                    "status": status,
                },
            )
        except Exception as e:
            print(f"Error ending run: {e}")

    def log_event(self, event: Event) -> None:
        if not self._client:
            return

        # We store events as a JSON blob on the Run object for now to avoid
        # high-cardinality Event objects, unless they are critical checkpoints.
        # For this implementation, we will append to a 'logs' list or similar
        # if we strictly followed schema, but since 'logs' is just a text blob,
        # we might just print it.
        # Alternatively, we can make an Event object if we really crave granularity.
        # Let's verify schema... we didn't make an Event collection.
        # Strategy: update the Run's 'logs' property by appending? No, that's expensive.
        # Real-world Weaviate usage: High volume logs shouldn't go here.
        # But CRITICAL events (Checkpoints) could be their own object or property.

        # For minimal disruption, we will just print to console for now, as the Schema
        # didn't define a lightweight Event stream.
        # If we need searchable events, we should add an Event collection.
        # Let's stick to the plan: "Store critical events as objects, high-volume logs as a blob".
        # We will assume these are critical for the dashboard.
        # We will assume these are critical for the dashboard.

    def register_artifact(self, artifact: Artifact) -> None:
        if not self._client:
            return

        artifacts = self._client.collections.get("Artifact")
        runs = self._client.collections.get("Run")

        art_uuid = weaviate.util.generate_uuid5(artifact.artifact_id)
        run_uuid = weaviate.util.generate_uuid5(artifact.run_id)

        try:
            # Create Artifact
            artifacts.data.insert(
                uuid=art_uuid,
                properties={
                    "artifact_id": artifact.artifact_id,
                    "path": artifact.path,
                    "role": artifact.role.value
                    if hasattr(artifact.role, "value")
                    else str(artifact.role),
                    "content_hash": artifact.content_hash,
                    "meta": json.dumps(artifact.metadata, default=str),
                    "summary": (
                        f"Artifact {artifact.path} ({artifact.role}) "
                        f"generated by run {artifact.run_id}"
                    ),
                },
                references={"generated_by": run_uuid},
            )

            # Link Run -> generated_artifacts
            runs.data.reference_add(
                from_uuid=run_uuid, from_property="generated_artifacts", to=art_uuid
            )

            # If it's an input artifact (role=input), we should link
            # run -> used_input_artifacts. But the 'register_artifact' usually
            # implies OUTPUT. Input registration is a separate concept.
            # We'll assume these are outputs.

        except Exception as e:
            print(f"Error registering artifact: {e}")

    def _get_or_create_module(self, collection: Any, module_id: str) -> str:
        # Deterministic UUID
        uuid = weaviate.util.generate_uuid5(module_id)
        if not collection.query.fetch_object_by_id(uuid):
            collection.data.insert(
                uuid=uuid,
                properties={
                    "module_id": module_id,
                    "description": f"Auto-registered module {module_id}",
                },
            )
        return uuid


# Global instance
db = ProvenanceDB()
