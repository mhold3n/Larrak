import logging
import os
import uuid
from datetime import datetime, timezone

import weaviate

log = logging.getLogger(__name__)


def _get_weaviate_client_from_url(weaviate_url: str):
    """Get Weaviate client from URL, handling Docker service names.

    Args:
        weaviate_url: URL like http://weaviate:8080 or http://localhost:8080

    Returns:
        WeaviateClient instance
    """
    # Parse URL
    if "://" in weaviate_url:
        url_clean = weaviate_url.replace("http://", "").replace("https://", "")
        url_parts = url_clean.split(":")
        host = url_parts[0] if url_parts else "localhost"
        port = int(url_parts[1]) if len(url_parts) > 1 else 8080
    else:
        host = "localhost"
        port = 8080

    grpc_port = 50051  # Internal Docker port (50052 is only for host access)

    # Check if connecting to a Docker service name (not localhost and no dots)
    is_service_name = host != "localhost" and "." not in host and not host.startswith("127.")

    if is_service_name:
        # For Docker service names, try connect_to_custom
        try:
            import weaviate.classes.init as wvi

            # Use connect_to_custom for service names
            return weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=False,
                grpc_host=host,
                grpc_port=grpc_port,
                grpc_secure=False,
            )
        except (AttributeError, ImportError, TypeError) as e:
            log.debug(f"connect_to_custom not available, trying local: {e}")
            # Fallback to local (may not work for service names)
            return weaviate.connect_to_local(port=port, grpc_port=grpc_port)
    else:
        # Standard localhost connection
        return weaviate.connect_to_local(port=port, grpc_port=grpc_port)


class ProvenanceClient:
    """Handles logging of orchestration runs to Weaviate."""

    def __init__(self, use_provenance: bool = True):
        self.enabled = use_provenance
        self.client = None
        self.run_uuid = None

        if self.enabled:
            self._connect()

    def _connect(self):
        try:
            # Use WEAVIATE_URL environment variable if available, otherwise default to localhost
            weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")

            # Use helper function to get client (handles Docker service names)
            self.client = _get_weaviate_client_from_url(weaviate_url)

            # Verify connection
            self.client.get_meta()
            log.info(f"Connected to Weaviate for provenance tracking at {weaviate_url}")
        except Exception as e:
            log.warning(f"Failed to connect to Weaviate. Provenance disabled. Error: {e}")
            self.client = None
            self.enabled = False

    def start_run(self, params: dict, tags: list[str] = None) -> str:
        """Create a new Run object in Weaviate."""
        if not self.enabled or not self.client:
            return ""

        run_id = str(uuid.uuid4())
        self.run_uuid = run_id

        try:
            runs = self.client.collections.get("Run")

            # Serialize params to avoiding nesting issues in kwargs
            # We store keys as args
            args_list = [
                f"{k}={v}" for k, v in params.items() if isinstance(v, (str, int, float, bool))
            ]

            runs.data.insert(
                properties={
                    "run_id": run_id,
                    "start_time": datetime.now(timezone.utc),
                    "status": "RUNNING",
                    "args": args_list,
                    "tags": tags or ["orchestrator"],
                    "env": os.getenv("CONDA_DEFAULT_ENV", "unknown"),
                },
                uuid=uuid.uuid5(uuid.NAMESPACE_DNS, run_id),  # Consistent UUID generation
            )
            log.info(f"Provenance: Started run {run_id}")
            return run_id
        except Exception as e:
            log.error(f"Provenance: Failed to start run: {e}")
            return ""

    def end_run(self, status: str = "COMPLETED"):
        """Update the Run object with end time and status."""
        if not self.enabled or not self.client or not self.run_uuid:
            return

        try:
            runs = self.client.collections.get("Run")
            # We need the obj UUID logic to match insert
            obj_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, self.run_uuid)

            runs.data.update(
                uuid=obj_uuid, properties={"end_time": datetime.now(timezone.utc), "status": status}
            )
            log.info(f"Provenance: Ended run {self.run_uuid} with status {status}")
        except Exception as e:
            log.error(f"Provenance: Failed to end run: {e}")

    def log_geometry(self, geometry_data: dict, run_id: str = None) -> str:
        """Log a geometry object generated during the run."""
        effective_run_uuid = run_id if run_id else self.run_uuid

        if not self.enabled or not self.client or not effective_run_uuid:
            import sys

            print(
                f"[DEBUG] Provenance LOGGING SKIPPED. Enabled={self.enabled}, Client={self.client}, RunID={effective_run_uuid}",
                file=sys.stderr,
            )
            return ""

        try:
            geometries = self.client.collections.get("Geometry")
            run_obj_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, effective_run_uuid)

            geom_id = str(uuid.uuid4())
            geom_obj_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, geom_id)

            properties = {
                "geometry_id": geom_id,
                "created_at": datetime.now(timezone.utc),
                **{
                    k: v
                    for k, v in geometry_data.items()
                    if k
                    in [
                        "voxel_file_path",
                        "mesh_file_path",
                        "volume_mm3",
                        "surface_area_mm2",
                        "mesh_hash",
                    ]
                },
            }

            geometries.data.insert(
                properties=properties, uuid=geom_obj_uuid, references={"generated_by": run_obj_uuid}
            )
            return geom_id
        except Exception as e:
            import sys
            import traceback

            print(f"[ERROR] Provenance log_geometry failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            log.error(f"Provenance: Failed to log geometry: {e}")
            return ""

    def log_constraint_check(self, geometry_id: str, check_data: dict) -> str:
        """Log a constraint check result for a geometry."""
        if not self.enabled or not self.client:
            return ""

        try:
            results = self.client.collections.get("ConstraintResult")
            geom_obj_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, geometry_id)

            result_id = str(uuid.uuid4())
            result_obj_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, result_id)

            properties = {
                "result_id": result_id,
                **{
                    k: v
                    for k, v in check_data.items()
                    if k in ["check_name", "passed", "margin_mm", "location_vector", "message"]
                },
            }

            results.data.insert(
                properties=properties,
                uuid=result_obj_uuid,
                references={"checked_on": geom_obj_uuid},
            )
            return result_id
        except Exception as e:
            log.error(f"Provenance: Failed to log constraint check: {e}")
            return ""

    def close(self):
        if self.client:
            self.client.close()
