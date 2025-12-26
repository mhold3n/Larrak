"""Model registry for tracking trained ML surrogates in Weaviate.

Provides functions to register, query, and manage trained model metadata
for integration with the orchestrator dashboard and provenance system.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    import weaviate
    import weaviate.classes.init as wvi
except ImportError:
    weaviate = None  # type: ignore[assignment]


def get_weaviate_client():
    """Get connected Weaviate client using environment configuration."""
    if weaviate is None:
        raise ImportError("weaviate-client not installed")

    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY", "")

    if "weaviate.cloud" in weaviate_url or "wcs.api.weaviate.io" in weaviate_url:
        # Cloud connection
        cluster_url = weaviate_url
        if "://" in cluster_url:
            cluster_url = cluster_url.split("://")[1]

        if weaviate_api_key:
            auth = wvi.Auth.api_key(weaviate_api_key)
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url, auth_credentials=auth
            )
        else:
            return weaviate.connect_to_weaviate_cloud(cluster_url=cluster_url)
    else:
        # Local connection
        return weaviate.connect_to_local(port=8080, grpc_port=50052)


def register_model(
    model_id: str,
    model_type: str,
    file_path: str | Path,
    n_samples: int,
    n_ensemble_members: int = 3,
    final_loss: float = 0.0,
    metadata: dict | None = None,
) -> str:
    """Register a trained model in Weaviate.

    Args:
        model_id: Unique identifier (e.g., 'thermal_surrogate_v1')
        model_type: Type of model ('thermal', 'structural', 'flow_coefficient')
        file_path: Path to the saved .pt file
        n_samples: Number of training samples used
        n_ensemble_members: Number of ensemble members
        final_loss: Final training loss
        metadata: Optional dict of hyperparameters

    Returns:
        UUID of the created object
    """
    client = get_weaviate_client()

    try:
        collection = client.collections.get("TrainedModel")

        # Check if model already exists
        existing = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("model_id").equal(model_id),
            limit=1,
        )

        if existing.objects:
            # Update existing
            uuid = existing.objects[0].uuid
            collection.data.update(
                uuid=uuid,
                properties={
                    "model_type": model_type,
                    "file_path": str(file_path),
                    "training_date": datetime.now(timezone.utc).isoformat(),
                    "n_samples": n_samples,
                    "n_ensemble_members": n_ensemble_members,
                    "final_loss": final_loss,
                    "workflow_stage": "surrogate",
                    "metadata": json.dumps(metadata or {}),
                },
            )
            print(f"[model_registry] Updated model: {model_id}")
            return str(uuid)
        else:
            # Create new
            uuid = collection.data.insert(
                properties={
                    "model_id": model_id,
                    "model_type": model_type,
                    "file_path": str(file_path),
                    "training_date": datetime.now(timezone.utc).isoformat(),
                    "n_samples": n_samples,
                    "n_ensemble_members": n_ensemble_members,
                    "final_loss": final_loss,
                    "workflow_stage": "surrogate",
                    "metadata": json.dumps(metadata or {}),
                }
            )
            print(f"[model_registry] Registered model: {model_id}")
            return str(uuid)

    finally:
        client.close()


def list_models() -> list[dict]:
    """List all registered trained models.

    Returns:
        List of model metadata dictionaries
    """
    client = get_weaviate_client()

    try:
        collection = client.collections.get("TrainedModel")
        result = collection.query.fetch_objects(limit=100)

        models = []
        for obj in result.objects:
            props = obj.properties
            models.append(
                {
                    "uuid": str(obj.uuid),
                    "model_id": props.get("model_id"),
                    "model_type": props.get("model_type"),
                    "file_path": props.get("file_path"),
                    "training_date": props.get("training_date"),
                    "n_samples": props.get("n_samples"),
                    "n_ensemble_members": props.get("n_ensemble_members"),
                    "final_loss": props.get("final_loss"),
                    "workflow_stage": props.get("workflow_stage"),
                }
            )

        return models

    finally:
        client.close()


def get_models_by_type(model_type: str) -> list[dict]:
    """Get all models of a specific type.

    Args:
        model_type: 'thermal', 'structural', or 'flow_coefficient'

    Returns:
        List of matching model metadata
    """
    client = get_weaviate_client()

    try:
        collection = client.collections.get("TrainedModel")
        result = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("model_type").equal(model_type),
            limit=10,
        )

        return [
            {
                "uuid": str(obj.uuid),
                "model_id": obj.properties.get("model_id"),
                "file_path": obj.properties.get("file_path"),
                "training_date": obj.properties.get("training_date"),
                "n_samples": obj.properties.get("n_samples"),
                "final_loss": obj.properties.get("final_loss"),
            }
            for obj in result.objects
        ]

    finally:
        client.close()


def get_models_for_dashboard() -> dict:
    """Get model summary formatted for the orchestrator dashboard.

    Returns:
        Dictionary with model counts and recent models by type
    """
    models = list_models()

    by_type = {}
    for m in models:
        mtype = m.get("model_type", "unknown")
        if mtype not in by_type:
            by_type[mtype] = []
        by_type[mtype].append(m)

    return {
        "total_count": len(models),
        "by_type": by_type,
        "workflow_stage": "surrogate",
    }
