"""Initialize Weaviate schema for Larrak provenance system.

Supports both local Weaviate and Weaviate Cloud (WCD).

Usage:
    # Local
    python scripts/init_schema.py

    # Weaviate Cloud
    WEAVIATE_URL=https://xxx.weaviate.cloud WEAVIATE_API_KEY=xxx python scripts/init_schema.py
"""

import logging
import os

import weaviate

from provenance.schema import create_schema

logging.basicConfig(level=logging.INFO)


def init_schema():
    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY", "")

    print(f"Connecting to Weaviate at {weaviate_url}...")

    try:
        # Detect if using Weaviate Cloud (WCD)
        if "weaviate.cloud" in weaviate_url or "wcs.api.weaviate.io" in weaviate_url:
            import weaviate.classes.init as wvi

            cluster_url = weaviate_url
            if not cluster_url.startswith("http"):
                cluster_url = f"https://{cluster_url}"

            if weaviate_api_key:
                auth = wvi.Auth.api_key(weaviate_api_key)
                client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=cluster_url,
                    auth_credentials=auth,
                )
            else:
                client = weaviate.connect_to_weaviate_cloud(cluster_url=cluster_url)
            print("Connected to Weaviate Cloud.")
        else:
            client = weaviate.connect_to_local(port=8080, grpc_port=50052)
            print("Connected to local Weaviate.")

        # Apply schema
        create_schema(client)
        print("Schema applied successfully.")

        client.close()
    except Exception as e:
        print(f"Failed to apply schema: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    init_schema()
