import logging

import weaviate

from provenance.schema import create_schema

logging.basicConfig(level=logging.INFO)


def init_schema():
    try:
        client = weaviate.connect_to_local(port=8080, grpc_port=50052)
        print("Connected to Weaviate.")

        # Apply schema
        create_schema(client)
        print("Schema applied successfully.")

        client.close()
    except Exception as e:
        print(f"Failed to apply schema: {e}")


if __name__ == "__main__":
    init_schema()
