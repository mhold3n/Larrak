import os

import weaviate

print("Debugging Weaviate Connection...")
# Hardcoded to what we expect in Docker
try:
    client = weaviate.connect_to_custom(
        http_host="weaviate",
        http_port=8080,
        http_secure=False,
        grpc_host="weaviate",
        grpc_port=50051,
        grpc_secure=False,
    )
    print(f"Client created. Connected: {client.is_connected()}")
    try:
        meta = client.get_meta()
        print("Meta:", meta)
    except Exception as em:
        print("Meta fail:", em)
    client.close()
except Exception as e:
    print(f"Failed to connect: {e}")
