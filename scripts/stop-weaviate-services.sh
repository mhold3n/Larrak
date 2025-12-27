#!/bin/bash
# Stop Weaviate and Outline API services

echo "Stopping Weaviate and Outline API..."

docker compose stop weaviate outline-api

echo "✅ Services stopped"
echo ""
echo "To start again: ./scripts/start-weaviate-services.sh"
echo "To remove containers: docker compose rm weaviate outline-api"
