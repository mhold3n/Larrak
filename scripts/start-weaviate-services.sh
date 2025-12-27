#!/bin/bash
# Start Weaviate and Outline API services
# Optional: Pass 'all' to start all services in docker-compose.yml

set -e

START_ALL=${1:-""}

echo "Starting Docker services..."
echo "=========================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Start services based on argument
if [ "$START_ALL" = "all" ]; then
    echo "Starting ALL services (larrak-dev, weaviate, outline-api, cem-service, openfoam, calculix)..."
    docker compose up -d
else
    echo "Starting Weaviate and Outline API only..."
    echo "(Use './scripts/start-weaviate-services.sh all' to start all services)"
    docker compose up -d weaviate outline-api
fi

echo ""
echo "Waiting for services to be healthy..."

# Wait for Weaviate
for i in {1..30}; do
    if curl -sf http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
        echo "✅ Weaviate is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Weaviate failed to start"
        docker compose logs weaviate
        exit 1
    fi
    echo "  Waiting for Weaviate... ($i/30)"
    sleep 2
done

# Wait for Outline API
for i in {1..20}; do
    if curl -sf http://localhost:5001/api/modules > /dev/null 2>&1; then
        echo "✅ Outline API is ready"
        break
    fi
    if [ $i -eq 20 ]; then
        echo "❌ Outline API failed to start"
        docker compose logs outline-api
        exit 1
    fi
    echo "  Waiting for Outline API... ($i/20)"
    sleep 2
done

echo ""
echo "✅ Services running!"
echo ""
echo "Active services:"
docker compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "Key endpoints:"
echo "  - Weaviate:    http://localhost:8080"
echo "  - Outline API: http://localhost:5001"
echo ""
echo "Next steps for first-time setup:"
echo "  1. Initialize schema: WEAVIATE_URL=http://localhost:8080 python scripts/init_schema.py"
echo "  2. Index codebase: WEAVIATE_URL=http://localhost:8080 python truthmaker/ingestion/code_scanner.py"
echo "  3. Reload IDE window (Cmd+Shift+P → 'Developer: Reload Window')"
echo ""
echo "Manage services:"
echo "  - View logs:  docker compose logs -f weaviate outline-api"
echo "  - Stop:       docker compose stop weaviate outline-api"
echo "  - Stop all:   docker compose stop"
echo ""
