#!/bin/bash
# Start the orchestrator dashboard with dockerized Weaviate API
# This script starts Weaviate and the outline-api (dashboard) services

set -e

echo "Starting Orchestrator Dashboard with Dockerized Weaviate..."
echo "=========================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Start Weaviate and outline-api services
echo "Starting Weaviate and Dashboard API services..."
docker compose up -d weaviate outline-api

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

# Wait for Outline API (Dashboard)
for i in {1..20}; do
    if curl -sf http://localhost:5001/api/modules > /dev/null 2>&1; then
        echo "✅ Dashboard API is ready"
        break
    fi
    if [ $i -eq 20 ]; then
        echo "❌ Dashboard API failed to start"
        docker compose logs outline-api
        exit 1
    fi
    echo "  Waiting for Dashboard API... ($i/20)"
    sleep 2
done

echo ""
echo "✅ Services running!"
echo ""
echo "Active services:"
docker compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}" weaviate outline-api
echo ""
echo "Key endpoints:"
echo "  - Weaviate:        http://localhost:8080"
echo "  - Dashboard API:   http://localhost:5001"
echo "  - Orchestrator Dashboard: http://localhost:5001/"
echo ""
echo "The dashboard is now ready to run optimizer routines with interface telemetry trace."
echo ""
echo "To view the dashboard, open: http://localhost:5001/"
echo "To start an optimization run, use the dashboard interface or POST to http://localhost:5001/api/start"
