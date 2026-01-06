#!/bin/bash
# Script to restart dashboard API and verify Weaviate connection

echo "=== Restarting Dashboard API Container ==="
docker compose restart outline-api

echo ""
echo "Waiting for container to restart..."
sleep 5

echo ""
echo "=== Checking Weaviate Connection in Logs ==="
docker compose logs outline-api --tail 50 | grep -i weaviate

echo ""
echo "=== Full Recent Logs (last 20 lines) ==="
docker compose logs outline-api --tail 20

echo ""
echo "=== Container Status ==="
docker compose ps outline-api
