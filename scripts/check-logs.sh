#!/bin/bash
# Check recent logs from the dashboard API container
# Run this on your Mac host (not in devcontainer)

set -e

echo "Checking Dashboard API Logs..."
echo "================================"
echo ""

# Check Weaviate connection
echo "1. Weaviate Connection Status:"
echo "-------------------------------"
docker compose logs outline-api 2>&1 | grep -i "weaviate\|provenance" | tail -10 || echo "  No Weaviate-related logs found"
echo ""

# Check for errors
echo "2. Recent Errors:"
echo "-----------------"
docker compose logs outline-api 2>&1 | grep -i "error\|failed\|exception" | tail -10 || echo "  No errors found"
echo ""

# Check orchestration runs
echo "3. Orchestration Runs:"
echo "----------------------"
docker compose logs outline-api 2>&1 | grep -i "orchestration\|run_id\|starting sequence" | tail -10 || echo "  No orchestration runs found"
echo ""

# Show last 30 lines of logs
echo "4. Last 30 Lines of Logs:"
echo "-------------------------"
docker compose logs outline-api --tail 30
echo ""

# Check container status
echo "5. Container Status:"
echo "-------------------"
docker compose ps outline-api weaviate
echo ""

# Check environment variable
echo "6. Environment Variables:"
echo "-------------------------"
docker compose exec outline-api env 2>/dev/null | grep WEAVIATE || echo "  Could not check environment (container may not be running)"
echo ""

echo "Done. If you see 'Failed to connect to Weaviate', restart the container:"
echo "  docker compose restart outline-api"
