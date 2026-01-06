# Launch Instructions for Dashboard

Since Docker is running on your host system but not accessible from within this container, please run these commands **on your host machine** (outside this container):

## Quick Start (Run on Host)

```bash
# Navigate to the project directory
cd /workspace  # or your project path

# Start the services
./scripts/start-dashboard.sh

# Or manually:
docker compose up -d weaviate outline-api
```

## Verify Services

After starting, verify the services are running:

```bash
# Check Weaviate
curl http://localhost:8080/v1/.well-known/ready

# Check Dashboard API  
curl http://localhost:5001/api/modules

# View running containers
docker compose ps
```

## Access the Dashboard

Once services are running, open in your browser:
- **Dashboard**: http://localhost:5001/
- **Weaviate**: http://localhost:8080

## Services Started

- ✅ **Weaviate** (port 8080) - Vector database
- ✅ **Dashboard API** (port 5001) - Orchestrator dashboard
- ✅ **WebSocket** (port 8765) - Telemetry trace (starts when optimization begins)

## View Logs

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f outline-api
docker compose logs -f weaviate
```

## Stop Services

```bash
docker compose stop weaviate outline-api
# or
docker compose down
```
