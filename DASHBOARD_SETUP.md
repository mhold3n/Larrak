# Orchestrator Dashboard Setup with Dockerized Weaviate

This document describes how to run the main dashboard with overlay, fully linked to the dockerized Weaviate API, ready for optimizer routine/sequence execution with interface telemetry trace.

## Quick Start

1. **Start the services:**
   ```bash
   ./scripts/start-dashboard.sh
   ```

2. **Access the dashboard:**
   - Open http://localhost:5001/ in your browser
   - The orchestrator dashboard will load with the overlay interface

3. **Start an optimization run:**
   - Use the dashboard interface to configure and start an optimization sequence
   - Or POST to `http://localhost:5001/api/start` with optimization parameters
   - The interface telemetry trace will be displayed in real-time via WebSocket

## Services

- **Weaviate**: Vector database running on port 8080
- **Dashboard API**: Flask API serving the orchestrator dashboard on port 5001
- **WebSocket Server**: Telemetry trace server on port 8765 (started automatically when optimization begins)

## Configuration

The dashboard API is configured to connect to Weaviate using the `WEAVIATE_URL` environment variable:
- In Docker: `http://weaviate:8080` (resolved via Docker DNS)
- Locally: `http://localhost:8080`

## Features

- ✅ Full Weaviate API integration via environment variable
- ✅ Docker networking support (service name resolution)
- ✅ Real-time telemetry trace via WebSocket
- ✅ Optimizer routine/sequence execution
- ✅ Interface overlay with live event tracking

## API Endpoints

- `GET /` - Orchestrator dashboard HTML
- `GET /api/modules` - List all modules with tools
- `GET /api/modules/<id>/tools` - Get tools for a module
- `POST /api/start` - Start optimization sequence with telemetry trace
- `POST /api/stop` - Stop running optimization
- `GET /api/optimization/steps` - Get optimization step history
- `GET /api/dataflows` - Get data flow connections

## Troubleshooting

If services fail to start:
1. Check Docker is running: `docker info`
2. View logs: `docker compose logs outline-api weaviate`
3. Verify ports are available: `netstat -an | grep -E '5001|8080|8765'`

## Next Steps

1. Initialize Weaviate schema (if first time):
   ```bash
   WEAVIATE_URL=http://localhost:8080 python scripts/init_schema.py
   ```

2. Index codebase (optional):
   ```bash
   WEAVIATE_URL=http://localhost:8080 python truthmaker/ingestion/code_scanner.py
   ```
