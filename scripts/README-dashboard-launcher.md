# Optimizer Dashboard Launcher

Complete pre-flight verification script for the Larrak optimizer telemetry dashboard.

## Features

The `launch-optimizer-dashboard.sh` script provides a **7-stage startup orchestration** that verifies every component before opening the dashboard:

### ✅ Stage 1: Pre-Flight Checks

- Verifies Docker is running
- Confirms Docker Compose is available
- Checks for port conflicts (5001, 8765)
- Offers to kill conflicting processes

### ✅ Stage 2: Docker Orchestration

- Launches Weaviate vector database
- Starts Flask API server (outline-api container)
- Optional: Start all services with `--all-services` flag

### ✅ Stage 3: Weaviate Health Check

- Polls `/v1/.well-known/ready` endpoint
- Verifies schema initialization
- Shows container logs on failure

### ✅ Stage 4: Flask API Testing

- Tests REST API endpoints (`/api/modules`, `/api/tools`, `/api/dataflows`)
- Confirms all critical routes are responsive
- Reports endpoint status individually

### ✅ Stage 5: WebSocket Verification

- Tests WebSocket server on port 8765
- Uses socket connectivity check
- Warns if telemetry streaming is unavailable

### ✅ Stage 6: Optimizer Integration Test

- Verifies `/api/start` endpoint exists
- Checks for adapter modules (CEM, SUR, SOL)
- Confirms optimizer can be triggered

### ✅ Stage 7: Status Report & Browser Launch

- Displays all active Docker containers
- Shows verified component checklist
- Opens dashboard in default browser
- Provides monitoring commands

## Usage

### Quick Start

```bash
# Launch dashboard with full verification
./scripts/launch-optimizer-dashboard.sh
```

### Options

**Skip browser launch:**

```bash
./scripts/launch-optimizer-dashboard.sh --no-browser
```

**Start all Docker services** (including CEM, OpenFOAM, CalculiX):

```bash
./scripts/launch-optimizer-dashboard.sh --all-services
```

**Combine flags:**

```bash
./scripts/launch-optimizer-dashboard.sh --all-services --no-browser
```

## Component Verification

The script ensures these **four critical elements** are functional:

| Element | Component | Verification Method |
|---------|-----------|-------------------|
| **(a)** | Docker containers | `docker compose ps` + health checks |
| **(b)** | Flask API (5001) | HTTP GET `/api/modules` |
| **(c)** | WebSocket (8765) | Socket connection test |
| **(d)** | Optimizer integration | Module detection + endpoint check |

## Output Example

```
╔════════════════════════════════════════════════════════╗
║  Larrak Optimizer Dashboard - Startup Orchestration   ║
╚════════════════════════════════════════════════════════╝

[1/7] Running pre-flight checks...
✅ Docker is running
✅ Docker Compose is available

[2/7] Starting Docker services...
   Starting essential services (weaviate, outline-api)...
✅ Docker containers launched

[3/7] Waiting for Weaviate vector database...
✅ Weaviate is ready (http://localhost:8080)
✅ Weaviate schema initialized

[4/7] Waiting for Flask API server...
✅ Flask API is ready (http://localhost:5001)
   Testing API endpoints...
   ✓ /api/modules
   ✓ /api/tools
   ✓ /api/dataflows

[5/7] Verifying WebSocket telemetry server...
✅ WebSocket server is ready (ws://localhost:8765)

[6/7] Testing optimizer integration...
✅ Optimizer API endpoint is configured
✅ Optimizer modules detected (CEM, Surrogate, Solver)

[7/7] Generating status report...

╔════════════════════════════════════════════════════════╗
║             🚀 Dashboard Ready to Launch 🚀            ║
╚════════════════════════════════════════════════════════╝

Verified Components:
   ✅ (a) Docker containers: Running
   ✅ (b) Flask API (port 5001): Responsive
   ✅ (c) WebSocket server (port 8765): Connected
   ✅ (d) Optimizer integration: Configured

Dashboard Endpoints:
   🌐 Main Dashboard:    http://localhost:5001/
   🔌 API Documentation: http://localhost:5001/api/modules
   📊 Vector Database:   http://localhost:8080/v1/schema
   📡 WebSocket:         ws://localhost:8765
```

## Troubleshooting

### Port Already in Use

Script will detect conflicts and offer to kill existing processes:

```
⚠️  Port 5001 (Flask API) is already in use
   Kill existing process? (y/n)
```

### Service Fails to Start

Script shows container logs automatically:

```
❌ Weaviate failed to start after 30 attempts

Container logs:
[recent logs displayed here]
```

### Manual Verification

Check service status:

```bash
docker compose ps
docker compose logs -f outline-api
```

Restart services:

```bash
docker compose restart weaviate outline-api
```

Stop everything:

```bash
docker compose stop
```

## Architecture

The script launches this stack:

```
┌─────────────────────────────────────┐
│  Browser (http://localhost:5001)   │
│  • Control panel inputs             │
│  • Real-time telemetry display     │
└──────────────┬──────────────────────┘
               │
        ┌──────┴───────┐
        │              │
    HTTP POST      WebSocket
   (params)       (events)
        │              │
┌───────▼──────────────▼──────────────┐
│  outline-api (Docker container)     │
│  • Flask API (port 5001)            │
│  • WebSocket Server (port 8765)     │
│  • Orchestrator                     │
└───────┬─────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  weaviate (Docker container)        │
│  • Vector DB (port 8080)            │
│  • Code symbols, provenance         │
└─────────────────────────────────────┘
```

## Related Scripts

- `scripts/start-dashboard.sh` - Basic startup (no health checks)
- `scripts/start-weaviate-services.sh` - Weaviate + API only
- `scripts/stop-weaviate-services.sh` - Shutdown script

## Requirements

- Docker Desktop running
- Docker Compose v2+
- Port 5001, 8080, 8765 available
- macOS or Linux (tested on macOS)
