# DevContainer Port Forwarding Setup

## Current Issue

The dashboard API (port 5001) and Weaviate (port 8080) are running in Docker containers on your host, but they're not accessible from inside the devcontainer via `localhost`.

## Solution: Add Port Forwarding

Update `.devcontainer/devcontainer.json` to forward the ports:

```json
{
  "name": "Larrak Dev",
  "dockerComposeFile": "../docker-compose.yml",
  "service": "larrak-dev",
  "workspaceFolder": "/workspace",
  "forwardPorts": [5001, 8080, 8765],
  "portsAttributes": {
    "5001": {
      "label": "Dashboard API",
      "onAutoForward": "notify"
    },
    "8080": {
      "label": "Weaviate",
      "onAutoForward": "silent"
    },
    "8765": {
      "label": "WebSocket Telemetry",
      "onAutoForward": "silent"
    }
  },
  ...
}
```

## Alternative: Access via Host Network

If port forwarding doesn't work, you can:
1. Access dashboard from your Mac browser: http://localhost:5001/
2. The devcontainer can still run Python code that connects to services
3. Use the dashboard from the host machine

## Test from Browser

The dashboard is fully functional when accessed from your Mac browser:
- Open: http://localhost:5001/
- The workflow orchestration will work
- WebSocket telemetry trace will connect to ws://localhost:8765
