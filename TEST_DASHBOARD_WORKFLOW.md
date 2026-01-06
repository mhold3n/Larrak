# Dashboard and Workflow Orchestration Test

## From DevContainer - Test Results

### 1. Dashboard API Access
```bash
curl http://localhost:5001/api/modules
```
✅ Should return JSON with all modules

### 2. Dashboard HTML
```bash
curl http://localhost:5001/
```
✅ Should return HTML for orchestrator dashboard

### 3. Workflow Orchestration Endpoints

#### Start Optimization Sequence
```bash
curl -X POST http://localhost:5001/api/start \
  -H "Content-Type: application/json" \
  -d '{
    "optimization": {
      "max_iterations": 5,
      "batch_size": 3
    },
    "budget": {
      "total_sim_calls": 10
    }
  }'
```

Expected response:
```json
{"status": "started", "run_id": "abc12345"}
```

#### Check Optimization Steps
```bash
curl http://localhost:5001/api/optimization/steps
```

#### Check Dataflows
```bash
curl http://localhost:5001/api/dataflows
```

### 4. WebSocket Telemetry Trace

The WebSocket server starts automatically when you POST to `/api/start`.
It runs on port 8765 and broadcasts execution events.

Test WebSocket connection (from browser console):
```javascript
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (event) => console.log('Event:', JSON.parse(event.data));
```

## Full Workflow Test

1. **Open Dashboard**: http://localhost:5001/
2. **Start Optimization**: Use dashboard UI or POST to `/api/start`
3. **Watch Telemetry**: WebSocket connects automatically to `ws://localhost:8765`
4. **View Events**: Real-time execution events appear in dashboard

## Verify Container Communication

The dashboard API connects to Weaviate using Docker service name:
- Inside container: `http://weaviate:8080` ✅
- From Mac: `http://localhost:8080` (may vary)

This is configured via `WEAVIATE_URL=http://weaviate:8080` in docker-compose.yml
