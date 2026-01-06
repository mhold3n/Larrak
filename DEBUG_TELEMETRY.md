# Debugging Telemetry Issues

## Problem
- Runs finish immediately
- No live telemetry displays
- Likely Weaviate connection issue

## Check Logs (Run on Mac)

```bash
# Dashboard API logs (most important)
docker compose logs outline-api --tail 100

# Weaviate logs
docker compose logs weaviate --tail 50

# Check for errors
docker compose logs outline-api | grep -i error
```

## Test Weaviate Connection from Container

```bash
# Test from inside dashboard container
docker compose exec outline-api python -c "
import os
os.environ['WEAVIATE_URL'] = 'http://weaviate:8080'
try:
    import weaviate
    client = weaviate.connect_to_local(port=8080, grpc_port=50052)
    print('✅ Weaviate connection successful')
    print(f'Ready: {client.is_ready()}')
    client.close()
except Exception as e:
    print(f'❌ Weaviate connection failed: {e}')
"
```

## Check Event Broadcasting

The WebSocket server should be broadcasting events. Check if events are being emitted:

```bash
# Check if WebSocket server is running
docker compose exec outline-api ps aux | grep python
```

## Common Issues

1. **Weaviate not accessible**: Container can't reach `http://weaviate:8080`
2. **Provenance client failing**: Silent failure in `provenance.start_run()`
3. **Orchestrator failing early**: Exception caught but not properly logged
4. **WebSocket not broadcasting**: Events emitted but not sent to clients
