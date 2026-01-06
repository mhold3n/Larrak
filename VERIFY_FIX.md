# Verify Weaviate Connection Fix

## Step 1: Restart Container

Run on your Mac terminal:
```bash
cd /path/to/your/project  # Navigate to project root
docker compose restart outline-api
```

## Step 2: Check Logs

```bash
# Check for Weaviate connection message
docker compose logs outline-api | grep -i weaviate

# Or see recent logs
docker compose logs outline-api --tail 50
```

## Expected Output

You should see one of these messages:
- ✅ `"Connected to Weaviate for provenance tracking at http://weaviate:8080"`
- ✅ `"Connected to Weaviate for provenance tracking"`

If you see:
- ❌ `"Failed to connect to Weaviate. Provenance disabled"`
- Check that Weaviate container is running: `docker compose ps weaviate`
- Check Weaviate logs: `docker compose logs weaviate --tail 20`

## Step 3: Test Dashboard

1. Open http://localhost:5001/ in your browser
2. Start an optimization run
3. Check browser console (F12) for WebSocket events
4. Verify telemetry appears in real-time

## Step 4: Monitor Logs During Run

In a separate terminal, watch logs in real-time:
```bash
docker compose logs -f outline-api
```

You should see:
- Orchestration starting
- Module events (CEM, SUR, SOL, etc.)
- Tool calls
- No immediate "Run finished" message
