# Fixed Weaviate Connection Issue

## Problem
The `ProvenanceClient` was using `weaviate.connect_to_local()` which connects to `localhost:8080`.
From inside Docker containers, it needs to connect to `weaviate:8080` (Docker service name).

## Solution
Updated `campro/orchestration/provenance.py` to:
1. Use `WEAVIATE_URL` environment variable
2. Parse the URL to extract host and port
3. Use `connect_to_custom` if available for Docker service names
4. Fall back to `connect_to_local` for localhost connections

## Next Steps

1. **Restart the dashboard API container** to pick up the changes:
   ```bash
   docker compose restart outline-api
   ```

2. **Verify Weaviate connection**:
   ```bash
   docker compose logs outline-api | grep -i weaviate
   ```
   Should see: "Connected to Weaviate for provenance tracking at http://weaviate:8080"

3. **Test a run again** from the dashboard
   - Should now see telemetry events
   - Run should not finish immediately
   - Events should appear in real-time

## Why This Fixes It

- Before: `connect_to_local()` → `localhost:8080` → ❌ Connection failed
- After: Uses `WEAVIATE_URL=http://weaviate:8080` → ✅ Connects via Docker DNS

The orchestrator was failing silently when provenance couldn't connect, causing immediate completion without events.
