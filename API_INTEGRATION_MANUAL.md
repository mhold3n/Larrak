# API Integration - Manual Steps

## Completed ✅

- [x] Added job queue imports to `dashboard/api.py` (lines 59-70)
- [x] Created all backend infrastructure

## Remaining: Add Endpoints to dashboard/api.py

**Location**: After line 265 in `dashboard/api.py`
(After the `/api/optimization/steps` endpoint)

**Copy-paste these endpoints**:

```python
# =========================================================================
# Job Queue Endpoints (Phase 2 - Architecture Refactor)
# =========================================================================

@app.route("/api/health")
def health_check() -> Response:
    """Health check endpoint for larrak-api."""
    redis_status = "available" if JOB_QUEUE_AVAILABLE and get_redis_connection() else "unavailable"
    return jsonify({
        "status": "healthy",
        "service": "larrak-api",
        "redis": redis_status,
        "job_queue": JOB_QUEUE_AVAILABLE
    })

@app.route("/api/runs/submit", methods=["POST"])
def submit_run() -> tuple[Response, int]:
    """Submit optimization run to job queue."""
    if not JOB_QUEUE_AVAILABLE:
        return jsonify({"error": "Job queue not available"}), 503

    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "No parameters provided"}), 400

        result = submit_optimization_job(params)
        if result:
            return jsonify(result), 202  # 202 Accepted
        else:
            return jsonify({"error": "Failed to queue job"}), 500
    except Exception as e:
        logging.error(f"Failed to submit job: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/runs/<job_id>")
def get_run_status_endpoint(job_id: str) -> tuple[Response, int] | Response:
    """Get status of a queued optimization run."""
    if not JOB_QUEUE_AVAILABLE:
        return jsonify({"error": "Job queue not available"}), 503

    status = get_job_status(job_id)
    if status:
        return jsonify(status)
    else:
        return jsonify({"error": f"Job {job_id} not found"}), 404

@app.route("/api/runs/<job_id>/result")
def get_run_result_endpoint(job_id: str) -> tuple[Response, int] | Response:
    """Get result of completed optimization run."""
    if not JOB_QUEUE_AVAILABLE:
        return jsonify({"error": "Job queue not available"}), 503

    result = get_job_result(job_id)
    if result:
        return jsonify({"result": result})
    else:
        status = get_job_status(job_id)
        if status:
            return json ify({
                "error": "Job not finished",
                "status": status.get("status")
            }), 425  # Too Early
        else:
            return jsonify({"error": f"Job {job_id} not found"}), 404
```

## Testing After Integration

```bash
# 1. Fix Docker credentials first
# Edit ~/.docker/config.json and remove "credsStore" line

# 2. Build and start services
docker compose build larrak-api
docker compose up -d redis larrak-api

# 3. Test health endpoint
curl http://localhost:8000/api/health

# Expected: {"status": "healthy", "service": "larrak-api", "redis": "available", "job_queue": true}
```

## Next: Add Worker Service

Once endpoints are integrated and API is running, add worker to `docker-compose.yml`:

```yaml
larrak-worker:
  build:
    context: .
    dockerfile: .devcontainer/Dockerfile
  command: python worker/run_worker.py
  environment:
    - REDIS_URL=redis://redis:6379
    - WEAVIATE_URL=http://weaviate:8080
    - CEM_SERVICE_URL=http://cem-service:50051
  depends_on:
    - redis
  networks:
    - larrak-internal
  volumes:
    - .:/workspace:cached
  restart: unless-stopped
```

Then:

```bash
docker compose up -d larrak-worker
docker compose logs -f larrak-worker
```
