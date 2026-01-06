"""
Add these endpoints to dashboard/api.py inside create_app() function.

Add after the existing API routes (around line 250):
"""

# Job Queue Endpoints (Phase 2 - Architecture Refactor)
from dashboard.job_queue import get_job_result, get_job_status, submit_optimization_job


@app.route("/api/runs/submit", methods=["POST"])
def submit_run():
    """Submit optimization run to job queue."""
    try:
        params = request.get_json()
        if not params:
            return jsonify({"error": "No parameters provided"}), 400

        result = submit_optimization_job(params)
        if result:
            return jsonify(result), 202  # 202 Accepted
        else:
            return jsonify({"error": "Job queue unavailable"}), 503
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/runs/<job_id>")
def get_run_status(job_id: str):
    """Get status of a queued optimization run."""
    status = get_job_status(job_id)
    if status:
        return jsonify(status)
    else:
        return jsonify({"error": f"Job {job_id} not found"}), 404


@app.route("/api/runs/<job_id>/result")
def get_run_result(job_id: str):
    """Get result of completed optimization run."""
    result = get_job_result(job_id)
    if result:
        return jsonify({"result": result})
    else:
        status = get_job_status(job_id)
        if status:
            return jsonify({"error": "Job not finished", "status": status["status"]}), 425
        else:
            return jsonify({"error": f"Job {job_id} not found"}), 404


@app.route("/api/health")
def health_check():
    """Health check endpoint for larrak-api."""
    return jsonify(
        {
            "status": "healthy",
            "service": "larrak-api",
            "redis": "available" if get_redis_connection() else "unavailable",
        }
    )
