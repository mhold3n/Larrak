"""Job queue integration for dashboard API.

Adds endpoints for:
- Submitting optimization jobs to Redis queue
- Checking job status
- Retrieving job results
"""

import logging
import os
from typing import Any

try:
    import redis
    from rq import Queue
    from rq.job import Job

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None  # type: ignore
    Queue = None  # type: ignore
    Job = None  # type: ignore

logger = logging.getLogger(__name__)

# Redis connection (lazy initialization)
_redis_conn = None
_job_queue = None


def get_redis_connection():
    """Get or create Redis connection."""
    global _redis_conn
    if _redis_conn is None and REDIS_AVAILABLE:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _redis_conn = redis.from_url(redis_url)
        logger.info(f"Connected to Redis at {redis_url}")
    return _redis_conn


def get_job_queue():
    """Get or create job queue."""
    global _job_queue
    if _job_queue is None and REDIS_AVAILABLE:
        conn = get_redis_connection()
        if conn:
            _job_queue = Queue("optimization", connection=conn)
            logger.info("Job queue initialized")
    return _job_queue


def submit_optimization_job(params: dict[str, Any]) -> dict[str, str] | None:
    """
    Submit optimization job to queue.

    Args:
        params: Optimization parameters

    Returns:
        Job info dict with job_id and status, or None if queue unavailable
    """
    if not REDIS_AVAILABLE:
        logger.warning("Redis not available, cannot queue job")
        return None

    queue = get_job_queue()
    if not queue:
        logger.error("Job queue not initialized")
        return None

    try:
        # Use string path to avoid importing heavy dependencies (CasADi/IPOPT) in the API
        # The worker will import this function when processing the job
        job_func_path = "campro.optimization.driver.solve_cycle"

        job = queue.enqueue(
            job_func_path,
            params,
            job_timeout="24h",
            result_ttl=86400,  # Keep results for 24 hours
            failure_ttl=86400,  # Keep failures for 24 hours
        )

        logger.info(f"Job {job.id} queued successfully")
        return {"job_id": job.id, "status": job.get_status(), "queue": "optimization"}
    except Exception as e:
        logger.error(f"Failed to queue job: {e}")
        return None


def get_job_status(job_id: str) -> dict[str, Any] | None:
    """
    Get status of a queued job.

    Args:
        job_id: Job identifier

    Returns:
        Job status dict or None if not found
    """
    if not REDIS_AVAILABLE:
        return None

    conn = get_redis_connection()
    if not conn:
        return None

    try:
        job = Job.fetch(job_id, connection=conn)

        return {
            "job_id": job.id,
            "status": job.get_status(),
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
            "progress": job.meta.get("progress", 0),
            "exc_info": job.exc_info if job.is_failed else None,
        }
    except Exception as e:
        logger.error(f"Failed to fetch job {job_id}: {e}")
        return None


def get_job_result(job_id: str) -> Any:
    """
    Get result of completed job.

    Args:
        job_id: Job identifier

    Returns:
        Job result or None
    """
    if not REDIS_AVAILABLE:
        return None

    conn = get_redis_connection()
    if not conn:
        return None

    try:
        job = Job.fetch(job_id, connection=conn)
        if job.is_finished:
            return job.result
        return None
    except Exception as e:
        logger.error(f"Failed to get result for job {job_id}: {e}")
        return None
