"""Worker process for job queue.

Runs optimization jobs from Redis queue.
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import redis
from rq import Queue, Worker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_conn = redis.from_url(REDIS_URL)


def main():
    """Run the worker."""
    logger.info(f"Starting worker, connecting to Redis at {REDIS_URL}")

    worker = Worker(["optimization", "default"], connection=redis_conn)

    # Initialize SocketIO for event broadcasting
    try:
        from flask_socketio import SocketIO

        from provenance.execution_events import add_listener

        # Connect to same Redis as API
        socketio = SocketIO(message_queue=REDIS_URL)

        def broadcast_event(event):
            try:
                # Emit to "execution_event" channel which API listens to (and clients)
                socketio.emit("execution_event", event.to_json())
            except Exception as e:
                logger.warning(f"Failed to emit event: {e}")

        add_listener(broadcast_event)
        logger.info("SocketIO event bridge initialized")

    except ImportError:
        logger.warning("flask_socketio or provenance not found, skipping event bridge")
    except Exception as e:
        logger.error(f"Failed to initialize event bridge: {e}")

    logger.info("Worker started, listening for jobs...")
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
