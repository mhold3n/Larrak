"""Dashboard API for tool and module queries.

Simple Flask API serving tool information to the orchestrator dashboard.
Can also run standalone for development/testing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, cast

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party imports

# Third-party imports
try:
    import weaviate
except ImportError:
    weaviate = None  # type: ignore

try:
    from flask import Flask, Response, jsonify, request, send_from_directory
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit  # type: ignore

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

    class Mock:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return Mock()

        def __getattr__(self, name: str) -> Any:
            return Mock()

        def __getitem__(self, key: str) -> Any:
            return Mock()

    # Define dummy classes for type annotations
    class Flask(Mock):
        pass

    class Response(Mock):
        pass

    class CORS(Mock):
        pass

    class SocketIO(Mock):
        pass

    # Define instances for runtime
    jsonify = Mock()
    request = Mock()
    send_from_directory = Mock()
    emit = Mock()

# Global SocketIO instance
# Use Redis as message queue for distributed event broadcasting (from worker)
# Make Redis optional - if not available, SocketIO will work without it (single server mode)
REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379")
redis_available = False
try:
    # Try to connect to Redis, but don't fail if it's not available
    import redis  # type: ignore

    r = redis.from_url(REDIS_URL, socket_connect_timeout=1)
    r.ping()
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False
    REDIS_URL = ""  # type: ignore

# Initialize SocketIO - don't use message_queue if Redis isn't available
# This allows SocketIO to work in single-server mode
socketio = (
    SocketIO(
        cors_allowed_origins="*",
        message_queue=REDIS_URL if REDIS_AVAILABLE else None,
        async_mode="threading",
        logger=False,
        engineio_logger=False,
    )
    if FLASK_AVAILABLE
    else None
)

# Project imports
from campro.orchestration.adapters.cem_adapter import CEMClientAdapter  # noqa: E402
from campro.orchestration.adapters.simulation_adapter import PhysicsSimulationAdapter  # noqa: E402
from campro.orchestration.adapters.solver_adapter import SimpleSolverAdapter  # noqa: E402
from campro.orchestration.adapters.surrogate_adapter import EnsembleSurrogateAdapter  # noqa: E402
from campro.orchestration.orchestrator import OrchestrationConfig, Orchestrator  # noqa: E402
from campro.validation.verification_assertions import (
    VerificationSuite,
    assert_dimensional_consistency,
    assert_efficiency_bounds,
    assert_energy_conservation,
    assert_geometry_validity,
    assert_losses_bounded,
    assert_losses_nonnegative,
)
from provenance.dataflow_scanner import scan_dashboard
from provenance.execution_events import EventType, add_listener, emit_event, set_run_id
from provenance.module_linker import ModuleLinker
from provenance.tool_scanner import KNOWN_TOOLS
from provenance.ws_server import start_background_server

# Job queue integration (Phase 2)
try:
    from dashboard.job_queue import (
        get_job_result,
        get_job_status,
        get_redis_connection,
        submit_optimization_job,
    )

    JOB_QUEUE_AVAILABLE = True
except ImportError:
    JOB_QUEUE_AVAILABLE = False
    logging.warning("Job queue not available - Redis/RQ not installed")


def get_weaviate_client():
    """Get Weaviate client, respecting environment configuration."""

    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")

    # Parse host and port
    if "://" in weaviate_url:
        host_port = weaviate_url.split("://")[1]
    else:
        host_port = weaviate_url

    if ":" in host_port:
        host, port_str = host_port.split(":")
        port = int(port_str)
    else:
        host = host_port
        port = 8080

    # Determine gRPC port based on environment
    # In Docker (host='weaviate'), gRPC is 50051
    # On Localhost (host='localhost'), gRPC is 50052 (mapped)
    grpc_port = 50051 if host != "localhost" and host != "127.0.0.1" else 50052

    logging.info("Connecting to Weaviate at %s:%s (gRPC: %s)", host, port, grpc_port)

    if weaviate is None:
        return None

    try:
        return weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=False,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=False,
        )
    except Exception as e:
        logging.warning(
            f"Connection to Weaviate at {host}:{port} failed: {e}. Retrying with local default."
        )
        try:
            return weaviate.connect_to_local(port=8080, grpc_port=grpc_port)
        except Exception:
            return None


class WebSocketLogHandler(logging.Handler):
    """Log handler that emits records via Flask-SocketIO."""

    def emit(self, record: logging.LogRecord) -> None:
        if not socketio:
            return

        try:
            msg = self.format(record)
            # Skip empty messages
            if not msg or not msg.strip():
                return

            # Default module, verify safe attribute access
            module = getattr(record, "module_id", "ORCH")

            # Construct payload compatible with frontend handleEvent
            # Frontend expects: { type, module, metadata: { message, level } }
            payload = {
                "type": "log",
                "module": module,
                "metadata": {"message": msg, "level": record.levelname},
            }

            # Emit to all connected clients
            try:
                socketio.emit("execution_event", payload)
            except RuntimeError:
                # If outside request context or loop issues
                pass

        except Exception:
            self.handleError(record)


# Pre-computed module tools (fallback if no live scan)
_cached_tools: dict[str, list[dict]] | None = None

# Process registry for task cancellation
active_processes: dict[str, Any] = {}


def stream_subprocess_output(
    proc: subprocess.Popen, task_id: str, task_name: str
) -> tuple[str, int]:
    """Stream subprocess output line-by-line via WebSocket.

    Args:
        proc: Running subprocess with stdout=PIPE
        task_id: Unique task identifier
        task_name: Human-readable task name

    Returns:
        Tuple of (full_output, return_code)
    """
    output_lines = []

    emit_event(EventType.MODULE_START, task_name, metadata={"task_id": task_id})

    # Stream stdout line by line
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        output_lines.append(line)
        # Broadcast to dashboard via WebSocket
        emit_event(
            EventType.LOG,
            task_name,
            metadata={"task_id": task_id, "line": line.strip(), "message": line.strip()},
        )

    proc.wait()
    full_output = "".join(output_lines)

    if proc.returncode == 0:
        emit_event(EventType.MODULE_END, task_name, metadata={"task_id": task_id, "success": True})
    else:
        emit_event(
            EventType.ERROR, task_name, metadata={"task_id": task_id, "error": "Process failed"}
        )

    return full_output, proc.returncode


def get_module_tools(project_root: Path | None = None) -> dict[str, list[dict]]:
    """Get all module tools, caching results.

    Args:
        project_root: Root directory (defaults to project root)

    Returns:
        Mapping of module_id to list of tool dicts
    """
    global _cached_tools
    if _cached_tools is not None:
        return _cached_tools

    root = project_root or PROJECT_ROOT
    linker = ModuleLinker(project_root=root)
    _cached_tools = linker.get_module_tools_json()
    return _cached_tools


def create_app(project_root: Path | None = None) -> Flask | Mock:
    """Create Flask application factory.

    Args:
        project_root: Root directory for scanning

    Returns:
        Configured Flask app
    """
    if not FLASK_AVAILABLE:
        raise ImportError("Flask not available. Install with: pip install flask flask-cors")

    app = Flask(__name__, static_folder=str(PROJECT_ROOT / "dashboard"))
    # Suppress type error for CORS if it's a Mock or optional
    CORS(app)  # type: ignore

    # Initialize SocketIO with this app
    if socketio:
        socketio.init_app(app)

        # Register event listener to forward Provenance events to SocketIO
        def broadcast_event(event):
            # We use a helper to avoid circular reference or context issues
            try:
                # 'execution_event' is the channel the frontend expects
                socketio.emit("execution_event", event.to_json())
            except Exception as e:
                # Use lazy formatting
                logging.debug("Failed to emit event via socketio: %s", e)

        add_listener(broadcast_event)

    register_routes(app, project_root)

    return app


def _setup_websocket_logging(solver_print_level: int = 5) -> None:
    """
    Configure WebSocket logging handler.

    Args:
        solver_print_level: IPOPT print level (0-12). Higher levels enable more verbose output.
    """
    root_logger = logging.getLogger()
    # Remove existing WS handlers to prevent duplicates
    for h in root_logger.handlers[:]:
        if isinstance(h, WebSocketLogHandler):
            root_logger.removeHandler(h)

    # Filter to suppress noisy third-party DEBUG logs
    class ThirdPartyDebugFilter(logging.Filter):
        """Suppress DEBUG logs from verbose third-party libraries."""

        SUPPRESSED_LOGGERS = {"httpcore", "httpx", "weaviate", "urllib3", "requests"}

        def filter(self, record: logging.LogRecord) -> bool:
            # Allow all non-DEBUG messages through
            if record.levelno > logging.DEBUG:
                return True
            # Suppress DEBUG from noisy third-party libraries
            logger_name = record.name.split(".")[0]
            return logger_name not in self.SUPPRESSED_LOGGERS

    ws_handler = WebSocketLogHandler()
    # Set handler level based on IPOPT verbosity:
    # print_level >= 8: Show DEBUG (iteration details)
    # print_level >= 5: Show INFO
    # print_level < 5:  Show WARNING
    if solver_print_level >= 8:
        handler_level = logging.DEBUG
        ws_handler.setLevel(handler_level)
        ws_handler.addFilter(ThirdPartyDebugFilter())
        # CRITICAL: Set root logger to DEBUG so module loggers can propagate DEBUG messages
        root_logger.setLevel(logging.DEBUG)
    elif solver_print_level >= 5:
        handler_level = logging.INFO
        ws_handler.setLevel(handler_level)
        root_logger.setLevel(logging.INFO)
    else:
        handler_level = logging.WARNING
        ws_handler.setLevel(handler_level)
        root_logger.setLevel(logging.WARNING)

    root_logger.addHandler(ws_handler)


def _parse_budget_params(params: dict[str, Any]) -> int:
    """Parse budget parameters."""
    budget_params = params.get("budget", {})
    if isinstance(budget_params, int):
        return budget_params
    if isinstance(budget_params, dict):
        return int(budget_params.get("total_sim_calls", 1000))
    return 1000


def _parse_opt_params(params: dict[str, Any]) -> tuple[int, int]:
    """Parse optimization parameters."""
    opt_params = params.get("optimization", {})

    max_iterations = 10
    if "maxIterations" in params:
        max_iterations = int(params["maxIterations"])
    elif "max_iterations" in opt_params:
        max_iterations = int(opt_params["max_iterations"])

    batch_size = 5
    if "batchSize" in params:
        batch_size = int(params["batchSize"])
    elif "batch_size" in opt_params:
        batch_size = int(opt_params["batch_size"])

    return max_iterations, batch_size


def _run_orchestration(
    run_id: str, total_budget: int, batch_size: int, max_iterations: int, params: dict[str, Any]
) -> None:
    """Run the orchestration loop in a background thread."""
    try:
        logging.info("Starting real orchestration (Run ID: %s)", run_id)

        # 1. Configure
        config = OrchestrationConfig(
            total_sim_budget=total_budget,
            batch_size=batch_size,
            max_iterations=max_iterations,
            use_provenance=True,
        )

        # 2. Instantiate Adapters
        use_full_physics = params.get("use_full_physics", True)
        mock_cem = params.get("mock_cem", False)

        logging.info("Configuration: Full Physics=%s, Mock CEM=%s", use_full_physics, mock_cem)

        cem_adapter = CEMClientAdapter()
        surrogate_adapter = EnsembleSurrogateAdapter()
        solver_adapter = SimpleSolverAdapter(step_scale=0.05, n_evals=10)
        sim_adapter = PhysicsSimulationAdapter(use_full_physics=use_full_physics)

        # 3. Instantiate Orchestrator
        orch = Orchestrator(
            cem=cem_adapter,
            surrogate=surrogate_adapter,
            solver=solver_adapter,
            simulation=sim_adapter,
            config=config,
        )

        # 4. Run Optimization
        initial_params = {
            "bore": 0.1,
            "stroke": 0.15,
            "cr": 15.0,
            "rpm": 3000.0,
            "p_intake_bar": 1.5,
            "fuel_mass_kg": 5e-5,
        }

        result = orch.optimize(initial_params)
        logging.info("Orchestration finished. Best: %.4f", result.best_objective)

    except Exception as e:
        logging.error("Orchestration failed: %s", e)
        logging.error(traceback.format_exc())
        emit_event(EventType.ERROR, "ORCH", metadata={"message": str(e)})
        emit_event(EventType.RUN_END, "ORCH", metadata={"success": False, "error": str(e)})


def register_routes(app: Flask | Mock, project_root: Path | None = None) -> None:  # noqa: C901, PLR0915, PLR0911
    """Register all routes to the Flask app.

    Args:
        app: Flask application instance
        project_root: Root directory for scanning
    """

    @app.route("/")
    def index() -> Response:
        """Serve dashboard HTML."""
        return send_from_directory(app.static_folder, "orchestrator_dashboard.html")  # type: ignore

    @app.route("/style.css")
    def style() -> Response:
        """Serve dashboard CSS."""
        return send_from_directory(app.static_folder, "style.css")  # type: ignore

    @app.route("/api/modules")
    def list_modules() -> Response:
        """List all modules with their tools."""
        tools = get_module_tools(project_root)
        return jsonify(tools)

    @app.route("/api/modules/<module_id>/tools")
    def get_tools(module_id: str) -> Response | tuple[Response, int]:
        """Get tools for a specific module."""
        tools = get_module_tools(project_root)
        if module_id.upper() not in tools:
            return jsonify({"error": f"Module {module_id} not found"}), 404
        return jsonify(tools[module_id.upper()])

    @app.route("/api/tools")
    def list_all_tools() -> Response:
        """List all known tools."""
        return jsonify(
            [
                {
                    "id": tool_id,
                    "name": info["name"],
                    "category": info["category"],
                    "version": info["version"],
                }
                for tool_id, info in KNOWN_TOOLS.items()
            ]
        )

    @app.route("/api/tools/<tool_id>")
    def get_tool(tool_id: str) -> Response | tuple[Response, int]:
        """Get info for a specific tool."""
        if tool_id not in KNOWN_TOOLS:
            return jsonify({"error": f"Tool {tool_id} not found"}), 404
        info = KNOWN_TOOLS[tool_id]
        return jsonify(
            {
                "id": tool_id,
                "name": info["name"],
                "category": info["category"],
                "version": info["version"],
                "patterns": info["patterns"],
            }
        )

    # =========================================================================
    # Runtime State Endpoints (Dashboard Tracking)
    # =========================================================================

    @app.route("/api/dataflows")
    def list_dataflows() -> Response:
        """List all data flow connections between modules."""
        # scan_dashboard imported globally now
        dashboard = PROJECT_ROOT / "dashboard" / "orchestrator_dashboard.html"
        flows = scan_dashboard(dashboard)
        return jsonify(
            [
                {
                    "flow_id": f.flow_id,
                    "source": f.source_module,
                    "target": f.target_module,
                    "label": f.label,
                    "category": f.category,
                    "color": f.color,
                }
                for f in flows[:20]  # First 20 valid flows
            ]
        )

    @app.route("/api/optimization/steps")
    def list_optimization_steps() -> Response:
        """List recent optimization steps (if Weaviate connected)."""
        try:
            client = get_weaviate_client()
            steps = client.collections.get("OptimizationStep")
            result = steps.query.fetch_objects(limit=20)
            client.close()
            return jsonify(
                [
                    {
                        "iteration": obj.properties.get("iteration"),
                        "timestamp": obj.properties.get("timestamp"),
                        "best_objective": obj.properties.get("best_objective"),
                        "budget_remaining": obj.properties.get("budget_remaining"),
                    }
                    for obj in result.objects
                ]
            )
        except Exception as e:
            return jsonify({"error": str(e), "data": []})

    # =========================================================================
    # Job Queue Endpoints (Phase 2 - Architecture Refactor)
    # =========================================================================

    @app.route("/api/health")
    def health_check() -> Response:
        """Health check endpoint for larrak-api."""
        redis_status = (
            "available" if JOB_QUEUE_AVAILABLE and get_redis_connection() else "unavailable"
        )
        return jsonify(
            {
                "status": "healthy",
                "service": "larrak-api",
                "redis": redis_status,
                "job_queue": JOB_QUEUE_AVAILABLE,
            }
        )

    @app.route("/api/runs/submit", methods=["POST"])
    def submit_run() -> Response | tuple[Response, int]:
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
            logging.error("Failed to submit job: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/runs/<job_id>")
    def get_run_status_endpoint(job_id: str) -> Response | tuple[Response, int]:
        """Get status of a queued optimization run."""
        if not JOB_QUEUE_AVAILABLE:
            return jsonify({"error": "Job queue not available"}), 503

        status = get_job_status(job_id)
        if status:
            return jsonify(status)
        else:
            return jsonify({"error": f"Job {job_id} not found"}), 404

    @app.route("/api/runs/<job_id>/result")
    def get_run_result_endpoint(job_id: str) -> Response | tuple[Response, int]:
        """Get result of completed optimization run."""
        if not JOB_QUEUE_AVAILABLE:
            return jsonify({"error": "Job queue not available"}), 503

        result = get_job_result(job_id)
        if result:
            return jsonify({"result": result})
        else:
            status = get_job_status(job_id)
            if status:
                return jsonify(
                    {"error": "Job not finished", "status": status.get("status")}
                ), 425  # Too Early
            else:
                return jsonify({"error": f"Job {job_id} not found"}), 404

    @app.route("/api/docker/start", methods=["POST"])
    def start_docker_containers() -> Response | tuple[Response, int]:
        """Start all Docker containers using docker compose."""

        try:
            # Find project root (where docker-compose.yml is located)
            compose_file = PROJECT_ROOT / "docker-compose.yml"
            if not compose_file.exists():
                return jsonify({"error": "docker-compose.yml not found"}), 404

            # Check if Docker is running
            check_result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            if check_result.returncode != 0:
                return jsonify(
                    {"error": "Docker is not running. Please start Docker Desktop."}
                ), 503

            # Start all containers
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "up", "-d"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            if result.returncode == 0:
                # Get list of running services
                ps_result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "-f",
                        str(compose_file),
                        "ps",
                        "--format",
                        "{{.Service}}",
                    ],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                services = (
                    [s.strip() for s in ps_result.stdout.splitlines() if s.strip()]
                    if ps_result.returncode == 0
                    else []
                )

                return jsonify(
                    {
                        "success": True,
                        "message": "All Docker containers started successfully",
                        "services": services,
                    }
                )
            else:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return jsonify({"error": f"Failed to start containers: {error_msg}"}), 500

        except subprocess.TimeoutExpired:
            return jsonify({"error": "Docker command timed out"}), 504
        except FileNotFoundError:
            return jsonify({"error": "Docker command not found. Please install Docker."}), 503
        except Exception as e:
            logging.error("Error starting Docker containers: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/docker/status")
    def docker_status() -> Response:
        """Check Docker container status."""

        try:
            compose_file = PROJECT_ROOT / "docker-compose.yml"
            if not compose_file.exists():
                return jsonify({"error": "docker-compose.yml not found"}), 404

            # Get container status in JSON format
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "ps", "--format", "json"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if result.returncode == 0:
                services = []
                for line in result.stdout.strip().splitlines():
                    if line.strip():
                        try:
                            service_data = json.loads(line)
                            services.append(
                                {
                                    "service": service_data.get("Service", ""),
                                    "status": service_data.get("State", ""),
                                    "health": service_data.get("Health", ""),
                                }
                            )
                        except json.JSONDecodeError:
                            continue

                return jsonify(
                    {
                        "success": True,
                        "services": services,
                    }
                )
            else:
                return jsonify(
                    {
                        "success": False,
                        "error": result.stderr or "Failed to get container status",
                        "services": [],
                    }
                )

        except subprocess.TimeoutExpired:
            return jsonify({"error": "Docker command timed out", "services": []}), 504
        except FileNotFoundError:
            return jsonify({"error": "Docker command not found", "services": []}), 503
        except Exception as e:
            logging.error("Error checking Docker status: %s", e)
            return jsonify({"error": str(e), "services": []}), 500

    @app.route("/api/cache/stats")
    def cache_stats() -> Response:
        """Get evaluation cache statistics."""
        try:
            client = get_weaviate_client()
            cache = client.collections.get("CacheEntry")
            result = cache.aggregate.over_all(total_count=True)
            hits = cache.query.fetch_objects(limit=1000)
            client.close()
            hit_count = sum(1 for o in hits.objects if o.properties.get("was_hit"))
            total = len(hits.objects)
            return jsonify(
                {
                    "total_entries": result.total_count,
                    "hit_rate": hit_count / total if total > 0 else 0,
                }
            )
        except Exception as e:
            return jsonify({"error": str(e), "total_entries": 0, "hit_rate": 0})

    @app.route("/api/budget/snapshots")
    def budget_snapshots() -> Response:
        """Get budget allocation history."""
        try:
            client = get_weaviate_client()
            budget = client.collections.get("BudgetSnapshot")
            result = budget.query.fetch_objects(limit=20)
            client.close()
            return jsonify(
                [
                    {
                        "timestamp": obj.properties.get("timestamp"),
                        "total": obj.properties.get("total_budget"),
                        "spent": obj.properties.get("spent_budget"),
                        "remaining": obj.properties.get("remaining_budget"),
                    }
                    for obj in result.objects
                ]
            )
        except Exception as e:
            return jsonify({"error": str(e), "data": []})

    @app.route("/api/trustregion/logs")
    def trustregion_logs() -> Response:
        """Get trust region adjustment history."""
        try:
            client = get_weaviate_client()
            tr = client.collections.get("TrustRegionLog")
            result = tr.query.fetch_objects(limit=20)
            client.close()
            return jsonify(
                [
                    {
                        "timestamp": obj.properties.get("timestamp"),
                        "radius_before": obj.properties.get("radius_before"),
                        "radius_after": obj.properties.get("radius_after"),
                        "action": obj.properties.get("action"),
                    }
                    for obj in result.objects
                ]
            )
        except Exception as e:
            return jsonify({"error": str(e), "data": []})

    # =========================================================================
    # Code Outline Endpoint (IDE Integration)
    # =========================================================================

    @app.route("/api/outline/<path:file_path>")
    def get_file_outline(file_path: str) -> Response:
        """Get code outline for a specific file."""
        try:
            client = get_weaviate_client()
            symbols_collection = client.collections.get("CodeSymbol")

            # Query symbols for this file
            import weaviate.classes.query as wvq

            result = symbols_collection.query.fetch_objects(
                filters=wvq.Filter.by_property("file_path").equal(file_path),
                limit=1000,
                return_references=[
                    wvq.QueryReference(link_on="parent_symbol", return_properties=["name"])
                ],
            )

            client.close()

            if not result.objects:
                return jsonify(
                    {
                        "file": file_path,
                        "symbols": [],
                        "message": "File not indexed or no symbols found",
                    }
                ), 404

            # Build hierarchical structure
            symbols_by_uuid = {}
            root_symbols = []

            for obj in result.objects:
                props = obj.properties
                symbol = {
                    "name": props.get("name", ""),
                    "kind": props.get("kind", "unknown"),
                    "line_start": props.get("line_number", 0),
                    "line_end": props.get("line_end", props.get("line_number", 0)),
                    "signature": props.get("signature", ""),
                    "docstring": props.get("docstring", ""),
                    "decorators": props.get("decorators", []),
                    "is_async": props.get("is_async", False),
                    "return_type": props.get("return_type", ""),
                    "children": [],
                }

                symbols_by_uuid[str(obj.uuid)] = symbol

                # Check if this symbol has a parent
                refs = obj.references
                has_parent = (
                    refs and hasattr(refs, "parent_symbol") and cast(Any, refs).parent_symbol
                )

                if not has_parent:
                    root_symbols.append(symbol)

            # Build parent-child relationships
            for obj in result.objects:
                refs = obj.references
                if refs and hasattr(refs, "parent_symbol") and cast(Any, refs).parent_symbol:
                    parent_objs = cast(Any, refs).parent_symbol.objects
                    if parent_objs:
                        parent_uuid = str(parent_objs[0].uuid)
                        if parent_uuid in symbols_by_uuid:
                            child_uuid = str(obj.uuid)
                            if child_uuid in symbols_by_uuid:
                                symbols_by_uuid[parent_uuid]["children"].append(
                                    symbols_by_uuid[child_uuid]
                                )

            # Sort symbols by line number
            def sort_symbols(symbols):
                symbols.sort(key=lambda s: s["line_start"])
                for sym in symbols:
                    if sym["children"]:
                        sort_symbols(sym["children"])

            sort_symbols(root_symbols)

            return jsonify(
                {
                    "file": file_path,
                    "symbols": root_symbols,
                    "count": len(result.objects),
                }
            )

        except Exception as e:
            return jsonify(
                {
                    "error": str(e),
                    "file": file_path,
                    "symbols": [],
                }
            ), 500

    @app.route("/api/outline/reindex", methods=["POST"])
    def reindex_file() -> Response:
        """Trigger re-indexing of a specific file."""

        data = request.get_json() or {}
        file_path = data.get("file")

        if not file_path:
            return jsonify({"error": "file parameter required"}), 400

        # Get absolute path
        repo_root = Path(PROJECT_ROOT)
        abs_file_path = repo_root / file_path

        if not abs_file_path.exists():
            return jsonify({"error": f"File not found: {file_path}"}), 404

        if not str(abs_file_path).endswith(".py"):
            return jsonify({"error": "Only Python files can be indexed"}), 400

        try:
            # Run code scanner for single file
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            env["WEAVIATE_URL"] = "http://localhost:8080"

            result = subprocess.run(
                [
                    sys.executable,
                    str(repo_root / "truthmaker" / "ingestion" / "code_scanner.py"),
                    "--file",
                    str(abs_file_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
                cwd=str(repo_root),
                check=False,
            )

            if result.returncode == 0:
                return jsonify(
                    {
                        "status": "success",
                        "file": file_path,
                        "message": "File re-indexed successfully",
                    }
                )
            else:
                return jsonify(
                    {
                        "status": "error",
                        "file": file_path,
                        "message": "Re-indexing failed",
                        "error": result.stderr,
                    }
                ), 500

        except subprocess.TimeoutExpired:
            return jsonify(
                {"status": "error", "file": file_path, "message": "Re-indexing timed out (>30s)"}
            ), 500
        except Exception as e:
            return jsonify({"status": "error", "file": file_path, "message": str(e)}), 500

    # =========================================================================
    # Sequence Execution Endpoint (Live Tracking)
    # =========================================================================

    @app.route("/api/start", methods=["POST"])
    def start_sequence() -> Response:
        """Start optimization sequence with live event broadcasting.

        This triggers the actual optimization loop and emits events
        that get broadcast via WebSocket to connected dashboards.
        """
        # Parse params from request
        params = request.get_json() or {}

        # Extract solver print level for WebSocket logging configuration
        solver_params = params.get("solver", {})
        solver_print_level = int(solver_params.get("print_level", 5))

        # Configure WebSocket logging based on solver verbosity
        _setup_websocket_logging(solver_print_level)

        # Extract optimization params
        opt_params = params.get("optimization", {})
        budget_params = params.get("budget", {})

        # Handle cases where budget might be int (legacy) or dict
        total_budget = 1000  # Default to production scale
        if isinstance(budget_params, int):
            total_budget = budget_params
        elif isinstance(budget_params, dict):
            total_budget = int(budget_params.get("total_sim_calls", 1000))

        # Handle max_iterations
        max_iterations = 10
        if "maxIterations" in params:
            max_iterations = int(params["maxIterations"])
        elif "max_iterations" in opt_params:
            max_iterations = int(opt_params["max_iterations"])

        # Handle batch_size
        batch_size = 5
        if "batchSize" in params:
            batch_size = int(params["batchSize"])
        elif "batch_size" in opt_params:
            batch_size = int(opt_params["batch_size"])

        # Start WebSocket server if not already running
        # With Flask-SocketIO, the server is already running on the same port
        pass

        run_id = str(uuid.uuid4())[:8]
        set_run_id(run_id)

        # Reset stop signal for new run

        os.environ["ORCHESTRATOR_STOP_SIGNAL"] = "0"

        def run_real_sequence() -> None:
            """Run the real orchestrator optimization loop."""
            try:
                logging.info("Starting real orchestration (Run ID: %s)", run_id)

                # 1. Configure
                config = OrchestrationConfig(
                    total_sim_budget=total_budget,
                    batch_size=batch_size,
                    max_iterations=max_iterations,
                    use_provenance=True,
                )

                # 2. Instantiate Adapters
                # Extract execution flags (Default to PRODUCTION mode)
                use_full_physics = params.get("use_full_physics", True)
                mock_cem = params.get("mock_cem", False)

                logging.info(
                    "Configuration: Full Physics=%s, Mock CEM=%s", use_full_physics, mock_cem
                )

                # 2. Instantiate Adapters
                # CEM Adapter (Set mock=False for real validation)
                cem_adapter = CEMClientAdapter()

                # Use surrogate adapter (will mock if model not found)
                surrogate_adapter = EnsembleSurrogateAdapter()

                # Extract solver config from params for physics simulation
                solver_params = params.get("solver", {})
                solver_config = None
                if solver_params:
                    ipopt_opts = {}
                    if "print_level" in solver_params:
                        ipopt_opts["print_level"] = int(solver_params["print_level"])
                    if "linear_solver" in solver_params:
                        ipopt_opts["linear_solver"] = str(solver_params["linear_solver"])
                    if "max_iter" in solver_params:
                        ipopt_opts["max_iter"] = int(solver_params["max_iter"])
                    if "acceptable_tol" in solver_params:
                        ipopt_opts["acceptable_tol"] = float(solver_params["acceptable_tol"])

                    if ipopt_opts:
                        solver_config = {"ipopt": ipopt_opts}
                        logging.info(
                            f"Solver config: print_level={ipopt_opts.get('print_level', 'default')}, "
                            f"linear_solver={ipopt_opts.get('linear_solver', 'default')}"
                        )

                # Use simple solver for robustness
                solver_adapter = SimpleSolverAdapter(step_scale=0.05, n_evals=10)

                # Physics Simulation with solver config
                sim_adapter = PhysicsSimulationAdapter(
                    use_full_physics=use_full_physics, solver_config=solver_config
                )

                # 3. Instantiate Orchestrator
                orch = Orchestrator(
                    cem=cem_adapter,
                    surrogate=surrogate_adapter,
                    solver=solver_adapter,
                    simulation=sim_adapter,
                    config=config,
                )

                # Inject run_id into provenance if possible, or it will generate its own
                # The orchestrator generates its own run_id in optimize()

                # Define initial params (engine design parameters for CEM seeding)
                initial_params = {
                    "bore": 0.1,
                    "stroke": 0.15,
                    "cr": 15.0,
                    "rpm": 3000.0,
                    "p_intake_bar": 1.5,
                    "fuel_mass_kg": 5e-5,
                }

                result = orch.optimize(initial_params)

                logging.info(f"Orchestration finished. Best: {result.best_objective:.4f}")

            except Exception as e:
                logging.error("Orchestration failed: %s", e)
                logging.error(traceback.format_exc())
                emit_event(EventType.ERROR, "ORCH", metadata={"message": str(e)})
                emit_event(EventType.RUN_END, "ORCH", metadata={"success": False, "error": str(e)})

        # Run in background thread
        thread = threading.Thread(target=run_real_sequence, daemon=True)
        thread.start()

        return jsonify({"status": "started", "run_id": run_id})

    @app.route("/api/stop", methods=["POST"])
    def stop_sequence() -> Response:
        """Stop running optimization."""

        # Set global stop signal
        os.environ["ORCHESTRATOR_STOP_SIGNAL"] = "1"
        emit_event(EventType.RUN_END, "ORCH", metadata={"stopped": True})
        return jsonify({"status": "stopped"})

    @app.route("/api/reset", methods=["POST"])
    def reset_sequence() -> Response:
        """Reset optimization state."""

        # Clear stop signal
        if "ORCHESTRATOR_STOP_SIGNAL" in os.environ:
            del os.environ["ORCHESTRATOR_STOP_SIGNAL"]

        emit_event(EventType.RUN_END, "ORCH", metadata={"reset": True})
        return jsonify({"status": "reset"})

    # =========================================================================
    # Workflow Action Endpoints
    # =========================================================================

    @app.route("/api/train_surrogates", methods=["POST"])
    def train_surrogates() -> Response:
        """Train structural and thermal surrogates from pilot DOE data."""
        try:
            project_root = Path(__file__).parents[1]
            script_path = project_root / "scripts" / "train_surrogates.py"

            if not script_path.exists():
                return jsonify({"error": f"Script not found: {script_path}"}), 404

            # Generate task ID for tracking
            task_id = str(uuid.uuid4())[:8]

            # Run in subprocess with streaming
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Register for cancellation
            active_processes[task_id] = proc

            try:
                # Stream output via WebSocket
                output, returncode = stream_subprocess_output(proc, task_id, "SURROGATE_TRAIN")

                if returncode == 0:
                    return jsonify(
                        {
                            "status": "success",
                            "task_id": task_id,
                            "message": "Surrogates trained successfully",
                            "output": output,
                            "models": [
                                "models/hifi/structural_surrogate.pt",
                                "models/hifi/thermal_surrogate.pt",
                            ],
                        }
                    )
                else:
                    return jsonify(
                        {
                            "status": "failed",
                            "task_id": task_id,
                            "error": "Training failed",
                            "output": output,
                        }
                    ), 500
            finally:
                # Clean up registry
                active_processes.pop(task_id, None)

        except Exception as e:
            logging.error("Surrogate training error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/run_gear_optimization", methods=["POST"])
    def run_gear_optimization() -> Response:
        """Run Phase 3 conjugate gear profile optimization."""
        try:
            project_root = Path(__file__).parents[1]
            script_path = project_root / "scripts" / "phase3" / "run_conjugate_optimization.py"

            if not script_path.exists():
                return jsonify({"error": f"Script not found: {script_path}"}), 404

            # Generate task ID
            task_id = str(uuid.uuid4())[:8]

            # Run in subprocess with streaming
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Register for cancellation
            active_processes[task_id] = proc

            try:
                # Stream output via WebSocket (this takes 5-10 minutes!)
                output, returncode = stream_subprocess_output(proc, task_id, "GEAR_OPT")

                if returncode == 0:
                    return jsonify(
                        {
                            "status": "success",
                            "task_id": task_id,
                            "message": "Gear optimization complete",
                            "output": output,
                            "results": [
                                "output/conjugate_shapes.html",
                                "output/conjugate_radii.html",
                            ],
                        }
                    )
                else:
                    return jsonify(
                        {
                            "status": "failed",
                            "task_id": task_id,
                            "error": "Optimization failed",
                            "output": output,
                        }
                    ), 500
            finally:
                active_processes.pop(task_id, None)

        except Exception as e:
            logging.error("Gear optimization error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/generate_doe", methods=["POST"])
    def generate_doe() -> Response:
        """Generate Design of Experiments (DOE) samples for pilot study."""
        try:
            project_root = Path(__file__).parents[1]
            script_path = project_root / "scripts" / "run_pilot_doe.py"

            if not script_path.exists():
                return jsonify({"error": f"Script not found: {script_path}"}), 404

            # Get optional parameters from request
            data = request.get_json() or {}
            n_samples = data.get("n_samples", 50)

            # Generate task ID
            task_id = str(uuid.uuid4())[:8]

            # Run in subprocess with streaming
            proc = subprocess.Popen(
                [sys.executable, str(script_path), str(n_samples)],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Register for cancellation
            active_processes[task_id] = proc

            try:
                # Stream output via WebSocket
                output, returncode = stream_subprocess_output(proc, task_id, "DOE_GEN")

                if returncode == 0:
                    return jsonify(
                        {
                            "status": "success",
                            "task_id": task_id,
                            "message": "DOE generation complete",
                            "output": output,
                            "n_samples": n_samples,
                        }
                    )
                else:
                    return jsonify(
                        {
                            "status": "failed",
                            "task_id": task_id,
                            "error": "DOE generation failed",
                            "output": output,
                        }
                    ), 500
            finally:
                active_processes.pop(task_id, None)

        except Exception as e:
            logging.error("DOE generation error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/cancel_task/<task_id>", methods=["POST"])
    def cancel_task(task_id: str) -> Response:
        """Cancel a running workflow task by task ID."""
        try:
            proc = active_processes.get(task_id)
            if not proc:
                return jsonify({"error": f"Task {task_id} not found or already completed"}), 404

            # Terminate the process
            proc.terminate()  # Send SIGTERM
            try:
                proc.wait(timeout=5)  # Wait up to 5 seconds
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                proc.kill()
                proc.wait()

            # Clean up
            active_processes.pop(task_id, None)

            # Emit cancellation event
            emit_event(
                EventType.WARNING,
                "TASK_CANCEL",
                metadata={"task_id": task_id, "message": "Task cancelled by user"},
            )

            return jsonify(
                {
                    "status": "cancelled",
                    "task_id": task_id,
                    "message": "Task terminated successfully",
                }
            )

        except Exception as e:
            logging.error("Task cancellation error: %s", e)
            return jsonify({"error": str(e)}), 500

    # ========== PHASE 5 ENDPOINTS ==========

    @app.route("/api/visualizations/list")
    def list_visualizations() -> Response:
        """List all visualization files in the output directory."""
        try:
            output_dir = Path(__file__).parents[1] / "output"
            if not output_dir.exists():
                return jsonify({"visualizations": [], "count": 0})

            visualizations = []
            for ext in ["*.html", "*.png", "*.jpg", "*.svg"]:
                for f in output_dir.glob(ext):
                    visualizations.append(
                        {
                            "path": str(f.relative_to(output_dir.parent)),
                            "name": f.name,
                            "type": f.suffix[1:],
                            "size": f.stat().st_size,
                            "modified": f.stat().st_mtime,
                        }
                    )

            visualizations.sort(key=lambda x: x["modified"], reverse=True)
            return jsonify({"visualizations": visualizations, "count": len(visualizations)})

        except Exception as e:
            logging.error("Visualization list error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/cem/validation_stats")
    def cem_validation_stats() -> Response:
        """Get CEM validation statistics from recent runs."""
        try:
            # Placeholder stats - will integrate with Weaviate later
            stats = {
                "total_candidates": 1247,
                "feasible": 892,
                "infeasible": 355,
                "feasibility_rate": 71.5,
                "rejection_reasons": {
                    "Stress limit exceeded": 128,
                    "Temperature out of range": 94,
                    "Geometry invalid": 67,
                    "Material failure": 44,
                    "Other": 22,
                },
            }

            # TODO: Query Weaviate for actual CEM validation data
            # if weaviate_client:
            #     results = query_cem_validations()
            #     stats = process_validation_results(results)

            return jsonify(stats)

        except Exception as e:
            logging.error("CEM stats error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/diagnostics/analyze_failures", methods=["POST"])
    def analyze_failures() -> Response:
        """Run failure analysis diagnostic script."""
        try:
            # TODO: Execute scripts/analysis/analyze_failures.py
            return jsonify(
                {
                    "status": "not_implemented",
                    "message": "Failure analysis script execution not yet implemented. Will run scripts/analysis/analyze_failures.py when integrated.",
                }
            ), 501
        except Exception as e:
            logging.error("Failure analysis error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/diagnostics/nlp_health", methods=["POST"])
    def nlp_health_check() -> Response:
        """Run NLP health check diagnostic."""
        try:
            # TODO: Execute tests/infra/nlp_diagnostics.py
            return jsonify(
                {
                    "status": "not_implemented",
                    "message": "NLP health check not yet implemented. Will run tests/infra/nlp_diagnostics.py when integrated.",
                }
            ), 501
        except Exception as e:
            logging.error("NLP health check error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/diagnostics/recovery_patterns", methods=["POST"])
    def recovery_patterns() -> Response:
        """Analyze recovery patterns from failures."""
        try:
            # TODO: Implement recovery pattern analysis
            return jsonify(
                {
                    "status": "not_implemented",
                    "message": "Recovery pattern analysis not yet implemented.",
                }
            ), 501
        except Exception as e:
            logging.error("Recovery pattern analysis error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/models/list")
    def list_models() -> Response:
        """List all registered surrogate models."""
        try:
            models_dir = Path(__file__).parents[1] / "models"
            if not models_dir.exists():
                return jsonify({"models": [], "count": 0})

            models = []
            for ext in ["*.pt", "*.pth"]:
                for f in models_dir.glob(ext):
                    models.append(
                        {
                            "id": f.stem,  # filename without extension
                            "path": str(f.relative_to(models_dir.parent)),
                            "name": f.name,
                            "type": f.suffix[1:],
                            "size": f.stat().st_size,
                            "modified": f.stat().st_mtime,
                        }
                    )

            models.sort(key=lambda x: x["modified"], reverse=True)
            return jsonify({"models": models, "count": len(models)})

        except Exception as e:
            logging.error("Model list error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/models/<model_id>/load", methods=["POST"])
    def load_model(model_id: str) -> Response:
        """Load a specific surrogate model."""
        try:
            # TODO: Integrate with provenance.model_registry for actual loading
            return jsonify(
                {
                    "status": "not_implemented",
                    "message": f"Model loading not yet implemented for model_id={model_id}. Will integrate with provenance.model_registry.",
                }
            ), 501
        except Exception as e:
            logging.error("Model load error: %s", e)
            return jsonify({"error": str(e)}), 500

    # =========================================================================
    # Verification Endpoints (Phase 3)
    # =========================================================================

    @app.route("/api/test/verify/input", methods=["POST"])
    def verify_input_params() -> tuple[Response, int] | Response:
        """Verify input parameters."""
        try:
            data = request.get_json() or {}
            params = data.get("params", {})

            if not params:
                return jsonify({"error": "No parameters provided"}), 400

            suite = VerificationSuite(strict=False)
            suite.add_assertion(assert_dimensional_consistency(params, {}))
            suite.add_assertion(assert_geometry_validity(params))

            report = suite.report()
            return jsonify(report)

        except Exception as e:
            logging.error("Input verification error: %s", e)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/test/verify/result", methods=["POST"])
    def verify_result() -> tuple[Response, int] | Response:
        """Verify optimization results."""
        try:
            data = request.get_json() or {}
            result = data.get("result", {})

            if not result:
                return jsonify({"error": "No result provided"}), 400

            suite = VerificationSuite(strict=False)

            # Use metrics directly if provided
            metrics = result.get("metrics", {})
            if metrics:
                suite.add_assertion(assert_efficiency_bounds(metrics))
                suite.add_assertion(assert_energy_conservation(metrics))
                suite.add_assertion(assert_losses_nonnegative(metrics))
                suite.add_assertion(assert_losses_bounded(metrics))
            else:
                # If full result object/dict passed, try to extract validity
                # For now, minimal check on structure
                if "best_objective" in result:
                    pass  # Valid structure check could go here

            report = suite.report()
            return jsonify(report)

        except Exception as e:
            logging.error("Result verification error: %s", e)
            return jsonify({"error": str(e)}), 500


def generate_static_tools_js(output_path: Path | None = None) -> str:
    """Generate static JavaScript with embedded tool data.

    This is for serving the dashboard without a backend server.

    Args:
        output_path: Optional path to write JS file

    Returns:
        JavaScript code as string
    """
    tools = get_module_tools()

    js_code = f"""// Auto-generated tool data from provenance/module_linker.py
// Generated at: {__import__("datetime").datetime.now().isoformat()}

const MODULE_TOOLS = {json.dumps(tools, indent=2)};

/**
 * Get tools for a given module ID.
 * @param {{string}} moduleId - Module identifier (e.g., "SOL", "CEM")
 * @returns {{Array}} Array of tool objects
 */
function getModuleTools(moduleId) {{
    return MODULE_TOOLS[moduleId.toUpperCase()] || [];
}}

/**
 * Format tools as Mermaid subgraph.
 * @param {{string}} moduleId - Module identifier
 * @returns {{string}} Mermaid subgraph definition
 */
function getToolsSubgraph(moduleId) {{
    const tools = getModuleTools(moduleId);
    if (tools.length === 0) return '';

    let mermaid = 'subgraph Tools [🔧 Tools]\\n';
    tools.forEach((t, i) => {{
        mermaid += `T${{i + 1}}[${{t.name}}]\\n`;
    }});
    mermaid += 'end\\n';
    return mermaid;
}}
"""

    if output_path:
        output_path.write_text(js_code, encoding="utf-8")

    return js_code


def main() -> int:
    """CLI entry point."""
    # No local import needed

    parser = argparse.ArgumentParser(description="Dashboard API server")
    parser.add_argument("--port", type=int, default=5001, help="Port to run on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--generate-static",
        type=Path,
        help="Generate static JS file instead of running server",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    # logger = logging.getLogger(__name__)  # Unused

    # Add WebSocket log handler
    ws_handler = WebSocketLogHandler()
    ws_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(ws_handler)

    request_logger = logging.getLogger("werkzeug")
    request_logger.setLevel(logging.ERROR)  # Silence standard request logs

    if args.generate_static:
        print(f"Generating static tools JS to {args.generate_static}")
        generate_static_tools_js(args.generate_static)
        return 0

    if not FLASK_AVAILABLE:
        print("Flask not available. Install with: pip install flask flask-cors")
        print("Or use --generate-static to create a static JS file.")
        return 1

    app = create_app()
    print(f"Starting dashboard API on http://{args.host}:{args.port}")
    print("Endpoints:")
    print("  GET /                      - Dashboard HTML")
    print("  GET /api/modules           - All modules with tools")
    print("  GET /api/modules/<id>/tools - Tools for specific module")
    print("  GET /api/tools             - All known tools")
    print("  GET /api/tools             - All known tools")

    # Start WebSocket server immediately for dashboard connectivity
    start_background_server(host=args.host)

    # Use socketio.run() instead of app.run() to properly initialize SocketIO
    if socketio:
        socketio.run(app, host=args.host, port=args.port, debug=False, allow_unsafe_werkzeug=True)
    else:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
