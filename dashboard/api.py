"""Dashboard API for tool and module queries.

Simple Flask API serving tool information to the orchestrator dashboard.
Can also run standalone for development/testing.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    from flask_socketio import SocketIO, emit

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None  # type: ignore
    Response = Any  # type: ignore
    SocketIO = None  # type: ignore
    emit = None  # type: ignore

# Global SocketIO instance
# Use Redis as message queue for distributed event broadcasting (from worker)
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
socketio = SocketIO(cors_allowed_origins="*", message_queue=REDIS_URL) if FLASK_AVAILABLE else None

# Project imports
from campro.orchestration.adapters.cem_adapter import CEMClientAdapter
from campro.orchestration.adapters.simulation_adapter import PhysicsSimulationAdapter
from campro.orchestration.adapters.solver_adapter import SimpleSolverAdapter
from campro.orchestration.adapters.surrogate_adapter import EnsembleSurrogateAdapter
from campro.orchestration.orchestrator import OrchestrationConfig, Orchestrator
from provenance.dataflow_scanner import scan_dashboard
from provenance.execution_events import (
    EventType,
    add_listener,
    emit_event,
    error,
    log_message,
    set_run_id,
    warning,
)
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
    import os

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

    logging.info(f"Connecting to Weaviate at {host}:{port} (gRPC: {grpc_port})")
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
        return weaviate.connect_to_local(port=8080, grpc_port=grpc_port)


class WebSocketLogHandler(logging.Handler):
    """Log handler that emits records via Flask-SocketIO."""

    def emit(self, record: logging.LogRecord) -> None:
        if not socketio:
            return

        try:
            msg = self.format(record)
            # Default module, verify safe attribute access
            module = getattr(record, "module_id", "ORCH")

            # Construct payload compatible with frontend
            payload = {"type": "log", "source": module, "text": msg, "level": record.levelname}

            # Emit to all connected clients
            # We use external=True if called from background thread
            try:
                socketio.emit("execution_event", payload)
            except RuntimeError:
                # If outside request context or loop issues?
                # socketio.emit usually handles thread-safety if using eventlet
                pass

        except Exception:
            self.handleError(record)


# Pre-computed module tools (fallback if no live scan)
_cached_tools: dict[str, list[dict]] | None = None


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


def create_app(project_root: Path | None = None) -> "Flask":
    """Create Flask application.

    Args:
        project_root: Root directory for scanning

    Returns:
        Configured Flask app
    """
    if not FLASK_AVAILABLE:
        raise ImportError("Flask not available. Install with: pip install flask flask-cors")

    app = Flask(__name__, static_folder=str(PROJECT_ROOT / "dashboard"))
    CORS(app)

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
                logging.debug(f"Failed to emit event via socketio: {e}")

        add_listener(broadcast_event)

    @app.route("/")
    def index() -> Response:
        """Serve dashboard HTML."""
        return send_from_directory(app.static_folder, "orchestrator_dashboard.html")  # type: ignore

    @app.route("/api/modules")
    def list_modules() -> Response:
        """List all modules with their tools."""
        tools = get_module_tools(project_root)
        return jsonify(tools)

    @app.route("/api/modules/<module_id>/tools")
    def get_tools(module_id: str) -> tuple[Response, int] | Response:
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
    def get_tool(tool_id: str) -> tuple[Response, int] | Response:
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
                return jsonify(
                    {"error": "Job not finished", "status": status.get("status")}
                ), 425  # Too Early
            else:
                return jsonify({"error": f"Job {job_id} not found"}), 404

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
                has_parent = refs and hasattr(refs, "parent_symbol") and refs.parent_symbol

                if not has_parent:
                    root_symbols.append(symbol)

            # Build parent-child relationships
            for obj in result.objects:
                refs = obj.references
                if refs and hasattr(refs, "parent_symbol") and refs.parent_symbol:
                    parent_objs = refs.parent_symbol.objects
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
        import os
        import subprocess

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
        # Attach to root logger or specific loggers
        root_logger = logging.getLogger()
        # Remove existing WS handlers to prevent duplicates
        for h in root_logger.handlers[:]:
            if isinstance(h, WebSocketLogHandler):
                root_logger.removeHandler(h)

        ws_handler = WebSocketLogHandler()
        ws_handler.setLevel(logging.INFO)
        root_logger.addHandler(ws_handler)

        # Parse params from request
        params = request.get_json() or {}

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
        import os

        os.environ["ORCHESTRATOR_STOP_SIGNAL"] = "0"

        def run_real_sequence() -> None:
            """Run the real orchestrator optimization loop."""
            try:
                logging.info(f"Starting real orchestration (Run ID: {run_id})")

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

                logging.info(f"Configuration: Full Physics={use_full_physics}, Mock CEM={mock_cem}")

                # 2. Instantiate Adapters
                # CEM Adapter (Set mock=False for real validation)
                cem_adapter = CEMClientAdapter(mock=mock_cem)

                # Use surrogate adapter (will mock if model not found)
                surrogate_adapter = EnsembleSurrogateAdapter()

                # Use simple solver for robustness
                solver_adapter = SimpleSolverAdapter(step_scale=0.05, n_evals=10)

                # Physics Simulation (Set use_full_physics=True for real solver)
                sim_adapter = PhysicsSimulationAdapter(use_full_physics=use_full_physics)

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

                # 4. Run Optimization
                # Define initial params
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
                logging.error(f"Orchestration failed: {e}")
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
        import os

        # Set global stop signal
        os.environ["ORCHESTRATOR_STOP_SIGNAL"] = "1"
        emit_event(EventType.RUN_END, "ORCH", metadata={"stopped": True})
        return jsonify({"status": "stopped"})

    @app.route("/api/reset", methods=["POST"])
    def reset_sequence() -> Response:
        """Reset optimization state."""
        import os

        # Clear stop signal
        if "ORCHESTRATOR_STOP_SIGNAL" in os.environ:
            del os.environ["ORCHESTRATOR_STOP_SIGNAL"]

        emit_event(EventType.RUN_END, "ORCH", metadata={"reset": True})
        return jsonify({"status": "reset"})

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
            logging.error(f"Visualization list error: {e}")
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
            logging.error(f"CEM stats error: {e}")
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
            logging.error(f"Failure analysis error: {e}")
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
            logging.error(f"NLP health check error: {e}")
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
            logging.error(f"Recovery pattern analysis error: {e}")
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
            logging.error(f"Model list error: {e}")
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
            logging.error(f"Model load error: {e}")
            return jsonify({"error": str(e)}), 500

    return app


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
    import argparse

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
    logger = logging.getLogger(__name__)

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

    # running in a separate process that doesn't share memory/events with the
    # orchestrator optimization thread.
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
