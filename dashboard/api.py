"""Dashboard API for tool and module queries.

Simple Flask API serving tool information to the orchestrator dashboard.
Can also run standalone for development/testing.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party imports (handled gracefully if missing)
try:
    import weaviate
    from flask import Flask, Response, jsonify, request, send_from_directory
    from flask_cors import CORS

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

    # Define dummy types for typing execution if Flask missing
    if TYPE_CHECKING:
        from flask import Flask, Response

# Project imports
from campro.orchestration.adapters.cem_adapter import CEMClientAdapter
from campro.orchestration.adapters.simulation_adapter import PhysicsSimulationAdapter
from campro.orchestration.adapters.solver_adapter import SimpleSolverAdapter
from campro.orchestration.adapters.surrogate_adapter import EnsembleSurrogateAdapter
from campro.orchestration.orchestrator import OrchestrationConfig, Orchestrator
from provenance.dataflow_scanner import scan_dashboard
from provenance.execution_events import (
    EventType,
    emit_event,
    error,
    log_message,
    module_end,
    module_start,
    set_run_id,
    step_end,
    step_start,
    warning,
)
from provenance.module_linker import ModuleLinker
from provenance.tool_scanner import KNOWN_TOOLS
from provenance.ws_server import start_background_server


class WebSocketLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Default module, verify safe attribute access
            module = getattr(record, "module_id", "ORCH")

            if record.levelno >= logging.ERROR:
                error(module, msg)
            elif record.levelno >= logging.WARNING:
                warning(module, msg)
            else:
                log_message(module, msg, level=record.levelname)

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
            client = weaviate.connect_to_local(port=8080, grpc_port=50052)
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

    @app.route("/api/cache/stats")
    def cache_stats() -> Response:
        """Get evaluation cache statistics."""
        try:
            client = weaviate.connect_to_local(port=8080, grpc_port=50052)
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
            client = weaviate.connect_to_local(port=8080, grpc_port=50052)
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
            client = weaviate.connect_to_local(port=8080, grpc_port=50052)
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
            client = weaviate.connect_to_local(port=8080, grpc_port=50052)
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

    @app.route("/api/outline/refresh", methods=["POST"])
    def refresh_outline() -> Response:
        """Trigger re-indexing of a specific file."""
        data = request.get_json() or {}
        file_path = data.get("file")

        if not file_path:
            return jsonify({"error": "file parameter required"}), 400

        # This would trigger the code scanner for a single file
        # For now, return a placeholder response
        return jsonify(
            {
                "status": "queued",
                "file": file_path,
                "message": "File re-indexing queued. This feature requires the file watcher service to be implemented.",
            }
        )

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
        total_budget = 20
        if isinstance(budget_params, int):
            total_budget = budget_params
        elif isinstance(budget_params, dict):
            total_budget = int(budget_params.get("total_sim_calls", 20))

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
        start_background_server()

        run_id = str(uuid.uuid4())[:8]
        set_run_id(run_id)

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
                # Use mock CEM for now to avoid needing external service
                cem_adapter = CEMClientAdapter(mock=True)

                # Use surrogate adapter (will mock if model not found)
                surrogate_adapter = EnsembleSurrogateAdapter()

                # Use simple solver for robustness
                solver_adapter = SimpleSolverAdapter(step_scale=0.05, n_evals=10)

                # Use physics simulation (1D or 0D)
                # Use 0D for speed in demo, set use_full_physics=True for real 1D
                sim_adapter = PhysicsSimulationAdapter(use_full_physics=False)

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
        """Stop running optimization (placeholder)."""
        emit_event(EventType.RUN_END, "ORCH", metadata={"stopped": True})
        return jsonify({"status": "stopped"})

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
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument(
        "--generate-static",
        type=Path,
        help="Generate static JS file instead of running server",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Add WebSocket log handler helper
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
    app.run(host=args.host, port=args.port, debug=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
