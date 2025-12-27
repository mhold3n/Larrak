"""Dashboard API for tool and module queries.

Simple Flask API serving tool information to the orchestrator dashboard.
Can also run standalone for development/testing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from provenance.module_linker import MODULE_DEFINITIONS, ModuleLinker
from provenance.tool_scanner import KNOWN_TOOLS

try:
    from flask import Flask, jsonify, send_from_directory
    from flask_cors import CORS

    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

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
    def index():
        """Serve dashboard HTML."""
        return send_from_directory(app.static_folder, "orchestrator_dashboard.html")

    @app.route("/api/modules")
    def list_modules():
        """List all modules with their tools."""
        tools = get_module_tools(project_root)
        return jsonify(tools)

    @app.route("/api/modules/<module_id>/tools")
    def get_tools(module_id: str):
        """Get tools for a specific module."""
        tools = get_module_tools(project_root)
        if module_id.upper() not in tools:
            return jsonify({"error": f"Module {module_id} not found"}), 404
        return jsonify(tools[module_id.upper()])

    @app.route("/api/tools")
    def list_all_tools():
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
    def get_tool(tool_id: str):
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
    def list_dataflows():
        """List all data flow connections between modules."""
        from provenance.dataflow_scanner import scan_dashboard

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
    def list_optimization_steps():
        """List recent optimization steps (if Weaviate connected)."""
        try:
            import weaviate

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
    def cache_stats():
        """Get evaluation cache statistics."""
        try:
            import weaviate

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
    def budget_snapshots():
        """Get budget allocation history."""
        try:
            import weaviate

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
    def trustregion_logs():
        """Get trust region adjustment history."""
        try:
            import weaviate

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
    parser.add_argument("--port", type=int, default=5000, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument(
        "--generate-static",
        type=Path,
        help="Generate static JS file instead of running server",
    )
    args = parser.parse_args()

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
