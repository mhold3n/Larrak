"""Module-to-tool linker for Weaviate.

Links discovered tools to their parent orchestrator modules,
enabling the dashboard to dynamically display tools per module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tool_scanner import KNOWN_TOOLS, ToolScanner

# Conditional Weaviate import
try:
    import weaviate

    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None  # type: ignore[assignment]


# Module definitions: orchestrator module ID → source files to scan
MODULE_DEFINITIONS = {
    "CEM": {
        "name": "CEM: Feasibility Generation",
        "entrypoint": "truthmaker/cem/client.py",
        "scan_paths": ["truthmaker/cem/"],
    },
    "SUR": {
        "name": "Surrogate: Inference Model",
        "entrypoint": "truthmaker/surrogates/inference/gated.py",
        "scan_paths": ["truthmaker/surrogates/"],
    },
    "SOL": {
        "name": "Solver: Local Refinement",
        "entrypoint": "campro/optimization/driver.py",
        "scan_paths": ["campro/optimization/", "campro/physics/casadi/"],
    },
    "ORCH": {
        "name": "Orchestrator",
        "entrypoint": "campro/orchestration/orchestrator.py",
        "scan_paths": ["campro/orchestration/"],
    },
    "BM": {
        "name": "Budget Manager",
        "entrypoint": "campro/orchestration/budget.py",
        "scan_paths": ["campro/orchestration/budget.py"],
    },
    "TR": {
        "name": "Trust Region",
        "entrypoint": "campro/orchestration/trust_region.py",
        "scan_paths": ["campro/orchestration/trust_region.py"],
    },
    "CACHE": {
        "name": "Eval Cache",
        "entrypoint": "campro/orchestration/cache.py",
        "scan_paths": ["campro/orchestration/cache.py"],
    },
    "SEL": {
        "name": "Candidate Selector",
        "entrypoint": "campro/orchestration/selector.py",
        "scan_paths": ["campro/orchestration/selector.py"],
    },
    "FOAM": {
        "name": "laplacianFoam (Thermal CFD)",
        "entrypoint": "Simulations/hifi/thermal_cfd.py",
        "scan_paths": ["Simulations/hifi/thermal_cfd.py"],
    },
    "CCX": {
        "name": "CalculiX (Structural FEA)",
        "entrypoint": "Simulations/hifi/gear_contact.py",
        "scan_paths": ["Simulations/hifi/gear_contact.py"],
    },
    "PORTFLOW": {
        "name": "simpleFoam (Port Flow CFD)",
        "entrypoint": "Simulations/hifi/port_flow_cfd.py",
        "scan_paths": ["Simulations/hifi/port_flow_cfd.py"],
    },
    "COMBUST": {
        "name": "reactingFoam (Combustion CFD)",
        "entrypoint": "Simulations/hifi/combustion_cfd.py",
        "scan_paths": ["Simulations/hifi/combustion_cfd.py"],
    },
    "TRUTH": {
        "name": "Truth DB",
        "entrypoint": "Simulations/hifi/result_parsers.py",
        "scan_paths": ["Simulations/hifi/"],
    },
    "WEAV": {
        "name": "Weaviate DB",
        "entrypoint": "provenance/db.py",
        "scan_paths": ["provenance/"],
    },
    "PROV": {
        "name": "Provenance Client",
        "entrypoint": "campro/orchestration/provenance.py",
        "scan_paths": ["campro/orchestration/provenance.py"],
    },
}


@dataclass
class ModuleToolLink:
    """Association between a module and its detected tools."""

    module_id: str
    module_name: str
    tool_ids: list[str]


class ModuleLinker:
    """Link modules to their detected tools."""

    def __init__(self, client: Any | None = None, project_root: Path | None = None):
        """Initialize linker.

        Args:
            client: Optional Weaviate client
            project_root: Root directory of the project
        """
        self.client = client
        self.project_root = project_root or Path.cwd()
        self.scanner = ToolScanner(client)

    def scan_module(self, module_id: str) -> ModuleToolLink:
        """Scan a module's source files for tool usage.

        Args:
            module_id: Module identifier (e.g., "SOL", "CEM")

        Returns:
            Module-tool link with detected tools
        """
        if module_id not in MODULE_DEFINITIONS:
            return ModuleToolLink(module_id=module_id, module_name="Unknown", tool_ids=[])

        module_def = MODULE_DEFINITIONS[module_id]
        detected_tools: set[str] = set()

        for scan_path in module_def["scan_paths"]:
            full_path = self.project_root / scan_path

            if full_path.is_file():
                matches = self.scanner.scan_file(full_path)
                for m in matches:
                    detected_tools.add(m.tool_id)
            elif full_path.is_dir():
                result = self.scanner.scan_directory(full_path)
                detected_tools.update(result.tools_found.keys())

        return ModuleToolLink(
            module_id=module_id,
            module_name=module_def["name"],
            tool_ids=sorted(detected_tools),
        )

    def scan_all_modules(self) -> dict[str, ModuleToolLink]:
        """Scan all defined modules for tool usage.

        Returns:
            Mapping of module_id to ModuleToolLink
        """
        results: dict[str, ModuleToolLink] = {}

        for module_id in MODULE_DEFINITIONS:
            results[module_id] = self.scan_module(module_id)

        return results

    def link_module_in_weaviate(self, module_id: str) -> bool:
        """Create/update module-tool links in Weaviate.

        Args:
            module_id: Module to link

        Returns:
            True if successful
        """
        if not self.client or not WEAVIATE_AVAILABLE:
            return False

        # Ensure tools exist
        tool_uuids = self.scanner.ensure_tools_in_weaviate()

        # Scan module
        link = self.scan_module(module_id)

        # Get or create module
        module_collection = self.client.collections.get("Module")
        module_uuid = hashlib.md5(module_id.encode()).hexdigest()

        try:
            existing = module_collection.query.fetch_object_by_id(module_uuid)
            if not existing:
                # Create module
                module_def = MODULE_DEFINITIONS.get(module_id, {})
                module_collection.data.insert(
                    properties={
                        "module_id": module_id,
                        "entrypoint": module_def.get("entrypoint", ""),
                        "description": module_def.get("name", module_id),
                    },
                    uuid=module_uuid,
                )
        except Exception:
            pass

        # Link tools to module
        tool_refs = [tool_uuids[tid] for tid in link.tool_ids if tid in tool_uuids]
        if tool_refs:
            try:
                module_collection.data.reference_add(
                    from_uuid=module_uuid,
                    from_property="uses_tools",
                    to=tool_refs,
                )
            except Exception:
                pass

        return True

    def get_module_tools_json(self) -> dict[str, list[dict]]:
        """Get all module tools as JSON for dashboard.

        Returns:
            Dictionary mapping module_id to list of tool info dicts
        """
        results: dict[str, list[dict]] = {}

        for module_id, link in self.scan_all_modules().items():
            tools = []
            for tool_id in link.tool_ids:
                if tool_id in KNOWN_TOOLS:
                    tool_info = KNOWN_TOOLS[tool_id]
                    tools.append(
                        {
                            "id": tool_id,
                            "name": tool_info["name"],
                            "category": tool_info["category"],
                            "version": tool_info["version"],
                        }
                    )
            results[module_id] = tools

        return results


def main() -> int:
    """CLI entry point."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Link modules to their tools")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--ingest", action="store_true", help="Ingest to Weaviate")
    args = parser.parse_args()

    client = None
    if args.ingest and WEAVIATE_AVAILABLE:
        try:
            client = weaviate.connect_to_local(port=8080, grpc_port=50052)
        except Exception as e:
            print(f"Warning: Could not connect to Weaviate: {e}")

    linker = ModuleLinker(client=client, project_root=args.project_root)

    if args.json:
        print(json.dumps(linker.get_module_tools_json(), indent=2))
    else:
        for module_id, link in linker.scan_all_modules().items():
            tools_str = ", ".join(KNOWN_TOOLS[t]["name"] for t in link.tool_ids if t in KNOWN_TOOLS)
            print(f"{module_id:12} → {tools_str or '(none)'}")

    if args.ingest and client:
        print("\nIngesting module-tool links to Weaviate...")
        for module_id in MODULE_DEFINITIONS:
            linker.link_module_in_weaviate(module_id)
        print("  Done.")
        client.close()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
