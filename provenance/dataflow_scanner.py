"""DataFlow scanner to extract trace definitions from dashboard HTML.

Parses the Mermaid diagram in orchestrator_dashboard.html and creates
DataFlow entries in Weaviate for each arrow/connection.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Conditional Weaviate import
try:
    import weaviate

    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None  # type: ignore[assignment]


@dataclass
class DataFlowEntry:
    """A parsed DataFlow from the Mermaid diagram."""

    flow_id: str
    source_module: str
    target_module: str
    label: str
    category: str
    color: str
    line_style: str  # "solid" or "dashed"


# Category color mapping from dashboard
CATEGORY_COLORS = {
    "#2e7d32": "optimization",  # green
    "#0097a7": "cache",  # cyan
    "#e65100": "hifi_dispatch",  # orange
    "#d84315": "hifi_store",  # dark orange
    "#7b1fa2": "data_flow",  # purple
    "#c62828": "provenance",  # red
}


def parse_mermaid_flows(html_content: str) -> list[DataFlowEntry]:
    """Parse DataFlow entries from Mermaid diagram in HTML.

    Args:
        html_content: HTML content containing Mermaid diagram

    Returns:
        List of DataFlowEntry objects
    """
    flows: list[DataFlowEntry] = []

    # Pattern for solid arrows: A ==>|label| B or A --> B
    solid_pattern = re.compile(r"(\w+)\s*(?:==>|-->)\s*\|([^|]+)\|\s*(\w+)", re.MULTILINE)
    # Pattern for dashed arrows: A -.->|label| B or A -.-|label| B
    dashed_pattern = re.compile(
        r"(\w+)\s*(?:-\.->|-\.->\||-\.-)\s*\|?([^|]*)\|?\s*(\w+)", re.MULTILINE
    )

    # Parse solid arrows
    for match in solid_pattern.finditer(html_content):
        source, label, target = match.groups()
        flow_id = f"{source.lower()}_to_{target.lower()}"
        flows.append(
            DataFlowEntry(
                flow_id=flow_id,
                source_module=source,
                target_module=target,
                label=label.strip(),
                category="optimization",  # Default, will be updated by linkStyle
                color="#2e7d32",
                line_style="solid",
            )
        )

    # Parse dashed arrows
    for match in dashed_pattern.finditer(html_content):
        source, label, target = match.groups()
        flow_id = f"{source.lower()}_to_{target.lower()}"
        # Skip if already exists (avoid duplicates)
        if any(f.flow_id == flow_id for f in flows):
            continue
        flows.append(
            DataFlowEntry(
                flow_id=flow_id,
                source_module=source,
                target_module=target,
                label=label.strip() if label else "",
                category="utility",
                color="#666666",
                line_style="dashed",
            )
        )

    # Parse linkStyle directives to update colors
    linkstyle_pattern = re.compile(r"linkStyle\s+(\d+)\s+stroke:([^,;]+)", re.MULTILINE)
    for match in linkstyle_pattern.finditer(html_content):
        index = int(match.group(1))
        color = match.group(2).strip()
        if index < len(flows):
            flows[index].color = color
            flows[index].category = CATEGORY_COLORS.get(color, "unknown")

    return flows


def scan_dashboard(dashboard_path: Path) -> list[DataFlowEntry]:
    """Scan dashboard HTML for DataFlow entries.

    Args:
        dashboard_path: Path to orchestrator_dashboard.html

    Returns:
        List of DataFlowEntry objects
    """
    content = dashboard_path.read_text(encoding="utf-8")
    return parse_mermaid_flows(content)


class DataFlowIngester:
    """Ingest DataFlow entries to Weaviate."""

    def __init__(self, client: Any | None = None):
        self.client = client

    def ingest_flows(self, flows: list[DataFlowEntry]) -> int:
        """Ingest DataFlow entries to Weaviate.

        Args:
            flows: List of DataFlowEntry objects

        Returns:
            Number of entries ingested
        """
        if not self.client or not WEAVIATE_AVAILABLE:
            return 0

        dataflow_collection = self.client.collections.get("DataFlow")
        count = 0

        for flow in flows:
            uuid_str = hashlib.md5(flow.flow_id.encode()).hexdigest()
            try:
                dataflow_collection.data.insert(
                    properties={
                        "flow_id": flow.flow_id,
                        "source_module": flow.source_module,
                        "target_module": flow.target_module,
                        "label": flow.label,
                        "category": flow.category,
                        "color": flow.color,
                        "line_style": flow.line_style,
                    },
                    uuid=uuid_str,
                )
                count += 1
            except Exception:
                # Already exists or error
                pass

        return count


def main() -> int:
    """CLI entry point."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Scan dashboard for DataFlow entries")
    parser.add_argument(
        "--dashboard",
        type=Path,
        default=Path(__file__).parent.parent / "dashboard" / "orchestrator_dashboard.html",
        help="Path to dashboard HTML",
    )
    parser.add_argument("--ingest", action="store_true", help="Ingest to Weaviate")
    args = parser.parse_args()

    if not args.dashboard.exists():
        print(f"Error: Dashboard not found at {args.dashboard}")
        return 1

    flows = scan_dashboard(args.dashboard)
    print(f"Found {len(flows)} DataFlow entries:")
    for f in flows:
        print(f"  {f.source_module} → {f.target_module}: {f.label} [{f.category}]")

    if args.ingest:
        client = None
        if WEAVIATE_AVAILABLE:
            try:
                client = weaviate.connect_to_local(port=8080, grpc_port=50052)
            except Exception as e:
                print(f"Warning: Could not connect to Weaviate: {e}")

        if client:
            ingester = DataFlowIngester(client)
            count = ingester.ingest_flows(flows)
            print(f"\nIngested {count} DataFlow entries to Weaviate")
            client.close()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
