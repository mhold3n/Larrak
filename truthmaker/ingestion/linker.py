"""Linker for creating Issue ↔ Symbol edges in Weaviate.

Reads the `linked_symbols_field` from GitHub Issues (synced from Projects custom fields)
and creates `Issue.relatesTo -> Symbol` cross-references.

The linked_symbols_field is expected to contain symbol identifiers in one of these formats:
- "module.ClassName.method_name" (fully qualified)
- "function_name" (will match any symbol with that name)
- "path/to/file.py::ClassName" (file-scoped)

Usage:
    python truthmaker/ingestion/linker.py

Environment Variables:
    WEAVIATE_URL: Weaviate endpoint (default: http://localhost:8080)
"""

from __future__ import annotations

import re
import sys
from typing import Any

import weaviate
import weaviate.classes.query as wvq


def parse_symbol_reference(ref: str) -> dict[str, Any]:
    """Parse a symbol reference string into search criteria.

    Supports formats:
    - "ClassName.method_name" -> search by name containing "ClassName.method_name"
    - "path/to/file.py::ClassName" -> search by file_path and name
    - "function_name" -> search by exact name
    """
    # Format: file.py::SymbolName
    if "::" in ref:
        file_path, symbol_name = ref.split("::", 1)
        return {"file_path": file_path.strip(), "name": symbol_name.strip()}

    # Otherwise just search by name
    return {"name": ref.strip()}


def find_symbols_by_criteria(
    client: weaviate.WeaviateClient,
    criteria: dict[str, Any],
) -> list[str]:
    """Find symbol UUIDs matching the given criteria."""
    symbols = client.collections.get("CodeSymbol")

    # Build filter based on criteria
    if "file_path" in criteria and "name" in criteria:
        # Exact match on both
        results = symbols.query.fetch_objects(
            filters=(
                wvq.Filter.by_property("file_path").equal(criteria["file_path"])
                & wvq.Filter.by_property("name").equal(criteria["name"])
            ),
            limit=10,
        )
    else:
        # Just match by name
        results = symbols.query.fetch_objects(
            filters=wvq.Filter.by_property("name").equal(criteria["name"]),
            limit=10,
        )

    return [str(obj.uuid) for obj in results.objects]


def link_issue_to_symbols(
    client: weaviate.WeaviateClient,
    issue_uuid: str,
    symbol_uuids: list[str],
) -> None:
    """Create relates_to references from an Issue to Symbols."""
    issues = client.collections.get("GitHubIssue")

    if not symbol_uuids:
        return

    # Update the issue with the symbol references
    issues.data.update(
        uuid=issue_uuid,
        references={"relates_to": symbol_uuids},
    )


def parse_linked_symbols_field(field_value: str) -> list[str]:
    """Parse the linked_symbols custom field value into symbol references.

    Expected formats:
    - Comma-separated: "func1, ClassName.method, path/file.py::Symbol"
    - Newline-separated
    - Semicolon-separated
    """
    if not field_value:
        return []

    # Split by common delimiters
    refs = re.split(r"[,;\n]+", field_value)
    return [r.strip() for r in refs if r.strip()]


def process_all_issues(client: weaviate.WeaviateClient) -> dict[str, int]:
    """Process all issues and create symbol links based on linked_symbols_field."""
    issues = client.collections.get("GitHubIssue")
    stats = {"processed": 0, "linked": 0, "symbols_found": 0}

    # Fetch all issues with their linked_symbols_field
    results = issues.query.fetch_objects(
        limit=1000,
        return_properties=["node_id", "title", "linked_symbols_field"],
    )

    for issue in results.objects:
        stats["processed"] += 1
        linked_field = issue.properties.get("linked_symbols_field", "")

        if not linked_field:
            continue

        # Parse the field into symbol references
        refs = parse_linked_symbols_field(linked_field)
        if not refs:
            continue

        # Find matching symbols
        all_symbol_uuids = []
        for ref in refs:
            criteria = parse_symbol_reference(ref)
            symbol_uuids = find_symbols_by_criteria(client, criteria)
            all_symbol_uuids.extend(symbol_uuids)
            stats["symbols_found"] += len(symbol_uuids)

        if all_symbol_uuids:
            # Create the links
            link_issue_to_symbols(client, str(issue.uuid), all_symbol_uuids)
            stats["linked"] += 1
            print(
                f"  Linked Issue '{issue.properties.get('title')}' "
                f"to {len(all_symbol_uuids)} symbols"
            )

    return stats


def suggest_links_from_body(client: weaviate.WeaviateClient) -> list[dict]:
    """Suggest Issue → Symbol links based on symbol names mentioned in issue body.

    This is a heuristic approach:
    1. Fetch all issues
    2. Search issue body for patterns that look like code references
    3. Match against known symbols
    4. Return suggestions (not automatically applied)
    """
    suggestions = []
    issues = client.collections.get("GitHubIssue")
    symbols = client.collections.get("CodeSymbol")

    # Get all symbol names for matching
    symbol_results = symbols.query.fetch_objects(
        limit=5000,
        return_properties=["name", "file_path"],
    )
    symbol_names = {
        sym.properties["name"]: {
            "uuid": str(sym.uuid),
            "file_path": sym.properties.get("file_path", ""),
        }
        for sym in symbol_results.objects
    }

    # Fetch issues
    issue_results = issues.query.fetch_objects(
        limit=1000,
        return_properties=["node_id", "title", "body"],
    )

    # Patterns that look like code references
    code_patterns = [
        r"`([A-Z][a-zA-Z0-9_]+)`",  # `ClassName`
        r"`([a-z_][a-z0-9_]+)`",  # `function_name`
        r"def\s+([a-z_][a-z0-9_]+)",  # def function_name
        r"class\s+([A-Z][a-zA-Z0-9_]+)",  # class ClassName
    ]

    for issue in issue_results.objects:
        body = issue.properties.get("body", "") or ""
        title = issue.properties.get("title", "")
        found_symbols = set()

        for pattern in code_patterns:
            matches = re.findall(pattern, body)
            for match in matches:
                if match in symbol_names:
                    found_symbols.add(match)

        if found_symbols:
            suggestions.append(
                {
                    "issue_uuid": str(issue.uuid),
                    "issue_title": title,
                    "suggested_symbols": [
                        {
                            "name": name,
                            "uuid": symbol_names[name]["uuid"],
                            "file_path": symbol_names[name]["file_path"],
                        }
                        for name in found_symbols
                    ],
                }
            )

    return suggestions


def main() -> int:
    """Main entry point."""
    import os

    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY", "")

    print(f"Connecting to Weaviate at {weaviate_url}...")

    # Detect if using Weaviate Cloud (WCD)
    if "weaviate.cloud" in weaviate_url or "wcs.api.weaviate.io" in weaviate_url:
        import weaviate.classes.init as wvi

        cluster_url = weaviate_url
        if not cluster_url.startswith("http"):
            cluster_url = f"https://{cluster_url}"

        if weaviate_api_key:
            auth = wvi.Auth.api_key(weaviate_api_key)
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=auth,
            )
        else:
            client = weaviate.connect_to_weaviate_cloud(cluster_url=cluster_url)
        print("Connected to Weaviate Cloud")
    else:
        client = weaviate.connect_to_local(port=8080, grpc_port=50052)
        print("Connected to local Weaviate")

    try:
        print("\n=== Processing Confirmed Links ===")
        stats = process_all_issues(client)
        print("\nResults:")
        print(f"  Issues processed: {stats['processed']}")
        print(f"  Issues linked: {stats['linked']}")
        print(f"  Symbols found: {stats['symbols_found']}")

        print("\n=== Generating Link Suggestions ===")
        suggestions = suggest_links_from_body(client)
        if suggestions:
            print(f"\nFound {len(suggestions)} issues with potential symbol matches:")
            for s in suggestions[:10]:  # Show first 10
                print(f"  Issue: {s['issue_title']}")
                for sym in s["suggested_symbols"][:3]:  # Show first 3 symbols
                    print(f"    -> {sym['name']} ({sym['file_path']})")
        else:
            print("No suggestions found.")

        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
