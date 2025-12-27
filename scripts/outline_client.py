#!/usr/bin/env python3
"""Outline client example for IDE integration.

This script demonstrates how to query the outline API and display
hierarchical code structure. It can be adapted for VS Code extensions,
CLI tools, or other IDE integrations.

Usage:
    # Start the dashboard API first
    python dashboard/api.py --port 5001

    # Then query a file outline
    python scripts/outline_client.py campro/optimization/orchestrator.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)


def get_file_outline(file_path: str, api_url: str = "http://localhost:5001") -> dict:
    """Fetch outline for a file from the API.

    Args:
        file_path: Relative path to the file
        api_url: Base URL of the API server

    Returns:
        JSON response with symbols hierarchy
    """
    url = f"{api_url}/api/outline/{file_path}"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # File not indexed yet, try to trigger indexing
            print(f"File not indexed yet. Triggering re-index...")
            refresh_response = requests.post(
                f"{api_url}/api/outline/refresh",
                json={"file": file_path},
                timeout=10,
            )
            if refresh_response.status_code == 200:
                print("Re-indexing queued. Please retry in a moment.")
            return {"error": "File not indexed", "retry": True}
        else:
            return {"error": response.text}

    except requests.exceptions.ConnectionError:
        return {"error": f"Could not connect to API at {api_url}. Is the dashboard API running?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


def print_symbol(symbol: dict, indent: int = 0) -> None:
    """Print a symbol with proper indentation.

    Args:
        symbol: Symbol dict with name, kind, line_start, etc.
        indent: Indentation level
    """
    prefix = "  " * indent

    # Build display string
    kind_icon = {
        "class": "🏛️",
        "function": "ƒ",
        "method": "⚙️",
        "variable": "📌",
    }.get(symbol["kind"], "•")

    async_marker = " async" if symbol.get("is_async") else ""
    decorators = symbol.get("decorators", [])
    decorator_str = f" @{'@'.join(decorators)}" if decorators else ""

    print(
        f"{prefix}{kind_icon} {symbol['name']}{async_marker}{decorator_str} "
        f"[L{symbol['line_start']}-{symbol['line_end']}]"
    )

    # Print signature if available and not too long
    signature = symbol.get("signature", "")
    if signature and len(signature) < 80:
        print(f"{prefix}   {signature}")

    # Print children recursively
    for child in symbol.get("children", []):
        print_symbol(child, indent + 1)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Query code outline from Weaviate")
    parser.add_argument("file_path", help="Relative path to the Python file")
    parser.add_argument("--api-url", default="http://localhost:5001", help="API server URL")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    # Normalize path
    file_path = Path(args.file_path)
    if file_path.is_absolute():
        # Try to make it relative to current directory
        try:
            file_path = file_path.relative_to(Path.cwd())
        except ValueError:
            print(f"Warning: Using absolute path: {file_path}")

    file_path_str = str(file_path).replace("\\", "/")  # Normalize for URL

    print(f"Querying outline for: {file_path_str}")
    print(f"API: {args.api_url}")
    print("-" * 60)

    result = get_file_outline(file_path_str, args.api_url)

    if "error" in result:
        print(f"Error: {result['error']}")
        if result.get("retry"):
            return 2  # Retry suggested
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    # Pretty print the outline
    symbols = result.get("symbols", [])
    if not symbols:
        print("No symbols found in file.")
        return 0

    print(f"File: {result['file']}")
    print(f"Symbols: {result['count']}")
    print()

    for symbol in symbols:
        print_symbol(symbol)

    return 0


if __name__ == "__main__":
    sys.exit(main())
