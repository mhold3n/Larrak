"""Code scanner for extracting symbols from Python files.

Walks the repository, parses Python files using AST, and upserts
File and Symbol information into Weaviate.

Usage:
    WEAVIATE_URL=http://localhost:8080 python truthmaker/ingestion/code_scanner.py

Environment Variables:
    WEAVIATE_URL: Weaviate endpoint (default: http://localhost:8080)
    REPO_PATH: Repository root (default: current directory)
    REPO_NAME: Repository name (default: Larrak)
    REPO_OWNER: Repository owner (default: mhold3n)
"""

from __future__ import annotations

import ast
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import weaviate
import weaviate.classes.query as wvq

# Directories to skip
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "PERMANENT_ARCHIVE",
    "obj",
    "bin",
}

# File extensions to index
PYTHON_EXTENSIONS = {".py"}


def get_or_create_repo(
    client: weaviate.WeaviateClient,
    repo_name: str,
    repo_owner: str,
) -> str:
    """Get or create a Repository object, return its UUID."""
    repos = client.collections.get("Repository")

    node_id = f"{repo_owner}/{repo_name}"

    result = repos.query.fetch_objects(
        filters=wvq.Filter.by_property("node_id").equal(node_id),
        limit=1,
    )

    properties = {
        "node_id": node_id,
        "name": repo_name,
        "owner": repo_owner,
        "url": f"https://github.com/{repo_owner}/{repo_name}",
        "default_branch": "main",
        "last_indexed": datetime.now(timezone.utc),
    }

    if result.objects:
        uuid = result.objects[0].uuid
        repos.data.update(uuid=uuid, properties=properties)
        return str(uuid)
    else:
        uuid = repos.data.insert(properties=properties)
        return str(uuid)


def upsert_file(
    client: weaviate.WeaviateClient,
    file_path: str,
    repo_uuid: str,
    language: str = "python",
    size_bytes: int = 0,
) -> str:
    """Upsert a CodeFile object, return its UUID."""
    files = client.collections.get("CodeFile")

    result = files.query.fetch_objects(
        filters=wvq.Filter.by_property("path").equal(file_path),
        limit=1,
    )

    properties = {
        "path": file_path,
        "language": language,
        "size_bytes": size_bytes,
        "last_indexed": datetime.now(timezone.utc),
    }

    references = {"in_repo": repo_uuid}

    if result.objects:
        uuid = result.objects[0].uuid
        files.data.update(uuid=uuid, properties=properties, references=references)
        return str(uuid)
    else:
        uuid = files.data.insert(properties=properties, references=references)
        return str(uuid)


def upsert_symbol(
    client: weaviate.WeaviateClient,
    name: str,
    kind: str,
    file_path: str,
    line_start: int,
    line_end: int,
    signature: str,
    docstring: str,
    file_uuid: str,
) -> str:
    """Upsert a CodeSymbol object, return its UUID."""
    import uuid as uuid_lib

    symbols = client.collections.get("CodeSymbol")

    # Generate deterministic UUID from file_path + name + line_start
    symbol_id = f"{file_path}::{name}::{line_start}"
    deterministic_uuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, symbol_id))

    properties = {
        "name": name,
        "file_path": file_path,
        "line_number": line_start,
        "signature": signature,
        "docstring": docstring[:1000] if docstring else "",  # Truncate long docstrings
        "code_content": "",  # Not extracting full code for now
    }

    references = {"defined_in_file": file_uuid}

    try:
        # Try to get existing object by deterministic UUID
        existing = symbols.query.fetch_object_by_id(uuid=deterministic_uuid)
        if existing:
            symbols.data.update(
                uuid=deterministic_uuid, properties=properties, references=references
            )
            return deterministic_uuid
    except Exception:
        pass  # Object doesn't exist, create it

    # Insert with deterministic UUID
    symbols.data.insert(properties=properties, references=references, uuid=deterministic_uuid)
    return deterministic_uuid


def extract_symbols_from_file(file_path: Path) -> list[dict]:
    """Parse a Python file and extract function/class definitions."""
    symbols = []

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  Warning: Could not parse {file_path}: {e}")
        return symbols

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # Build signature
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)
            signature = f"def {node.name}({', '.join(args)})"
            if node.returns:
                signature += f" -> {ast.unparse(node.returns)}"

            symbols.append(
                {
                    "name": node.name,
                    "kind": "function",
                    "line_start": node.lineno,
                    "line_end": node.end_lineno or node.lineno,
                    "signature": signature,
                    "docstring": ast.get_docstring(node) or "",
                }
            )

        elif isinstance(node, ast.ClassDef):
            # Build class signature with bases
            bases = [ast.unparse(b) for b in node.bases]
            signature = f"class {node.name}"
            if bases:
                signature += f"({', '.join(bases)})"

            symbols.append(
                {
                    "name": node.name,
                    "kind": "class",
                    "line_start": node.lineno,
                    "line_end": node.end_lineno or node.lineno,
                    "signature": signature,
                    "docstring": ast.get_docstring(node) or "",
                }
            )

            # Also extract methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    method_args = []
                    for arg in item.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        method_args.append(arg_str)
                    method_sig = f"def {node.name}.{item.name}({', '.join(method_args)})"
                    if item.returns:
                        method_sig += f" -> {ast.unparse(item.returns)}"

                    symbols.append(
                        {
                            "name": f"{node.name}.{item.name}",
                            "kind": "method",
                            "line_start": item.lineno,
                            "line_end": item.end_lineno or item.lineno,
                            "signature": method_sig,
                            "docstring": ast.get_docstring(item) or "",
                        }
                    )

    return symbols


def scan_repository(
    client: weaviate.WeaviateClient,
    repo_path: Path,
    repo_name: str,
    repo_owner: str,
) -> dict:
    """Scan the repository and index all Python files and symbols."""
    stats = {"files": 0, "symbols": 0}

    # Get or create repo
    repo_uuid = get_or_create_repo(client, repo_name, repo_owner)
    print(f"Repository UUID: {repo_uuid}")

    # Walk the directory
    for root, dirs, files in os.walk(repo_path):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            ext = Path(filename).suffix
            if ext not in PYTHON_EXTENSIONS:
                continue

            file_path = Path(root) / filename
            relative_path = str(file_path.relative_to(repo_path))

            # Get file size
            try:
                size_bytes = file_path.stat().st_size
            except OSError:
                size_bytes = 0

            # Upsert file
            file_uuid = upsert_file(client, relative_path, repo_uuid, "python", size_bytes)
            stats["files"] += 1

            # Extract and upsert symbols
            symbols = extract_symbols_from_file(file_path)
            for sym in symbols:
                upsert_symbol(
                    client,
                    sym["name"],
                    sym["kind"],
                    relative_path,
                    sym["line_start"],
                    sym["line_end"],
                    sym["signature"],
                    sym["docstring"],
                    file_uuid,
                )
                stats["symbols"] += 1

            if symbols:
                print(f"  {relative_path}: {len(symbols)} symbols")

    return stats


def main() -> int:
    """Main entry point."""
    repo_path = Path(os.environ.get("REPO_PATH", ".")).resolve()
    repo_name = os.environ.get("REPO_NAME", "Larrak")
    repo_owner = os.environ.get("REPO_OWNER", "mhold3n")
    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY", "")

    print(f"Scanning repository: {repo_path}")
    print(f"Repository: {repo_owner}/{repo_name}")

    print(f"Connecting to Weaviate at {weaviate_url}...")

    # Detect if using Weaviate Cloud (WCD)
    if "weaviate.cloud" in weaviate_url or "wcs.api.weaviate.io" in weaviate_url:
        # Weaviate Cloud connection
        import weaviate.classes.init as wvi

        # Extract cluster URL (add https:// if missing)
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
            # Try without auth (for read-only clusters)
            client = weaviate.connect_to_weaviate_cloud(cluster_url=cluster_url)
        print("Connected to Weaviate Cloud")
    else:
        # Local Weaviate connection
        client = weaviate.connect_to_local(port=8080, grpc_port=50052)
        print("Connected to local Weaviate")

    try:
        stats = scan_repository(client, repo_path, repo_name, repo_owner)
        print(f"\nIndexing complete:")
        print(f"  Files indexed: {stats['files']}")
        print(f"  Symbols indexed: {stats['symbols']}")
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
