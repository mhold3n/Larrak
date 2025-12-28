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
    decorators: list[str] | None = None,
    is_async: bool = False,
    return_type: str | None = None,
    parameters: str | None = None,
    parent_uuid: str | None = None,
) -> str:
    """Upsert a CodeSymbol object, return its UUID."""
    import json
    import uuid as uuid_lib

    symbols = client.collections.get("CodeSymbol")

    # Generate deterministic UUID from file_path + name + line_start
    symbol_id = f"{file_path}::{name}::{line_start}"
    deterministic_uuid = str(uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, symbol_id))

    properties = {
        "name": name,
        "kind": kind,
        "file_path": file_path,
        "line_number": line_start,
        "line_end": line_end,
        "signature": signature,
        "docstring": docstring[:1000] if docstring else "",
        "code_content": "",
        "decorators": decorators or [],
        "is_async": is_async,
        "return_type": return_type or "",
        "parameters": parameters or "",
    }

    references = {"defined_in_file": file_uuid}
    if parent_uuid:
        references["parent_symbol"] = parent_uuid

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
    """Parse a Python file and extract function/class definitions with outline metadata."""
    import json

    symbols = []

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  Warning: Could not parse {file_path}: {e}")
        return symbols

    # Use ast.iter_child_nodes for hierarchy instead of ast.walk
    def extract_from_body(nodes, parent_name=None):
        for node in nodes:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                # Extract decorators
                decorators = [ast.unparse(d) for d in node.decorator_list]

                # Build parameter JSON
                params = []
                for arg in node.args.args:
                    param = {"name": arg.arg}
                    if arg.annotation:
                        param["type"] = ast.unparse(arg.annotation)
                    params.append(param)

                # Build signature
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    args.append(arg_str)
                signature = f"def {node.name}({', '.join(args)})"

                return_type = None
                if node.returns:
                    return_type = ast.unparse(node.returns)
                    signature += f" -> {return_type}"

                full_name = f"{parent_name}.{node.name}" if parent_name else node.name
                kind = "method" if parent_name else "function"

                symbols.append(
                    {
                        "name": full_name,
                        "kind": kind,
                        "line_start": node.lineno,
                        "line_end": node.end_lineno or node.lineno,
                        "signature": signature,
                        "docstring": ast.get_docstring(node) or "",
                        "decorators": decorators,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "return_type": return_type,
                        "parameters": json.dumps(params),
                        "parent_name": parent_name,
                    }
                )

            elif isinstance(node, ast.ClassDef):
                # Extract decorators
                decorators = [ast.unparse(d) for d in node.decorator_list]

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
                        "decorators": decorators,
                        "is_async": False,
                        "return_type": None,
                        "parameters": "",
                        "parent_name": None,
                    }
                )

                # Extract methods recursively with parent reference
                extract_from_body(node.body, parent_name=node.name)

    extract_from_body(tree.body)
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

            # First pass: create all symbols and build UUID map
            symbol_uuids = {}
            for sym in symbols:
                sym_uuid = upsert_symbol(
                    client,
                    sym["name"],
                    sym["kind"],
                    relative_path,
                    sym["line_start"],
                    sym["line_end"],
                    sym["signature"],
                    sym["docstring"],
                    file_uuid,
                    decorators=sym.get("decorators"),
                    is_async=sym.get("is_async", False),
                    return_type=sym.get("return_type"),
                    parameters=sym.get("parameters"),
                )
                symbol_uuids[sym["name"]] = sym_uuid
                stats["symbols"] += 1

            # Second pass: update parent references for methods
            for sym in symbols:
                if sym.get("parent_name") and sym["parent_name"] in symbol_uuids:
                    parent_uuid = symbol_uuids[sym["parent_name"]]
                    child_uuid = symbol_uuids[sym["name"]]
                    # Update the child with parent reference
                    symbols_collection = client.collections.get("CodeSymbol")
                    try:
                        symbols_collection.data.reference_add(
                            from_uuid=child_uuid,
                            from_property="parent_symbol",
                            to=parent_uuid,
                        )
                    except Exception:
                        pass  # Reference may already exist

            if symbols:
                print(f"  {relative_path}: {len(symbols)} symbols")

    return stats


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Scan Python code and index symbols into Weaviate")
    parser.add_argument("--file", type=str, help="Index a single file instead of entire repository")
    args = parser.parse_args()

    repo_path = Path(os.environ.get("REPO_PATH", ".")).resolve()
    repo_name = os.environ.get("REPO_NAME", "Larrak")
    repo_owner = os.environ.get("REPO_OWNER", "mhold3n")
    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key = os.environ.get("WEAVIATE_API_KEY", "")

    # Single file mode
    if args.file:
        file_path = Path(args.file).resolve()
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return 1

        if not str(file_path).endswith(".py"):
            print(f"Error: Only Python files: {file_path}")
            return 1

        print(f"Indexing single file: {file_path}")
    else:
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
        if args.file:
            # Single-file mode
            file_path = Path(args.file).resolve()
            try:
                relative_path = str(file_path.relative_to(repo_path))
            except ValueError:
                print(f"Error: File must be in repository: {file_path}")
                return 1

            # Get/create repo
            repo_uuid = get_or_create_repo(client, repo_name, repo_owner)

            # Index single file
            size_bytes = file_path.stat().st_size
            file_uuid = upsert_file(client, relative_path, repo_uuid, "python", size_bytes)
            symbols = extract_symbols_from_file(file_path)

            # Upsert symbols with two-pass for parent refs
            symbol_uuids = {}
            for sym in symbols:
                sym_uuid = upsert_symbol(
                    client,
                    sym["name"],
                    sym["kind"],
                    relative_path,
                    sym["line_start"],
                    sym["line_end"],
                    sym["signature"],
                    sym["docstring"],
                    file_uuid,
                    decorators=sym.get("decorators"),
                    is_async=sym.get("is_async", False),
                    return_type=sym.get("return_type"),
                    parameters=sym.get("parameters"),
                )
                symbol_uuids[sym["name"]] = sym_uuid

            # Second pass: parent references
            for sym in symbols:
                if sym.get("parent_name") and sym["parent_name"] in symbol_uuids:
                    parent_uuid = symbol_uuids[sym["parent_name"]]
                    child_uuid = symbol_uuids[sym["name"]]
                    symbols_collection = client.collections.get("CodeSymbol")
                    try:
                        symbols_collection.data.reference_add(
                            from_uuid=child_uuid,
                            from_property="parent_symbol",
                            to=parent_uuid,
                        )
                    except Exception:
                        pass

            print(f"✅ Indexed {len(symbols)} symbols from {relative_path}")
            return 0
        else:
            # Full repository scan
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
