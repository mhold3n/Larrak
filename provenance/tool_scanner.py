"""AST-based scanner for tool and dependency detection.

Scans Python files for imports and subprocess calls to detect
which external tools (CasADi, OpenFOAM, PyTorch, etc.) are used.
Results are indexed in Weaviate for dashboard display.
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Conditional Weaviate import
try:
    import weaviate

    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None  # type: ignore[assignment]


# Known tool definitions with detection patterns
KNOWN_TOOLS = {
    # Optimization
    "casadi": {
        "name": "CasADi",
        "category": "optimization",
        "patterns": [r"import casadi", r"from casadi"],
        "version": "3.6",
    },
    "ipopt": {
        "name": "IPOPT",
        "category": "optimization",
        "patterns": [r"nlpsol.*ipopt", r"ipopt\."],
        "version": "3.14",
    },
    "ma57": {
        "name": "MA57 / HSL",
        "category": "optimization",
        "patterns": [r"linear_solver.*ma57", r"ma57", r"hsllib"],
        "version": "2024.05",
    },
    # ML / Surrogates
    "pytorch": {
        "name": "PyTorch",
        "category": "ml",
        "patterns": [r"import torch", r"from torch"],
        "version": "2.0+",
    },
    "numpy": {
        "name": "NumPy",
        "category": "utility",
        "patterns": [r"import numpy", r"from numpy"],
        "version": "1.24+",
    },
    # CFD / FEA
    "openfoam": {
        "name": "OpenFOAM 11",
        "category": "cfd",
        "patterns": [r"laplacianFoam", r"simpleFoam", r"reactingFoam", r"openfoam"],
        "version": "11",
    },
    "calculix": {
        "name": "CalculiX",
        "category": "fea",
        "patterns": [r"ccx", r"calculix", r"CalculiX"],
        "version": "2.21",
    },
    "gmsh": {
        "name": "Gmsh",
        "category": "fea",
        "patterns": [r"gmsh", r"\.inp"],
        "version": "4.11",
    },
    # Infrastructure
    "weaviate": {
        "name": "Weaviate",
        "category": "database",
        "patterns": [r"import weaviate", r"from weaviate"],
        "version": "4.0+",
    },
    "docker": {
        "name": "Docker",
        "category": "infrastructure",
        "patterns": [r"docker", r"subprocess.*openfoam"],
        "version": "24+",
    },
    "grpc": {
        "name": "gRPC",
        "category": "infrastructure",
        "patterns": [r"import grpc", r"_pb2\.py", r"grpc\."],
        "version": "1.50+",
    },
    # Caching
    "hashlib": {
        "name": "hashlib",
        "category": "utility",
        "patterns": [r"import hashlib", r"hashlib\.sha"],
        "version": "stdlib",
    },
    "lru_cache": {
        "name": "functools.lru_cache",
        "category": "utility",
        "patterns": [r"lru_cache", r"@cache"],
        "version": "stdlib",
    },
}


@dataclass
class ToolMatch:
    """A detected tool usage."""

    tool_id: str
    name: str
    category: str
    file_path: str
    line_number: int
    match_text: str


@dataclass
class ScanResult:
    """Result of scanning a file or directory."""

    files_scanned: int = 0
    tools_found: dict[str, list[ToolMatch]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class ToolScanner:
    """Scan Python files for tool and dependency usage."""

    def __init__(self, client: Any | None = None):
        """Initialize scanner with optional Weaviate client."""
        self.client = client
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for efficiency."""
        for tool_id, tool_info in KNOWN_TOOLS.items():
            self._compiled_patterns[tool_id] = [
                re.compile(p, re.IGNORECASE) for p in tool_info["patterns"]
            ]

    def scan_file(self, path: Path) -> list[ToolMatch]:
        """Scan a single file for tool usage.

        Args:
            path: Path to Python file

        Returns:
            List of detected tool matches
        """
        matches: list[ToolMatch] = []

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            for line_num, line in enumerate(lines, start=1):
                for tool_id, patterns in self._compiled_patterns.items():
                    for pattern in patterns:
                        if pattern.search(line):
                            tool_info = KNOWN_TOOLS[tool_id]
                            matches.append(
                                ToolMatch(
                                    tool_id=tool_id,
                                    name=tool_info["name"],
                                    category=tool_info["category"],
                                    file_path=str(path),
                                    line_number=line_num,
                                    match_text=line.strip()[:100],
                                )
                            )
                            break  # One match per tool per line is enough

        except Exception as e:
            # Log but don't fail
            pass

        return matches

    def scan_directory(self, root: Path, exclude: list[str] | None = None) -> ScanResult:
        """Scan directory tree for tool usage.

        Args:
            root: Root directory to scan
            exclude: Glob patterns to exclude

        Returns:
            Scan results with all tool matches
        """
        result = ScanResult()
        exclude = exclude or ["**/venv/**", "**/__pycache__/**", "**/node_modules/**"]

        py_files = list(root.rglob("*.py"))

        for py_file in py_files:
            # Skip excluded patterns
            skip = False
            for pattern in exclude:
                if py_file.match(pattern):
                    skip = True
                    break
            if skip:
                continue

            result.files_scanned += 1
            matches = self.scan_file(py_file)

            for match in matches:
                if match.tool_id not in result.tools_found:
                    result.tools_found[match.tool_id] = []
                result.tools_found[match.tool_id].append(match)

        return result

    def ensure_tools_in_weaviate(self) -> dict[str, str]:
        """Ensure all known tools exist in Weaviate.

        Returns:
            Mapping of tool_id to Weaviate UUID
        """
        if not self.client or not WEAVIATE_AVAILABLE:
            return {}

        tool_collection = self.client.collections.get("Tool")
        tool_uuids: dict[str, str] = {}

        for tool_id, tool_info in KNOWN_TOOLS.items():
            # Generate deterministic UUID from tool_id
            uuid_str = hashlib.md5(tool_id.encode()).hexdigest()

            try:
                # Check if exists
                existing = tool_collection.query.fetch_object_by_id(uuid_str)
                if existing:
                    tool_uuids[tool_id] = uuid_str
                    continue
            except Exception:
                pass

            # Create new tool entry
            try:
                tool_collection.data.insert(
                    properties={
                        "tool_id": tool_id,
                        "name": tool_info["name"],
                        "version": tool_info["version"],
                        "category": tool_info["category"],
                        "import_pattern": "|".join(tool_info["patterns"]),
                    },
                    uuid=uuid_str,
                )
                tool_uuids[tool_id] = uuid_str
            except Exception as e:
                pass

        return tool_uuids


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Scan codebase for tool usage")
    parser.add_argument("paths", nargs="+", help="Directories or files to scan")
    parser.add_argument("--ingest", action="store_true", help="Ingest to Weaviate")
    args = parser.parse_args()

    client = None
    if args.ingest and WEAVIATE_AVAILABLE:
        try:
            client = weaviate.connect_to_local(port=8080, grpc_port=50052)
        except Exception as e:
            print(f"Warning: Could not connect to Weaviate: {e}")

    scanner = ToolScanner(client)

    for path_str in args.paths:
        path = Path(path_str)
        if path.is_dir():
            result = scanner.scan_directory(path)
        elif path.is_file():
            matches = scanner.scan_file(path)
            result = ScanResult(files_scanned=1)
            for m in matches:
                if m.tool_id not in result.tools_found:
                    result.tools_found[m.tool_id] = []
                result.tools_found[m.tool_id].append(m)
        else:
            print(f"Warning: {path} not found")
            continue

        print(f"\n=== Scanned {result.files_scanned} files in {path} ===")
        for tool_id, matches in sorted(result.tools_found.items()):
            tool = KNOWN_TOOLS[tool_id]
            print(f"  {tool['name']}: {len(matches)} occurrences")

    if args.ingest and client:
        print("\nIngesting tools to Weaviate...")
        tool_uuids = scanner.ensure_tools_in_weaviate()
        print(f"  Indexed {len(tool_uuids)} tools")
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
