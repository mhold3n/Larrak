"""AST-based scanner for engineering element extraction and Weaviate ingestion.

This module scans Python files for engineering-relevant elements (constants,
material properties, physics functions) and indexes them in Weaviate for
dynamic codebase tracking.

Usage:
    # Scan a directory
    python -m provenance.engineering_scanner --path campro/

    # Scan only changed files (for pre-commit)
    python -m provenance.engineering_scanner --changed-only path/to/file.py

    # Generate coverage report
    python -m provenance.engineering_scanner --report
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Conditional Weaviate import - graceful degradation if not available
try:
    import weaviate

    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None  # type: ignore[assignment]


@dataclass
class EngElement:
    """Extracted engineering element."""

    element_id: str
    element_type: str  # constant, property, function, class
    file_path: str
    line_start: int
    line_end: int
    name: str
    unit: str | None = None
    source_citation: str | None = None
    docstring: str | None = None
    raw_code: str = ""
    uncertainty: float | None = None
    valid_range: str | None = None  # JSON string


@dataclass
class ScanStats:
    """Statistics from a scan operation."""

    files_scanned: int = 0
    elements_found: int = 0
    elements_ingested: int = 0
    files_with_elements: int = 0
    errors: list[str] = field(default_factory=list)


class EngineeringScanner:
    """Scan Python files for engineering elements and ingest to Weaviate."""

    # Patterns indicating engineering-relevant code
    ENGINEERING_PATTERNS = [
        r"PhysicalConstant\(",
        r"MaterialProperty\(",
        r"PropertySource\(",
        r"SafetyFactor\(",  # Safety factor definitions
        r"SF_[A-Z]",  # Safety factor instances (e.g., SF_PISTON_FATIGUE)
        r'unit\s*[=:]\s*["\']',
        r'source\s*[=:]\s*["\']',
        r"uncertainty",
        r"valid_range",
    ]

    def __init__(self, client: Any | None = None):
        """Initialize scanner.

        Args:
            client: Optional Weaviate client. If None, operates in dry-run mode.
        """
        self.client = client
        self.elements_collection = None
        if client and WEAVIATE_AVAILABLE:
            try:
                self.elements_collection = client.collections.get("EngineeringElement")
            except Exception:
                pass  # Collection may not exist yet

    def scan_file(self, path: Path) -> list[EngElement]:
        """Parse file and extract engineering elements.

        Args:
            path: Path to Python file

        Returns:
            List of extracted engineering elements
        """
        elements: list[EngElement] = []

        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
            return elements

        # Quick check if file likely contains engineering elements
        if not any(re.search(p, source) for p in self.ENGINEERING_PATTERNS):
            return elements

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"Warning: Syntax error in {path}: {e}")
            return elements

        lines = source.splitlines()

        for node in ast.walk(tree):
            # Check module-level assignments for PhysicalConstant
            if isinstance(node, ast.Assign):
                # Check for PhysicalConstant
                elem = self._extract_physical_constant(node, path, lines)
                if elem:
                    elements.append(elem)
                # Check for SafetyFactor
                elem = self._extract_safety_factor(node, path, lines)
                if elem:
                    elements.append(elem)

            # Check function definitions with physics/engineering docstrings
            elif isinstance(node, ast.FunctionDef):
                elem = self._extract_documented_function(node, path, lines)
                if elem:
                    elements.append(elem)

            # Check class definitions (MaterialProperty, etc.)
            elif isinstance(node, ast.ClassDef):
                elem = self._extract_engineering_class(node, path, lines)
                if elem:
                    elements.append(elem)

        return elements

    def _extract_physical_constant(
        self, node: ast.Assign, path: Path, lines: list[str]
    ) -> EngElement | None:
        """Extract PhysicalConstant assignments."""
        # Check if this is a PhysicalConstant call
        if not isinstance(node.value, ast.Call):
            return None

        func = node.value.func
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr

        if func_name != "PhysicalConstant":
            return None

        # Get the target name
        if not node.targets or not isinstance(node.targets[0], ast.Name):
            return None

        name = node.targets[0].id

        # Extract keyword arguments
        kwargs = {}
        for kw in node.value.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant):
                kwargs[kw.arg] = kw.value.value

        # Generate element ID
        element_id = self._generate_element_id(str(path), name, node.lineno)

        # Get source code lines
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        raw_code = "\n".join(lines[line_start - 1 : line_end])

        return EngElement(
            element_id=element_id,
            element_type="constant",
            file_path=str(path),
            line_start=line_start,
            line_end=line_end,
            name=name,
            unit=kwargs.get("unit"),
            source_citation=kwargs.get("source"),
            uncertainty=kwargs.get("uncertainty"),
            valid_range=str(kwargs.get("valid_range")) if "valid_range" in kwargs else None,
            docstring=kwargs.get("notes"),
            raw_code=raw_code,
        )

    def _extract_documented_function(
        self, node: ast.FunctionDef, path: Path, lines: list[str]
    ) -> EngElement | None:
        """Extract functions with engineering-relevant docstrings."""
        docstring = ast.get_docstring(node)
        if not docstring:
            return None

        # Check for engineering keywords in docstring
        engineering_keywords = [
            "unit:",
            "units:",
            "temperature",
            "pressure",
            "physics",
            "thermodynamic",
            "combustion",
            "kinematic",
            "dynamic",
            "returns:",
            "args:",
            "parameters",
            "uncertainty",
            "tolerance",
        ]

        doc_lower = docstring.lower()
        if not any(kw in doc_lower for kw in engineering_keywords):
            return None

        # Extract unit info from docstring if present
        unit = None
        unit_match = re.search(r"unit[s]?:\s*(\S+)", docstring, re.IGNORECASE)
        if unit_match:
            unit = unit_match.group(1)

        element_id = self._generate_element_id(str(path), node.name, node.lineno)
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno

        return EngElement(
            element_id=element_id,
            element_type="function",
            file_path=str(path),
            line_start=line_start,
            line_end=line_end,
            name=node.name,
            unit=unit,
            docstring=docstring[:500],  # Truncate long docstrings
            raw_code="\n".join(lines[line_start - 1 : min(line_start + 10, line_end)]),
        )

    def _extract_engineering_class(
        self, node: ast.ClassDef, path: Path, lines: list[str]
    ) -> EngElement | None:
        """Extract engineering-related class definitions."""
        # Check for engineering class patterns
        engineering_classes = [
            "MaterialProperty",
            "PropertySource",
            "MaterialDatabase",
            "PhysicsModel",
            "CombustionModel",
            "ThermoODE",
        ]

        # Check if this class inherits from or is an engineering class
        is_engineering = node.name in engineering_classes
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id in engineering_classes:
                is_engineering = True
                break

        if not is_engineering:
            return None

        docstring = ast.get_docstring(node)
        element_id = self._generate_element_id(str(path), node.name, node.lineno)
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno

        return EngElement(
            element_id=element_id,
            element_type="class",
            file_path=str(path),
            line_start=line_start,
            line_end=line_end,
            name=node.name,
            docstring=docstring[:500] if docstring else None,
            raw_code="\n".join(lines[line_start - 1 : min(line_start + 20, line_end)]),
        )

    def _extract_safety_factor(
        self, node: ast.Assign, path: Path, lines: list[str]
    ) -> EngElement | None:
        """Extract SafetyFactor assignments."""
        if not isinstance(node.value, ast.Call):
            return None

        func = node.value.func
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr

        if func_name != "SafetyFactor":
            return None

        # Get the target name
        if not node.targets or not isinstance(node.targets[0], ast.Name):
            return None

        name = node.targets[0].id

        # Extract keyword arguments
        kwargs = {}
        for kw in node.value.keywords:
            if kw.arg:
                if isinstance(kw.value, ast.Constant):
                    kwargs[kw.arg] = kw.value.value
                elif isinstance(kw.value, ast.Attribute):
                    # Handle enum values like FailureMode.FATIGUE
                    kwargs[kw.arg] = (
                        f"{kw.value.value.id}.{kw.value.attr}"
                        if isinstance(kw.value.value, ast.Name)
                        else kw.value.attr
                    )

        # Generate element ID
        element_id = self._generate_element_id(str(path), name, node.lineno)

        # Get source code lines
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        raw_code = "\n".join(lines[line_start - 1 : line_end])

        # Get safety factor value for documentation
        sf_value = kwargs.get("value", "")
        failure_mode = kwargs.get("failure_mode", "")

        return EngElement(
            element_id=element_id,
            element_type="safety_factor",
            file_path=str(path),
            line_start=line_start,
            line_end=line_end,
            name=name,
            unit=f"SF={sf_value}",
            source_citation=kwargs.get("source"),
            docstring=f"{failure_mode}: {kwargs.get('notes', '')}",
            raw_code=raw_code,
        )

    def _generate_element_id(self, file_path: str, name: str, line: int) -> str:
        """Generate deterministic element ID."""
        content = f"{file_path}:{name}:{line}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def ingest_element(self, element: EngElement) -> str | None:
        """Insert or update element in Weaviate.

        Args:
            element: Engineering element to ingest

        Returns:
            UUID if successful, None otherwise
        """
        if self.elements_collection is None or not WEAVIATE_AVAILABLE:
            return None

        uuid = weaviate.util.generate_uuid5(element.element_id)

        try:
            self.elements_collection.data.insert(
                uuid=uuid,
                properties={
                    "element_id": element.element_id,
                    "element_type": element.element_type,
                    "file_path": element.file_path,
                    "line_range": f"{element.line_start}-{element.line_end}",
                    "name": element.name,
                    "unit": element.unit or "",
                    "source_citation": element.source_citation or "",
                    "uncertainty": element.uncertainty or 0.0,
                    "valid_range": element.valid_range or "",
                    "compliance_status": "tracked",
                    "last_verified": datetime.now(timezone.utc).isoformat(),
                    "docstring": element.docstring or "",
                },
            )
            return uuid
        except Exception as e:
            # May fail if element already exists - try update
            try:
                self.elements_collection.data.update(
                    uuid=uuid,
                    properties={
                        "last_verified": datetime.now(timezone.utc).isoformat(),
                        "unit": element.unit or "",
                        "source_citation": element.source_citation or "",
                    },
                )
                return uuid
            except Exception:
                print(f"Warning: Could not ingest element {element.name}: {e}")
                return None

    def scan_directory(
        self,
        root: Path,
        exclude: list[str] | None = None,
        ingest: bool = True,
    ) -> ScanStats:
        """Scan directory tree and return coverage stats.

        Args:
            root: Root directory to scan
            exclude: Patterns to exclude
            ingest: Whether to ingest to Weaviate

        Returns:
            Scan statistics
        """
        exclude = exclude or ["__pycache__", "cem", "PERMANENT_ARCHIVE", ".git", "miniforge3"]
        stats = ScanStats()

        for py_file in root.rglob("*.py"):
            # Check exclusions
            if any(ex in str(py_file) for ex in exclude):
                continue

            stats.files_scanned += 1
            file_elements = self.scan_file(py_file)

            if file_elements:
                stats.files_with_elements += 1
                for elem in file_elements:
                    stats.elements_found += 1
                    if ingest:
                        uuid = self.ingest_element(elem)
                        if uuid:
                            stats.elements_ingested += 1

        return stats

    def scan_files(
        self,
        files: list[Path],
        ingest: bool = True,
    ) -> ScanStats:
        """Scan specific files.

        Args:
            files: List of files to scan
            ingest: Whether to ingest to Weaviate

        Returns:
            Scan statistics
        """
        stats = ScanStats()

        for py_file in files:
            if not py_file.suffix == ".py":
                continue

            stats.files_scanned += 1
            file_elements = self.scan_file(py_file)

            if file_elements:
                stats.files_with_elements += 1
                for elem in file_elements:
                    stats.elements_found += 1
                    if ingest:
                        uuid = self.ingest_element(elem)
                        if uuid:
                            stats.elements_ingested += 1

        return stats


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scan Python files for engineering elements and index in Weaviate"
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Directory to scan",
    )
    parser.add_argument(
        "--changed-only",
        nargs="*",
        type=Path,
        help="Scan only these specific files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan without ingesting to Weaviate",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate coverage report",
    )

    args = parser.parse_args()

    # Connect to Weaviate if not dry-run
    client = None
    if not args.dry_run and WEAVIATE_AVAILABLE:
        try:
            # Note: gRPC port mapped to 50052 in docker-compose
            client = weaviate.connect_to_local(port=8080, grpc_port=50052)
        except Exception as e:
            print(f"Warning: Could not connect to Weaviate: {e}")
            print("Running in dry-run mode.")

    scanner = EngineeringScanner(client)

    try:
        if args.changed_only is not None:
            # Scan specific files
            files = [Path(f) for f in args.changed_only if f]
            stats = scanner.scan_files(files, ingest=not args.dry_run)
        elif args.path:
            # Scan directory
            stats = scanner.scan_directory(args.path, ingest=not args.dry_run)
        else:
            # Default: scan campro and truthmaker
            project_root = Path(__file__).parent.parent
            stats = ScanStats()

            for subdir in ["campro", "truthmaker"]:
                path = project_root / subdir
                if path.exists():
                    sub_stats = scanner.scan_directory(path, ingest=not args.dry_run)
                    stats.files_scanned += sub_stats.files_scanned
                    stats.elements_found += sub_stats.elements_found
                    stats.elements_ingested += sub_stats.elements_ingested
                    stats.files_with_elements += sub_stats.files_with_elements

        # Print results
        print(f"\n{'=' * 50}")
        print("Engineering Element Scan Results")
        print(f"{'=' * 50}")
        print(f"Files scanned:        {stats.files_scanned}")
        print(f"Files with elements:  {stats.files_with_elements}")
        print(f"Elements found:       {stats.elements_found}")
        print(f"Elements ingested:    {stats.elements_ingested}")
        if stats.files_scanned > 0:
            coverage = (stats.files_with_elements / stats.files_scanned) * 100
            print(f"Coverage:             {coverage:.1f}%")
        print(f"{'=' * 50}")

        return 0

    finally:
        if client:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
