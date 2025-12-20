#!/usr/bin/env python3
"""
Synchronize requirements.txt with environment.yml.

This script ensures that pip requirements.txt stays in sync with the conda
environment.yml file, preventing dependency drift between conda and pip users.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class Dependency(NamedTuple):
    """A parsed dependency with name and version constraint."""

    name: str
    constraint: str  # e.g., ">=3.6.0" or "" if no constraint


def parse_version_constraint(dep_str: str) -> Dependency:
    """
    Parse a dependency string into name and version constraint.

    Args:
        dep_str: Dependency string like "numpy>=1.24.0" or "pytest"

    Returns:
        Dependency with name and constraint
    """
    # Match package name followed by optional version constraint
    match = re.match(r"^([a-zA-Z0-9_-]+)(.*)?$", dep_str.strip())
    if match:
        name = match.group(1)
        constraint = match.group(2) or ""
        return Dependency(name=name, constraint=constraint.strip())
    return Dependency(name=dep_str.strip(), constraint="")


def parse_environment_yml(env_file: Path) -> dict[str, Dependency]:
    """
    Parse environment.yml and extract all dependencies.

    Args:
        env_file: Path to environment.yml

    Returns:
        Dictionary mapping package names to Dependency objects
    """
    dependencies: dict[str, Dependency] = {}

    if not env_file.exists():
        return dependencies

    try:
        import yaml  # type: ignore

        with open(env_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return dependencies

        deps = data.get("dependencies", [])
        if deps is None:
            deps = []

        for dep in deps:
            if isinstance(dep, str):
                parsed = parse_version_constraint(dep)
                # Skip python version and pip itself
                if parsed.name.lower() not in ("python", "pip"):
                    dependencies[parsed.name.lower()] = parsed
            elif isinstance(dep, dict):
                # Handle pip section
                if "pip" in dep:
                    pip_deps = dep["pip"]
                    if pip_deps:
                        for pip_dep in pip_deps:
                            if isinstance(pip_dep, str):
                                parsed = parse_version_constraint(pip_dep)
                                dependencies[parsed.name.lower()] = parsed

    except ImportError:
        # Fallback: simple parsing without yaml
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "- pip:" in line or "pip:" in line:
                    continue
                if line.startswith("- "):
                    dep_str = line.replace("- ", "").strip()
                    parsed = parse_version_constraint(dep_str)
                    if parsed.name.lower() not in ("python", "pip"):
                        dependencies[parsed.name.lower()] = parsed

    return dependencies


def parse_requirements_txt(req_file: Path) -> dict[str, Dependency]:
    """
    Parse requirements.txt and extract all dependencies.

    Args:
        req_file: Path to requirements.txt

    Returns:
        Dictionary mapping package names to Dependency objects
    """
    dependencies: dict[str, Dependency] = {}

    if not req_file.exists():
        return dependencies

    with open(req_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments
            if not line or line.startswith("#"):
                continue
            # Skip options like -e, -r, etc.
            if line.startswith("-"):
                continue

            parsed = parse_version_constraint(line)
            dependencies[parsed.name.lower()] = parsed

    return dependencies


def generate_requirements_content(
    env_deps: dict[str, Dependency],
    existing_header: str | None = None,
) -> str:
    """
    Generate requirements.txt content from environment.yml dependencies.

    Args:
        env_deps: Dependencies parsed from environment.yml
        existing_header: Optional header comment to preserve

    Returns:
        Generated requirements.txt content
    """
    header = existing_header or """\
# Larrak Dependencies
#
# IMPORTANT: This project requires conda for proper installation:
#   conda create -n larrak python=3.11
#   conda activate larrak
#   conda install -c conda-forge casadi ipopt numpy scipy matplotlib
#
# This file is auto-generated from environment.yml.
# Run: python scripts/sync_requirements.py --generate
#
# If you must use pip, run: pip install -r requirements.txt
# Then verify installation: python scripts/check_environment.py
"""

    lines = [header.strip(), ""]

    # Sort dependencies alphabetically
    sorted_deps = sorted(env_deps.items(), key=lambda x: x[0])

    for dep in sorted_deps:
        # dep is a tuple (name, Dependency)
        dependency = dep[1]
        if dependency.constraint:
            lines.append(f"{dependency.name}{dependency.constraint}")
        else:
            lines.append(dependency.name)

    lines.append("")  # Trailing newline
    return "\n".join(lines)


def check_sync(
    env_file: Path,
    req_file: Path,
) -> tuple[bool, list[str]]:
    """
    Check if requirements.txt is in sync with environment.yml.

    Args:
        env_file: Path to environment.yml
        req_file: Path to requirements.txt

    Returns:
        Tuple of (is_synced, list of discrepancy messages)
    """
    env_deps = parse_environment_yml(env_file)
    req_deps = parse_requirements_txt(req_file)

    discrepancies: list[str] = []

    # Check for packages in env but not in requirements
    for name in env_deps:
        if name not in req_deps:
            discrepancies.append(
                f"Package '{name}' in environment.yml but not in "
                "requirements.txt"
            )

    # Check for packages in requirements but not in env
    for name in req_deps:
        if name not in env_deps:
            discrepancies.append(
                f"Package '{name}' in requirements.txt but not in "
                "environment.yml"
            )

    # Check for version constraint mismatches
    for name, env_dep in env_deps.items():
        if name in req_deps:
            req_dep = req_deps[name]
            env_constraint = env_dep.constraint
            req_constraint = req_dep.constraint
            if env_constraint != req_constraint:
                discrepancies.append(
                    f"Version mismatch for '{name}': "
                    f"env={env_constraint or '(none)'}, "
                    f"req={req_constraint or '(none)'}"
                )

    return len(discrepancies) == 0, discrepancies


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Synchronize requirements.txt with environment.yml",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if files are in sync (exit code 1 if not)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate requirements.txt from environment.yml",
    )
    parser.add_argument(
        "--env-file",
        default="environment.yml",
        help="Path to environment.yml (default: environment.yml)",
    )
    parser.add_argument(
        "--req-file",
        default="requirements.txt",
        help="Path to requirements.txt (default: requirements.txt)",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    env_file = Path(args.env_file)
    if not env_file.is_absolute():
        env_file = project_root / env_file

    req_file = Path(args.req_file)
    if not req_file.is_absolute():
        req_file = project_root / req_file

    if args.check:
        is_synced, discrepancies = check_sync(env_file, req_file)
        if is_synced:
            print("✓ requirements.txt is in sync with environment.yml")
            return 0
        else:
            print("✗ requirements.txt is NOT in sync with environment.yml:")
            for msg in discrepancies:
                print(f"  - {msg}")
            print(
                "\nRun 'python scripts/sync_requirements.py --generate' "
                "to fix."
            )
            return 1

    elif args.generate:
        env_deps = parse_environment_yml(env_file)
        if not env_deps:
            print(f"Error: No dependencies found in {env_file}")
            return 1

        content = generate_requirements_content(env_deps)
        with open(req_file, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✓ Generated {req_file} with {len(env_deps)} dependencies")
        return 0

    else:
        # Default: show sync status
        is_synced, discrepancies = check_sync(env_file, req_file)
        if is_synced:
            print("✓ requirements.txt is in sync with environment.yml")
        else:
            print("Sync status:")
            for msg in discrepancies:
                print(f"  - {msg}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
