"""Tests for sync_requirements.py script."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

# Import the module under test
import sys
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from sync_requirements import (
    Dependency,
    parse_version_constraint,
    parse_environment_yml,
    parse_requirements_txt,
    check_sync,
    generate_requirements_content,
)


class TestParseVersionConstraint:
    """Test version constraint parsing."""

    def test_simple_package(self):
        """Test parsing package without version."""
        dep = parse_version_constraint("numpy")
        assert dep.name == "numpy"
        assert dep.constraint == ""

    def test_package_with_gte(self):
        """Test parsing package with >= constraint."""
        dep = parse_version_constraint("numpy>=1.24.0")
        assert dep.name == "numpy"
        assert dep.constraint == ">=1.24.0"

    def test_package_with_range(self):
        """Test parsing package with version range."""
        dep = parse_version_constraint("python>=3.9,<3.12")
        assert dep.name == "python"
        assert dep.constraint == ">=3.9,<3.12"

    def test_package_with_whitespace(self):
        """Test parsing package with leading/trailing whitespace."""
        dep = parse_version_constraint("  scipy>=1.10.0  ")
        assert dep.name == "scipy"
        assert dep.constraint == ">=1.10.0"


class TestParseEnvironmentYml:
    """Test environment.yml parsing."""

    def test_parse_simple_env(self):
        """Test parsing a simple environment.yml."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False,
        ) as f:
            f.write("""
name: test
channels:
  - conda-forge
dependencies:
  - python>=3.9
  - numpy>=1.24.0
  - scipy>=1.10.0
  - pip:
    - typeguard
    - radon>=5.0.0
""")
            f.flush()
            env_file = Path(f.name)

        try:
            deps = parse_environment_yml(env_file)
            # python should be excluded
            assert "python" not in deps
            assert "numpy" in deps
            assert deps["numpy"].constraint == ">=1.24.0"
            assert "scipy" in deps
            assert "typeguard" in deps
            assert deps["typeguard"].constraint == ""
            assert "radon" in deps
            assert deps["radon"].constraint == ">=5.0.0"
        finally:
            env_file.unlink()

    def test_parse_nonexistent_file(self):
        """Test parsing a nonexistent file returns empty dict."""
        deps = parse_environment_yml(Path("/nonexistent/file.yml"))
        assert deps == {}


class TestParseRequirementsTxt:
    """Test requirements.txt parsing."""

    def test_parse_simple_requirements(self):
        """Test parsing a simple requirements.txt."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False,
        ) as f:
            f.write("""
# This is a comment
numpy>=1.24.0
scipy>=1.10.0
matplotlib
""")
            f.flush()
            req_file = Path(f.name)

        try:
            deps = parse_requirements_txt(req_file)
            assert "numpy" in deps
            assert deps["numpy"].constraint == ">=1.24.0"
            assert "scipy" in deps
            assert "matplotlib" in deps
            assert deps["matplotlib"].constraint == ""
        finally:
            req_file.unlink()


class TestCheckSync:
    """Test sync checking."""

    def test_in_sync(self):
        """Test files that are in sync."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / "environment.yml"
            req_file = Path(tmpdir) / "requirements.txt"

            env_file.write_text("""
name: test
dependencies:
  - numpy>=1.24.0
  - scipy>=1.10.0
""")
            req_file.write_text("""
numpy>=1.24.0
scipy>=1.10.0
""")

            is_synced, discrepancies = check_sync(env_file, req_file)
            assert is_synced is True
            assert len(discrepancies) == 0

    def test_missing_in_requirements(self):
        """Test package in env but not requirements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / "environment.yml"
            req_file = Path(tmpdir) / "requirements.txt"

            env_file.write_text("""
name: test
dependencies:
  - numpy>=1.24.0
  - scipy>=1.10.0
""")
            req_file.write_text("""
numpy>=1.24.0
""")

            is_synced, discrepancies = check_sync(env_file, req_file)
            assert is_synced is False
            assert len(discrepancies) == 1
            assert "scipy" in discrepancies[0]


class TestGenerateRequirementsContent:
    """Test requirements.txt generation."""

    def test_generate_content(self):
        """Test generating requirements content."""
        deps = {
            "numpy": Dependency("numpy", ">=1.24.0"),
            "scipy": Dependency("scipy", ">=1.10.0"),
            "matplotlib": Dependency("matplotlib", ""),
        }

        content = generate_requirements_content(deps)

        assert "numpy>=1.24.0" in content
        assert "scipy>=1.10.0" in content
        assert "matplotlib" in content
        # Should be sorted alphabetically
        lines = [l for l in content.split("\n") if l and not l.startswith("#")]
        assert lines[0] == "matplotlib"
        assert lines[1] == "numpy>=1.24.0"
        assert lines[2] == "scipy>=1.10.0"
