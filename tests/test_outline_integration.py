"""Integration tests for outline API."""

import json
from pathlib import Path

import pytest
import weaviate


@pytest.fixture
def weaviate_client():
    """Create a Weaviate client for testing."""
    client = weaviate.connect_to_local(port=8080, grpc_port=50052)
    yield client
    client.close()


def test_code_scanner_extracts_symbols(tmp_path):
    """Test that code scanner extracts symbols with new properties."""
    # Create a test Python file
    test_file = tmp_path / "test_module.py"
    test_file.write_text("""
\"\"\"Test module for outline.\"\"\"

@dataclass
class TestClass:
    \"\"\"A test class.\"\"\"

    @property
    def test_method(self) -> str:
        \"\"\"A test method.\"\"\"
        return "test"

    async def async_method(self, param: int) -> None:
        \"\"\"An async method.\"\"\"
        pass

def standalone_function(x: float, y: float) -> float:
    \"\"\"A standalone function.\"\"\"
    return x + y
""")

    # Extract symbols
    from truthmaker.ingestion.code_scanner import extract_symbols_from_file

    symbols = extract_symbols_from_file(test_file)

    # Verify we extracted the class
    class_symbol = next((s for s in symbols if s["name"] == "TestClass"), None)
    assert class_symbol is not None
    assert class_symbol["kind"] == "class"
    assert "@dataclass" in class_symbol["decorators"]

    # Verify we extracted the methods
    method_symbol = next((s for s in symbols if s["name"] == "TestClass.test_method"), None)
    assert method_symbol is not None
    assert method_symbol["kind"] == "method"
    assert "@property" in method_symbol["decorators"]
    assert method_symbol["is_async"] is False
    assert method_symbol["return_type"] == "str"

    async_method_symbol = next((s for s in symbols if s["name"] == "TestClass.async_method"), None)
    assert async_method_symbol is not None
    assert async_method_symbol["is_async"] is True
    assert async_method_symbol["return_type"] == "None"

    # Verify we extracted the function
    func_symbol = next((s for s in symbols if s["name"] == "standalone_function"), None)
    assert func_symbol is not None
    assert func_symbol["kind"] == "function"
    assert func_symbol["return_type"] == "float"

    # Verify parameters are JSON-encoded
    params = json.loads(func_symbol["parameters"])
    assert len(params) == 2
    assert params[0]["name"] == "x"
    assert params[0]["type"] == "float"


def test_outline_api_hierarchy(weaviate_client):
    """Test that outline API returns hierarchical structure."""
    from dashboard.api import create_app

    app = create_app()
    client = app.test_client()

    # Assuming campro/optimization/orchestrator.py has been indexed
    response = client.get("/api/outline/campro/optimization/orchestrator.py")

    if response.status_code == 404:
        pytest.skip("File not indexed in test environment")

    assert response.status_code == 200
    data = response.get_json()

    assert "file" in data
    assert "symbols" in data
    assert "count" in data

    # Check that we have symbols
    symbols = data["symbols"]
    assert len(symbols) > 0

    # Check that classes have methods as children
    class_symbols = [s for s in symbols if s["kind"] == "class"]
    if class_symbols:
        # At least one class should have methods
        has_children = any(s["children"] for s in class_symbols)
        assert has_children, "Classes should have child methods"


def test_outline_api_not_found():
    """Test that outline API returns 404 for non-existent files."""
    from dashboard.api import create_app

    app = create_app()
    client = app.test_client()

    response = client.get("/api/outline/nonexistent/file.py")

    assert response.status_code == 404
    data = response.get_json()
    assert "error" in data or "message" in data


def test_outline_refresh_endpoint():
    """Test outline refresh endpoint."""
    from dashboard.api import create_app

    app = create_app()
    client = app.test_client()

    response = client.post(
        "/api/outline/refresh",
        json={"file": "test/file.py"},
        content_type="application/json",
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "queued"
    assert data["file"] == "test/file.py"
