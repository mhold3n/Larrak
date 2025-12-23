"""
Integration test for .NET CEM gRPC service.

Tests Python CEMClient talking to actual .NET Larrak.CEM service.
Requires:
- .NET 8.0 SDK installed
- Larrak.CEM project built

Usage:
    pytest tests/integration/test_cem_grpc_integration.py -v

    # Or with service already running:
    pytest tests/integration/test_cem_grpc_integration.py -v --cem-port 5051
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_addoption(parser):
    """Add CEM port option."""
    parser.addoption(
        "--cem-port",
        action="store",
        default=None,
        help="Port of running CEM service (skip auto-start)",
    )


@pytest.fixture(scope="module")
def cem_service(request):
    """
    Fixture that starts CEM service for tests.

    If --cem-port is provided, skips starting and uses existing service.
    """
    from truthmaker.cem.resolver import cem_runtime

    port = request.config.getoption("--cem-port")

    if port is not None:
        # Use existing service
        yield int(port)
        return

    # Check if .NET SDK available
    if not cem_runtime.has_dotnet_sdk():
        pytest.skip(".NET SDK not available")

    # Build and start service
    try:
        cem_runtime.build(force=False)
        cem_runtime.start_service()

        # Wait for service to be ready
        time.sleep(2)

        yield cem_runtime.port

    finally:
        cem_runtime.stop_service()


@pytest.fixture
def cem_client(cem_service):
    """Create CEMClient connected to test service."""
    from truthmaker.cem import CEMClient

    # Connect to the service
    client = CEMClient(
        mock=False,
        host="localhost",
        port=cem_service,
    )
    yield client
    client.close()


class TestCEMGrpcIntegration:
    """Integration tests for .NET CEM service."""

    @pytest.mark.integration
    def test_get_version(self, cem_client):
        """Can get version from running service."""
        version = cem_client.get_version()

        assert version is not None
        assert "cem_version" in version or hasattr(version, "cem_version")

    @pytest.mark.integration
    def test_health_check(self, cem_client):
        """Health check returns healthy."""
        status = cem_client.health_check()

        assert status is True or (hasattr(status, "healthy") and status.healthy)

    @pytest.mark.integration
    def test_validate_motion_simple(self, cem_client):
        """Motion validation works for simple sinusoidal profile."""
        # Simple sinusoidal motion
        theta = np.linspace(0, 2 * np.pi, 360)
        x_profile = 50 * np.sin(theta) + 100  # 50mm amplitude, 100mm offset

        report = cem_client.validate_motion(x_profile, theta)

        assert hasattr(report, "is_valid")
        assert hasattr(report, "violations")

    @pytest.mark.integration
    def test_get_thermo_envelope(self, cem_client):
        """Thermo envelope returns valid ranges."""
        envelope = cem_client.get_thermo_envelope(
            bore=0.1,
            stroke=0.1,
            cr=15.0,
            rpm=3000.0,
        )

        assert hasattr(envelope, "boost_range") or hasattr(envelope, "boost_min")
        assert hasattr(envelope, "feasible")

    @pytest.mark.integration
    def test_get_gear_initial_guess(self, cem_client):
        """Gear initial guess returns valid geometry."""
        theta = np.linspace(0, 2 * np.pi, 360)
        x_target = 50 * np.sin(theta) + 100  # mm

        guess = cem_client.get_gear_initial_guess(x_target, theta)

        assert guess is not None
        # Should have rp, rr, c arrays
        assert hasattr(guess, "rp") or "rp" in guess

    @pytest.mark.integration
    def test_round_trip_performance(self, cem_client):
        """Measure round-trip latency."""
        import time

        theta = np.linspace(0, 2 * np.pi, 360)
        x_profile = 50 * np.sin(theta) + 100

        # Warm up
        cem_client.validate_motion(x_profile, theta)

        # Measure
        n_calls = 10
        start = time.perf_counter()
        for _ in range(n_calls):
            cem_client.validate_motion(x_profile, theta)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / n_calls) * 1000
        print(f"\nAverage round-trip latency: {avg_latency_ms:.2f} ms")

        # Should be under 100ms for local gRPC
        assert avg_latency_ms < 100, f"Latency too high: {avg_latency_ms}ms"


class TestCEMRuntimeResolver:
    """Tests for CEM runtime resolution (without starting service)."""

    def test_resolver_detects_dotnet(self):
        """Resolver can detect .NET SDK."""
        from truthmaker.cem.resolver import cem_runtime

        # Just check it doesn't crash
        has_sdk = cem_runtime.has_dotnet_sdk()
        assert isinstance(has_sdk, bool)

    def test_resolver_finds_project(self):
        """Resolver can find Larrak.CEM project."""
        from truthmaker.cem.resolver import cem_runtime

        project_path = cem_runtime.project_path()

        if project_path is not None:
            assert project_path.exists()
            assert (project_path / "Larrak.CEM.sln").exists() or (
                project_path / "src" / "Larrak.CEM.API"
            ).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
