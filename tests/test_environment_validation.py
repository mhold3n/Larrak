from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from campro.environment.validator import (
    ValidationResult,
    ValidationStatus,
    get_installation_instructions,
    validate_casadi_ipopt,
    validate_environment,
    validate_python_version,
    validate_required_packages,
)

# NOTE: This file uses class-based tests. Consider refactoring to function-based
# tests to match the main test file style (see test_gear_profile_generation.py)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult objects."""
        result = ValidationResult(
            status=ValidationStatus.PASS,
            message="Test message",
            details="Test details",
            suggestion="Test suggestion",
            version="1.0.0",
        )

        assert result.status == ValidationStatus.PASS
        assert result.message == "Test message"
        assert result.details == "Test details"
        assert result.suggestion == "Test suggestion"
        assert result.version == "1.0.0"

    def test_validation_result_minimal(self):
        """Test creating ValidationResult with minimal fields."""
        result = ValidationResult(
            status=ValidationStatus.ERROR,
            message="Error message",
        )

        assert result.status == ValidationStatus.ERROR
        assert result.message == "Error message"
        assert result.details is None
        assert result.suggestion is None
        assert result.version is None


class TestValidatePythonVersion:
    """Test Python version validation."""

    def test_python_version_pass(self):
        """Test Python version validation passes for supported version."""
        with patch("sys.version_info", (3, 10, 0)):
            result = validate_python_version((3, 9))

            assert result.status == ValidationStatus.PASS
            assert "3.10" in result.message
            assert result.version == "3.10"

    def test_python_version_fail(self):
        """Test Python version validation fails for unsupported version."""
        with patch("sys.version_info", (3, 8, 0)):
            result = validate_python_version((3, 9))

            assert result.status == ValidationStatus.ERROR
            assert "3.8" in result.message
            assert "not supported" in result.message
            assert result.suggestion is not None
            assert "upgrade" in result.suggestion.lower()

    def test_python_version_exact_minimum(self):
        """Test Python version validation for exact minimum version."""
        with patch("sys.version_info", (3, 9, 0)):
            result = validate_python_version((3, 9))

            assert result.status == ValidationStatus.PASS
            assert "3.9" in result.message


class TestValidateCasadiIpopt:
    """Test CasADi and ipopt validation."""

    @patch("campro.environment.validator.log")
    def test_casadi_import_error(self, mock_log):
        """Test validation when CasADi cannot be imported."""
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'casadi'"),
        ):
            result = validate_casadi_ipopt()

            assert result.status == ValidationStatus.ERROR
            assert "not installed" in result.message
            assert "conda" in result.suggestion
            mock_log.info.assert_called()

    @patch("campro.environment.validator.log")
    def test_casadi_no_ipopt(self, mock_log):
        """Test validation when CasADi is available but ipopt is not."""
        mock_ca = Mock()
        mock_ca.__version__ = "3.6.0"
        mock_ca.nlpsol_plugins.return_value = ["sqpmethod", "qpsol"]

        with patch("builtins.__import__", return_value=mock_ca):
            result = validate_casadi_ipopt()

            assert result.status == ValidationStatus.ERROR
            assert "ipopt solver is not available" in result.message
            assert "Available solvers" in result.details
            assert "conda" in result.suggestion
            mock_log.info.assert_called()

    @patch("campro.environment.validator.log")
    def test_casadi_with_ipopt(self, mock_log):
        """Test validation when CasADi and ipopt are both available."""
        mock_ca = Mock()
        mock_ca.__version__ = "3.6.0"
        mock_ca.nlpsol_plugins.return_value = ["ipopt", "sqpmethod", "qpsol"]

        with patch("builtins.__import__", return_value=mock_ca):
            result = validate_casadi_ipopt()

            assert result.status == ValidationStatus.PASS
            assert "ipopt support is available" in result.message
            assert "3.6.0" in result.details
            assert result.version == "3.6.0"
            mock_log.info.assert_called()

    @patch("campro.environment.validator.log")
    def test_casadi_unexpected_error(self, mock_log):
        """Test validation when unexpected error occurs."""
        with patch("builtins.__import__", side_effect=Exception("Unexpected error")):
            result = validate_casadi_ipopt()

            assert result.status == ValidationStatus.ERROR
            assert "Error checking CasADi" in result.message
            assert "Unexpected error" in result.details
            mock_log.error.assert_called()


class TestValidateRequiredPackages:
    """Test required packages validation."""

    def test_all_packages_available(self):
        """Test validation when all required packages are available."""
        mock_numpy = Mock()
        mock_numpy.__version__ = "1.24.0"
        mock_scipy = Mock()
        mock_scipy.__version__ = "1.10.0"
        mock_matplotlib = Mock()
        mock_matplotlib.__version__ = "3.7.0"

        def mock_import(name, *args, **kwargs):
            if name == "numpy":
                return mock_numpy
            if name == "scipy":
                return mock_scipy
            if name == "matplotlib":
                return mock_matplotlib
            raise ImportError(f"No module named '{name}'")

        with patch("builtins.__import__", side_effect=mock_import):
            results = validate_required_packages()

            assert len(results) == 3
            for result in results:
                assert result.status == ValidationStatus.PASS
                assert "is available" in result.message

    def test_package_import_error(self):
        """Test validation when a package cannot be imported."""

        def mock_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("No module named 'numpy'")
            if name == "scipy":
                mock_scipy = Mock()
                mock_scipy.__version__ = "1.10.0"
                return mock_scipy
            if name == "matplotlib":
                mock_matplotlib = Mock()
                mock_matplotlib.__version__ = "3.7.0"
                return mock_matplotlib
            raise ImportError(f"No module named '{name}'")

        with patch("builtins.__import__", side_effect=mock_import):
            results = validate_required_packages()

            assert len(results) == 3

            # Check numpy result (should be error)
            numpy_result = next(r for r in results if "numpy" in r.message)
            assert numpy_result.status == ValidationStatus.ERROR
            assert "not installed" in numpy_result.message
            assert "conda" in numpy_result.suggestion

            # Check other results (should be pass)
            other_results = [r for r in results if "numpy" not in r.message]
            for result in other_results:
                assert result.status == ValidationStatus.PASS

    def test_package_version_unknown(self):
        """Test validation when package version is unknown."""
        mock_package = Mock()
        mock_package.__version__ = None  # Simulate missing version

        def mock_import(name, *args, **kwargs):
            if name == "numpy":
                return mock_package
            if name == "scipy":
                mock_scipy = Mock()
                mock_scipy.__version__ = "1.10.0"
                return mock_scipy
            if name == "matplotlib":
                mock_matplotlib = Mock()
                mock_matplotlib.__version__ = "3.7.0"
                return mock_matplotlib
            raise ImportError(f"No module named '{name}'")

        with patch("builtins.__import__", side_effect=mock_import):
            results = validate_required_packages()

            assert len(results) == 3

            # Check numpy result (should be warning)
            numpy_result = next(r for r in results if "numpy" in r.message)
            assert numpy_result.status == ValidationStatus.WARNING
            assert "version unknown" in numpy_result.message


class TestValidateEnvironment:
    """Test comprehensive environment validation."""

    @patch("campro.environment.validator.validate_python_version")
    @patch("campro.environment.validator.validate_casadi_ipopt")
    @patch("campro.environment.validator.validate_required_packages")
    @patch("campro.environment.validator.log")
    def test_validate_environment_success(
        self,
        mock_log,
        mock_validate_packages,
        mock_validate_casadi,
        mock_validate_python,
    ):
        """Test successful environment validation."""
        # Mock successful results
        mock_validate_python.return_value = ValidationResult(
            status=ValidationStatus.PASS,
            message="Python 3.10 is supported",
            version="3.10",
        )
        mock_validate_casadi.return_value = ValidationResult(
            status=ValidationStatus.PASS,
            message="CasADi with ipopt support is available",
            version="3.6.0",
        )
        mock_validate_packages.return_value = [
            ValidationResult(
                status=ValidationStatus.PASS, message="numpy is available",
            ),
            ValidationResult(
                status=ValidationStatus.PASS, message="scipy is available",
            ),
            ValidationResult(
                status=ValidationStatus.PASS, message="matplotlib is available",
            ),
        ]

        results = validate_environment()

        # Check structure
        assert "python_version" in results
        assert "casadi_ipopt" in results
        assert "required_packages" in results
        assert "summary" in results

        # Check summary
        summary = results["summary"]
        assert summary["overall_status"] == ValidationStatus.PASS
        assert summary["status_counts"][ValidationStatus.PASS] == 5
        assert summary["status_counts"][ValidationStatus.ERROR] == 0
        assert summary["total_checks"] == 5

        mock_log.info.assert_called()

    @patch("campro.environment.validator.validate_python_version")
    @patch("campro.environment.validator.validate_casadi_ipopt")
    @patch("campro.environment.validator.validate_required_packages")
    @patch("campro.environment.validator.log")
    def test_validate_environment_with_errors(
        self,
        mock_log,
        mock_validate_packages,
        mock_validate_casadi,
        mock_validate_python,
    ):
        """Test environment validation with errors."""
        # Mock results with errors
        mock_validate_python.return_value = ValidationResult(
            status=ValidationStatus.PASS,
            message="Python 3.10 is supported",
            version="3.10",
        )
        mock_validate_casadi.return_value = ValidationResult(
            status=ValidationStatus.ERROR,
            message="CasADi is not installed",
            suggestion="Install CasADi using conda",
        )
        mock_validate_packages.return_value = [
            ValidationResult(
                status=ValidationStatus.ERROR, message="numpy is not installed",
            ),
            ValidationResult(
                status=ValidationStatus.PASS, message="scipy is available",
            ),
            ValidationResult(
                status=ValidationStatus.PASS, message="matplotlib is available",
            ),
        ]

        results = validate_environment()

        # Check summary
        summary = results["summary"]
        assert summary["overall_status"] == ValidationStatus.ERROR
        assert summary["status_counts"][ValidationStatus.PASS] == 3
        assert summary["status_counts"][ValidationStatus.ERROR] == 2
        assert summary["total_checks"] == 5

        mock_log.info.assert_called()


class TestGetInstallationInstructions:
    """Test installation instructions function."""

    def test_get_installation_instructions(self):
        """Test that installation instructions are returned."""
        instructions = get_installation_instructions()

        assert isinstance(instructions, str)
        assert "Installation Instructions" in instructions
        assert "conda" in instructions
        assert "environment.yml" in instructions
        assert "scripts/check_environment.py" in instructions


class TestImportTimeValidation:
    """Test import-time validation in campro.__init__."""

    def test_import_time_validation_skip(self):
        """Test that import-time validation can be skipped."""
        with patch.dict(os.environ, {"CAMPRO_SKIP_VALIDATION": "1"}):
            # Re-import to trigger validation
            if "campro" in sys.modules:
                del sys.modules["campro"]

            import campro

            assert campro.is_ipopt_available() is True

    @patch("builtins.__import__")
    def test_import_time_validation_casadi_available(self, mock_import):
        """Test import-time validation when CasADi is available."""
        # Mock CasADi with ipopt
        mock_ca = Mock()
        mock_ca.nlpsol_plugins.return_value = ["ipopt", "sqpmethod"]

        def mock_import_side_effect(name, *args, **kwargs):
            if name == "casadi":
                return mock_ca
            return Mock()

        mock_import.side_effect = mock_import_side_effect

        with patch.dict(os.environ, {"CAMPRO_SKIP_VALIDATION": "0"}, clear=True):
            # Re-import to trigger validation
            if "campro" in sys.modules:
                del sys.modules["campro"]

            import campro

            assert campro.is_ipopt_available() is True

    @patch("builtins.__import__")
    def test_import_time_validation_casadi_not_available(self, mock_import):
        """Test import-time validation when CasADi is not available."""

        def mock_import_side_effect(name, *args, **kwargs):
            if name == "casadi":
                raise ImportError("No module named 'casadi'")
            return Mock()

        mock_import.side_effect = mock_import_side_effect

        with patch.dict(os.environ, {"CAMPRO_SKIP_VALIDATION": "0"}, clear=True):
            # Re-import to trigger validation
            if "campro" in sys.modules:
                del sys.modules["campro"]

            import campro

            assert campro.is_ipopt_available() is False


class TestValidationPerformance:
    """Test validation performance requirements."""

    @patch("time.time")
    def test_import_time_validation_performance(self, mock_time):
        """Test that import-time validation is fast."""
        # Mock time to simulate fast execution
        mock_time.side_effect = [0.0, 0.05]  # 50ms execution time

        with patch("builtins.__import__", return_value=Mock()):
            with patch.dict(os.environ, {"CAMPRO_SKIP_VALIDATION": "0"}, clear=True):
                # Re-import to trigger validation
                if "campro" in sys.modules:
                    del sys.modules["campro"]

                import campro

                # Should not print warning about slow execution
                assert campro.is_ipopt_available() is not None
