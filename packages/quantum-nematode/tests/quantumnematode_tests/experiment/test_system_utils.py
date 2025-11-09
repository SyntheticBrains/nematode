"""Tests for system utility functions."""

import sys
from unittest.mock import patch

from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.experiment.system_utils import (
    capture_system_info,
    get_device_type_string,
    get_package_version,
    get_python_version,
    get_qiskit_version,
    get_torch_version,
)


class TestPythonVersion:
    """Test Python version retrieval."""

    def test_get_python_version(self):
        """Test getting Python version string."""
        version = get_python_version()

        assert isinstance(version, str)
        assert len(version.split(".")) == 3  # Major.minor.micro
        assert (
            version == f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )


class TestPackageVersion:
    """Test package version retrieval."""

    def test_get_package_version_installed(self):
        """Test getting version of installed package."""
        with patch("quantumnematode.experiment.system_utils.version", return_value="1.0.0"):
            ver = get_package_version("test-package")
            assert ver == "1.0.0"

    def test_get_package_version_not_installed(self):
        """Test handling package not found."""
        with patch("quantumnematode.experiment.system_utils.version") as mock_version:
            from importlib.metadata import PackageNotFoundError

            mock_version.side_effect = PackageNotFoundError()
            ver = get_package_version("nonexistent-package")
            assert ver is None


class TestQiskitVersion:
    """Test Qiskit version retrieval."""

    def test_get_qiskit_version_installed(self):
        """Test getting Qiskit version when installed."""
        with patch(
            "quantumnematode.experiment.system_utils.get_package_version",
            return_value="1.0.0",
        ):
            ver = get_qiskit_version()
            assert ver == "1.0.0"

    def test_get_qiskit_version_not_installed(self):
        """Test fallback when Qiskit not installed."""
        with patch(
            "quantumnematode.experiment.system_utils.get_package_version",
            return_value=None,
        ):
            ver = get_qiskit_version()
            assert ver == "unknown"


class TestTorchVersion:
    """Test PyTorch version retrieval."""

    def test_get_torch_version_installed(self):
        """Test getting PyTorch version when installed."""
        with patch(
            "quantumnematode.experiment.system_utils.get_package_version",
            return_value="2.1.0",
        ):
            ver = get_torch_version()
            assert ver == "2.1.0"

    def test_get_torch_version_not_installed(self):
        """Test handling PyTorch not installed."""
        with patch(
            "quantumnematode.experiment.system_utils.get_package_version",
            return_value=None,
        ):
            ver = get_torch_version()
            assert ver is None


class TestDeviceTypeString:
    """Test device type string conversion."""

    def test_device_type_cpu(self):
        """Test converting CPU device type."""
        result = get_device_type_string(DeviceType.CPU)
        assert result == "cpu"

    def test_device_type_gpu(self):
        """Test converting GPU device type."""
        result = get_device_type_string(DeviceType.GPU)
        assert result == "gpu"

    def test_device_type_qpu(self):
        """Test converting QPU device type."""
        result = get_device_type_string(DeviceType.QPU)
        assert result == "qpu"


class TestCaptureSystemInfo:
    """Test complete system info capture."""

    def test_capture_system_info_cpu(self):
        """Test capturing system info for CPU device."""
        with (
            patch(
                "quantumnematode.experiment.system_utils.get_python_version",
                return_value="3.12.0",
            ),
            patch(
                "quantumnematode.experiment.system_utils.get_qiskit_version",
                return_value="1.0.0",
            ),
            patch(
                "quantumnematode.experiment.system_utils.get_torch_version",
                return_value="2.1.0",
            ),
        ):
            info = capture_system_info(DeviceType.CPU)

            assert info["python_version"] == "3.12.0"
            assert info["qiskit_version"] == "1.0.0"
            assert info["torch_version"] == "2.1.0"
            assert info["device_type"] == "cpu"
            assert info["qpu_backend"] is None

    def test_capture_system_info_qpu(self):
        """Test capturing system info for QPU device."""
        with (
            patch(
                "quantumnematode.experiment.system_utils.get_python_version",
                return_value="3.12.0",
            ),
            patch(
                "quantumnematode.experiment.system_utils.get_qiskit_version",
                return_value="1.0.0",
            ),
            patch(
                "quantumnematode.experiment.system_utils.get_torch_version",
                return_value=None,
            ),
        ):
            info = capture_system_info(DeviceType.QPU, qpu_backend="ibm_brisbane")

            assert info["device_type"] == "qpu"
            assert info["qpu_backend"] == "ibm_brisbane"
            assert info["torch_version"] is None

    def test_capture_system_info_gpu(self):
        """Test capturing system info for GPU device."""
        with (
            patch(
                "quantumnematode.experiment.system_utils.get_python_version",
                return_value="3.11.5",
            ),
            patch(
                "quantumnematode.experiment.system_utils.get_qiskit_version",
                return_value="0.45.0",
            ),
            patch(
                "quantumnematode.experiment.system_utils.get_torch_version",
                return_value="2.0.1",
            ),
        ):
            info = capture_system_info(DeviceType.GPU)

            assert info["python_version"] == "3.11.5"
            assert info["device_type"] == "gpu"
            assert info["torch_version"] == "2.0.1"
