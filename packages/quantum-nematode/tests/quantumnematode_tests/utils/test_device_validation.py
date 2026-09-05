"""Tests for device selection validation.

``DeviceType`` serves both the PyTorch backend and the Qiskit backend, so a
device that is spelled correctly is not necessarily usable by the brain that
was selected. These tests cover both failure modes the validator exists to
catch: a PyTorch-only accelerator handed to a Qiskit-backed brain, and an
accelerator this build cannot provide.
"""

from __future__ import annotations

import pytest
import torch
from quantumnematode.brain.arch.dtypes import DeviceType
from quantumnematode.utils.device import (
    _accepted_devices,
    _is_quantum_brain,
    validate_device_selection,
)

# 'qrc' is registered with families ("quantum", "classical") — the hybrid case.
_QUANTUM_BRAIN = "qvarcircuit"
_HYBRID_BRAIN = "qrc"
_TORCH_BRAIN = "mlpppo"


class TestDeviceTypeMapping:
    """`DeviceType` maps to the torch device strings the brains construct."""

    def test_mps_maps_to_mps(self) -> None:
        assert DeviceType.MPS.to_torch_device_str() == "mps"

    def test_gpu_still_maps_to_cuda(self) -> None:
        """GPU is not silently redirected to another vendor's accelerator."""
        assert DeviceType.GPU.to_torch_device_str() == "cuda"

    def test_cpu_and_qpu_map_to_cpu(self) -> None:
        assert DeviceType.CPU.to_torch_device_str() == "cpu"
        assert DeviceType.QPU.to_torch_device_str() == "cpu"


class TestBrainFamilyDetection:
    """Family detection reads registry metadata, not a hand-kept list."""

    def test_quantum_brain_detected(self) -> None:
        assert _is_quantum_brain(_QUANTUM_BRAIN)

    def test_hybrid_counts_as_quantum(self) -> None:
        """Its device value still reaches the simulator, so the risk applies."""
        assert _is_quantum_brain(_HYBRID_BRAIN)

    def test_torch_brain_not_quantum(self) -> None:
        assert not _is_quantum_brain(_TORCH_BRAIN)

    def test_unknown_brain_defers_to_construction(self) -> None:
        """An unknown name is reported by brain construction, not here."""
        assert not _is_quantum_brain("no_such_brain")


class TestQuantumBrainsRejectTorchOnlyAccelerators:
    """The silent-wrong-backend path this validator exists to close."""

    @pytest.mark.parametrize("brain", [_QUANTUM_BRAIN, _HYBRID_BRAIN])
    def test_mps_rejected(self, brain: str) -> None:
        with pytest.raises(ValueError, match="PyTorch-only accelerator") as excinfo:
            validate_device_selection(DeviceType.MPS, brain)
        message = str(excinfo.value)
        assert "mps" in message
        assert brain in message
        # The message must name what the user can actually use instead.
        assert "cpu" in message

    @pytest.mark.parametrize("device", [DeviceType.CPU, DeviceType.GPU, DeviceType.QPU])
    def test_quantum_brains_keep_their_own_devices(self, device: DeviceType) -> None:
        """`gpu` here selects Aer's GPU device, which validates itself."""
        validate_device_selection(device, _QUANTUM_BRAIN)

    def test_accepted_devices_excludes_torch_only(self) -> None:
        assert _accepted_devices(_QUANTUM_BRAIN) == ["cpu", "gpu", "qpu"]

    def test_accepted_devices_for_torch_brain_includes_mps(self) -> None:
        assert "mps" in _accepted_devices(_TORCH_BRAIN)


class TestAcceleratorAvailability:
    """An accelerator this build cannot provide fails before construction."""

    def test_cpu_needs_no_check(self) -> None:
        validate_device_selection(DeviceType.CPU, _TORCH_BRAIN)

    @pytest.mark.skipif(torch.cuda.is_available(), reason="requires a build without CUDA")
    def test_cuda_unavailable_names_device_and_alternative(self) -> None:
        with pytest.raises(ValueError, match="not available in this PyTorch build") as excinfo:
            validate_device_selection(DeviceType.GPU, _TORCH_BRAIN)
        message = str(excinfo.value)
        assert "gpu" in message
        assert "--device" in message

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="requires an available MPS backend",
    )
    def test_mps_accepted_for_torch_brain_when_available(self) -> None:
        validate_device_selection(DeviceType.MPS, _TORCH_BRAIN)

    @pytest.mark.skipif(
        torch.backends.mps.is_available(),
        reason="requires a build without MPS",
    )
    def test_mps_unavailable_names_device(self) -> None:
        with pytest.raises(ValueError, match="not available in this PyTorch build") as excinfo:
            validate_device_selection(DeviceType.MPS, _TORCH_BRAIN)
        assert "mps" in str(excinfo.value)


class TestValidationDerivesFromRegistry:
    """Coverage follows registration, so new architectures are not missed."""

    def test_every_registered_quantum_brain_rejects_torch_only_devices(self) -> None:
        """No hand-maintained list: the check reads each brain's family tags.

        A new quantum architecture is covered the moment it registers, which
        is the property that makes this validation durable rather than a
        snapshot of the brains that existed when it was written.
        """
        from quantumnematode.brain.arch._registry import get_all_registrations

        quantum = [
            name
            for name, registration in get_all_registrations().items()
            if "quantum" in registration.families
        ]
        assert quantum, "expected at least one registered quantum brain"

        for name in quantum:
            with pytest.raises(ValueError, match="PyTorch-only accelerator"):
                validate_device_selection(DeviceType.MPS, name)

    def test_non_quantum_brains_are_not_rejected_on_family_grounds(self) -> None:
        from quantumnematode.brain.arch._registry import get_all_registrations

        classical = [
            name
            for name, registration in get_all_registrations().items()
            if "quantum" not in registration.families
        ]
        assert classical

        # Either it passes, or it fails on availability — never on family.
        family_rejections = []
        for name in classical:
            try:
                validate_device_selection(DeviceType.MPS, name)
            except ValueError as exc:
                if "PyTorch-only accelerator" in str(exc):
                    family_rejections.append(name)
        assert family_rejections == []


class TestQuantumAcceleratorsAreLeftToQiskit:
    """`gpu` on a quantum brain selects Aer's device, not torch's."""

    @pytest.mark.skipif(torch.cuda.is_available(), reason="requires a build without CUDA")
    def test_gpu_accepted_for_quantum_brain_without_torch_cuda(self) -> None:
        """Torch's view of CUDA is not evidence about qiskit-aer-gpu.

        Rejecting here would refuse a configuration that works whenever the
        Aer GPU wheel is installed but the torch build is CPU-only.
        """
        validate_device_selection(DeviceType.GPU, _QUANTUM_BRAIN)
