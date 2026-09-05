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

    @pytest.mark.parametrize("device", [DeviceType.CPU, DeviceType.QPU])
    def test_quantum_brains_keep_their_own_devices(self, device: DeviceType) -> None:
        """Devices that place no tensors on an accelerator stay accepted."""
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


class TestAvailabilityAppliesToQuantumBrainsToo:
    """Being Qiskit-backed does not mean a brain avoids torch."""

    def test_quantum_brains_that_build_torch_modules_are_checked(self) -> None:
        """Most `quantum`-family brains also allocate torch actors and critics.

        Skipping the availability check for them would reinstate the raw
        `Torch not compiled with CUDA enabled` assertion this validation
        exists to replace.
        """
        from quantumnematode.brain.arch._registry import get_all_registrations

        registrations = get_all_registrations()
        assert "quantum" in registrations["equivariantquantum"].families

        if torch.cuda.is_available():  # pragma: no cover — host-dependent
            pytest.skip("requires a build without CUDA")

        with pytest.raises(ValueError, match="not available in this PyTorch build"):
            validate_device_selection(DeviceType.GPU, "equivariantquantum")

    @pytest.mark.skipif(torch.cuda.is_available(), reason="requires a build without CUDA")
    def test_every_torch_using_quantum_brain_is_checked(self) -> None:
        """Guards the whole set, not just the one brain review happened to name."""
        torch_using_quantum = [
            "equivariantquantum",
            "hybridquantum",
            "hybridquantumcortex",
            "qliflstm",
            "qrc",
            "qsnnppo",
            "qsnnreinforce",
        ]
        for name in torch_using_quantum:
            with pytest.raises(ValueError, match="not available in this PyTorch build"):
                validate_device_selection(DeviceType.GPU, name)

    def test_qpu_is_never_blocked_by_torch_availability(self) -> None:
        """`qpu` places no tensors on an accelerator, so it needs no check."""
        validate_device_selection(DeviceType.QPU, _QUANTUM_BRAIN)
        validate_device_selection(DeviceType.QPU, _TORCH_BRAIN)
