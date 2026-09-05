"""Validate a user-selected device against the brain that will use it.

``DeviceType`` is shared by two backends that accept different members, so a
value being spelled correctly does not make it usable:

- **PyTorch brains** place tensors on the device; the backend has to actually
  be present in the running build.
- **Quantum brains** pass ``device.value`` to Qiskit as a simulator selector.
  Aer accepts an unrecognised device string *without raising*, and ``"MPS"``
  collides with its own Matrix Product State method — so an unchecked
  selection would not fail, it would run against a meaningless backend and
  record that backend's name as though it were real.

Both checks therefore run before the brain is constructed, so a bad selection
produces a message naming the alternatives instead of a torch assertion
surfacing from somewhere inside a brain's ``__init__``.
"""

from __future__ import annotations

from quantumnematode.brain.arch._registry import get_registration
from quantumnematode.brain.arch.dtypes import DeviceType

QUANTUM_FAMILY = "quantum"

# Devices whose meaning is specific to the PyTorch backend. Aer would take
# these strings without complaint, which is exactly why they are listed.
_TORCH_ONLY_DEVICES = frozenset({DeviceType.MPS})

# Devices every brain can take: CPU is universal, and QPU is the quantum
# backend's own selector (non-quantum brains fall back to CPU tensors, which
# is long-standing behaviour this module deliberately leaves alone).
_UNIVERSAL_DEVICES = frozenset({DeviceType.CPU, DeviceType.QPU})


def _is_quantum_brain(brain_name: str) -> bool:
    """Whether ``brain_name`` reaches a Qiskit backend.

    Read from the plugin registry's family tags rather than a hand-kept list,
    so a newly registered architecture is covered by its registration alone.
    A brain tagged both quantum and classical counts as quantum: its device
    value still reaches the simulator, so the collision applies to it too.
    """
    try:
        registration = get_registration(brain_name)
    except ValueError:
        # An unknown brain name is not this module's error to raise; brain
        # construction reports it with the list of valid names.
        return False
    return QUANTUM_FAMILY in registration.families


def _accepted_devices(brain_name: str) -> list[str]:
    """Devices the given brain can be run with, for error messages."""
    quantum = _is_quantum_brain(brain_name)
    return sorted(
        d.value
        for d in DeviceType
        if d in _UNIVERSAL_DEVICES or not (quantum and d in _TORCH_ONLY_DEVICES)
    )


def _torch_backend_available(device: DeviceType) -> bool:
    """Whether the running torch build can provide ``device``."""
    import torch

    if device is DeviceType.GPU:
        return torch.cuda.is_available()
    if device is DeviceType.MPS:
        return torch.backends.mps.is_available()
    return True


def _alternative_hint(device: DeviceType) -> str:
    """Suggest the accelerator a user on this host most likely wants."""
    import torch

    if device is DeviceType.GPU and torch.backends.mps.is_available():
        return (
            " On Apple silicon use '--device mps' instead, though note that CPU is faster "
            "for this project's model sizes."
        )
    if device is DeviceType.MPS and torch.cuda.is_available():
        return " On a CUDA host use '--device gpu' instead."
    return " Use '--device cpu'."


def validate_device_selection(device: DeviceType, brain_name: str) -> None:
    """Raise ``ValueError`` if ``device`` cannot serve ``brain_name``.

    Two failure modes, checked in this order:

    1. **Wrong backend.** A PyTorch-only accelerator selected for a quantum
       brain. Checked first because it holds regardless of what hardware is
       present — reporting it as "unavailable" would send the user hunting for
       a driver instead of correcting the flag.
    2. **Absent backend.** An accelerator this torch build cannot provide.

    The availability check applies to **every** brain, quantum ones included.
    Being tagged ``quantum`` does not mean a brain is Qiskit-only: most of them
    also build torch actors and critics at construction and place those tensors
    on this device, so skipping the check for them reinstates exactly the raw
    ``Torch not compiled with CUDA enabled`` assertion this validation exists to
    replace. A host with a GPU-enabled Aer but a deliberately CPU-only torch
    build is refused here; that is the accepted cost, and it fails with a
    message naming the problem rather than a traceback from inside a brain.

    CPU needs no check at all.
    """
    if device is DeviceType.CPU:
        return

    if _is_quantum_brain(brain_name) and device in _TORCH_ONLY_DEVICES:
        msg = (
            f"Device '{device.value}' is a PyTorch-only accelerator and cannot be used with "
            f"the '{brain_name}' brain, which runs on a Qiskit backend. Qiskit would accept "
            f"'{device.value.upper()}' without error and run against a meaningless backend. "
            f"Accepted devices for '{brain_name}': {', '.join(_accepted_devices(brain_name))}."
        )
        raise ValueError(msg)

    if not _torch_backend_available(device):
        msg = (
            f"Device '{device.value}' was requested but is not available in this PyTorch "
            f"build.{_alternative_hint(device)}"
        )
        raise ValueError(msg)
