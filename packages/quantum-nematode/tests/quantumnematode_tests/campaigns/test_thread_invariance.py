"""Results are bit-identical across BLAS thread counts.

The campaign runner pins its children to one thread per numerical library so
that W concurrent workers do not each open a full thread pool and
oversubscribe the machine. That is only legitimate if pinning cannot change
what a run computes.

Thread count is **not** inert in general — parallel reductions can reorder
float accumulation — so the property is pinned here rather than assumed. If a
future architecture uses tensors large enough for torch to take a threaded
reduction path, this test fails, and the runner's guarantee that a
campaign-launched run equals a hand-launched one has to be revisited before
anything else does.

Shapes mirror what the simulation actually runs: a per-step forward at batch
1, a PPO minibatch forward, the gradients from a backward pass, and a
connectome-scale matmul.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

_SEED = 2026
_THREAD_COUNTS = (1, 2, 4)

# The mlpppo C3 actor: 13 sensory features -> 64 -> 64 -> (speed, turn).
_IN_FEATURES = 13
_HIDDEN = 64
_OUT_FEATURES = 2
_MINIBATCH = 128
_CONNECTOME_NEURONS = 302


@pytest.fixture
def restore_thread_count() -> Iterator[None]:
    """Leave the interpreter's thread count as this test found it."""
    original = torch.get_num_threads()
    yield
    torch.set_num_threads(original)


def _digest(tensor: torch.Tensor) -> str:
    """Hash raw tensor bytes — equality here is bit equality, not closeness."""
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def _fingerprints(threads: int) -> dict[str, str]:
    """Compute every fingerprint at a given thread count."""
    torch.set_num_threads(threads)

    torch.manual_seed(_SEED)
    actor = torch.nn.Sequential(
        torch.nn.Linear(_IN_FEATURES, _HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(_HIDDEN, _HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(_HIDDEN, _OUT_FEATURES),
    )

    torch.manual_seed(_SEED + 1)
    single = torch.randn(_IN_FEATURES)
    minibatch = torch.randn(_MINIBATCH, _IN_FEATURES)

    with torch.no_grad():
        step_forward = _digest(actor(single))
        batch_forward = _digest(actor(minibatch))

    actor.zero_grad()
    actor(minibatch).pow(2).mean().backward()
    gradients = _digest(
        torch.cat([p.grad.reshape(-1) for p in actor.parameters() if p.grad is not None]),
    )

    torch.manual_seed(_SEED + 2)
    connectome = torch.nn.Linear(_CONNECTOME_NEURONS, _CONNECTOME_NEURONS)
    with torch.no_grad():
        connectome_forward = _digest(connectome(torch.randn(_MINIBATCH, _CONNECTOME_NEURONS)))

    return {
        "step_forward": step_forward,
        "batch_forward": batch_forward,
        "gradients": gradients,
        "connectome_forward": connectome_forward,
    }


@pytest.mark.usefixtures("restore_thread_count")
class TestThreadInvariance:
    """Pinning threads changes speed, never results."""

    def test_all_paths_bit_identical_across_thread_counts(self) -> None:
        reference = _fingerprints(_THREAD_COUNTS[0])
        for threads in _THREAD_COUNTS[1:]:
            assert _fingerprints(threads) == reference, (
                f"thread count {threads} changed results; the campaign runner pins "
                f"children to one thread and relies on this being inert"
            )

    def test_fingerprints_are_distinct(self) -> None:
        """Guard against the comparison passing because everything hashes alike."""
        fingerprints = _fingerprints(1)
        assert len(set(fingerprints.values())) == len(fingerprints)
