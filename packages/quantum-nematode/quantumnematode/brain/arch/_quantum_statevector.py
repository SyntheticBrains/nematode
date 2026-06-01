"""Differentiable torch statevector simulator for small quantum circuits.

A minimal, autograd-friendly statevector simulator (designed for ``<= ~10``
qubits) used by the equivariant quantum brain. Gates are applied as tensor
contractions over a complex state of shape ``(batch, 2**n_qubits)``; gradients
flow through to the gate parameters via torch autograd
(backprop-through-simulator), so the circuit trains under PPO **without** the
parameter-shift rule. The simulator is validated against Qiskit on fixed
circuits in the tests.

Gate parameters may be a scalar tensor (shared across the batch) or a
``(batch,)`` tensor (per-sample, e.g. data-encoding angles); both broadcast.

Convention: qubit ``0`` is the most-significant bit of the flat ``2**n`` index
(big-endian). Expectation values are reported per *logical* qubit, so the
internal endianness convention does not leak into observables.
"""

from __future__ import annotations

import torch

_COMPLEX: torch.dtype = torch.complex64


def _as_complex(real_tensor: torch.Tensor) -> torch.Tensor:
    return real_tensor.to(_COMPLEX)


# ---------------------------------------------------------------------------
# Single-qubit gate matrices: theta is () or (B,) real -> (..., 2, 2) complex
# ---------------------------------------------------------------------------


def rx(theta: torch.Tensor) -> torch.Tensor:
    """RX(theta) = exp(-i theta/2 X)."""
    half = theta / 2
    c = _as_complex(torch.cos(half))
    mis = -1j * _as_complex(torch.sin(half))
    top = torch.stack([c, mis], dim=-1)
    bot = torch.stack([mis, c], dim=-1)
    return torch.stack([top, bot], dim=-2)


def ry(theta: torch.Tensor) -> torch.Tensor:
    """RY(theta) = exp(-i theta/2 Y)."""
    half = theta / 2
    c = _as_complex(torch.cos(half))
    s = _as_complex(torch.sin(half))
    top = torch.stack([c, -s], dim=-1)
    bot = torch.stack([s, c], dim=-1)
    return torch.stack([top, bot], dim=-2)


def rz(theta: torch.Tensor) -> torch.Tensor:
    """RZ(theta) = exp(-i theta/2 Z)."""
    half = theta / 2
    em = torch.exp(-1j * _as_complex(half))
    ep = torch.exp(1j * _as_complex(half))
    zero = torch.zeros_like(em)
    top = torch.stack([em, zero], dim=-1)
    bot = torch.stack([zero, ep], dim=-1)
    return torch.stack([top, bot], dim=-2)


# ---------------------------------------------------------------------------
# Two-qubit gate matrices: theta is () or (B,) real -> (..., 4, 4) complex
# Row/col index = 2*bit(q1) + bit(q2), q1 the more-significant qubit.
# ---------------------------------------------------------------------------


def rxx(theta: torch.Tensor) -> torch.Tensor:
    """RXX(theta) = exp(-i theta/2 X⊗X)."""
    half = theta / 2
    c = _as_complex(torch.cos(half))
    mis = -1j * _as_complex(torch.sin(half))
    z = torch.zeros_like(c)
    r0 = torch.stack([c, z, z, mis], dim=-1)
    r1 = torch.stack([z, c, mis, z], dim=-1)
    r2 = torch.stack([z, mis, c, z], dim=-1)
    r3 = torch.stack([mis, z, z, c], dim=-1)
    return torch.stack([r0, r1, r2, r3], dim=-2)


def rzz(theta: torch.Tensor) -> torch.Tensor:
    """RZZ(theta) = exp(-i theta/2 Z⊗Z)."""
    half = theta / 2
    em = torch.exp(-1j * _as_complex(half))
    ep = torch.exp(1j * _as_complex(half))
    z = torch.zeros_like(em)
    r0 = torch.stack([em, z, z, z], dim=-1)
    r1 = torch.stack([z, ep, z, z], dim=-1)
    r2 = torch.stack([z, z, ep, z], dim=-1)
    r3 = torch.stack([z, z, z, em], dim=-1)
    return torch.stack([r0, r1, r2, r3], dim=-2)


def h_matrix(device: torch.device) -> torch.Tensor:
    """Hadamard."""
    inv_sqrt2 = 1.0 / (2.0**0.5)
    return inv_sqrt2 * torch.tensor(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=_COMPLEX,
        device=device,
    )


def cz_matrix(device: torch.device) -> torch.Tensor:
    """Controlled-Z (4x4, diagonal)."""
    return torch.diag(torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=_COMPLEX, device=device))


# ---------------------------------------------------------------------------
# State construction + gate application
# ---------------------------------------------------------------------------


def zero_state(batch: int, n_qubits: int, device: torch.device) -> torch.Tensor:
    """Return the |0...0> state, shape (batch, 2**n_qubits)."""
    state = torch.zeros((batch, 2**n_qubits), dtype=_COMPLEX, device=device)
    state[:, 0] = 1.0
    return state


def apply_1q(state: torch.Tensor, gate: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """Apply a single-qubit gate to ``qubit``.

    ``state``: ``(B, 2**n)``. ``gate``: ``(2, 2)`` (shared) or ``(B, 2, 2)``
    (per-sample). Returns the new ``(B, 2**n)`` state.
    """
    batch = state.shape[0]
    left = 2**qubit
    right = 2 ** (n_qubits - 1 - qubit)
    s = state.reshape(batch, left, 2, right)
    if gate.dim() == 2:  # noqa: PLR2004 — (2,2) shared gate
        out = torch.einsum("ik,blkr->blir", gate, s)
    else:  # (B,2,2) per-sample gate
        out = torch.einsum("bik,blkr->blir", gate, s)
    return out.reshape(batch, 2**n_qubits)


def apply_2q(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubit_a: int,
    qubit_b: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply a two-qubit gate to ``(qubit_a, qubit_b)`` with ``qubit_a < qubit_b``.

    ``gate``: ``(4, 4)`` (shared) or ``(B, 4, 4)`` (per-sample), indexed with
    ``qubit_a`` the more-significant qubit. Returns the new ``(B, 2**n)`` state.
    """
    if qubit_a >= qubit_b:
        msg = f"apply_2q requires qubit_a < qubit_b, got ({qubit_a}, {qubit_b})"
        raise ValueError(msg)
    batch = state.shape[0]
    a = 2**qubit_a
    mid = 2 ** (qubit_b - qubit_a - 1)
    c = 2 ** (n_qubits - 1 - qubit_b)
    s = state.reshape(batch, a, 2, mid, 2, c)
    if gate.dim() == 2:  # noqa: PLR2004 — (4,4) shared gate
        g = gate.reshape(2, 2, 2, 2)
        out = torch.einsum("ijkl,bakmlc->baimjc", g, s)
    else:  # (B,4,4) per-sample gate
        g = gate.reshape(-1, 2, 2, 2, 2)
        out = torch.einsum("bijkl,bakmlc->baimjc", g, s)
    return out.reshape(batch, 2**n_qubits)


# ---------------------------------------------------------------------------
# Observables: <Z_q>, <X_q> per logical qubit -> (B,) real
# ---------------------------------------------------------------------------


def expect_z(state: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """<Z_qubit> over the batch, shape (B,)."""
    batch = state.shape[0]
    left = 2**qubit
    right = 2 ** (n_qubits - 1 - qubit)
    s = state.reshape(batch, left, 2, right)
    prob = (s.conj() * s).real
    return prob[:, :, 0, :].sum(dim=(1, 2)) - prob[:, :, 1, :].sum(dim=(1, 2))


def expect_x(state: torch.Tensor, qubit: int, n_qubits: int) -> torch.Tensor:
    """<X_qubit> over the batch, shape (B,)."""
    batch = state.shape[0]
    left = 2**qubit
    right = 2 ** (n_qubits - 1 - qubit)
    s = state.reshape(batch, left, 2, right)
    cross = (s[:, :, 0, :].conj() * s[:, :, 1, :]).sum(dim=(1, 2))
    return 2.0 * cross.real
