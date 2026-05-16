"""Transgenerational memory substrate — inheritable behavioural-bias dataclass.

Provides :class:`TransgenerationalMemory`, the heritable substrate
that the ``TransgenerationalInheritance`` strategy
(:class:`quantumnematode.evolution.transgenerational_inheritance.TransgenerationalInheritance`)
threads from the F0 elite through F1/F2/F3 generations. The
substrate carries a per-action additive logit bias that, when set on
``LSTMPPOBrain.tei_prior``, augments the actor's logits before
softmax at every step of every episode — biasing offspring action
distributions toward avoidance independently of trained weights.

Three fields:

- ``logit_bias``: ``torch.Tensor`` of shape ``(num_actions,)``,
  dtype ``float32``. Clamped to ``|x| ≤ 2.0`` post-construction
  (Boltzmann ratio cap ≈ 7.4x). The size matches the brain's
  ``num_actions`` (4 for the default ``DEFAULT_ACTIONS`` set;
  forward-compatible with ``SIX_ACTIONS``).
- ``lineage_depth``: ``int``, 0 for F0, incremented by 1 at each
  ``inherit_from`` call.
- ``source_genome_id``: ``str``, the F0 elite's genome ID — the
  authoritative provenance anchor for the cascade.

The dataclass is ``frozen=True`` so cross-generation aliasing
cannot mutate an ancestor's substrate; ``__post_init__`` validates
shape + dtype and clamps the bias via ``object.__setattr__`` (the
canonical Python pattern for frozen-dataclass post-init mutation).

Three operations:

- :meth:`apply_to_logits`: additive ``logits + self.logit_bias``,
  broadcast over leading batch/sequence dimensions, preserving
  input shape/dtype/device, no in-place mutation.
- :meth:`inherit_from`: class method producing a child substrate
  as ``child.logit_bias = parents[0].logit_bias * decay_factor``,
  with ``lineage_depth`` incremented and ``source_genome_id``
  inherited. Decay is multiplicative at the generation boundary.
- :func:`save` / :func:`load`: ``.tei.pt`` round-trip via
  ``torch.save`` / ``torch.load``.

One stub:

- :func:`extract_from_brain`: contractual placeholder that raises
  ``NotImplementedError``. The functional implementation requires
  env coupling (a pathogen lawn at a known position + probe-position
  generator + brain-policy adapter) and lands alongside the F0
  Substrate Extraction Pipeline in a follow-up commit. The stub
  preserves the public-API surface so callers can import the symbol
  today.

See the OpenSpec change ``openspec/changes/add-transgenerational-
memory/`` for the full design rationale + cascade semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


# Maximum absolute element-wise magnitude of ``logit_bias``. Caps the
# Boltzmann ratio at ``e^2 ≈ 7.4x`` so a strong bias cannot collapse
# exploration. Kaletsky F2 ≈ 0.55 choice index corresponds to a
# ~3x action-probability tilt, so the cap leaves headroom while
# preventing pathological deterministic policies.
LOGIT_BIAS_CLAMP: float = 2.0


@dataclass(frozen=True)
class TransgenerationalMemory:
    """Inheritable behavioural-bias substrate carried across generations.

    Stores a per-action additive logit bias that biases an actor's
    action distribution at every step of every episode. The bias is
    extracted from an F0 elite policy via :func:`extract_from_brain`
    and multiplicatively decayed across F1/F2/F3 generations via
    :meth:`inherit_from`.

    Parameters
    ----------
    logit_bias : torch.Tensor
        1-D tensor of shape ``(num_actions,)`` and dtype ``float32``.
        Each element SHALL satisfy ``|x| ≤ LOGIT_BIAS_CLAMP`` after
        construction; values are clamped automatically in
        ``__post_init__`` and the input tensor is NOT mutated in place.
    lineage_depth : int
        Inheritance generation (0 for F0, 1 for F1, etc.). Incremented
        by 1 at each :meth:`inherit_from` call. SHALL be non-negative.
    source_genome_id : str
        F0 elite genome ID — the provenance anchor for the cascade.
        Inherited unchanged across all generations.

    Raises
    ------
    ValueError
        If ``logit_bias.ndim != 1``, ``logit_bias.dtype != torch.float32``,
        or ``lineage_depth < 0``.

    Notes
    -----
    ``frozen=True`` prevents cross-generation aliasing mutation. The
    clamping pass inside ``__post_init__`` uses ``object.__setattr__``
    (the canonical Python pattern for frozen-dataclass post-init
    mutation, since frozen dataclasses cannot reassign fields
    directly). The original caller-provided ``logit_bias`` tensor is
    cloned before clamping so the caller's tensor is never modified
    in place.
    """

    logit_bias: torch.Tensor
    lineage_depth: int
    source_genome_id: str

    def __post_init__(self) -> None:
        """Validate shape/dtype/depth and clamp ``logit_bias`` element-wise.

        Runs after dataclass field assignment. Raises ``ValueError`` on
        invalid inputs (non-1-D logit_bias, non-float32 dtype, or
        negative ``lineage_depth``). Clones the input ``logit_bias``
        and clamps the copy to ``[-LOGIT_BIAS_CLAMP, +LOGIT_BIAS_CLAMP]``,
        then reassigns via ``object.__setattr__`` (frozen-dataclass
        canonical post-init mutation pattern) so the caller's tensor
        is never mutated in place.
        """
        # Shape validation: must be 1-D so broadcast over (batch, seq, num_actions)
        # logits is unambiguous. Higher-dim biases would broadcast non-obviously.
        if self.logit_bias.ndim != 1:
            msg = (
                f"TransgenerationalMemory.logit_bias must be 1-D "
                f"(shape (num_actions,)); got ndim={self.logit_bias.ndim} "
                f"with shape {tuple(self.logit_bias.shape)} and "
                f"dtype {self.logit_bias.dtype}."
            )
            raise ValueError(msg)
        # Dtype validation: float32 is the actor's logit dtype across all
        # LSTMPPO call sites; mixing dtypes would silently cast at the
        # addition site and break torch.equal-based round-trip tests.
        if self.logit_bias.dtype != torch.float32:
            msg = (
                f"TransgenerationalMemory.logit_bias must have dtype "
                f"torch.float32; got dtype={self.logit_bias.dtype} "
                f"with shape {tuple(self.logit_bias.shape)} and "
                f"ndim={self.logit_bias.ndim}."
            )
            raise ValueError(msg)
        if self.lineage_depth < 0:
            msg = (
                f"TransgenerationalMemory.lineage_depth must be >= 0 "
                f"(0 for F0, incremented at each inherit_from call); "
                f"got {self.lineage_depth}."
            )
            raise ValueError(msg)
        # Clone-then-clamp: the input tensor is NEVER mutated. clone()
        # detaches autograd history too, which is desired — the substrate
        # is non-trainable additive state, not a gradient-flowing parameter.
        clamped = self.logit_bias.detach().clone().clamp_(-LOGIT_BIAS_CLAMP, LOGIT_BIAS_CLAMP)
        # Frozen dataclasses cannot reassign fields directly; the canonical
        # post-init mutation pattern uses object.__setattr__.
        object.__setattr__(self, "logit_bias", clamped)

    def apply_to_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Return ``logits + self.logit_bias`` without in-place mutation.

        Broadcasts over leading batch / sequence dimensions: a
        ``(num_actions,)`` bias added to ``(batch, seq, num_actions)``
        logits broadcasts elementwise over the actions axis. The
        returned tensor is a distinct object — neither the input
        ``logits`` nor ``self.logit_bias`` is mutated.

        Parameters
        ----------
        logits : torch.Tensor
            Actor logits of shape ``(..., num_actions)``. The trailing
            dimension SHALL match ``self.logit_bias.shape[0]``; mismatch
            raises ``RuntimeError`` at the addition site (PyTorch
            broadcast rule), with a clear message.

        Returns
        -------
        torch.Tensor
            New tensor of the same shape, dtype, and device as
            ``logits``, with the bias added element-wise.
        """
        # `+` is non-in-place on tensors; returns a new tensor.
        return logits + self.logit_bias

    @classmethod
    def inherit_from(
        cls,
        parents: Sequence[TransgenerationalMemory],
        decay_factor: float,
    ) -> TransgenerationalMemory:
        """Produce a child substrate decayed from a single parent (top-1 elite).

        ``child.logit_bias = parents[0].logit_bias * decay_factor``;
        ``child.lineage_depth = parents[0].lineage_depth + 1``;
        ``child.source_genome_id = parents[0].source_genome_id``.

        The multi-parent ``parents`` signature is forward-compatible
        with future strategies (tournament, soft-elite top-k) but the
        current single-elite-broadcast semantics use only ``parents[0]``.
        This matches ``LamarckianInheritance.assign_parent``'s
        round-robin contract under ``elite_count=1``.

        Parameters
        ----------
        parents : Sequence[TransgenerationalMemory]
            At least one parent substrate. The single-elite rule uses
            ``parents[0]``.
        decay_factor : float
            Multiplicative decay applied to the parent's ``logit_bias``.
            Constrained to ``[0.0, 1.0]``; values outside raise
            ``ValueError``.

        Returns
        -------
        TransgenerationalMemory
            Child substrate with depth ``parent.lineage_depth + 1``.
            Post-construction clamp still applies (so a decayed bias
            that re-enters the ``[-2.0, 2.0]`` range from a higher
            magnitude is preserved).

        Raises
        ------
        ValueError
            If ``parents`` is empty, or ``decay_factor`` is outside
            ``[0.0, 1.0]``.
        """
        if not parents:
            msg = (
                "TransgenerationalMemory.inherit_from requires at least one "
                "parent substrate; got an empty sequence."
            )
            raise ValueError(msg)
        if not 0.0 <= decay_factor <= 1.0:
            msg = (
                f"TransgenerationalMemory.inherit_from decay_factor must be "
                f"in [0.0, 1.0]; got {decay_factor}."
            )
            raise ValueError(msg)
        parent = parents[0]
        return cls(
            logit_bias=parent.logit_bias * decay_factor,
            lineage_depth=parent.lineage_depth + 1,
            source_genome_id=parent.source_genome_id,
        )


def save(substrate: TransgenerationalMemory, path: Path) -> None:
    """Serialise a substrate to disk via ``torch.save`` at the ``.tei.pt`` path.

    Stores all three fields (``logit_bias`` tensor, ``lineage_depth``
    int, ``source_genome_id`` str) in a dict so :func:`load` can
    reconstruct the dataclass byte-equivalently. The parent
    directory is created if missing (mirrors ``LamarckianInheritance``'s
    capture-side behaviour).

    Parameters
    ----------
    substrate : TransgenerationalMemory
        The substrate to serialise.
    path : Path
        Destination ``.tei.pt`` file. Parent directory created if
        missing. The path-builder
        (``TransgenerationalInheritance.checkpoint_path``) returns
        the canonical ``inheritance/gen-NNN/genome-<gid>.tei.pt``
        layout; callers SHOULD use it to avoid drift between writer
        and reader.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "logit_bias": substrate.logit_bias,
        "lineage_depth": substrate.lineage_depth,
        "source_genome_id": substrate.source_genome_id,
    }
    torch.save(payload, path)


def load(path: Path) -> TransgenerationalMemory:
    """Deserialise a substrate from disk; reconstruct the dataclass.

    The returned instance SHALL be byte-equivalent to the original
    that was saved: ``logit_bias`` tensor equal element-wise,
    ``lineage_depth`` equal, ``source_genome_id`` equal.

    Parameters
    ----------
    path : Path
        Source ``.tei.pt`` file. SHALL exist.

    Returns
    -------
    TransgenerationalMemory
        Reconstructed substrate (post-construction clamp re-applies,
        but a previously-clamped tensor is idempotent under the
        re-clamp).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist. The message includes the missing
        path so the operator can diagnose.
    """
    if not path.exists():
        msg = (
            f"TransgenerationalMemory.load: substrate file not found at "
            f"{path}. Check the F0 substrate extraction pipeline ran "
            f"successfully and the path matches "
            f"TransgenerationalInheritance.checkpoint_path's output."
        )
        raise FileNotFoundError(msg)
    # weights_only=False because the payload is a dict of mixed types
    # (tensor + int + str), not just a tensor state_dict. The .tei.pt
    # files are produced by our own pipeline (never user-supplied), so
    # the unpickle risk is bounded by the same trust assumption that
    # applies to the rest of the project's .pt artefacts.
    payload = torch.load(path, weights_only=False)
    return TransgenerationalMemory(
        logit_bias=payload["logit_bias"],
        lineage_depth=payload["lineage_depth"],
        source_genome_id=payload["source_genome_id"],
    )


def extract_from_brain(
    brain: object,
    env: object,
    probe_positions: Sequence[tuple[int, int]],
    rng_seed: int,
) -> TransgenerationalMemory:
    """Telemetry-pass placeholder. Functional implementation lands with the F0 pipeline.

    The functional implementation requires env coupling (a pathogen
    lawn at a known position, a probe-position generator, and a
    brain-policy adapter that extracts action distributions from
    actor logits) which belongs to the F0 Substrate Extraction
    Pipeline in the evolution loop. This stub preserves the public
    API surface so downstream callers can import the symbol today;
    the loop integration commit will replace the body.

    See the OpenSpec change's "F0 Telemetry-Pass Extraction" and
    "F0 Substrate Extraction Pipeline" requirements for the
    contract this function will satisfy.

    Raises
    ------
    NotImplementedError
        Always. Use the F0 Substrate Extraction Pipeline (loop-side
        integration, a follow-up commit) instead.
    """
    msg = (
        "TransgenerationalMemory.extract_from_brain is a placeholder; the "
        "functional implementation requires env coupling and lands with the "
        "F0 Substrate Extraction Pipeline in a follow-up commit. See "
        "openspec/changes/add-transgenerational-memory/."
    )
    raise NotImplementedError(msg)
