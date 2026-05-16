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

One telemetry pass:

- :func:`extract_from_brain`: probes the trained F0 brain over a
  deterministic sequence of synthetic ``BrainParams`` representing
  "near-pathogen" sensory states, records the empirical action
  distribution, and returns a substrate whose ``logit_bias`` is the
  log-deviation from uniform. Deterministic on the supplied
  ``rng_seed`` so the F0 extraction is reproducible. Invoked by the
  ``EvolutionLoop``'s F0 Substrate Extraction Pipeline after the
  F0 generation's ``optimizer.tell`` + ``select_parents`` identifies
  the elite.

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
    probe_params: Sequence[object],
    rng_seed: int,
    source_genome_id: str,
) -> TransgenerationalMemory:
    """Telemetry-pass: extract the F0 elite's behavioural-bias substrate.

    Runs the trained F0 brain over a deterministic set of probe
    ``BrainParams`` instances (each representing a sensory state near
    a pathogen lawn), records the resulting action probabilities, and
    returns a ``TransgenerationalMemory`` whose ``logit_bias`` captures
    the policy's deviation from a uniform distribution. The deviation
    is computed in log space: ``logit_bias[i] = log(mean_probs[i]) -
    log(1/num_actions)``. Positive entries indicate actions the F0
    elite has learned to prefer in the probed contexts; negative
    entries indicate actions it has learned to avoid.

    The deterministic-on-seed contract is satisfied because (a) the
    brain's RNG is reseeded via ``prepare_episode()`` semantics
    (LSTMPPO zeroes the LSTM hidden state) before probing, (b) the
    probe params are passed in by the caller as a deterministic
    sequence, and (c) the ``rng_seed`` is used to seed any internal
    sampling (the brain itself uses a seeded RNG).

    Parameters
    ----------
    brain : object
        Trained brain with a ``prepare_episode()`` method and a
        ``run_brain(params, reward, input_data, *, top_only,
        top_randomize) -> list[ActionData]`` method whose ``run_brain``
        records action probabilities accessible via the brain's
        ``latest_data.action_probabilities`` attribute. Typed as
        ``object`` rather than ``Brain`` to avoid importing the
        Protocol here (which would create an import cycle with
        ``agent`` ↔ ``brain`` modules).
    probe_params : Sequence[object]
        A deterministic sequence of ``BrainParams`` instances
        representing the sensory states to probe. The caller (the F0
        Substrate Extraction Pipeline in ``EvolutionLoop``)
        constructs these to reflect "near a pathogen lawn" contexts
        — typically synthetic ``BrainParams`` with non-zero
        ``predator_gradient_strength``. Typed as ``object`` for the
        same import-cycle reason as ``brain``.
    rng_seed : int
        Seed used to ensure deterministic action sampling within
        the brain's ``run_brain`` calls. The same seed always
        produces the same logit_bias for the same brain weights and
        probe sequence.
    source_genome_id : str
        Provenance anchor for the cascade — the F0 elite's
        ``genome_id``. Stored in the returned substrate and inherited
        unchanged across all decay generations.

    Returns
    -------
    TransgenerationalMemory
        Substrate with ``lineage_depth=0``, ``source_genome_id`` set
        to the F0 elite's ID, and ``logit_bias`` capturing the
        averaged action-probability deviation from uniform.
    """
    # Validate inputs BEFORE any side effects on the brain (the
    # subsequent ``prepare_episode`` resets LSTM hidden state and
    # the rng reseed mutates the brain's RNG). Rejecting an empty
    # probe sequence here keeps the brain unchanged on the error
    # path, matching standard defensive-programming convention.
    if not probe_params:
        msg = (
            "extract_from_brain requires at least one probe_params element; got an empty sequence."
        )
        raise ValueError(msg)

    # Side-effect note: this function mutates the brain's internal
    # state (LSTM hidden state via ``prepare_episode``, RNG via
    # the reseed, and — under commit-3's PPO consistency snapshot
    # — the ``_tei_prior_rollout_snapshot_*`` fields). The F0
    # Substrate Extraction Pipeline in ``EvolutionLoop`` constructs
    # a fresh throwaway brain via ``encoder.decode`` + ``load_weights``,
    # so these mutations don't leak. Callers reusing the brain
    # afterward should be aware.

    # Defensive: prepare_episode resets internal recurrent state so
    # the probe sequence starts from a known initial state regardless
    # of any prior rollout history on the brain.
    brain.prepare_episode()  # type: ignore[attr-defined]

    # Reseed the brain's RNG so the probe pass is deterministic on
    # ``rng_seed`` regardless of any prior random sampling.
    if hasattr(brain, "rng"):
        from quantumnematode.utils.seeding import get_rng

        brain.rng = get_rng(rng_seed)  # type: ignore[attr-defined]

    # Probe each position once; accumulate the sampled action's
    # one-hot encoding (the brain doesn't expose action_probs
    # post-run via a stable attribute, so we approximate the mean
    # action distribution from the empirical sampled-action
    # frequency across probes — a Monte-Carlo estimate of the
    # policy's action probabilities at the probed contexts).
    num_actions = brain.num_actions  # type: ignore[attr-defined]
    action_counts = torch.zeros(num_actions, dtype=torch.float32)
    for params in probe_params:
        actions = brain.run_brain(  # type: ignore[attr-defined]
            params,
            reward=None,
            input_data=None,
            top_only=False,
            top_randomize=False,
        )
        # ActionData.action is the sampled enum; map to index via the
        # brain's action_set.
        sampled_action = actions[0].action
        action_idx = brain.action_set.index(sampled_action)  # type: ignore[attr-defined]
        action_counts[action_idx] += 1

    # Convert counts to empirical probabilities and compute the
    # log-deviation from uniform. eps prevents log(0) when an action
    # was never sampled in the probe set.
    eps = 1e-8
    mean_probs = action_counts / float(len(probe_params))
    uniform_log = float(torch.log(torch.tensor(1.0 / num_actions)))
    logit_bias = torch.log(mean_probs + eps) - uniform_log
    # The substrate's __post_init__ will clamp to LOGIT_BIAS_CLAMP.
    return TransgenerationalMemory(
        logit_bias=logit_bias,
        lineage_depth=0,
        source_genome_id=source_genome_id,
    )
