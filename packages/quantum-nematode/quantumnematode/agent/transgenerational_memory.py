"""Transgenerational memory substrate — inheritable behavioural-bias dataclass.

Provides :class:`TransgenerationalMemory`, the heritable substrate
that the ``TransgenerationalInheritance`` strategy threads from the
F0 elite through F1/F2/F3 generations.

Two substrate forms, selected via the optional ``bias_network``
field:

- **Constant logit-bias**. When ``bias_network is None`` the
  substrate's effective bias is the per-action constant
  ``logit_bias`` tensor of shape ``(num_actions,)``.
- **Sensory-conditional bias-network**. When ``bias_network`` is set
  to a ``torch.nn.Sequential``, the effective bias is computed per
  step as ``bias_network(sensory_input)`` where ``sensory_input`` is
  a 1-D tensor of features named in ``input_features``. This lets
  the substrate express *context-conditional* biases (avoid when
  near a pathogen gradient, forage otherwise) which the constant
  form structurally cannot.

Both forms clamp the effective bias at ``|x| ≤ LOGIT_BIAS_CLAMP``
(Boltzmann ratio cap ≈ 7.4x). The dataclass is ``frozen=True`` so
cross-generation aliasing cannot mutate an ancestor's substrate;
``__post_init__`` validates inputs and deep-copies (for the
bias-network) or clamp-clones (for the logit_bias tensor) via
``object.__setattr__`` (the canonical Python pattern for
frozen-dataclass post-init mutation).

The ``inherit_from`` cascade supports three decay shapes
(geometric / linear / sigmoid) configured at the strategy layer.
The cascade scales every weight tensor in the bias-network (or the
constant logit_bias tensor, when in that form) by the decay
schedule's per-generation factor.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


# Maximum absolute element-wise magnitude of the effective substrate
# bias (whether produced by the constant ``logit_bias`` or by
# ``bias_network(sensory_input)``). Caps the Boltzmann ratio at
# ``e^6 ≈ 403x`` so a strong bias can dominate fresh-init policy
# noise at F1+ (K=0). Kaletsky F2 ≈ 0.55 choice index corresponds
# to a ~3x action-probability tilt; a tighter cap saturates
# FORWARD/STAY and blunts substrate signal at F1+ even when the
# bias_network produces stronger raw outputs. The cap was raised
# from 2.0 (e^2 ≈ 7.4x) to 6.0 after empirical evidence of
# saturation as the mechanical bottleneck. 6.0 still prevents
# pathological deterministic policies (a 403x tilt is high but
# not infinity).
LOGIT_BIAS_CLAMP: float = 6.0

DecayShape = Literal["geometric", "linear", "sigmoid"]
_SUPPORTED_DECAY_SHAPES: tuple[str, ...] = ("geometric", "linear", "sigmoid")
_SIGMOID_DECAY_K: float = 2.0
_SIGMOID_DECAY_MIDPOINT: float = 1.0
_SUFFIX_SIN = "_sin"
_SUFFIX_COS = "_cos"


def _resolve_feature_value(params: object, feature_name: str) -> float:
    """Read one feature from a ``BrainParams`` instance, handling ``_sin``/``_cos`` suffixes.

    ``feature_name`` either names a raw field on ``params`` (e.g.
    ``"predator_gradient_strength"``) or names a derived transform of a
    radian-valued field via the ``_sin`` / ``_cos`` suffix (e.g.
    ``"predator_gradient_direction_sin"`` reads ``predator_gradient_direction``
    and returns ``math.sin`` of it). Missing or ``None`` raw values
    default to ``0.0`` (the same convention the brain's feature
    pipeline uses for absent sensory channels).
    """
    if feature_name.endswith(_SUFFIX_SIN):
        base = feature_name[: -len(_SUFFIX_SIN)]
        raw = getattr(params, base, None)
        return math.sin(float(raw)) if raw is not None else 0.0
    if feature_name.endswith(_SUFFIX_COS):
        base = feature_name[: -len(_SUFFIX_COS)]
        raw = getattr(params, base, None)
        return math.cos(float(raw)) if raw is not None else 0.0
    raw = getattr(params, feature_name, None)
    return float(raw) if raw is not None else 0.0


def build_sensory_input(params: object, input_features: Sequence[str]) -> torch.Tensor:
    """Pack the configured sensory features from a ``BrainParams`` into a 1-D tensor.

    The order in ``input_features`` is preserved. ``_sin``/``_cos``
    suffixes are resolved via :func:`_resolve_feature_value`. The
    returned tensor is float32 (matching ``LOGIT_BIAS_CLAMP``'s dtype)
    and lives on CPU; callers move it to the brain's device if needed.
    """
    values = [_resolve_feature_value(params, name) for name in input_features]
    return torch.tensor(values, dtype=torch.float32)


@dataclass(frozen=True)
class TransgenerationalMemory:
    """Inheritable behavioural-bias substrate carried across generations.

    Stores either a per-action constant logit bias or a sensory-
    conditional parametric bias-network. The substrate is extracted
    from an F0 elite policy via :func:`extract_from_brain` and
    multiplicatively decayed across F1/F2/F3 generations via
    :meth:`inherit_from`.

    Parameters
    ----------
    logit_bias : torch.Tensor
        1-D tensor of shape ``(num_actions,)`` and dtype ``float32``.
        Used as the effective bias when ``bias_network is None``;
        used only as a placeholder / fallback dtype anchor when
        ``bias_network`` is set. Each element SHALL satisfy
        ``|x| ≤ LOGIT_BIAS_CLAMP`` after construction; values are
        clamped automatically in ``__post_init__`` and the input
        tensor is NOT mutated in place.
    lineage_depth : int
        Inheritance generation (0 for F0, 1 for F1, etc.). Incremented
        by 1 at each :meth:`inherit_from` call. SHALL be non-negative.
    source_genome_id : str
        F0 elite genome ID — the provenance anchor for the cascade.
        Inherited unchanged across all generations.
    bias_network : torch.nn.Sequential | None
        Optional sensory-conditional parametric bias-network. When
        set, :meth:`apply_to_logits` returns
        ``logits + clamp(bias_network(sensory_input))``; when ``None``,
        falls back to the constant ``logits + logit_bias`` path.
        The caller-passed module is ``copy.deepcopy``'d in
        ``__post_init__`` so caller-side mutations cannot bleed across
        generations (mirrors the ``logit_bias.detach().clone()``
        clamp-clone pattern). Default ``None`` selects the constant
        logit-bias form.
    input_features : tuple[str, ...]
        Names of the sensory features the ``bias_network`` consumes,
        in the order it expects them. Each entry is either a raw
        ``BrainParams`` field name OR a derived transform via the
        ``_sin`` / ``_cos`` suffix (e.g.
        ``"predator_gradient_direction_sin"`` reads
        ``predator_gradient_direction`` and applies ``math.sin``).
        Persisted in the ``.tei.pt`` payload so F1+ workers can
        validate sensory-input shape at load time. Default empty
        tuple is valid only when ``bias_network is None``.

    Raises
    ------
    ValueError
        If ``logit_bias.ndim != 1``, ``logit_bias.dtype != torch.float32``,
        ``lineage_depth < 0``, or ``bias_network is not None`` paired
        with an empty ``input_features`` (the bias-network needs
        named inputs to evaluate against a ``BrainParams``).

    Notes
    -----
    ``frozen=True`` prevents cross-generation aliasing mutation. The
    deep-copy + clamp passes inside ``__post_init__`` use
    ``object.__setattr__`` (the canonical Python pattern for
    frozen-dataclass post-init mutation, since frozen dataclasses
    cannot reassign fields directly). The original caller-provided
    inputs are never mutated.
    """

    logit_bias: torch.Tensor
    lineage_depth: int
    source_genome_id: str
    bias_network: nn.Sequential | None = None
    input_features: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate inputs; clamp ``logit_bias`` and deep-copy ``bias_network``.

        Runs after dataclass field assignment. Raises ``ValueError`` on
        invalid inputs. Clones+clamps the ``logit_bias`` tensor;
        ``copy.deepcopy``'s the ``bias_network`` (when set) so caller-
        side mutations to the source module cannot bleed into the
        stored substrate. Reassigns via ``object.__setattr__`` because
        frozen dataclasses cannot reassign fields directly.
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
        if self.bias_network is not None and not self.input_features:
            msg = (
                "TransgenerationalMemory.bias_network is set but "
                "input_features is empty. The bias-network needs named "
                "BrainParams fields (or _sin / _cos derived transforms) "
                "to evaluate against per-step sensory input; pass a "
                "non-empty tuple matching the network's input-layer width."
            )
            raise ValueError(msg)
        # Clone-then-clamp the logit_bias: the input tensor is NEVER
        # mutated. clone() detaches autograd history too — the
        # substrate is non-trainable additive state, not a gradient-
        # flowing parameter.
        clamped = self.logit_bias.detach().clone().clamp_(-LOGIT_BIAS_CLAMP, LOGIT_BIAS_CLAMP)
        object.__setattr__(self, "logit_bias", clamped)
        # Deep-copy the bias_network so caller-side mutations to the
        # source module cannot bleed across generations. Detach all
        # parameters from autograd because the substrate is frozen
        # non-trainable state.
        if self.bias_network is not None:
            cloned_net = copy.deepcopy(self.bias_network)
            for param in cloned_net.parameters():
                param.requires_grad_(False)  # noqa: FBT003 - PyTorch's in-place API takes a positional bool
            object.__setattr__(self, "bias_network", cloned_net)
        # Normalise input_features to a tuple (tolerates list input).
        object.__setattr__(self, "input_features", tuple(self.input_features))

    def apply_to_logits(
        self,
        logits: torch.Tensor,
        sensory_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``logits + effective_bias`` without in-place mutation.

        Two paths:

        - When ``bias_network is None`` (constant form): returns
          ``logits + self.logit_bias``. The ``sensory_input`` argument
          is ignored — supplied for signature compatibility with the
          sensory-conditional form. Broadcasts over leading batch /
          sequence dimensions.
        - When ``bias_network`` is set (sensory-conditional form):
          returns ``logits + clamp(bias_network(sensory_input))`` where
          the clamp is element-wise to ``[-LOGIT_BIAS_CLAMP, +LOGIT_BIAS_CLAMP]``.
          ``sensory_input`` SHALL be a 1-D tensor matching the
          bias-network's expected input width (typically
          ``len(self.input_features)``). The returned tensor is a
          distinct object — neither input is mutated.

        Parameters
        ----------
        logits : torch.Tensor
            Actor logits of shape ``(..., num_actions)``. The trailing
            dimension SHALL match the bias's first dimension.
        sensory_input : torch.Tensor | None
            1-D tensor of sensory features in the order specified by
            ``self.input_features``. Required when ``bias_network`` is
            set; ignored when ``None``. Build it with
            :func:`build_sensory_input`.

        Returns
        -------
        torch.Tensor
            New tensor of the same shape, dtype, and device as
            ``logits``, with the effective bias added element-wise.

        Raises
        ------
        ValueError
            When ``bias_network`` is set and ``sensory_input is None``
            (the bias-network path is non-degenerate; a missing
            sensory_input is a programming bug, not a recoverable
            condition).
        """
        if self.bias_network is None:
            # Constant-form path: `+` is non-in-place on tensors.
            return logits + self.logit_bias
        if sensory_input is None:
            msg = (
                "TransgenerationalMemory.apply_to_logits: bias_network is set "
                "but sensory_input is None. The sensory-conditional substrate "
                "needs a per-step input tensor. Build it with "
                "transgenerational_memory.build_sensory_input(brain_params, "
                "self.input_features) at the runner call site."
            )
            raise ValueError(msg)
        # eval() ensures dropout/batchnorm (if ever added) are inactive;
        # parameters are already requires_grad=False (set in __post_init__),
        # so a torch.no_grad() context isn't strictly needed but is cheap
        # insurance against gradient leakage from upstream training code.
        with torch.no_grad():
            raw_bias = self.bias_network(sensory_input.to(logits.device))
        clamped_bias = raw_bias.clamp(-LOGIT_BIAS_CLAMP, LOGIT_BIAS_CLAMP)
        return logits + clamped_bias

    @classmethod
    def inherit_from(
        cls,
        parents: Sequence[TransgenerationalMemory],
        decay_factor: float,
        decay_shape: DecayShape = "geometric",
    ) -> TransgenerationalMemory:
        """Produce a child substrate decayed from a single parent (top-1 elite).

        The cascade scales every weight tensor in the substrate by a
        per-generation factor determined by ``decay_shape``:

        - ``geometric`` (default): factor = ``decay_factor`` every
          generation. Cumulative effect at child depth ``d``:
          ``decay_factor ** d``.
        - ``linear``: factor at child depth ``d`` is
          ``max(0, 1 - d * (1 - decay_factor))``. The cumulative
          factor reaches zero at ``d = 1 / (1 - decay_factor)``;
          downstream generations carry a zero substrate.
        - ``sigmoid``: cumulative scale at depth ``d`` is
          ``sigmoid(K * (M - d))`` with fixed ``K=2``, ``M=1`` —
          **explicitly independent of `decay_factor`**. Slow-then-fast
          schedule with ``cum(0)≈0.881``, ``cum(1)=0.500``,
          ``cum(2)≈0.119``, ``cum(3)≈0.018``. Reserved as a
          sensitivity-analysis alternative; calibration by
          ``decay_factor`` only applies to ``geometric`` and ``linear``.

        For non-geometric shapes, the per-generation factor depends on
        the *cumulative* depth of the child (``parent.lineage_depth + 1``),
        so the resulting bias scales monotonically with depth.

        Both substrate forms (constant ``logit_bias`` and
        ``bias_network``) are decayed in lockstep — the constant
        tensor is scaled element-wise, and the bias-network's parameters
        + buffers are each scaled by the same per-generation factor.

        Parameters
        ----------
        parents : Sequence[TransgenerationalMemory]
            At least one parent substrate. Single-elite rule uses
            ``parents[0]``.
        decay_factor : float
            Decay strength in ``[0.0, 1.0]``; values outside raise
            ``ValueError``. Anchors all three decay shapes.
        decay_shape : Literal["geometric", "linear", "sigmoid"]
            Selects the per-generation factor formula. Default
            ``"geometric"``.

        Returns
        -------
        TransgenerationalMemory
            Child substrate with depth ``parent.lineage_depth + 1``.
            Post-construction clamp still applies.

        Raises
        ------
        ValueError
            If ``parents`` is empty, ``decay_factor`` is outside
            ``[0.0, 1.0]``, or ``decay_shape`` is not a supported value.
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
        if decay_shape not in _SUPPORTED_DECAY_SHAPES:
            msg = (
                f"TransgenerationalMemory.inherit_from decay_shape must be "
                f"one of {_SUPPORTED_DECAY_SHAPES}; got {decay_shape!r}."
            )
            raise ValueError(msg)
        parent = parents[0]
        child_depth = parent.lineage_depth + 1
        # Cumulative scale factor applied to *parent* state to produce
        # *child* state. For geometric this is just ``decay_factor``;
        # for linear/sigmoid the schedule depends on the child's depth
        # in the cascade so each generation correctly tracks total decay.
        scale = _resolve_decay_scale(decay_shape, decay_factor, child_depth, parent.lineage_depth)
        # Scale the constant logit_bias (used at apply-time only when
        # bias_network is None; under the bias-network path it's a
        # legacy artefact preserved for round-trip compatibility).
        scaled_logit_bias = parent.logit_bias * scale
        # Construct the child substrate; ``__post_init__`` deep-copies
        # the ``bias_network`` so the parent's module is never mutated
        # — we don't need to deep-copy here. Then scale the child's
        # (already-cloned) bias_network in-place so we only pay one
        # deep-copy per generation step (the prior double-deep-copy
        # was a latency leak).
        child = cls(
            logit_bias=scaled_logit_bias,
            lineage_depth=child_depth,
            source_genome_id=parent.source_genome_id,
            bias_network=parent.bias_network,
            input_features=parent.input_features,
        )
        if child.bias_network is not None:
            with torch.no_grad():
                for tensor in child.bias_network.parameters():
                    tensor.mul_(scale)
                for buffer in child.bias_network.buffers():
                    buffer.mul_(scale)
        return child


def _resolve_decay_scale(
    decay_shape: DecayShape,
    decay_factor: float,
    child_depth: int,
    parent_depth: int,
) -> float:
    """Return the per-step decay scale to apply to parent state.

    Computes the *delta* (parent → child) cumulative factor so that
    applying it once to parent state yields the correct child state:

    - geometric: child_cumulative / parent_cumulative
      = decay_factor**child_depth / decay_factor**parent_depth
      = decay_factor
    - linear: scale = max(0, 1 - child_depth * (1-decay_factor))
              / max(eps, 1 - parent_depth * (1-decay_factor))
    - sigmoid: scale = sigmoid(K * (mid - child_depth))
               / sigmoid(K * (mid - parent_depth))

    Mathematically all three forms preserve the invariant
    ``child_state == cumulative_factor(child_depth) * F0_state``.
    """
    if decay_shape == "geometric":
        return decay_factor
    eps = 1e-12
    if decay_shape == "linear":
        slope = 1.0 - decay_factor
        child_cum = max(0.0, 1.0 - child_depth * slope)
        parent_cum = max(eps, 1.0 - parent_depth * slope)
        return child_cum / parent_cum
    # sigmoid
    child_cum = 1.0 / (1.0 + math.exp(_SIGMOID_DECAY_K * (child_depth - _SIGMOID_DECAY_MIDPOINT)))
    parent_cum = 1.0 / (1.0 + math.exp(_SIGMOID_DECAY_K * (parent_depth - _SIGMOID_DECAY_MIDPOINT)))
    if parent_cum < eps:
        return 0.0
    return child_cum / parent_cum


def _build_bias_network_from_spec(spec: dict) -> nn.Sequential:
    """Reconstruct an ``nn.Sequential`` bias-network from a serialised spec.

    The persisted spec is a small dict — input/hidden/output dims +
    activation name + the saved ``state_dict``. We rebuild the module
    architecture, then load the state_dict so the weights are
    byte-equivalent to the saved substrate. Mirrors the rebuild
    pattern in :class:`LamarckianInheritance` for trained weights.
    """
    input_dim = int(spec["input_dim"])
    hidden_dim = int(spec["hidden_dim"])
    output_dim = int(spec["output_dim"])
    activation = str(spec["activation"])
    layers: list[nn.Module] = []
    if hidden_dim <= 0:
        # Linear projection only (closed-form least-squares path).
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(_resolve_activation(activation))
        layers.append(nn.Linear(hidden_dim, output_dim))
    net = nn.Sequential(*layers)
    net.load_state_dict(spec["state_dict"])
    return net


def _resolve_activation(name: str) -> nn.Module:
    """Map an activation name to an ``nn.Module``.

    Supported names match the Pydantic schema's
    ``bias_network.activation`` Literal: ``tanh / relu / gelu``.
    """
    table: dict[str, type[nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
    }
    if name not in table:
        msg = f"Unknown bias_network.activation {name!r}; expected one of {sorted(table)}."
        raise ValueError(msg)
    return table[name]()


def _describe_bias_network(net: nn.Sequential) -> dict:
    """Serialise a bias-network ``nn.Sequential`` into a save-friendly spec.

    Records architecture metadata (dims + activation) alongside the
    state_dict so :func:`_build_bias_network_from_spec` can rebuild
    the module on load. The architecture introspection assumes the
    canonical shapes (linear-only or linear-act-linear); a future
    capacity bump would extend this descriptor.
    """
    linears = [m for m in net if isinstance(m, nn.Linear)]
    activations = [m for m in net if not isinstance(m, nn.Linear)]
    if not linears:
        msg = "bias_network must contain at least one nn.Linear layer; got empty Sequential."
        raise ValueError(msg)
    input_dim = linears[0].in_features
    output_dim = linears[-1].out_features
    if len(linears) == 1:
        hidden_dim = 0
        activation_name = "tanh"  # not actually used; recorded for completeness
    else:
        hidden_dim = linears[0].out_features
        if not activations:
            msg = (
                "bias_network with hidden_dim > 0 must include an activation between linears; "
                "got a Sequential without an activation module."
            )
            raise ValueError(msg)
        activation_name = type(activations[0]).__name__.lower()
        if activation_name not in {"tanh", "relu", "gelu"}:
            msg = (
                f"Unsupported bias_network activation {type(activations[0]).__name__}; "
                f"expected Tanh / ReLU / GELU."
            )
            raise ValueError(msg)
    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "activation": activation_name,
        "state_dict": net.state_dict(),
    }


def save(substrate: TransgenerationalMemory, path: Path) -> None:
    """Serialise a substrate to disk via ``torch.save`` at the ``.tei.pt`` path.

    The payload is a dict with the substrate's fields. When
    ``bias_network`` is set, the payload additionally includes a
    ``bias_network_spec`` dict describing the architecture + the
    state_dict (see :func:`_describe_bias_network`). The
    ``input_features`` tuple is always recorded (empty when the
    legacy ``logit_bias`` path is in use).

    The parent directory is created if missing (mirrors
    ``LamarckianInheritance``'s capture-side behaviour).

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
    payload: dict[str, object] = {
        "logit_bias": substrate.logit_bias,
        "lineage_depth": substrate.lineage_depth,
        "source_genome_id": substrate.source_genome_id,
        "input_features": list(substrate.input_features),
    }
    if substrate.bias_network is not None:
        payload["bias_network_spec"] = _describe_bias_network(substrate.bias_network)
    torch.save(payload, path)


def load(path: Path) -> TransgenerationalMemory:
    """Deserialise a substrate from disk; reconstruct the dataclass.

    The returned instance SHALL be byte-equivalent to the original
    that was saved. The ``bias_network`` is rebuilt from the
    persisted spec when present; ``input_features`` is restored as
    a tuple.

    Parameters
    ----------
    path : Path
        Source ``.tei.pt`` file. SHALL exist.

    Returns
    -------
    TransgenerationalMemory
        Reconstructed substrate (post-construction clamp + deep-copy
        re-apply, but a previously-clamped tensor / module is
        idempotent under those operations).

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
    # (tensor + int + str + optional state_dict). The .tei.pt files
    # are produced by our own pipeline (never user-supplied), so the
    # unpickle risk is bounded by the same trust assumption that
    # applies to the rest of the project's .pt artefacts.
    payload = torch.load(path, weights_only=False)
    bias_network: nn.Sequential | None = None
    spec = payload.get("bias_network_spec")
    if spec is not None:
        bias_network = _build_bias_network_from_spec(spec)
    input_features = tuple(payload.get("input_features", ()))
    return TransgenerationalMemory(
        logit_bias=payload["logit_bias"],
        lineage_depth=payload["lineage_depth"],
        source_genome_id=payload["source_genome_id"],
        bias_network=bias_network,
        input_features=input_features,
    )


def extract_from_brain(  # noqa: PLR0913 - the bias-network fit knobs are deliberately kwargs
    brain: object,
    probe_params: Sequence[object],
    rng_seed: int,
    source_genome_id: str,
    *,
    bias_network_spec: dict | None = None,
    input_features: Sequence[str] = (),
    fit_epochs: int = 50,
    fit_lr: float = 0.05,
    samples_per_probe: int = 10,
) -> TransgenerationalMemory:
    """Telemetry-pass: extract the F0 elite's behavioural-bias substrate.

    Two paths, selected by ``bias_network_spec``:

    - **Constant logit-bias** (``bias_network_spec is None``):
      probes the trained F0 brain over ``probe_params``, accumulates
      empirical sampled-action counts, and returns a substrate whose
      ``logit_bias`` is the log-deviation from uniform.
    - **Sensory-conditional bias-network** (``bias_network_spec``
      provided): probes the brain ``samples_per_probe`` times per
      ``probe_params`` entry under the configured seed, fits the
      bias-network MLP against ``(sensory_input -> logit_offset_from_uniform)``
      with Adam over ``fit_epochs`` epochs (or closed-form
      least-squares when the spec's ``hidden_dim == 0``), and returns
      a substrate carrying the fitted MLP. The ``logit_bias``
      tensor in the sensory-conditional path is the per-probe-averaged
      offset (an unused artefact preserved for round-trip + diagnostics).

    The deterministic-on-seed contract is satisfied because (a) the
    brain's RNG is reseeded via ``prepare_episode()`` (zeroes the
    LSTM hidden state) before probing, (b) the probe params are
    passed in deterministically, (c) ``rng_seed`` seeds internal
    sampling, and (d) the MLP fit uses ``torch.manual_seed(rng_seed)``
    before the optimiser is constructed.

    Parameters
    ----------
    brain : object
        Trained brain. Duck-type contract: ``prepare_episode``,
        ``run_brain``, ``action_set``, ``num_actions``, optional
        ``rng``.
    probe_params : Sequence[object]
        Deterministic sequence of ``BrainParams`` instances. Caller
        (the F0 substrate-extraction pipeline) constructs these from
        the env-derived probe ring.
    rng_seed : int
        Seed for deterministic action sampling + MLP fit. Same seed
        always produces the same substrate.
    source_genome_id : str
        Provenance anchor for the cascade — the F0 elite's
        ``genome_id``.
    bias_network_spec : dict | None
        When set, must contain ``input_dim`` (= ``len(input_features)``),
        ``hidden_dim``, ``output_dim`` (= ``brain.num_actions``), and
        ``activation`` (Literal["tanh", "relu", "gelu"]). Triggers the
        sensory-conditional path. When ``None``, the constant
        logit-bias path runs.
    input_features : Sequence[str]
        Names of the ``BrainParams`` features (incl. ``_sin`` / ``_cos``
        suffixes) consumed by the bias-network. Required when
        ``bias_network_spec`` is set; ignored when ``None``.
    fit_epochs : int
        Adam epochs for the MLP fit. Ignored under the linear-projection
        (``hidden_dim == 0``) or legacy paths.
    fit_lr : float
        Adam learning rate for the MLP fit.
    samples_per_probe : int
        Per-probe action samples for the empirical distribution.
        Under the constant logit-bias path this is implicitly 1; the
        sensory-conditional path defaults to 10 to reduce variance
        in the MLP fit target.

    Returns
    -------
    TransgenerationalMemory
        Substrate with ``lineage_depth=0`` and the configured
        substrate form populated.
    """
    if not probe_params:
        msg = (
            "extract_from_brain requires at least one probe_params element; got an empty sequence."
        )
        raise ValueError(msg)
    if bias_network_spec is not None and not input_features:
        msg = (
            "extract_from_brain: bias_network_spec is set but input_features is empty. "
            "The MLP needs named BrainParams fields to read sensory state at each probe."
        )
        raise ValueError(msg)

    brain.prepare_episode()  # type: ignore[attr-defined]
    if hasattr(brain, "rng"):
        from quantumnematode.utils.seeding import get_rng

        brain.rng = get_rng(rng_seed)  # type: ignore[attr-defined]

    num_actions = brain.num_actions  # type: ignore[attr-defined]
    eps = 1e-8

    # Probe the brain. Constant-form path: one sample per probe →
    # action_counts. Sensory-conditional path: samples_per_probe
    # samples per probe → per-probe empirical distribution recorded
    # alongside the probe params for the MLP fit.
    #
    # Hidden-state reset between probes. The LSTM/GRU recurrent state
    # updates on every ``run_brain`` call, so the policy at probe N
    # depends on the trajectory through probes 0..N-1. At F1+ runtime
    # the substrate is queried per-step with no shared recurrent
    # context across the probe ring — the fit target distribution and
    # the deployment input distribution must match. Reset
    # ``prepare_episode()`` between probes so each probe is sampled
    # against a stateless recurrent context, mirroring deployment.
    n_samples = samples_per_probe if bias_network_spec is not None else 1
    per_probe_counts: list[torch.Tensor] = []
    for params in probe_params:
        counts = torch.zeros(num_actions, dtype=torch.float32)
        for _ in range(n_samples):
            brain.prepare_episode()  # type: ignore[attr-defined]
            actions = brain.run_brain(  # type: ignore[attr-defined]
                params,
                reward=None,
                input_data=None,
                top_only=False,
                top_randomize=False,
            )
            sampled_action = actions[0].action
            action_idx = brain.action_set.index(sampled_action)  # type: ignore[attr-defined]
            counts[action_idx] += 1
        per_probe_counts.append(counts / float(n_samples))

    # Stack per-probe empirical distributions → (num_probes, num_actions).
    probs = torch.stack(per_probe_counts)
    uniform_log = float(torch.log(torch.tensor(1.0 / num_actions)))
    # logit_offset per probe: log(probs + eps) - log(1 / num_actions).
    logit_offsets = torch.log(probs + eps) - uniform_log

    if bias_network_spec is None:
        # Constant-form path: average across probes; the substrate
        # carries the per-action constant tensor; bias_network=None.
        mean_logit_bias = logit_offsets.mean(dim=0)
        return TransgenerationalMemory(
            logit_bias=mean_logit_bias,
            lineage_depth=0,
            source_genome_id=source_genome_id,
            bias_network=None,
            input_features=(),
        )

    # Sensory-conditional path: build the MLP, fit it against
    # (sensory_input -> logit_offset_from_uniform).
    # Seed BEFORE constructing the network so its initial parameter
    # init is deterministic on ``rng_seed`` (Linear's default init uses
    # the global RNG); the seed is re-applied before the Adam optimiser
    # is built inside ``_fit_bias_network`` for the same reason.
    torch.manual_seed(rng_seed)
    bias_network = _build_empty_bias_network(
        input_dim=int(bias_network_spec["input_dim"]),
        hidden_dim=int(bias_network_spec["hidden_dim"]),
        output_dim=int(bias_network_spec["output_dim"]),
        activation=str(bias_network_spec["activation"]),
    )
    sensory_matrix = torch.stack(
        [build_sensory_input(params, list(input_features)) for params in probe_params],
    )
    _fit_bias_network(
        bias_network,
        sensory_matrix=sensory_matrix,
        target_offsets=logit_offsets,
        rng_seed=rng_seed,
        fit_epochs=fit_epochs,
        fit_lr=fit_lr,
    )
    # Legacy artefact: also record the probe-averaged logit_offset as
    # logit_bias so the saved substrate has a meaningful fallback if
    # ever loaded back without bias_network_spec by mistake.
    mean_logit_bias = logit_offsets.mean(dim=0)
    return TransgenerationalMemory(
        logit_bias=mean_logit_bias,
        lineage_depth=0,
        source_genome_id=source_genome_id,
        bias_network=bias_network,
        input_features=tuple(input_features),
    )


def _build_empty_bias_network(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    """Construct an un-fit ``nn.Sequential`` matching the configured architecture."""
    if hidden_dim <= 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        _resolve_activation(activation),
        nn.Linear(hidden_dim, output_dim),
    )


def _fit_bias_network(  # noqa: PLR0913 - fit knobs are deliberately explicit kwargs
    bias_network: nn.Sequential,
    *,
    sensory_matrix: torch.Tensor,
    target_offsets: torch.Tensor,
    rng_seed: int,
    fit_epochs: int,
    fit_lr: float,
) -> None:
    """Fit the bias-network in-place against per-probe sensory → logit-offset pairs.

    Linear-projection (single ``nn.Linear``) uses closed-form least-
    squares via ``torch.linalg.lstsq`` — deterministic on inputs,
    no seed-dependent optimisation.

    MLPs use Adam with ``torch.manual_seed(rng_seed)`` before
    construction. Loss is MSE between predicted and empirical
    log-offsets. ``fit_epochs`` is the budget; an early-exit on
    plateau is intentionally absent (the substrate-diversity
    tripwire T2 prefers identical compute spend across calibration
    seeds so the per-pair CoV reflects genuine policy difference,
    not stochastic optimiser stopping).
    """
    linears = [m for m in bias_network if isinstance(m, nn.Linear)]
    if len(linears) == 1:
        # Closed-form: y = X W^T + b → solve [X | 1] [W, b]^T = y.
        x_aug = torch.cat([sensory_matrix, torch.ones(sensory_matrix.shape[0], 1)], dim=1)
        solution = torch.linalg.lstsq(x_aug, target_offsets).solution
        weight = solution[:-1].T  # (out, in)
        bias = solution[-1]
        with torch.no_grad():
            linears[0].weight.copy_(weight)
            linears[0].bias.copy_(bias)
        return
    torch.manual_seed(rng_seed)
    optimizer = torch.optim.Adam(bias_network.parameters(), lr=fit_lr)
    loss_fn = nn.MSELoss()
    for _ in range(fit_epochs):
        optimizer.zero_grad()
        predicted = bias_network(sensory_matrix)
        loss = loss_fn(predicted, target_offsets)
        loss.backward()
        optimizer.step()
