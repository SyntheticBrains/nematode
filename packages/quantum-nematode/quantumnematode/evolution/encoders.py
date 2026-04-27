"""Genome encoder protocol and concrete implementations for the evolution loop.

The encoder is the boundary between the optimiser (which sees flat numpy
parameter arrays) and the brain (which has structured weight tensors).  Each
encoder owns the round-trip:

- ``initial_genome(sim_config, *, rng)`` — build a fresh brain, serialize its
  weights into a ``Genome``.
- ``decode(genome, sim_config, *, seed=None)`` — instantiate a fresh brain
  via :func:`evolution.brain_factory.instantiate_brain_from_sim_config`, then
  apply the genome's weights via the brain's ``WeightPersistence`` protocol.
- ``genome_dim(sim_config)`` — the number of float parameters in a genome
  for the given config.  Constructs a brain to count, so call once at startup.

Component selection is **dynamic**: encoders ask the brain for all weight
components and filter out the denylist :data:`NON_GENOME_COMPONENTS`.  This
picks up conditional components like MLPPPO's ``gate_weights`` automatically
and survives future component additions without encoder changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np
import torch

from quantumnematode.brain.weights import WeightPersistence
from quantumnematode.evolution.brain_factory import instantiate_brain_from_sim_config
from quantumnematode.evolution.genome import Genome

if TYPE_CHECKING:
    from quantumnematode.brain.arch._brain import Brain
    from quantumnematode.brain.weights import WeightComponent
    from quantumnematode.utils.config_loader import SimulationConfig


# Components that are NOT part of the genome.  Optimizer state belongs to
# the training process; episode counters belong to runtime state.  Encoders
# discover all weight components dynamically and filter this set out.
NON_GENOME_COMPONENTS: frozenset[str] = frozenset(
    {"optimizer", "actor_optimizer", "critic_optimizer", "training_state"},
)


def _as_weight_persistence(brain: Brain) -> WeightPersistence:
    """Narrow ``Brain`` to ``WeightPersistence`` with a clear error.

    The general :class:`~quantumnematode.brain.arch._brain.Brain` protocol does
    not require ``WeightPersistence``, but every brain registered in
    ``ENCODER_REGISTRY`` must implement it.  ``cast`` is used after a runtime
    isinstance check so static type checkers see the narrower interface.
    """
    if not isinstance(brain, WeightPersistence):
        msg = (
            f"Brain {type(brain).__name__} does not implement WeightPersistence; "
            "it cannot be used as an evolution target.  Add WeightPersistence "
            "support to the brain class before registering an encoder for it."
        )
        raise TypeError(msg)
    return cast("WeightPersistence", brain)


@runtime_checkable
class GenomeEncoder(Protocol):
    """Protocol for serialising a brain's weights into a flat genome and back."""

    brain_name: str

    def initial_genome(
        self,
        sim_config: SimulationConfig,
        *,
        rng: np.random.Generator,
    ) -> Genome:
        """Build a fresh brain and serialize its weights into a Genome."""
        ...

    def decode(
        self,
        genome: Genome,
        sim_config: SimulationConfig,
        *,
        seed: int | None = None,
    ) -> Brain:
        """Construct a fresh brain and apply the genome's weights to it.

        ``seed`` is forwarded to
        :func:`~quantumnematode.evolution.brain_factory.instantiate_brain_from_sim_config`,
        which patches ``BrainConfig.seed`` so the brain's ``__init__``
        seeds globals to the fitness function's ``seed``.
        """
        ...

    def genome_dim(self, sim_config: SimulationConfig) -> int:
        """Return the number of float parameters in a genome for this brain config.

        Constructs a fresh brain to count parameters; call once at loop
        initialisation, not per-generation.
        """
        ...


# ---------------------------------------------------------------------------
# Flatten / unflatten helpers (private)
# ---------------------------------------------------------------------------


def _flatten_components(
    components: dict[str, WeightComponent],
) -> tuple[np.ndarray, dict[str, dict[str, tuple[tuple[int, ...], str]]]]:
    """Flatten a set of weight components into a single float32 numpy array.

    Walks components in deterministic key-sorted order and, within each
    component, walks tensors in deterministic key-sorted order.  Returns the
    flat array plus a shape map suitable for :func:`_unflatten_components`.

    Parameters
    ----------
    components
        Mapping of component name → ``WeightComponent``.

    Returns
    -------
    tuple
        ``(flat_array, shape_map)`` where ``shape_map`` is
        ``{component_name: {tensor_key: (shape_tuple, dtype_str)}}``.
    """
    pieces: list[np.ndarray] = []
    shape_map: dict[str, dict[str, tuple[tuple[int, ...], str]]] = {}
    for comp_name in sorted(components):
        comp = components[comp_name]
        comp_shapes: dict[str, tuple[tuple[int, ...], str]] = {}
        for tensor_key in sorted(comp.state):
            tensor = comp.state[tensor_key]
            if not isinstance(tensor, torch.Tensor):
                # Skip non-tensor values (e.g. integer counters in
                # training_state).  In practice the denylist already
                # excludes these components, but defend anyway.
                continue
            arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            comp_shapes[tensor_key] = (tuple(tensor.shape), str(tensor.dtype))
            pieces.append(arr.reshape(-1))
        if comp_shapes:
            shape_map[comp_name] = comp_shapes

    flat = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return flat, shape_map


def _unflatten_components(
    params: np.ndarray,
    shape_map: dict[str, dict[str, tuple[tuple[int, ...], str]]],
    template: dict[str, WeightComponent],
) -> dict[str, WeightComponent]:
    """Reconstruct weight components from a flat array and a shape map.

    Reuses ``template`` (the same components freshly extracted from the brain)
    so any non-tensor state (rare but defensive) is preserved.  Tensor values
    are replaced from ``params``, walking in the same deterministic order as
    :func:`_flatten_components`.

    Parameters
    ----------
    params
        Flat array produced by :func:`_flatten_components`.
    shape_map
        Shape/dtype map from the same call.
    template
        Dict of fresh ``WeightComponent`` instances from the brain we are
        decoding into; used to preserve non-tensor sub-state and the
        ``WeightComponent`` envelope.

    Returns
    -------
    dict[str, WeightComponent]
        Components ready to pass into ``brain.load_weight_components``.
    """
    cursor = 0
    out: dict[str, WeightComponent] = {}
    for comp_name in sorted(shape_map):
        comp_shapes = shape_map[comp_name]
        new_state: dict[str, object] = dict(template[comp_name].state)
        for tensor_key in sorted(comp_shapes):
            shape, dtype_str = comp_shapes[tensor_key]
            size = int(np.prod(shape)) if shape else 1
            chunk = params[cursor : cursor + size]
            cursor += size
            torch_dtype = getattr(torch, dtype_str.removeprefix("torch."))
            tensor = torch.from_numpy(np.ascontiguousarray(chunk)).reshape(shape).to(torch_dtype)
            new_state[tensor_key] = tensor
        # Reuse the WeightComponent class via the template.
        comp = template[comp_name]
        out[comp_name] = type(comp)(name=comp.name, state=new_state, metadata=comp.metadata)

    return out


def _select_genome_components(
    components: dict[str, WeightComponent],
) -> dict[str, WeightComponent]:
    """Filter out non-genome components per :data:`NON_GENOME_COMPONENTS`."""
    return {k: v for k, v in components.items() if k not in NON_GENOME_COMPONENTS}


# ---------------------------------------------------------------------------
# Concrete encoders
# ---------------------------------------------------------------------------


class _ClassicalPPOEncoder:
    """Shared encoder logic for MLPPPOBrain and LSTMPPOBrain.

    Both brains have the same weight-component contract (``WeightPersistence``)
    and the same post-load reset requirements (``_episode_count = 0`` then
    ``_update_learning_rate()``).  This base class implements that flow once.
    Concrete subclasses pin ``brain_name``.
    """

    brain_name: str = ""  # subclasses override

    def initial_genome(
        self,
        sim_config: SimulationConfig,
        *,
        rng: np.random.Generator,  # noqa: ARG002 - reserved for future RNG-driven sampling
    ) -> Genome:
        brain = instantiate_brain_from_sim_config(sim_config)
        wp = _as_weight_persistence(brain)
        components = _select_genome_components(wp.get_weight_components())
        params, shape_map = _flatten_components(components)
        return Genome(
            params=params,
            genome_id="",  # filled in by the loop via genome_id_for
            parent_ids=[],
            generation=0,
            birth_metadata={"shape_map": shape_map, "brain_name": self.brain_name},
        )

    def decode(
        self,
        genome: Genome,
        sim_config: SimulationConfig,
        *,
        seed: int | None = None,
    ) -> Brain:
        brain = instantiate_brain_from_sim_config(sim_config, seed=seed)
        wp = _as_weight_persistence(brain)
        # Re-extract components so we have a fresh template (with current shapes
        # for the current sim_config — important if e.g. sensory_modules differ).
        template = _select_genome_components(wp.get_weight_components())
        shape_map = genome.birth_metadata.get("shape_map")
        if shape_map is None:
            # Fallback: derive shape_map from the fresh template.  This handles
            # genomes whose birth_metadata is empty (e.g. produced by an
            # optimiser sampling a flat vector directly).
            _, shape_map = _flatten_components(template)
        components = _unflatten_components(genome.params, shape_map, template)
        wp.load_weight_components(components)

        # Reset runtime state so a freshly-decoded brain matches a freshly-
        # constructed one.  ``_episode_count`` belongs to ``training_state``
        # (deliberately excluded from the genome) but is consulted by the LR
        # scheduler at runtime; without resetting it, a genome captured at
        # episode 800 would inherit a stale count and the LR scheduler would
        # be in the wrong regime.  Calling ``_update_learning_rate()`` after
        # the count reset brings the LR back into sync.  These attributes
        # belong to the concrete brain subclass (MLPPPOBrain / LSTMPPOBrain),
        # not to the general Brain protocol — hence the type: ignore.
        brain._episode_count = 0  # type: ignore[attr-defined]  # noqa: SLF001
        brain._update_learning_rate()  # type: ignore[attr-defined]  # noqa: SLF001

        return brain

    def genome_dim(self, sim_config: SimulationConfig) -> int:
        brain = instantiate_brain_from_sim_config(sim_config)
        wp = _as_weight_persistence(brain)
        components = _select_genome_components(wp.get_weight_components())
        params, _ = _flatten_components(components)
        return int(params.size)


class MLPPPOEncoder(_ClassicalPPOEncoder):
    """Encoder for :class:`~quantumnematode.brain.arch.mlpppo.MLPPPOBrain`.

    Picks up ``policy``, ``value``, and conditional ``gate_weights`` (when
    ``_feature_gating: true`` is configured).  Excludes ``optimizer`` and
    ``training_state`` per :data:`NON_GENOME_COMPONENTS`.
    """

    brain_name = "mlpppo"


class LSTMPPOEncoder(_ClassicalPPOEncoder):
    """Encoder for :class:`~quantumnematode.brain.arch.lstmppo.LSTMPPOBrain`.

    Picks up ``lstm``, ``layer_norm``, ``policy``, ``value`` (all four
    learned-weight components).  Excludes ``actor_optimizer``,
    ``critic_optimizer``, and ``training_state``.  Per-episode hidden state
    (``_pending_h_state``, ``_pending_c_state``) and ``_step_count`` reset in
    ``prepare_episode()`` (called by the runner before each episode), so the
    encoder doesn't handle those.
    """

    brain_name = "lstmppo"


ENCODER_REGISTRY: dict[str, type[GenomeEncoder]] = {
    MLPPPOEncoder.brain_name: MLPPPOEncoder,
    LSTMPPOEncoder.brain_name: LSTMPPOEncoder,
}


def get_encoder(brain_name: str) -> GenomeEncoder:
    """Look up an encoder by brain name; raise a helpful ValueError if missing.

    Parameters
    ----------
    brain_name
        Brain identifier as it appears in YAML (e.g. ``"mlpppo"``).

    Returns
    -------
    GenomeEncoder
        A fresh encoder instance.

    Raises
    ------
    ValueError
        If ``brain_name`` is not registered.  The error message lists the
        registered names.
    """
    encoder_cls = ENCODER_REGISTRY.get(brain_name)
    if encoder_cls is None:
        registered = sorted(ENCODER_REGISTRY)
        msg = (
            f"No encoder for brain {brain_name!r}. Registered: {registered}. "
            "Quantum brains are not currently supported."
        )
        raise ValueError(msg)
    return encoder_cls()
