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

from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import numpy as np
import torch

from quantumnematode.brain.weights import WeightPersistence
from quantumnematode.evolution.brain_factory import instantiate_brain_from_sim_config
from quantumnematode.evolution.genome import Genome

if TYPE_CHECKING:
    from quantumnematode.brain.arch._brain import Brain
    from quantumnematode.brain.weights import WeightComponent
    from quantumnematode.utils.config_loader import ParamSchemaEntry, SimulationConfig


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

    def genome_stds(self, sim_config: SimulationConfig) -> list[float] | None:
        """Return per-parameter standard deviations for the optimiser, or None.

        When None, the optimiser uses a uniform sigma across all
        dimensions — appropriate when genome dimensions cluster around
        zero with similar variance (e.g. neural network weights).

        When the dimensions live on materially different scales (e.g.
        mixed hyperparameter schemas with log-scale floats, tight
        linear floats, and ints), encoders SHOULD return per-parameter
        stds so the optimiser explores each dimension proportionally
        to its bound range.
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

    def genome_stds(
        self,
        sim_config: SimulationConfig,  # noqa: ARG002
    ) -> list[float] | None:
        """Weight encoders use uniform sigma across all dimensions.

        Network weights are roughly normalised by initialisation and
        cluster around zero with similar variance — uniform sigma is
        appropriate.  Returning None tells the optimiser to use its
        default sigma scaling.
        """
        return None


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


# ---------------------------------------------------------------------------
# Hyperparameter encoder + dispatch
# ---------------------------------------------------------------------------


def build_birth_metadata(sim_config: SimulationConfig) -> dict[str, Any]:
    """Build the birth_metadata dict for a hyperparameter-evolution genome.

    Single source of truth shared between
    :meth:`HyperparameterEncoder.initial_genome` and the EvolutionLoop's
    Genome-construction sites.  Returns a dict with a ``param_schema``
    key containing a list of plain dicts (NOT Pydantic instances) so it
    pickles cheaply across worker processes without requiring a Pydantic
    dependency on the worker decode path.

    When ``sim_config.hyperparam_schema is None`` (weight-evolution
    runs), returns an empty dict so the empty-``birth_metadata``
    behaviour used by weight encoders is preserved verbatim.
    """
    if sim_config.hyperparam_schema is None:
        return {}
    return {
        "param_schema": [entry.model_dump() for entry in sim_config.hyperparam_schema],
    }


class HyperparameterEncoder:
    """Brain-agnostic encoder that evolves brain-config hyperparameters.

    Each :class:`Genome` slot corresponds to one entry in
    ``sim_config.hyperparam_schema``; the encoder reads the schema from
    ``Genome.birth_metadata["param_schema"]`` (the genome is the source
    of truth in worker processes) and applies the type-appropriate
    transform:

    - float: clip to bounds, then exp() if log_scale=True
    - int: clip to bounds, then int(round(value))
    - bool: value > 0.0
    - categorical: values[int(round(value)) mod len(values)]

    The decoded values are applied to ``sim_config.brain.config`` via
    ``model_copy(update={...})`` and a fresh brain is constructed via
    :func:`instantiate_brain_from_sim_config`.  No weights are loaded —
    the brain is freshly initialised on every evaluation, and the
    train phase of :class:`LearnedPerformanceFitness` brings it from
    random init to learned policy.

    The encoder is brain-agnostic: it works for any brain via
    ``sim_config.brain.config`` patching.  ``brain_name`` is set to the
    empty string ``""`` to satisfy the runtime_checkable GenomeEncoder
    protocol's ``brain_name: str`` typing while signalling
    brain-agnosticism — the empty string never collides with a real
    brain name.

    This encoder is selected via :func:`select_encoder` based on
    ``sim_config.hyperparam_schema`` presence, not via
    :data:`ENCODER_REGISTRY` lookup — that registry is reserved for
    brain-keyed encoders.
    """

    brain_name: str = ""

    def initial_genome(
        self,
        sim_config: SimulationConfig,
        *,
        rng: np.random.Generator,
    ) -> Genome:
        """Sample one float per schema slot from the per-type initial distribution."""
        if sim_config.hyperparam_schema is None:
            msg = (
                "HyperparameterEncoder.initial_genome requires "
                "sim_config.hyperparam_schema to be set."
            )
            raise ValueError(msg)

        params = np.empty(len(sim_config.hyperparam_schema), dtype=np.float32)
        for i, entry in enumerate(sim_config.hyperparam_schema):
            params[i] = self._sample_initial(entry, rng)

        return Genome(
            params=params,
            genome_id="",  # filled in by the loop via genome_id_for
            parent_ids=[],
            generation=0,
            birth_metadata=build_birth_metadata(sim_config),
        )

    def decode(
        self,
        genome: Genome,
        sim_config: SimulationConfig,
        *,
        seed: int | None = None,
    ) -> Brain:
        """Patch sim_config.brain.config with decoded values, build fresh brain."""
        schema_dump = genome.birth_metadata.get("param_schema")
        if schema_dump is None:
            msg = (
                "HyperparameterEncoder.decode requires "
                "genome.birth_metadata['param_schema'] to be populated. "
                "Did the EvolutionLoop call build_birth_metadata when "
                "constructing the Genome?"
            )
            raise ValueError(msg)
        if sim_config.brain is None:
            msg = "HyperparameterEncoder.decode requires sim_config.brain to be set."
            raise ValueError(msg)

        if len(genome.params) != len(schema_dump):
            msg = (
                f"Genome param length ({len(genome.params)}) does not match "
                f"birth_metadata param_schema length ({len(schema_dump)})."
            )
            raise ValueError(msg)

        updates: dict[str, Any] = {}
        for entry_dict, value in zip(schema_dump, genome.params, strict=True):
            updates[entry_dict["name"]] = self._decode_one(entry_dict, float(value))

        # Construct a fresh SimulationConfig with the patched brain config.
        # NEVER mutate the input sim_config in place.
        new_brain_cfg = sim_config.brain.config.model_copy(update=updates)
        new_container = sim_config.brain.model_copy(update={"config": new_brain_cfg})
        new_sim_config = sim_config.model_copy(update={"brain": new_container})

        return instantiate_brain_from_sim_config(new_sim_config, seed=seed)

    def genome_dim(self, sim_config: SimulationConfig) -> int:
        """Return the number of evolved hyperparameter slots.

        Equal to ``len(sim_config.hyperparam_schema)``.  Does NOT
        construct a brain — fast path for loop initialisation.
        """
        if sim_config.hyperparam_schema is None:
            msg = (
                "HyperparameterEncoder.genome_dim requires sim_config.hyperparam_schema to be set."
            )
            raise ValueError(msg)
        return len(sim_config.hyperparam_schema)

    def genome_stds(self, sim_config: SimulationConfig) -> list[float] | None:
        """Return per-parameter standard deviations matched to bound widths.

        For each schema entry:

        - float (linear): std = (high - low) / 6, so ±3 stds at sigma=1.0
          spans the full bound range.
        - float (log_scale): std = (log(high) - log(low)) / 6, since
          the genome lives in log-space.
        - int: std = (high - low) / 6, treating the integer range as
          continuous.  Decode clips and rounds.
        - bool: std = 1.0 (genome value is ±1; sigma=1 gives full
          coverage of both poles in expectation).
        - categorical: std = max(1.0, len(values) / 6) so ±3 stds spans
          all categorical bins.

        Without per-parameter stds, a single uniform sigma cannot be
        appropriate across mixed-scale dimensions: too large for tight
        ranges (samples saturate at bounds) or too small for wide
        ranges (no exploration).
        """
        if sim_config.hyperparam_schema is None:
            msg = (
                "HyperparameterEncoder.genome_stds requires sim_config.hyperparam_schema to be set."
            )
            raise ValueError(msg)
        # ±3 stds should cover the full bound range, so std = width / 6.
        bound_coverage_std_divisor = 6.0
        stds: list[float] = []
        for entry in sim_config.hyperparam_schema:
            if entry.type in {"float", "int"}:
                if entry.bounds is None:  # pragma: no cover — validator-enforced
                    msg = f"{entry.type} entry missing bounds"
                    raise ValueError(msg)
                low, high = entry.bounds
                width = float(np.log(high) - np.log(low)) if entry.log_scale else float(high - low)
                stds.append(width / bound_coverage_std_divisor)
            elif entry.type == "bool":
                stds.append(1.0)
            elif entry.type == "categorical":
                if entry.values is None:  # pragma: no cover — validator-enforced
                    msg = "categorical entry missing values"
                    raise ValueError(msg)
                stds.append(max(1.0, len(entry.values) / bound_coverage_std_divisor))
            else:  # pragma: no cover — validator-enforced
                msg = f"Unknown param schema type: {entry.type!r}"
                raise ValueError(msg)
        return stds

    @staticmethod
    def _sample_initial(
        entry: ParamSchemaEntry,
        rng: np.random.Generator,
    ) -> float:
        """Sample one initial genome float for ``entry`` from the per-type prior.

        - float: uniform-in-bounds (or log-uniform when ``log_scale=True``)
        - int: uniform-in-bounds (returned as float, decode rounds)
        - bool: ±1 uniform (decode thresholds at 0)
        - categorical: random index in ``[0, len(values))``
        """
        # ParamSchemaEntry validator guarantees bounds is set for float/int
        # and values is set for categorical, but the type checker doesn't
        # see that — explicit branches with locals satisfy both.
        if entry.type == "float":
            if entry.bounds is None:  # pragma: no cover — validator-enforced invariant
                msg = "float entry missing bounds (validator should prevent this)"
                raise ValueError(msg)
            low, high = entry.bounds
            if entry.log_scale:
                return float(rng.uniform(np.log(low), np.log(high)))
            return float(rng.uniform(low, high))
        if entry.type == "int":
            if entry.bounds is None:  # pragma: no cover — validator-enforced invariant
                msg = "int entry missing bounds (validator should prevent this)"
                raise ValueError(msg)
            low, high = entry.bounds
            return float(rng.uniform(low, high))
        if entry.type == "bool":
            return float(rng.choice([-1.0, 1.0]))
        # categorical
        if entry.values is None:  # pragma: no cover — validator-enforced invariant
            msg = "categorical entry missing values (validator should prevent this)"
            raise ValueError(msg)
        return float(rng.integers(0, len(entry.values)))

    @staticmethod
    def _decode_one(entry_dict: dict[str, Any], value: float) -> Any:  # noqa: ANN401
        """Apply the per-type decode transform.

        Single-method dispatch by ``entry_dict['type']``.  Adding a new
        type means adding a new branch here plus a Literal extension to
        :class:`ParamSchemaEntry.type`.
        """
        entry_type = entry_dict["type"]
        if entry_type == "float":
            low, high = entry_dict["bounds"]
            if entry_dict.get("log_scale", False):
                # genome value is in log-space; clip in log-space then exp.
                log_low, log_high = float(np.log(low)), float(np.log(high))
                clipped_log = max(log_low, min(log_high, value))
                return float(np.exp(clipped_log))
            # Linear float: clip in linear-space.
            return float(max(low, min(high, value)))
        if entry_type == "int":
            low, high = entry_dict["bounds"]
            clipped = max(low, min(high, value))
            return round(clipped)
        if entry_type == "bool":
            return value > 0.0
        if entry_type == "categorical":
            values = entry_dict["values"]
            idx = round(value) % len(values)
            return values[idx]
        msg = f"Unknown param schema type: {entry_type!r}"
        raise ValueError(msg)


def select_encoder(sim_config: SimulationConfig) -> GenomeEncoder:
    """Public dispatch helper: pick the right encoder for a sim_config.

    - When ``sim_config.hyperparam_schema is not None``, return a
      :class:`HyperparameterEncoder` directly (NOT via registry —
      brain-agnostic encoders don't pollute the brain-keyed registry).
      This works for any brain in ``BRAIN_CONFIG_MAP`` even if the
      brain has no weight encoder in :data:`ENCODER_REGISTRY`.
    - Otherwise (the weight-evolution path), return
      :func:`get_encoder(sim_config.brain.name)`.

    Raises
    ------
    ValueError
        If ``hyperparam_schema is None`` and the brain name has no
        weight encoder.  Same message as :func:`get_encoder`.
    """
    if sim_config.hyperparam_schema is not None:
        return HyperparameterEncoder()
    # Weight-evolution path: caller is responsible for ensuring
    # sim_config.brain is non-None (the CLI entry point guards this
    # before loop construction).
    if sim_config.brain is None:  # pragma: no cover — caller-precondition violation
        msg = (
            "select_encoder requires sim_config.brain to be set when "
            "hyperparam_schema is None.  The CLI entry point guards this "
            "at startup; programmatic callers must do the same."
        )
        raise ValueError(msg)
    return get_encoder(sim_config.brain.name)
