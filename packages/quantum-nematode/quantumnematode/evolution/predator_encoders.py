"""Predator-side genome encoder + registry for the M5 co-evolution loop.

Parallel to :data:`quantumnematode.evolution.encoders.ENCODER_REGISTRY` but
isolated: the predator stack has its own brain factory + brain class, and
mixing the two registries would let an agent-side dispatch accidentally
select a predator encoder (or vice versa). Keeping
:data:`PREDATOR_ENCODER_REGISTRY` as a separate top-level mapping makes the
boundary explicit.

Why an override class rather than a parameter to `_ClassicalPPOEncoder`?

The parent's three core methods (`initial_genome`, `decode`, `genome_dim`)
all call :func:`evolution.brain_factory.instantiate_brain_from_sim_config`,
which dispatches through agent-side `setup_brain_model`
(`BRAIN_CONFIG_MAP` only knows the 19 registered AGENT brain types).
Plain subclassing wouldn't reach an `MLPPPOPredatorBrain`. The predator
encoder therefore overrides each of the three methods to call
:func:`evolution._predator_brain_factory.instantiate_predator_brain_from_sim_config`
instead, while reusing the parent's brain-agnostic flatten/unflatten
helpers (:func:`_flatten_components`, :func:`_unflatten_components`,
:func:`_select_genome_components`) verbatim — those operate on
`WeightPersistence` components, which the predator brain implements in
exactly the same shape (policy + value tensors).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quantumnematode.evolution._predator_brain_factory import (
    instantiate_predator_brain_from_sim_config,
)
from quantumnematode.evolution.encoders import (
    GenomeEncoder,
    _ClassicalPPOEncoder,
    _flatten_components,
    _select_genome_components,
    _unflatten_components,
)
from quantumnematode.evolution.genome import Genome

if TYPE_CHECKING:
    import numpy as np

    from quantumnematode.utils.config_loader import SimulationConfig


class MLPPPOPredatorEncoder(_ClassicalPPOEncoder):
    """Encoder for :class:`MLPPPOPredatorBrain` weight evolution.

    Overrides the three brain-construction methods so they call
    :func:`instantiate_predator_brain_from_sim_config` (predator-side
    factory) rather than the agent-side
    :func:`instantiate_brain_from_sim_config` inherited from
    :class:`_ClassicalPPOEncoder`. Everything else — the
    `WeightPersistence` round-trip, the `NON_GENOME_COMPONENTS` filter,
    `genome_stds` (None — uniform sigma), `genome_bounds` (None —
    unbounded for sep-CMA-ES) — falls through to the parent unchanged.

    The post-load runtime-state reset present in
    :meth:`_ClassicalPPOEncoder.decode` (``brain._episode_count = 0``
    plus ``brain._update_learning_rate()``) is INTENTIONALLY skipped
    here: :class:`MLPPPOPredatorBrain` is frozen-weight at evaluation
    time (no inner-loop PPO training, no LR scheduler, no episode
    counter) per design.md D13. Calling the agent-brain helpers would
    raise `AttributeError`.

    Pinning ``brain_name = "mlpppo_predator"`` matches the dispatcher
    Literal extension in :class:`PredatorBrainConfigSchema` and the
    :data:`PREDATOR_ENCODER_REGISTRY` key — single source of truth.
    """

    brain_name = "mlpppo_predator"

    def initial_genome(
        self,
        sim_config: SimulationConfig,
        *,
        rng: np.random.Generator,  # noqa: ARG002 — reserved (uniform with parent)
    ) -> Genome:
        """Build a fresh predator brain and serialise its weights.

        Construction order matches the parent:

        1. Build a fresh brain via the predator factory (no seed — parent
           leaves the brain at its orthogonal-init values).
        2. Filter out non-genome components (optimizer, training_state).
           The predator brain has neither, so the filter is a no-op
           today — kept for symmetry and future-proofing.
        3. Flatten the surviving tensors into a deterministic float32
           array via :func:`_flatten_components`.
        4. Wrap into a :class:`Genome` with shape_map + brain_name in
           ``birth_metadata`` so :meth:`decode` can round-trip without
           re-extracting the template shape from a fresh brain.
        """
        brain = instantiate_predator_brain_from_sim_config(sim_config)
        components = _select_genome_components(brain.get_weight_components())
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
    ) -> Any:  # noqa: ANN401 — see class docstring rationale
        """Construct a fresh predator brain and apply the genome's weights.

        Differences from :meth:`_ClassicalPPOEncoder.decode`:

        - Uses the predator factory.
        - Skips the agent-brain runtime-state reset (`_episode_count`,
          `_update_learning_rate`) — see class docstring rationale.
        - Returns :class:`MLPPPOPredatorBrain`, NOT :class:`Brain`. The
          parent Protocol's return annotation is `Brain` (agent-side)
          but the predator brain is a different Protocol entirely
          (:class:`PredatorBrain`); annotating `Any` here keeps both
          static checkers happy and runtime callers correct (the
          actual return is always an `MLPPPOPredatorBrain` instance).

        The shape_map is read from `genome.birth_metadata` when present;
        otherwise it is re-derived from the freshly-constructed brain's
        components (handles genomes produced by an optimiser sampling a
        flat vector directly with empty birth_metadata, e.g. CMA-ES).
        """
        brain = instantiate_predator_brain_from_sim_config(sim_config, seed=seed)
        # Re-extract for fresh template (brain-agnostic — works because
        # MLPPPOPredatorBrain implements WeightPersistence with the same
        # surface as the agent brains).
        template = _select_genome_components(brain.get_weight_components())
        shape_map = genome.birth_metadata.get("shape_map")
        if shape_map is None:
            _, shape_map = _flatten_components(template)
        components = _unflatten_components(genome.params, shape_map, template)
        brain.load_weight_components(components)
        return brain

    def genome_dim(self, sim_config: SimulationConfig) -> int:
        """Count predator-brain genome floats by constructing a fresh brain."""
        brain = instantiate_predator_brain_from_sim_config(sim_config)
        components = _select_genome_components(brain.get_weight_components())
        params, _ = _flatten_components(components)
        return int(params.size)


PREDATOR_ENCODER_REGISTRY: dict[str, type[GenomeEncoder]] = {
    MLPPPOPredatorEncoder.brain_name: MLPPPOPredatorEncoder,
}


def get_predator_encoder(brain_name: str) -> GenomeEncoder:
    """Look up a predator encoder by brain name; raise on unknown.

    Parallel to :func:`quantumnematode.evolution.encoders.get_encoder`
    but isolated — agent-side `ENCODER_REGISTRY` is NOT consulted as a
    fallback.

    Raises
    ------
    ValueError
        If ``brain_name`` is not in :data:`PREDATOR_ENCODER_REGISTRY`.
    """
    encoder_cls = PREDATOR_ENCODER_REGISTRY.get(brain_name)
    if encoder_cls is None:
        registered = sorted(PREDATOR_ENCODER_REGISTRY)
        msg = f"No predator encoder for brain {brain_name!r}. Registered: {registered}."
        raise ValueError(msg)
    return encoder_cls()
