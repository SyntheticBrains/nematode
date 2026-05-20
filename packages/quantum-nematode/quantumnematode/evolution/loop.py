"""Evolution loop with parallel fitness evaluation and pickle resume.

The :class:`EvolutionLoop` ties everything together:

- Asks the optimiser for a population of candidate parameter vectors.
- Wraps each candidate as a :class:`Genome` with a deterministic ID and
  ``parent_ids`` listing every member of the previous generation (a
  uniform convention across CMA-ES and GA, since neither optimiser
  exposes per-child parent provenance to the loop).
- Evaluates fitness in parallel via ``multiprocessing.Pool`` (when
  ``parallel_workers > 1``), with a SIGINT-ignore worker init so Ctrl-C
  cleanly stops the parent.
- Reports the **negated** fitness to the optimiser (which minimises) since
  our fitness functions return success rate which we maximise.
- Records each evaluation in the lineage CSV.
- Pickles a checkpoint every ``checkpoint_every`` generations.
- Writes ``best_params.json`` and ``history.csv`` on completion.

Resume reconstructs the optimiser, RNG, lineage path, and current generation
from a pickle checkpoint without re-running prior generations.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import pickle
import signal
from contextlib import suppress
from multiprocessing import Pool
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from quantumnematode.evolution.encoders import build_birth_metadata
from quantumnematode.evolution.genome import Genome, genome_id_for
from quantumnematode.evolution.inheritance import InheritanceStrategy, NoInheritance
from quantumnematode.evolution.lineage import LineageTracker
from quantumnematode.optimizers.evolutionary import EvolutionResult

if TYPE_CHECKING:
    from multiprocessing.pool import Pool as PoolType

    from quantumnematode.brain.arch._brain import Brain, BrainParams
    from quantumnematode.env.env import DynamicForagingEnvironment
    from quantumnematode.evolution.encoders import GenomeEncoder
    from quantumnematode.evolution.fitness import FitnessFunction
    from quantumnematode.optimizers.evolutionary import EvolutionaryOptimizer
    from quantumnematode.utils.config_loader import (
        EvolutionConfig,
        ProbeRingConfig,
        SafeProbesConfig,
        SimulationConfig,
    )

logger = logging.getLogger(__name__)


# Pickle checkpoint version.  Bump if the checkpoint dict shape changes in
# a backwards-incompatible way; `_load_checkpoint` validates this and
# refuses incompatible checkpoints.  v2 adds ``selected_parent_ids``
# (list of genome IDs whose Lamarckian checkpoints survive into the next
# generation) and ``inheritance`` (the literal "none"/"lamarckian" string
# the run was launched with — used to reject mismatched-setting resumes).
# v3 adds ``gens_without_improvement`` (int counter for the early-stop
# check) and ``last_best_fitness`` (float | None — the previous
# generation's best, used by the counter to detect strict improvement on
# resume) and extends the ``inheritance`` literal to include "baldwin".
CHECKPOINT_VERSION = 3


# ---------------------------------------------------------------------------
# Worker (module-level so it's picklable for multiprocessing.Pool)
# ---------------------------------------------------------------------------


def _init_worker(log_level: int) -> None:
    """Initialise a multiprocessing worker.

    Workers ignore SIGINT so the parent process handles Ctrl+C gracefully.
    Without this, each worker would crash with KeyboardInterrupt and spew
    a traceback.

    Performance setup:

    - ``torch.set_num_threads(1)``: with ``parallel_workers > 1`` each
      forked worker would otherwise default to multi-threaded BLAS,
      oversubscribing the CPU and slowing every worker down.  Single
      thread per worker leaves coordination to ``multiprocessing.Pool``.
    - Per-step agent/runner logging is silenced to WARNING by setting the
      shared ``quantumnematode.logging_config`` logger.  Both
      ``agent.runners`` and ``agent.agent`` import that logger by name
      (``from quantumnematode.logging_config import logger``), so it's
      the level on that one shared logger that controls per-step output.
      Without this, the ``isEnabledFor(INFO)`` gates in
      ``StandardEpisodeRunner.run`` (runners.py:773-784, 734-737) would
      still fire and build f-strings every step at ``--log-level INFO``,
      defeating the perf gate.  The evolution loop's own progress logs
      are unaffected because they go through this module's own logger.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    logging.getLogger("quantumnematode.logging_config").setLevel(logging.WARNING)
    # Loop-module logger keeps the parent's verbosity so generation-level
    # progress still surfaces in the worker logs.
    logger.setLevel(log_level)


def _evaluate_in_worker(args: tuple) -> float:
    """Evaluate fitness for one genome in a worker process.

    Args
    ----
    args
        Tuple of ``(params_array, sim_config, encoder, fitness, episodes, seed,
        generation, index, parent_ids, warm_start_path_override,
        weight_capture_path, tei_prior_source, diagnostics_path)``.  All
        elements must be picklable.  The four trailing ``Path | None`` / tuple
        fields are used by the inheritance + telemetry steps:

        - ``warm_start_path_override`` and ``weight_capture_path``: per the
          Lamarckian/Baldwin pattern. When both are ``None`` (the no-
          inheritance case) the call site is byte-equivalent to a
          frozen-weight evolution run.
        - ``tei_prior_source``: ``(f0_substrate_path, decay_factor,
          lineage_depth)`` triple for transgenerational F1+ workers, or
          ``None`` otherwise. When set, the worker forwards it to
          ``fitness.evaluate`` as the same-named kwarg; ``fitness.evaluate``
          loads the F0 substrate, applies ``inherit_from`` ``lineage_depth``
          times, and sets ``brain.tei_prior`` post-decode (mirrors the
          ``warm_start_path_override`` / ``weight_capture_path``
          forwarding pattern).
        - ``diagnostics_path``: per-session ``eval_diagnostics.jsonl``
          path. When non-``None``, ``LearnedPerformanceFitness.evaluate``
          appends one per-genome row recording ``success_rate``,
          ``survival_rate``, deaths, successes, composite, and the
          scalar fitness it returned. Used by the decision-gate
          machinery (T1 envelope check, T3 M6-floor-to-beat, post-hoc
          aggregator). ``EpisodicSuccessRate`` does not accept this
          kwarg; the worker checks the signature dynamically before
          forwarding (so co-evolution dispatch on either fitness
          class works without per-class branching).

        ``encoder`` and ``fitness`` are class instances pickled by
        reference to their class definitions; concrete encoders/fitness
        functions in this module are top-level classes and pickle cleanly.

    Returns
    -------
    float
        Fitness value for the genome (positive = better, in [0, 1] for
        :class:`EpisodicSuccessRate`).
    """
    (
        params,
        sim_config,
        encoder,
        fitness,
        episodes,
        seed,
        generation,
        index,
        parent_ids,
        warm_start_path_override,
        weight_capture_path,
        tei_prior_source,
        diagnostics_path,
    ) = args
    genome = Genome(
        params=np.asarray(params, dtype=np.float32),
        genome_id=genome_id_for(generation, index, parent_ids),
        parent_ids=parent_ids,
        generation=generation,
        birth_metadata=build_birth_metadata(sim_config),
    )
    # Only LearnedPerformanceFitness accepts the inheritance + diagnostics
    # kwargs; EpisodicSuccessRate's signature accepts the inheritance ones
    # for ABI symmetry but ignores them, and does NOT accept
    # ``diagnostics_path``. Detect by signature rather than type-import to
    # avoid coupling the worker to a specific fitness class. When all
    # optional kwargs are None (no-inheritance, no-diagnostics case) we
    # drop them entirely so the call shape is byte-equivalent to a
    # frozen-weight evolution run.
    if (
        warm_start_path_override is not None
        or weight_capture_path is not None
        or tei_prior_source is not None
        or diagnostics_path is not None
    ):
        # Build kwargs dynamically to avoid passing diagnostics_path to
        # fitness functions that don't accept it (EpisodicSuccessRate).
        kwargs = {
            "episodes": episodes,
            "seed": seed,
            "warm_start_path_override": warm_start_path_override,
            "weight_capture_path": weight_capture_path,
            "tei_prior_source": tei_prior_source,
        }
        import inspect

        if "diagnostics_path" in inspect.signature(fitness.evaluate).parameters:
            kwargs["diagnostics_path"] = diagnostics_path
        return fitness.evaluate(genome, sim_config, encoder, **kwargs)
    return fitness.evaluate(genome, sim_config, encoder, episodes=episodes, seed=seed)


# ---------------------------------------------------------------------------
# F0 probe ring helpers (M6.11 — env-derived probe geometry)
# ---------------------------------------------------------------------------


def _compute_probe_gradient(
    probe_pos: tuple[int, int],
    predator_pos: tuple[int, int],
) -> tuple[float, float]:
    """Return ``(strength, direction)`` of the predator gradient at a probe position.

    Pure function, unit-testable without env mocks. Mirrors the env's
    convention that gradient strength falls off with Manhattan distance
    and direction points from the probe toward the predator (i.e. the
    direction the agent would move TO reach the predator).

    Parameters
    ----------
    probe_pos : tuple[int, int]
        Grid coordinate of the probe position (where the synthetic
        agent's sensory snapshot is being computed).
    predator_pos : tuple[int, int]
        Grid coordinate of the source predator.

    Returns
    -------
    tuple[float, float]
        ``(predator_gradient_strength, predator_gradient_direction)``.
        Strength is in ``[0.0, 1.0]`` (1.0 at zero distance, monotonically
        decreasing). Direction is in radians via ``atan2(predator_y -
        probe_y, predator_x - probe_x)``; matches the env-side convention
        for predator-gradient channel emission.
    """
    px, py = predator_pos
    x, y = probe_pos
    manhattan = abs(px - x) + abs(py - y)
    strength = 1.0 / (1.0 + manhattan)
    direction = math.atan2(py - y, px - x)
    return strength, direction


def _manhattan_ring_offsets(radius: int) -> list[tuple[int, int]]:
    """Return the ``4 * radius`` ``(dx, dy)`` offsets on the L1 ring at distance ``radius``.

    For ``radius > 0`` the offsets satisfy ``|dx| + |dy| == radius`` and
    are emitted in counter-clockwise order starting from ``(radius, 0)``:
    east → north → west → south. For ``radius == 0`` returns a single
    ``(0, 0)`` offset.

    Used by the M6.11 probe-ring builder to enumerate integer cells at
    a fixed Manhattan distance from a stationary predator. Manhattan
    geometry matches the env's gradient-strength falloff
    (``1 / (1 + manhattan)``); a Euclidean ring would produce probes
    at variable Manhattan distance, breaking the spec scenario that
    ties probe distance to ``damage_radius + radius_offset``.
    """
    if radius <= 0:
        return [(0, 0)]
    # Build the four L1-perimeter quadrants in counter-clockwise order
    # via list comprehensions (extending a single list inline is the
    # idiomatic perf-conscious form per ruff PERF401).
    offsets: list[tuple[int, int]] = []
    # East→North quadrant: dx from +radius to 0; dy = radius - |dx|.
    offsets.extend((dx, radius - dx) for dx in range(radius, 0, -1))
    # North→West quadrant: dy from +radius to 0; dx = -(radius - dy).
    offsets.extend((-(radius - dy), dy) for dy in range(radius, 0, -1))
    # West→South quadrant: dx from -radius to 0; dy = -(radius - |dx|).
    offsets.extend((dx, -(radius + dx)) for dx in range(-radius, 0, 1))
    # South→East quadrant: dy from -radius to 0; dx = radius + dy.
    offsets.extend((radius + dy, dy) for dy in range(-radius, 0, 1))
    return offsets


def _sample_ring_offsets(radius: int, count: int) -> list[tuple[int, int]]:
    """Down-sample the L1 ring at ``radius`` to ``count`` evenly-spaced offsets.

    If ``count`` equals or exceeds the ring perimeter (``4 * radius``)
    the full ring is returned. Otherwise samples ``count`` cells at
    evenly-spaced perimeter indices via integer index stepping so the
    result is deterministic and angularly even.
    """
    full = _manhattan_ring_offsets(radius)
    if count >= len(full):
        return full
    # Step through the full perimeter at len(full) / count cadence.
    step = len(full) / count
    return [full[round(i * step) % len(full)] for i in range(count)]


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class EvolutionLoop:
    """Orchestrates the optimiser ↔ fitness ↔ lineage feedback loop.

    Parameters
    ----------
    optimizer
        :class:`CMAESOptimizer` or :class:`GeneticAlgorithmOptimizer` instance,
        already constructed with the correct ``num_params``.
    encoder
        Genome encoder (decides genome shape; produces brains from genomes).
    fitness
        Fitness function (decides success criterion).
    sim_config
        Full simulation config (passed to encoder.decode and fitness.evaluate).
    evolution_config
        Evolution-specific knobs (population, generations, parallel_workers,
        checkpoint_every).
    output_dir
        Directory to write ``checkpoint.pkl``, ``best_params.json``,
        ``history.csv``, and ``lineage.csv``.  Created if it doesn't exist.
    rng
        Numpy Generator for non-optimiser RNG (per-generation seed derivation).
    log_level
        Forwarded to worker init.
    """

    def __init__(  # noqa: PLR0913 - explicit deps are clearer than a config object
        self,
        optimizer: EvolutionaryOptimizer,
        encoder: GenomeEncoder,
        fitness: FitnessFunction,
        sim_config: SimulationConfig,
        evolution_config: EvolutionConfig,
        output_dir: Path,
        rng: np.random.Generator,
        log_level: int = logging.WARNING,
        inheritance: InheritanceStrategy | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.encoder = encoder
        self.fitness = fitness
        self.sim_config = sim_config
        self.evolution_config = evolution_config
        self.output_dir = output_dir
        self.rng = rng
        self.log_level = log_level
        # Inheritance strategy for per-genome weight capture/warm-start.
        # Defaults to NoInheritance() so the loop runs as a frozen-weight
        # evolution loop when no strategy is supplied.  Use a None
        # sentinel + late-construction (rather than ``= NoInheritance()``
        # in the signature) to avoid a shared default instance across
        # calls.
        self.inheritance: InheritanceStrategy = (
            inheritance if inheritance is not None else NoInheritance()
        )

        # Reject silent divergence between the supplied inheritance instance
        # and ``evolution_config.inheritance``.  Without this guard the
        # checkpoint would record one value while runtime behaviour follows
        # the other (lineage rows + selected_parent_ids derive from the
        # instance via ``kind()``; the literal stored on disk derives from
        # the config string).  Resume would then load with the config-side
        # literal but operate under whatever new instance the caller hands
        # ``EvolutionLoop`` next time, hiding the mismatch.
        _expected_kind = {
            "none": "none",
            "lamarckian": "weights",
            "baldwin": "trait",
            "transgenerational": "transgenerational",
            "weights+transgenerational": "weights+transgenerational",
        }[self.evolution_config.inheritance]
        if self.inheritance.kind() != _expected_kind:
            msg = (
                f"Inheritance instance mismatch: evolution_config.inheritance="
                f"{self.evolution_config.inheritance!r} expects "
                f"kind()={_expected_kind!r}, but the supplied strategy reports "
                f"kind()={self.inheritance.kind()!r}.  Pass an instance whose "
                "kind() matches the config string (or omit the argument so "
                "the loop defaults to NoInheritance and set "
                "evolution.inheritance: none in the YAML)."
            )
            raise ValueError(msg)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lineage_path = self.output_dir / "lineage.csv"
        self._checkpoint_path = self.output_dir / "checkpoint.pkl"
        self._lineage = LineageTracker(self._lineage_path)

        # Generation index and prior-generation IDs are mutated in run().
        self._generation = 0
        self._prev_generation_ids: list[str] = []
        # Inheritance bookkeeping: IDs of genomes whose weight checkpoints
        # are about to be inherited by the next generation's children.
        # Empty until the first generation completes select_parents().
        self._selected_parent_ids: list[str] = []
        # Early-stop bookkeeping: counter incremented each generation in
        # which best_fitness fails to strictly improve over the previous
        # generation's best.  Reset to 0 on any strict improvement.
        # ``_last_best_fitness`` is None until gen 1 completes (gen-1
        # bootstrap branch in the counter-update logic treats the first
        # generation as an "improvement").  Both persisted in the
        # checkpoint pickle so resume preserves the saturation state.
        self._gens_without_improvement: int = 0
        self._last_best_fitness: float | None = None

        # Transgenerational-inheritance bookkeeping: path to the F0
        # elite's ``.tei.pt`` substrate file, populated by the F0
        # Substrate Extraction Pipeline at the end of gen 0. F1+
        # worker tuples use this path to construct ``tei_prior_source``
        # for ``fitness.evaluate`` (a follow-up addition). The
        # attribute is persisted in the checkpoint pickle so resume
        # from gen 1+ can recover the path without re-running F0.
        # ``None`` outside of transgenerational runs and before the
        # F0 extraction completes.
        self._tei_f0_substrate_path: Path | None = None

    # ---- Checkpoint / Resume ---------------------------------------------

    def _save_checkpoint(self) -> None:
        """Pickle optimiser + RNG + generation + lineage path + version."""
        # ``inheritance`` is persisted as the literal "none"/"lamarckian"/
        # "baldwin" string (not the strategy instance) so resume can
        # compare against the resolved current value and reject
        # mismatches.  The constructor's kind()-vs-config guard above
        # ensures this string is always consistent with
        # ``self.inheritance.kind()`` at the moment we write it.
        inheritance_value = self.evolution_config.inheritance
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "optimizer": self.optimizer,
            "generation": self._generation,
            "prev_generation_ids": self._prev_generation_ids,
            "selected_parent_ids": self._selected_parent_ids,
            "inheritance": inheritance_value,
            "gens_without_improvement": self._gens_without_improvement,
            "last_best_fitness": self._last_best_fitness,
            "rng_state": self.rng.bit_generator.state,
            "lineage_path": str(self._lineage_path),
            # Transgenerational F0 substrate path: stored as a string
            # (or None) so the pickle is portable across runs that
            # rebase the output_dir. Loaders reconstruct the Path.
            "tei_f0_substrate_path": (
                str(self._tei_f0_substrate_path)
                if self._tei_f0_substrate_path is not None
                else None
            ),
        }
        # Atomic write via tmp file + rename.
        tmp_path = self._checkpoint_path.with_suffix(".pkl.tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle)
        tmp_path.replace(self._checkpoint_path)

    def _load_checkpoint(self, path: Path) -> None:
        """Restore optimiser, RNG, and generation from a pickle file.

        Raises
        ------
        ValueError
            If the checkpoint's ``checkpoint_version`` doesn't match
            :data:`CHECKPOINT_VERSION`.  Refusing incompatible checkpoints
            avoids silent state corruption when the loop's schema changes.
            Also raised when the checkpoint's recorded ``inheritance``
            differs from the resolved current ``EvolutionConfig.inheritance``
            (mid-run inheritance changes are not supported).
        """
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - trusted local file

        version = payload.get("checkpoint_version")
        if version != CHECKPOINT_VERSION:
            msg = (
                f"Incompatible checkpoint version: expected {CHECKPOINT_VERSION}, "
                f"found {version}.  Refusing to resume."
            )
            raise ValueError(msg)

        # Validate the resume-critical key set present in the payload
        # BEFORE assigning any of them.  ``_save_checkpoint`` writes
        # every key in this list unconditionally for any v3 payload, so
        # a payload that passed the version check but is missing any of
        # them is structurally inconsistent (probably hand-edited or
        # written by a buggy older revision claiming to be v3).  Fail
        # fast with a descriptive message naming the missing key rather
        # than silently defaulting to "none" / [] / 0 / None — those
        # defaults would corrupt resume semantics in subtle ways:
        # ``inheritance`` defaulting to "none" would mask a Lamarckian
        # checkpoint; ``selected_parent_ids`` defaulting to [] would
        # silently drop the next generation's warm-start parent assignment;
        # the early-stop fields defaulting to 0 / None would falsify the
        # saturation-tracking history.  Listed in the same order as the
        # save-side ``payload`` dict for grep-discoverability.
        #
        # NOTE: keys added to ``_save_checkpoint`` *after* the v3 schema
        # was fixed (e.g. ``tei_f0_substrate_path``) intentionally do
        # NOT appear here — those use ``.get(key, default)`` at the
        # assignment site so legacy v3 payloads (pre-additive-field)
        # load cleanly without a ``CHECKPOINT_VERSION`` bump. Only keys
        # that were part of the original v3 schema are required.
        required_keys = (
            "optimizer",
            "generation",
            "prev_generation_ids",
            "selected_parent_ids",
            "inheritance",
            "gens_without_improvement",
            "last_best_fitness",
            "rng_state",
        )
        for required_key in required_keys:
            if required_key not in payload:
                msg = (
                    f"Checkpoint claims version {CHECKPOINT_VERSION} but is "
                    f"missing required key {required_key!r}. All v3 payloads "
                    "MUST contain every key written by ``_save_checkpoint`` "
                    "(the value may be None / [] / 0 where appropriate; the "
                    "key itself must exist). Refusing to resume from a "
                    "structurally inconsistent checkpoint."
                )
                raise ValueError(msg)

        # Reject mid-run inheritance changes.  Fires BEFORE any optimiser
        # state is restored so an inadvertent --inheritance override
        # doesn't waste compute on a corrupted run.
        checkpoint_inheritance = payload["inheritance"]
        current_inheritance = self.evolution_config.inheritance
        if checkpoint_inheritance != current_inheritance:
            msg = (
                f"Checkpoint inheritance setting ({checkpoint_inheritance!r}) "
                f"differs from resolved current value ({current_inheritance!r}). "
                "Mid-run inheritance changes are not supported. Either start "
                "a fresh run or restore the original setting (in YAML or "
                "via --inheritance) before resuming."
            )
            raise ValueError(msg)

        self.optimizer = payload["optimizer"]
        self._generation = payload["generation"]
        self._prev_generation_ids = payload["prev_generation_ids"]
        self._selected_parent_ids = payload["selected_parent_ids"]
        self._gens_without_improvement = payload["gens_without_improvement"]
        self._last_best_fitness = payload["last_best_fitness"]
        self.rng.bit_generator.state = payload["rng_state"]
        # Transgenerational F0 substrate path: ``.get`` with None default
        # for backwards compatibility with v3 checkpoints (pre-TEI).
        # The field is additive — no CHECKPOINT_VERSION bump needed.
        tei_path_str = payload.get("tei_f0_substrate_path")
        self._tei_f0_substrate_path = Path(tei_path_str) if tei_path_str else None

    # ---- Inheritance helpers --------------------------------------------

    def _inheritance_active(self) -> bool:
        """Return True iff the active strategy uses per-genome weight checkpoints.

        Gates the weight-IO code paths in the loop: per-genome
        ``checkpoint_path`` computation, the GC step, and the
        warm-start lookup. Fires for ``LamarckianInheritance``
        (kind=``"weights"``) AND ``LamarckianTransgenerationalInheritance``
        (kind=``"weights+transgenerational"``, the composed mode that
        reuses the Lamarckian warm-start path alongside substrate flow).
        """
        return self.inheritance.kind() in {"weights", "weights+transgenerational"}

    def _inheritance_records_lineage(self) -> bool:
        """Return True iff the active strategy populates the lineage CSV's `inherited_from`.

        Gates the per-generation ``select_parents`` call and the
        ``_selected_parent_ids`` update. Fires for every non-no-op
        strategy: ``LamarckianInheritance``, ``BaldwinInheritance``,
        ``TransgenerationalInheritance``, and
        ``LamarckianTransgenerationalInheritance`` (composed).
        Only ``NoInheritance`` returns ``False``.
        """
        return self.inheritance.kind() != "none"

    def _substrate_inheritance_active(self) -> bool:
        """Return True iff the active strategy uses the TEI substrate-flow pipeline.

        Gates the F0 Substrate Extraction Pipeline (F0 weight capture
        + post-eval extraction + ``.tei.pt`` save + F0 ``.pt`` GC).
        Distinct from ``_inheritance_active()`` (which gates the
        weight-IO checkpoint flow); both predicates fire simultaneously
        under composed mode so the loop runs BOTH the weight-IO
        path AND the substrate-flow path in parallel.

        Fires for ``TransgenerationalInheritance`` (kind=
        ``"transgenerational"``, pure-TEI) AND
        ``LamarckianTransgenerationalInheritance`` (kind=
        ``"weights+transgenerational"``, composed).
        """
        return self.inheritance.kind() in {"transgenerational", "weights+transgenerational"}

    def _combined_inheritance_active(self) -> bool:
        """Return True iff the active strategy is the composed mode.

        Used to gate behaviour that fires ONLY under composed mode
        (e.g. suppressing the F0 substrate-extraction pipeline's
        internal `.pt` GC pass at lines 855-866 of
        ``_run_f0_substrate_extraction`` — under composed mode the
        F0 elite's `.pt` must survive that pass for F1 children to
        warm-start from; the main-loop Lamarckian GC at the end of
        the same generation iteration then keeps only the elite per
        ``_inheritance_active()``'s widened scope).
        """
        return self.inheritance.kind() == "weights+transgenerational"

    def _append_per_gen_elite(
        self,
        *,
        generation: int,
        genome_id: str,
        params: np.ndarray,
        fitness: float,
    ) -> None:
        """Append one ``(generation, genome_id, params, fitness)`` row to ``per_gen_elites.jsonl``.

        JSON-Lines format so the file can be opened append-mode and
        truncated by external tooling without parsing the whole structure.
        Each line is a self-contained JSON object — readers can iterate
        the file and skip malformed lines without disqualifying the rest.

        Idempotent on ``generation``: if the file already contains a row
        for ``generation``, the new row is skipped with a debug log.
        This guards against duplicate snapshots when a run resumes from
        a checkpoint written *before* the post-eval append (the
        ``checkpoint_every`` interval means most generations land in
        that window). Without the guard, the resumed run would re-append
        for the same gen and the offline evaluator would see two
        elites for the same (gen, seed).

        Post-hoc evaluators (e.g.
        ``scripts/campaigns/transgenerational_per_gen_eval.py``) read
        this artifact to reconstruct each generation's elite genome
        offline; without it, only the FINAL elite is recoverable from
        ``best_params.json``.
        """
        import json as _json

        path = self.output_dir / "per_gen_elites.jsonl"
        if path.exists():
            # Scan for an existing entry at this generation. Readers
            # silently skip malformed lines (see ``_read_per_gen_elites``
            # in the evaluator); we mirror that here so a corrupted
            # tail-line doesn't block the dedupe check. Each guard
            # handles a distinct malformed shape:
            #   - JSONDecodeError: not valid JSON at all
            #   - non-dict (a list/scalar that happens to parse): skip
            #   - missing 'generation' key: skip
            #   - non-int-castable 'generation' value: skip
            target = int(generation)
            with path.open(encoding="utf-8") as handle:
                for raw in handle:
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        existing = _json.loads(stripped)
                    except _json.JSONDecodeError:
                        continue
                    if not isinstance(existing, dict):
                        continue
                    if "generation" not in existing:
                        continue
                    try:
                        existing_gen = int(existing["generation"])
                    except (TypeError, ValueError):
                        continue
                    if existing_gen == target:
                        logger.debug(
                            "per_gen_elites: row for generation=%d already exists; "
                            "skipping append (resume idempotency).",
                            target,
                        )
                        return
        row = {
            "generation": int(generation),
            "genome_id": str(genome_id),
            "params": [float(p) for p in np.asarray(params).ravel()],
            "fitness": float(fitness),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_json.dumps(row) + "\n")

    def _gc_inheritance_dir(self, generation: int, keep_ids: list[str]) -> None:
        """Garbage-collect non-survivor checkpoints in one inheritance directory.

        Deletes every weight-checkpoint file in
        ``inheritance/gen-{generation:03d}/`` whose genome ID is NOT in
        ``keep_ids``.  No-op when the directory does not exist (e.g. on
        the very first GC pass before any inheritance file has been
        written, or when ``generation < 0``).  Weight files are matched
        by the canonical ``genome-<gid>.pt`` name shape; the ``<gid>``
        token is extracted via ``stem.removeprefix("genome-")`` (cleaner
        than regex for the fixed pattern).

        Substrate files (``genome-<gid>.tei.pt``) are SKIPPED — their
        lifecycle is owned by the F0 substrate-extraction pipeline, not
        the main-loop Lamarckian GC.  Under composed mode the F0
        substrate must survive past F0 because F2/F3 children inherit
        it with ``decay_factor^2`` / ``decay_factor^3``; treating
        ``genome-X.tei.pt`` as a weight checkpoint here would strip only
        one suffix (``stem`` becomes ``genome-X.tei``) and leave a
        bogus genome ID that never matches the keep-set, deleting the
        substrate.
        """
        if generation < 0:
            return
        gen_dir = self.output_dir / "inheritance" / f"gen-{generation:03d}"
        if not gen_dir.exists():
            return
        keep = set(keep_ids)
        for path in gen_dir.glob("genome-*.pt"):
            if path.name.endswith(".tei.pt"):
                continue
            gid = path.stem.removeprefix("genome-")
            if gid not in keep:
                path.unlink(missing_ok=True)
        # Cosmetic: remove the directory itself if it's now empty.  Keeps
        # ``inheritance/`` clean during a run (so an operator can inspect
        # mid-run and see only the surviving generation directories).
        # ``rmdir`` raises OSError on non-empty dirs; suppress because
        # that's the expected case when ``keep`` was non-empty.
        with suppress(OSError):
            gen_dir.rmdir()

    def _run_f0_substrate_extraction(
        self,
        elite_id: str,
        gen_ids: list[str],
        solutions: list,
    ) -> None:
        """Run the F0 Substrate Extraction Pipeline.

        Invoked once per run, immediately after F0's
        ``select_parents`` identifies the elite. Steps:

        1. Locate the elite's params (from the F0 ``solutions`` batch
           via ``gen_ids.index(elite_id)``).
        2. Decode a fresh brain from the elite genome (matches the
           shape the F0 worker constructed).
        3. Load the elite's captured ``.pt`` weights into the fresh
           brain.
        4. Invoke ``extract_from_brain`` to compute the substrate's
           ``logit_bias`` via the deterministic probe pass.
        5. Save the substrate to
           ``inheritance/gen-000/genome-{elite_id}.tei.pt``.
        6. GC every F0 ``.pt`` weight file; only the ``.tei.pt`` is
           retained for the cascade.

        Defensive: if the elite's captured ``.pt`` is missing
        (unexpected — every F0 worker should have written one via
        the ``weight_capture_path`` kwarg per
        ``_resolve_per_child_inheritance``), log a warning and skip
        the substrate save. F1+ workers downstream will then have
        no substrate to load, which the loop's
        ``_substrate_inheritance_active()`` flow handles
        gracefully (a follow-up commit's worker integration will
        detect the missing file and emit a clear error).
        """
        # Imports here to avoid heavy module-load cost when TEI is unused.
        from quantumnematode.agent.transgenerational_memory import (
            extract_from_brain,
        )
        from quantumnematode.agent.transgenerational_memory import (
            save as save_substrate,
        )
        from quantumnematode.brain.weights import load_weights

        # Locate elite params from the F0 solutions batch.
        try:
            elite_idx = gen_ids.index(elite_id)
        except ValueError:
            logger.warning(
                "F0 substrate extraction: elite_id=%s not found in gen_ids; "
                "skipping substrate save.",
                elite_id,
            )
            return
        elite_params = solutions[elite_idx]

        # Decode the elite genome into a fresh brain. ``encoder.decode``
        # is invoked at the F0 worker too, but we re-invoke here with
        # a deterministic seed (same as ``run()``'s rng seeding pattern)
        # to ensure shape parity. The elite's TRAINED weights are then
        # loaded from disk into this fresh brain.
        elite_genome = Genome(
            params=np.asarray(elite_params, dtype=np.float32),
            genome_id=elite_id,
            parent_ids=list(self._prev_generation_ids),
            generation=0,
            birth_metadata=build_birth_metadata(self.sim_config),
        )
        # Use a deterministic seed for the brain construction; the brain's
        # initial weights are overwritten by load_weights below, so the
        # seed only affects ancillary RNG state (e.g., torch.manual_seed
        # at construction). A fixed seed keeps the substrate extraction
        # deterministic across runs.
        brain = self.encoder.decode(elite_genome, self.sim_config, seed=0)

        # Path to the F0 elite's captured weight file (written by the
        # F0 worker via the ``weight_capture_path`` kwarg per
        # ``_resolve_per_child_inheritance``'s transgenerational branch).
        elite_pt_path = self.output_dir / "inheritance" / "gen-000" / f"genome-{elite_id}.pt"
        if not elite_pt_path.exists():
            logger.warning(
                "F0 substrate extraction: expected elite weights at %s but file is "
                "missing; skipping substrate save. F1+ workers will have no "
                "substrate to load.",
                elite_pt_path,
            )
            return
        load_weights(brain, elite_pt_path)

        # Build the probe params sequence. M6.11 path: when the
        # transgenerational config has a ``probe_ring`` sub-block AND
        # the env has stationary predators, ``_build_f0_probe_params``
        # generates env-derived ring positions. Otherwise falls back
        # to the M6 legacy synthetic-probe path.
        probe_env = self._build_probe_env_if_configured()
        probe_params = self._build_f0_probe_params(brain=brain, env=probe_env)

        # Resolve the substrate form from the YAML-configured
        # ``transgenerational`` block. When ``bias_network`` is set
        # the F0 extraction follows the sensory-conditional path
        # (MLP-fit substrate); when None it falls back to the M6
        # legacy constant ``logit_bias`` path (byte-equivalent).
        # ``extraction_seed`` is also threaded from config so the
        # per-seed MLP init RNG is distinct across calibration
        # seeds — without this the substrate-diversity tripwire
        # underestimates pairwise CoV (every calibration seed
        # shares the same MLP-init RNG → artificially low diversity).
        cfg_tg = self.evolution_config.transgenerational
        bias_network_spec: dict | None = None
        input_features: tuple[str, ...] = ()
        extraction_seed = 424242
        if cfg_tg is not None:
            extraction_seed = cfg_tg.extraction_seed
            if cfg_tg.bias_network is not None:
                bias_network_spec = {
                    "input_dim": len(cfg_tg.bias_network.input_features),
                    "hidden_dim": cfg_tg.bias_network.hidden_dim,
                    "output_dim": brain.num_actions,  # type: ignore[attr-defined]
                    "activation": cfg_tg.bias_network.activation,
                }
                input_features = tuple(cfg_tg.bias_network.input_features)

        substrate = extract_from_brain(
            brain=brain,
            probe_params=probe_params,
            rng_seed=extraction_seed,
            source_genome_id=elite_id,
            bias_network_spec=bias_network_spec,
            input_features=input_features,
        )

        # Save the substrate at the canonical ``.tei.pt`` path. This
        # path is hardcoded here (NOT routed through
        # ``self.inheritance.checkpoint_path``) because the strategy's
        # ``checkpoint_path`` returns the WEIGHTS path under composed
        # mode (``LamarckianTransgenerationalInheritance`` returns
        # the canonical ``.pt`` for F1+ warm-start use). Asking the
        # composed strategy for the substrate path would collide with
        # the elite's weights file: substrate bytes would overwrite
        # the weights ``.pt`` and F1 children would fail to warm-start.
        # The substrate path-builder lives here in the loop because
        # only the loop knows it needs a ``.tei.pt`` (substrate file)
        # vs the strategy's ``.pt`` (weights file).
        substrate_path = self.output_dir / "inheritance" / "gen-000" / f"genome-{elite_id}.tei.pt"
        save_substrate(substrate, substrate_path)

        # Record the path so F1+ workers (a follow-up commit's
        # worker-tuple extension) can compute ``tei_prior_source``
        # without re-deriving the elite ID from ``_selected_parent_ids``
        # (which gets overwritten at each generation's
        # ``select_parents`` call). Persisted in the checkpoint pickle.
        self._tei_f0_substrate_path = substrate_path

        # GC every F0 ``.pt`` weight file — only the ``.tei.pt`` is
        # retained for the cascade. We reuse ``_gc_inheritance_dir``
        # with ``keep_ids=[]`` to clear all genome-*.pt entries; the
        # method only matches files with the ``genome-`` prefix and
        # extension matching the strategy's checkpoint_path output,
        # so the ``.tei.pt`` we just wrote is NOT affected (different
        # suffix). For safety we additionally filter explicitly: walk
        # the directory and delete only ``.pt`` files (not ``.tei.pt``).
        #
        # Composed mode: this GC SHALL be SKIPPED. Composed mode
        # reuses the Lamarckian per-child weight-IO flow, so F1+
        # children warm-start from the F0 elite's ``.pt`` — it MUST
        # survive this pipeline. The main-loop GC at ``run()``'s
        # post-``select_parents`` block (around line 1670) already
        # keeps only the F0 elite per ``_inheritance_active()``'s
        # widened scope (composed mode now satisfies that predicate),
        # so the net effect under composed mode is byte-equivalent to
        # pure Lamarckian: exactly one F0 ``.pt`` survives at the
        # moment gen 1 begins, indexed by the elite's genome_id,
        # alongside the ``.tei.pt`` substrate. Pure-TEI
        # (kind=``"transgenerational"``) is unchanged — main-loop GC
        # does NOT fire for that kind, so
        # this inline GC remains load-bearing.
        if not self._combined_inheritance_active():
            gen_dir = self.output_dir / "inheritance" / "gen-000"
            for path in gen_dir.iterdir():
                if path.suffix == ".pt" and not path.name.endswith(".tei.pt"):
                    path.unlink(missing_ok=True)

    def _build_probe_env_if_configured(self) -> DynamicForagingEnvironment | None:
        """Construct a transient env for F0 probe-ring sampling, or return None.

        Returns ``None`` when no ``probe_ring`` sub-block is configured
        (legacy M6 synthetic-probe path stays active). Otherwise
        returns a freshly-constructed env from
        ``self.sim_config.environment`` whose stationary predators the
        probe-ring builder reads. The env is throwaway — used only
        for predator-coordinate enumeration; no episode is run on it.

        Probe env seed: hardcoded ``seed=0``, deliberately
        independent of the campaign seed (which sweeps 42-45 across
        the M6.9+ full campaign). Rationale: the substrate-diversity
        tripwire T2 measures pairwise CoV across the 4 calibration
        seeds' extracted bias-network ``state_dict()`` tensors. If
        the probe env's predator positions varied with campaign
        seed, T2 would conflate "different brain policy" (the M6
        attractor signature we want to detect) with "different
        probe geometry" (noise from the seeded env layout). Pinning
        the probe-env seed makes substrate diversity attributable
        to the brain-policy difference alone.
        """
        cfg = self.sim_config.evolution
        if cfg is None or cfg.transgenerational is None:
            return None
        if cfg.transgenerational.probe_ring is None:
            return None
        if self.sim_config.environment is None:
            # Defensive: should never happen under a transgenerational
            # run (the YAML schema requires an env block when predators
            # are configured) but guard explicitly so a misconfigured
            # config fails over to the legacy synthetic-probe path
            # rather than crashing.
            return None
        from quantumnematode.env.theme import Theme
        from quantumnematode.utils.config_loader import create_env_from_config

        # Theme.HEADLESS — no episode is run on this env (it's used
        # only for predator-coordinate enumeration), so the ASCII
        # render path would be unused wall-time anyway. Mirrors the
        # convention used in evolution/fitness.py for evaluation envs.
        return create_env_from_config(
            self.sim_config.environment,
            seed=0,
            theme=Theme.HEADLESS,
        )

    def _build_f0_probe_params(
        self,
        brain: Brain | None = None,
        env: DynamicForagingEnvironment | None = None,
    ) -> list[BrainParams]:
        """Build a deterministic sequence of probe ``BrainParams`` for F0 telemetry.

        Two paths:

        - **Env-derived probe ring** (M6.11): when ``env`` is supplied
          AND the transgenerational config has a ``probe_ring``
          sub-block AND the env has at least one stationary predator,
          generate ``probe_ring.count`` ring positions around each
          stationary predator at distance ``predator.damage_radius +
          probe_ring.radius_offset``, evenly angular-distributed. For
          each ring position compute ``(predator_gradient_strength,
          predator_gradient_direction)`` via :func:`_compute_probe_gradient`
          relative to its source predator. Optionally emit two ring
          iterations (food-gradient set vs zeroed) when
          ``probe_ring.include_food_gradient_variants`` is True.
        - **Legacy M6 synthetic** (fallback): three hardcoded probes
          with varying food-gradient strengths and zero predator
          gradient. Used when no ``probe_ring`` is configured (M6
          byte-equivalent) or when no stationary predators exist on
          the env.

        When ``brain`` is provided, the probe params are filled with
        zero-valued ``stam_state`` sized/typed to match the brain's
        runtime feature pipeline. Without this, the brain's input
        shape (which depends on the env-derived STAM dim) silently
        diverges from the synthetic BrainParams shape (which defaults
        to the registry's hard-coded STAM ``classical_dim=11``), and
        ``feature_norm`` raises at the first probe step.
        """
        # Imports here to avoid coupling loop module load to the agent
        # subpackage when TEI is unused.
        from quantumnematode.brain.arch import BrainParams
        from quantumnematode.brain.modules import SENSORY_MODULES, ModuleName
        from quantumnematode.env import Direction

        # Derive the brain's effective STAM dim from its known input_dim
        # minus the sum of non-STAM module classical_dims. This pins the
        # synthetic ``stam_state`` to the same shape that the agent
        # runner constructs at training time, where the env's
        # ``stam_dim_from_env`` is the source of truth. See the
        # ``STAMSensoryModule.to_classical`` branch that returns
        # ``np.zeros(self.classical_dim)`` when ``params.stam_state is
        # None`` — the registry instance's ``classical_dim`` defaults to
        # 11 (4-channel mode), which doesn't match a 2-channel env's
        # 7-element STAM state.
        stam_state: tuple[float, ...] | None = None
        # ``sensory_modules`` and ``input_dim`` are LSTMPPO/MLPPPO
        # implementation attributes, not on the Brain protocol;
        # ``getattr`` with sentinel + hasattr guards preserve the
        # backwards-compat path for brains that lack them.
        brain_modules = getattr(brain, "sensory_modules", None) if brain is not None else None
        brain_input_dim = getattr(brain, "input_dim", None) if brain is not None else None
        if brain_modules is not None and brain_input_dim is not None:
            non_stam_total = 0
            has_stam = False
            for m in brain_modules:
                if m == ModuleName.STAM:
                    has_stam = True
                    continue
                sensory_module = SENSORY_MODULES.get(m)
                if sensory_module is not None:
                    non_stam_total += sensory_module.classical_dim
                else:
                    # Match ``extract_classical_features``'s 2-feature
                    # fallback for unknown modules.
                    non_stam_total += 2
            if has_stam:
                effective_stam_dim = int(brain_input_dim) - non_stam_total
                if effective_stam_dim < 0:
                    logger.warning(
                        "F0 probe: brain.input_dim=%d is smaller than the sum of "
                        "non-STAM module classical_dims=%d. Falling back to "
                        "stam_state=None (the registry default will be used).",
                        brain_input_dim,
                        non_stam_total,
                    )
                else:
                    stam_state = tuple(0.0 for _ in range(effective_stam_dim))

        # M6.11 env-derived probe ring path. Falls back to the legacy
        # synthetic-probe path when ``env`` is None, no ``probe_ring``
        # config is set, or the env has no stationary predators.
        probe_ring_cfg: ProbeRingConfig | None = None
        cfg = self.sim_config.evolution
        if cfg is not None and cfg.transgenerational is not None:
            probe_ring_cfg = cfg.transgenerational.probe_ring
        if env is not None and probe_ring_cfg is not None:
            ring_probes = self._build_ring_probes(env, probe_ring_cfg, stam_state)
            if ring_probes:
                return ring_probes

        # Legacy M6 fallback: three synthetic probes with varying
        # food-gradient strengths and zero predator gradient. Active
        # when ``env`` is None, ``probe_ring`` is unset, or the env
        # has no stationary predators (e.g. non-pathogen-lawn envs).
        return [
            BrainParams(
                food_gradient_strength=0.3,
                food_gradient_direction=float(np.pi / 2),
                predator_gradient_strength=0.0,
                predator_gradient_direction=0.0,
                stam_state=stam_state,
                agent_direction=Direction.UP,
            ),
            BrainParams(
                food_gradient_strength=0.5,
                food_gradient_direction=float(np.pi),
                predator_gradient_strength=0.0,
                predator_gradient_direction=0.0,
                stam_state=stam_state,
                agent_direction=Direction.UP,
            ),
            BrainParams(
                food_gradient_strength=0.1,
                food_gradient_direction=0.0,
                predator_gradient_strength=0.0,
                predator_gradient_direction=0.0,
                stam_state=stam_state,
                agent_direction=Direction.UP,
            ),
        ]

    def _build_ring_probes(
        self,
        env: DynamicForagingEnvironment,
        probe_ring_cfg: ProbeRingConfig,
        stam_state: tuple[float, ...] | None,
    ) -> list[BrainParams]:
        """Build the M6.11 env-derived probe ring for F0 substrate extraction.

        Returns an empty list when the env has no stationary predators
        (caller falls back to the legacy synthetic-probe path). Each
        stationary predator contributes ``probe_ring.count`` probes
        sampled from a Manhattan-distance ring at exact L1 distance
        ``predator.damage_radius + probe_ring.radius_offset``. When
        ``include_food_gradient_variants`` is True, emits two probes
        per ring position (food-gradient zero + food-gradient set).

        When ``probe_ring.safe_probes`` is set, additional probes are
        appended at positions FAR from any predator (low
        ``predator_gradient_strength``) with varying food-gradient
        strengths. These give the substrate's bias-network MLP
        examples of safe-zone behaviour, so it can fit a
        *conditional* response (predator-near → avoid;
        predator-far + food-near → forage) rather than the
        unconditional "always-LEFT" bias that pilot-1 produced.
        """
        from quantumnematode.brain.arch import BrainParams
        from quantumnematode.env import Direction
        from quantumnematode.env.env import PredatorType

        stationary = [p for p in env.predators if p.predator_type is PredatorType.STATIONARY]
        if not stationary:
            return []
        ring_probes: list[BrainParams] = []
        for predator in stationary:
            radius = int(predator.damage_radius) + int(probe_ring_cfg.radius_offset)
            offsets = _sample_ring_offsets(radius, probe_ring_cfg.count)
            for dx, dy in offsets:
                probe_x = int(predator.position[0]) + dx
                probe_y = int(predator.position[1]) + dy
                strength, direction = _compute_probe_gradient(
                    (probe_x, probe_y),
                    (int(predator.position[0]), int(predator.position[1])),
                )
                # Default probe: food-gradient zero (pathogen-isolated).
                ring_probes.append(
                    BrainParams(
                        food_gradient_strength=0.0,
                        food_gradient_direction=0.0,
                        predator_gradient_strength=strength,
                        predator_gradient_direction=direction,
                        stam_state=stam_state,
                        agent_direction=Direction.UP,
                    ),
                )
                if probe_ring_cfg.include_food_gradient_variants:
                    # Second probe at the same position: same predator
                    # gradient but a moderate non-zero food gradient so
                    # the substrate sees the joint (predator + food)
                    # context too.
                    ring_probes.append(
                        BrainParams(
                            food_gradient_strength=0.3,
                            food_gradient_direction=float(np.pi / 2),
                            predator_gradient_strength=strength,
                            predator_gradient_direction=direction,
                            stam_state=stam_state,
                            agent_direction=Direction.UP,
                        ),
                    )
        # Append safe-zone probes when configured. These probe the
        # substrate at positions FAR from any predator, with varying
        # food-gradient strengths, so the MLP learns predator-low →
        # forage behaviour distinct from the near-predator → avoid
        # behaviour the ring captures.
        if probe_ring_cfg.safe_probes is not None:
            ring_probes.extend(
                self._build_safe_probes(env, probe_ring_cfg.safe_probes, stam_state),
            )
        return ring_probes

    def _build_safe_probes(
        self,
        env: DynamicForagingEnvironment,
        safe_cfg: SafeProbesConfig,
        stam_state: tuple[float, ...] | None,
    ) -> list[BrainParams]:
        """Build N probes at positions far from any stationary predator.

        Samples ``safe_cfg.count`` positions from the grid where the L1
        (Manhattan) distance to every stationary predator is at least
        ``safe_cfg.min_predator_distance``. Each probe has predator_gradient_strength
        computed from the actual env geometry (which will be small but
        not necessarily zero — the gradient_decay_constant tail still
        reaches), and food_gradient_strength sampled across [0, 1] in
        even increments so the MLP sees the full food-only response
        surface.

        Returns an empty list (with a logger warning) when no positions
        on the grid satisfy ``min_predator_distance`` — typically only
        happens when predators saturate the grid, which the ring-probe
        path would also have struggled with.
        """
        from quantumnematode.brain.arch import BrainParams
        from quantumnematode.env import Direction
        from quantumnematode.env.env import PredatorType

        stationary = [p for p in env.predators if p.predator_type is PredatorType.STATIONARY]
        # Collect candidate positions: every grid cell at min L1 distance from all predators.
        grid_size = env.grid_size
        candidates: list[tuple[int, int]] = []
        for x in range(grid_size):
            for y in range(grid_size):
                ok = True
                for p in stationary:
                    if (
                        abs(x - int(p.position[0])) + abs(y - int(p.position[1]))
                        < safe_cfg.min_predator_distance
                    ):
                        ok = False
                        break
                if ok:
                    candidates.append((x, y))
        if not candidates:
            logger.warning(
                "safe_probes: no grid positions at L1 distance >= %d from any predator; "
                "skipping safe-probe set.",
                safe_cfg.min_predator_distance,
            )
            return []
        # Sample ``count`` positions evenly from the candidate set
        # (deterministic — uses the candidate index, not RNG, so the
        # probe set is identical across calls for the same env).
        step = max(1, len(candidates) // safe_cfg.count)
        sampled = candidates[::step][: safe_cfg.count]
        safe_probes: list[BrainParams] = []
        for i, (probe_x, probe_y) in enumerate(sampled):
            # Compute the actual predator_gradient_strength + direction
            # the env would report at this position (uses
            # _compute_probe_gradient against the NEAREST stationary
            # predator to get the strongest signal — even at
            # min_predator_distance the gradient_decay tail isn't zero).
            if stationary:
                nearest = min(
                    stationary,
                    key=lambda p: (
                        abs(probe_x - int(p.position[0])) + abs(probe_y - int(p.position[1]))
                    ),
                )
                pred_strength, pred_direction = _compute_probe_gradient(
                    (probe_x, probe_y),
                    (int(nearest.position[0]), int(nearest.position[1])),
                )
            else:
                pred_strength, pred_direction = 0.0, 0.0
            # Vary food_gradient_strength across the probe set evenly
            # in [0.1, 1.0] so the MLP sees the response surface.
            food_strength = 0.1 + 0.9 * (i / max(1, safe_cfg.count - 1))
            safe_probes.append(
                BrainParams(
                    food_gradient_strength=food_strength,
                    food_gradient_direction=float(np.pi / 4),
                    predator_gradient_strength=pred_strength,
                    predator_gradient_direction=pred_direction,
                    stam_state=stam_state,
                    agent_direction=Direction.UP,
                ),
            )
        return safe_probes

    def _build_per_gen_sim_config(self, gen: int) -> SimulationConfig:
        """Build the per-generation ``sim_config`` for worker dispatch.

        Under transgenerational inheritance with a ``lawn_schedule``,
        produces a Pydantic-v2 ``model_copy`` of the base ``sim_config``
        with two overrides applied from the schedule entry for the
        current generation:

        - ``environment.predators.enabled`` ← ``pathogen_lawns_enabled``
          (pathogen lawn on/off for this generation).
        - ``evolution.learn_episodes_per_eval`` ← ``ppo_train_episodes``
          (train-phase episode count override — 0 is permitted under
          TEI's F1+ inheritance-without-retraining design, handled by
          the train-phase bypass in ``LearnedPerformanceFitness.evaluate``).

        Base ``sim_config`` is NEVER mutated — each generation's workers
        receive a distinct ``model_copy`` instance. Outside of
        transgenerational runs (or when the config block is absent),
        returns the base ``sim_config`` unchanged so behaviour is
        byte-equivalent for non-TEI strategies.

        The ``evolution`` override is layered on top of
        ``self.sim_config.evolution`` (what non-TEI workers already see),
        not ``self.evolution_config`` — keeping the worker's view of the
        evolution block consistent across TEI and non-TEI runs.
        """
        cfg = self.evolution_config
        # Bypass the schedule entirely when transgenerational is absent
        # OR explicitly disabled (the TEI-off control arm of a paired
        # ablation): the pairing validator pins ``enabled=false`` to
        # ``inheritance=none``, so the control arm receives the base
        # sim_config every generation and the substrate is the only
        # cross-arm difference.
        if cfg.transgenerational is None or not cfg.transgenerational.enabled:
            return self.sim_config
        # Find the schedule entry for the current generation. The
        # coverage validator on EvolutionConfig already enforced that
        # every generation in [0, generations) has exactly one matching
        # entry, so this lookup MUST succeed — fail loud if a future
        # validator regression lets a gap-y schedule through.
        entry = next(
            (e for e in cfg.transgenerational.lawn_schedule if e.generation == gen),
            None,
        )
        if entry is None:
            msg = (
                f"transgenerational.lawn_schedule has no entry for gen={gen}; "
                f"the EvolutionConfig coverage validator should have rejected "
                f"this schedule at config-load."
            )
            raise RuntimeError(msg)
        # Base the per-gen evolution copy on what workers already see
        # under non-TEI runs: ``self.sim_config.evolution`` (the raw
        # YAML-or-merged block). TEI runs use ``LearnedPerformanceFitness``
        # (CLI guard rejects success_rate under inheritance: transgenerational)
        # which requires ``sim_config.evolution`` to be set — see
        # ``LearnedPerformanceFitness.evaluate``'s rejection. The raise
        # documents that invariant.
        if self.sim_config.evolution is None:
            msg = (
                "transgenerational runs require sim_config.evolution to be set "
                "(LearnedPerformanceFitness rejects None); reached "
                "_build_per_gen_sim_config without it."
            )
            raise RuntimeError(msg)
        updated_evolution = self.sim_config.evolution.model_copy(
            update={"learn_episodes_per_eval": entry.ppo_train_episodes},
        )
        # Build the per-gen overrides. The environment's predators
        # block is optional in the base sim_config (None → no predators);
        # under TEI we expect predators to be configured already.
        env = self.sim_config.environment
        if env is None or env.predators is None:
            logger.warning(
                "transgenerational lawn_schedule expects an "
                "environment.predators block but none is configured; "
                "lawn enable/disable will have no effect for gen=%d.",
                gen,
            )
            return self.sim_config.model_copy(
                update={"evolution": updated_evolution},
            )
        updated_predators = env.predators.model_copy(
            update={"enabled": entry.pathogen_lawns_enabled},
        )
        updated_env = env.model_copy(update={"predators": updated_predators})
        return self.sim_config.model_copy(
            update={
                "environment": updated_env,
                "evolution": updated_evolution,
            },
        )

    def _compute_tei_prior_source(
        self,
        gen: int,
    ) -> tuple[Path, float, int] | None:
        """Compute the per-generation ``tei_prior_source`` for worker tuples.

        Same value for every child within a single generation (the F0
        substrate is one shared resource; ``decay_factor`` and
        ``lineage_depth`` are gen-level). At F0 (gen 0) or non-TEI
        runs, returns ``None``.

        At F1+: requires both an active transgenerational config (for
        ``decay_factor``) and a populated ``_tei_f0_substrate_path``
        attribute (populated by the F0 Substrate Extraction Pipeline
        at end of gen 0, persisted in the checkpoint pickle for resume).
        """
        cfg = self.evolution_config
        # Bypass when transgenerational is absent OR explicitly disabled
        # (TEI-off control arm), and at F0 (the elite IS the substrate
        # source, not consumer).
        if cfg.transgenerational is None or not cfg.transgenerational.enabled or gen == 0:
            return None
        if self._tei_f0_substrate_path is None:
            # Defensive: F0 extraction must have populated this before
            # the loop reaches gen 1. If it didn't (e.g., F0 weights
            # missing on disk), the extraction pipeline logged a warning
            # earlier; F1+ workers proceed without a substrate (None
            # passed through, no bias applied).
            logger.warning(
                "transgenerational at gen=%d but _tei_f0_substrate_path "
                "is unset; F1+ workers will have no substrate to load.",
                gen,
            )
            return None
        return (
            self._tei_f0_substrate_path,
            cfg.transgenerational.decay_factor,
            gen,
        )

    def _resolve_per_child_inheritance(  # noqa: PLR0911
        self,
        child_idx: int,
        gen: int,
        gid: str,
    ) -> tuple[Path | None, Path | None, str]:
        """Compute one child's (parent_warm_start, child_capture_path, inherited_from).

        Five-branch switch on the strategy's ``kind()``:

        - ``"none"`` → returns ``(None, None, "")``.  The loop's
          per-child step short-circuits to from-scratch evaluation
          identically to a frozen-weight evolution run.
        - ``"trait"`` (Baldwin) → returns ``(None, None, parent_id)``
          where ``parent_id`` is the prior generation's elite
          (``self._selected_parent_ids[0]`` if set, else ``""`` for
          gen 0).  No checkpoint paths are computed; the child trains
          from-scratch but its lineage row records the elite ID.
        - ``"transgenerational"`` → returns ``(None, capture_path, parent_id)``
          where ``capture_path`` is non-None at gen 0 (F0 weight
          capture for the post-eval substrate extraction pipeline)
          and ``None`` at gen 1+. Pure-TEI does not use per-child
          weight-IO inheritance paths at F1+; the F0 substrate is
          captured by a separate loop-level pipeline that fires AFTER
          F0 evaluation completes, reads the captured ``.pt`` files,
          extracts the bias, saves the substrate as ``.tei.pt``, and
          GCs the F0 ``.pt`` files. F1+ workers receive the substrate
          path via a separate ``tei_prior_source`` kwarg into
          ``fitness.evaluate``, not via ``weight_capture_path``.
        - ``"weights"`` (Lamarckian) → returns the full
          ``(parent_warm_start, child_capture_path, parent_id)``
          tuple.  ``parent_warm_start`` is the ``Path`` to the parent's
          pre-saved checkpoint, or ``None`` if (a) gen 0, (b)
          ``assign_parent`` returned ``None``, or (c) the parent file
          is unexpectedly missing on disk (defensive fallback with a
          ``logger.warning``).
        - ``"weights+transgenerational"`` (composed) → returns the
          SAME tuple shape as ``"weights"`` (per-child warm-start path
          + capture path + parent ID). The composed-mode branch
          handles both F0 capture AND F1+ warm-start through the
          Lamarckian-pattern weight-IO path; the substrate flow is
          orthogonal and goes through ``_compute_tei_prior_source`` /
          ``_run_f0_substrate_extraction`` as it would for pure-TEI.
          F1+ workers receive BOTH ``warm_start_path_override`` (from
          this branch) AND ``tei_prior_source`` (from the substrate-
          flow path) in their ``fitness.evaluate`` call.
        """
        kind = self.inheritance.kind()
        if kind == "none":
            return None, None, ""
        if kind == "trait":
            parent_id = self._selected_parent_ids[0] if self._selected_parent_ids else ""
            return None, None, parent_id
        if kind == "transgenerational":
            parent_id = self._selected_parent_ids[0] if self._selected_parent_ids else ""
            # Special-case F0: capture trained weights to disk so the
            # post-eval F0 Substrate Extraction Pipeline can re-load
            # the elite's brain and run the telemetry pass. F1+
            # children inherit the substrate via a kwarg path, not via
            # ``weight_capture_path``, so capture is disabled there.
            # The ``.pt`` extension here matches Lamarckian's capture
            # path; the substrate-extraction pipeline reads these
            # ``.pt`` files and rewrites the result as ``.tei.pt`` (per
            # ``TransgenerationalInheritance.checkpoint_path``), then
            # GCs the ``.pt`` files. Distinct extensions prevent on-disk
            # collisions.
            if gen == 0:
                capture_path = (
                    self.output_dir / "inheritance" / f"gen-{gen:03d}" / f"genome-{gid}.pt"
                )
                return None, capture_path, parent_id
            return None, None, parent_id
        # Both ``"weights"`` (Lamarckian) and ``"weights+transgenerational"``
        # (composed) follow the per-child warm-start + capture pattern:
        # both write per-genome ``.pt`` files for every child (including
        # F1+), and both warm-start each F1+ child from its parent's
        # saved checkpoint. Composed mode additionally threads the F0
        # substrate through ``tei_prior_source`` (handled by
        # ``_compute_tei_prior_source``), but that's an orthogonal flow.
        # The strategy's ``checkpoint_path`` returns the canonical
        # ``inheritance/gen-NNN/genome-<gid>.pt`` path for both kinds.
        child_capture_path = self.inheritance.checkpoint_path(self.output_dir, gen, gid)
        parent_id = self.inheritance.assign_parent(child_idx, self._selected_parent_ids)
        if parent_id is None:
            return None, child_capture_path, ""
        candidate = self.inheritance.checkpoint_path(self.output_dir, gen - 1, parent_id)
        # Path.exists() guard: defensive against unexpected on-disk
        # state on resume.  Fall back to from-scratch rather than
        # crashing the whole generation.
        if candidate is not None and candidate.exists():
            return candidate, child_capture_path, parent_id
        logger.warning(
            "Lamarckian parent checkpoint missing for child idx=%d gen=%d "
            "(expected parent_id=%s at %s) — falling back to from-scratch.",
            child_idx,
            gen,
            parent_id,
            candidate,
        )
        return None, child_capture_path, ""

    # ---- Main loop --------------------------------------------------------

    # PLR0915: this method is the canonical sequential
    # ask→eval→tell→inherit→checkpoint pipeline.  The statements are the
    # documented optional paths (parallel-pool vs sequential, no-op vs
    # active inheritance branching); inline statements keep the
    # per-genome data flow that downstream readers (and the spec)
    # reason about visible at one read.
    def run(self, *, resume_from: Path | None = None) -> EvolutionResult:  # noqa: C901, PLR0912, PLR0915
        """Run the evolution loop, optionally resuming from a checkpoint.

        Parameters
        ----------
        resume_from
            Path to a pickle checkpoint produced by a prior run.  If supplied,
            the optimiser, RNG, and generation index are restored before the
            loop continues.  The lineage CSV is appended (not truncated).

        Returns
        -------
        EvolutionResult
            The best parameters and fitness found across all generations.
        """
        if resume_from is not None:
            self._load_checkpoint(resume_from)
            logger.info(
                "Resuming evolution from checkpoint: generation=%d",
                self._generation,
            )

        cfg = self.evolution_config
        # Brain-keyed encoders set brain_name to a real brain name
        # (e.g. "mlpppo"); brain-agnostic encoders use the empty string.
        # Fall back to sim_config.brain.name in the empty case so all
        # runs record the actual brain identity in lineage.csv and
        # best_params.json regardless of encoder type.  sim_config.brain
        # is guaranteed non-None here: the CLI entry point validates it
        # before constructing the loop.
        brain_type = self.encoder.brain_name or (
            self.sim_config.brain.name if self.sim_config.brain else ""
        )

        pool: PoolType | None = None
        if cfg.parallel_workers > 1:
            pool = Pool(
                processes=cfg.parallel_workers,
                initializer=_init_worker,
                initargs=(self.log_level,),
            )

        try:
            while self._generation < cfg.generations:
                gen = self._generation
                logger.info("Generation %d / %d", gen, cfg.generations)

                # Per-generation sim_config: under transgenerational, apply
                # the lawn_schedule overrides (pathogen_lawns_enabled,
                # ppo_train_episodes) via ``model_copy`` so the base
                # sim_config is never mutated. Each generation's workers
                # receive a distinct copy. When transgenerational config
                # is absent (the default for all non-TEI runs), every
                # worker receives the unmodified base sim_config —
                # byte-equivalent to the prior behaviour.
                sim_config_for_gen = self._build_per_gen_sim_config(gen)

                # Compute the F1+ tei_prior_source once per generation
                # (same value for every child within a generation). At
                # F0 (gen 0) or non-TEI runs, this is None.
                tei_prior_source_for_gen = self._compute_tei_prior_source(gen)

                # Ask optimiser for population
                solutions = self.optimizer.ask()

                # Wrap as Genomes with deterministic IDs and shared parent_ids.
                # Every gen-N candidate has the entire gen-(N-1) ID list as
                # parents — a uniform convention across CMA-ES and GA, since
                # neither optimiser exposes per-child parent provenance.
                parent_ids = list(self._prev_generation_ids)
                gen_ids: list[str] = []
                eval_args: list[tuple] = []
                # Per-child inherited-from tracking for the lineage CSV.
                # Empty string means "from-scratch" (no-inheritance, gen 0,
                # or fallback when the parent file is missing).
                inherited_from_per_child: list[str] = []
                for idx, params in enumerate(solutions):
                    gid = genome_id_for(gen, idx, parent_ids)
                    gen_ids.append(gid)
                    # Per-evaluation seed derived from the parent rng so each
                    # evaluation in the population is seeded distinctly but
                    # reproducibly.
                    eval_seed = int(self.rng.integers(0, 2**31 - 1))

                    # Per-child inheritance resolution: returns (None, None, "")
                    # under no-op so the worker call shape and fitness
                    # behaviour are byte-equivalent to a frozen-weight
                    # evolution run.
                    parent_warm_start, child_capture_path, inherited_from = (
                        self._resolve_per_child_inheritance(idx, gen, gid)
                    )
                    inherited_from_per_child.append(inherited_from)

                    # eval_diagnostics.jsonl: per-genome telemetry
                    # (success_rate + survival_rate + deaths + ...).
                    # Single per-session file; workers append. Used by
                    # the decision-gate machinery (T1 envelope check,
                    # T3 M6-floor-to-beat, post-hoc aggregator). Only
                    # ``LearnedPerformanceFitness`` accepts it; other
                    # fitness functions (incl. test stubs) get ``None``
                    # so the worker's signature-dispatch elides the
                    # kwarg before forwarding. Signature-check at the
                    # loop layer rather than the worker layer so the
                    # worker pickling boundary doesn't need to import
                    # ``LearnedPerformanceFitness``.
                    diagnostics_path = (
                        self.output_dir / "eval_diagnostics.jsonl"
                        if "diagnostics_path" in inspect.signature(self.fitness.evaluate).parameters
                        else None
                    )
                    eval_args.append(
                        (
                            np.asarray(params, dtype=np.float32),
                            sim_config_for_gen,
                            self.encoder,
                            self.fitness,
                            cfg.episodes_per_eval,
                            eval_seed,
                            gen,
                            idx,
                            parent_ids,
                            parent_warm_start,
                            child_capture_path,
                            tei_prior_source_for_gen,
                            diagnostics_path,
                        ),
                    )

                # Evaluate fitness in parallel or sequentially.
                if pool is not None:
                    fitnesses = pool.map(_evaluate_in_worker, eval_args)
                else:
                    fitnesses = [_evaluate_in_worker(args) for args in eval_args]

                # Record lineage (positive fitness = success rate).
                for idx, (gid, fit) in enumerate(zip(gen_ids, fitnesses, strict=True)):
                    genome = Genome(
                        params=np.asarray(solutions[idx], dtype=np.float32),
                        genome_id=gid,
                        parent_ids=parent_ids,
                        generation=gen,
                        birth_metadata=build_birth_metadata(self.sim_config),
                    )
                    self._lineage.record(
                        genome,
                        fitness=fit,
                        brain_type=brain_type,
                        inherited_from=inherited_from_per_child[idx],
                    )

                # Tell optimiser (negate: optimisers minimise, our fitness maximises).
                neg_fitnesses = [-f for f in fitnesses]
                self.optimizer.tell(list(solutions), neg_fitnesses)

                # Early-stop counter update.  Placed BEFORE the inheritance
                # guards so the counter reflects the just-evaluated
                # generation by the time the early-stop check fires at the
                # end of the body.
                #
                # Compare ``current_best`` against the **previous
                # generation's** best (``prev``), not against a running
                # maximum.  A noisy regression-then-recovery trajectory like
                # ``[0.3, 0.5, 0.4, 0.45, 0.46]`` should reset the counter
                # at gen 4 (0.45 > 0.4 = prev gen) instead of penalising
                # every gen that fails to surpass the all-time peak — the
                # spec says "strictly greater than the prior generation's"
                # and we honour that literally.
                #
                # ``prev is None`` is the gen-1 bootstrap branch: no
                # previous to compare against, treat as an "improvement"
                # (counter stays 0).
                #
                # ``_last_best_fitness`` is assigned UNCONDITIONALLY at the
                # end so the next generation's ``prev`` is the actual prior
                # value, regardless of whether this generation improved
                # over its prior or not.
                prev = self._last_best_fitness
                current_best = max(fitnesses)
                if prev is None or current_best > prev:
                    self._gens_without_improvement = 0
                else:
                    self._gens_without_improvement += 1
                self._last_best_fitness = current_best

                # Inheritance step (two-guard split):
                #
                # 1. Lineage-tracking guard: any non-no-op strategy
                #    (Lamarckian OR Baldwin) updates ``_selected_parent_ids``
                #    so the next generation's children can populate the
                #    lineage CSV's ``inherited_from`` column.
                # 2. Weight-IO GC guard: weight-flow strategies
                #    (Lamarckian AND composed) write per-genome
                #    checkpoints, so only they need GC. Phase one
                #    clears all remaining files in the previous-
                #    generation directory (keep=[]) because the gen-N
                #    children that inherited from them have just
                #    finished evaluating, so those checkpoints are no
                #    longer needed; no-op when gen is zero. Phase two
                #    keeps only ``next_selected`` in the current-
                #    generation directory so the about-to-evaluate
                #    gen-(N+1) children can read their parent file.
                #    Steady-state disk usage after this step is at most
                #    ``inheritance_elite_count`` files, bounded over
                #    the whole run.
                next_selected: list[str] = []
                if self._inheritance_records_lineage():
                    next_selected = self.inheritance.select_parents(
                        gen_ids,
                        list(fitnesses),
                        gen,
                    )
                    self._selected_parent_ids = next_selected
                    if self._inheritance_active():
                        # Drop the previous-gen elites whose children just ran.
                        self._gc_inheritance_dir(gen - 1, [])
                        # Keep only the about-to-be-parents in current gen.
                        self._gc_inheritance_dir(gen, next_selected)

                # Per-gen elite snapshot: append the top-1 elite's
                # (generation, genome_id, params, fitness) to a
                # JSON-Lines artifact at ``per_gen_elites.jsonl``. Used
                # by post-hoc evaluators (e.g. the transgenerational
                # per-gen choice-index evaluator) that need to
                # reconstruct each generation's elite genome offline.
                # Runs UNCONDITIONALLY — including NoInheritance arms
                # (the TEI-off control) — so the paired-arm decision
                # gate can compare both sides. Under
                # ``_inheritance_records_lineage`` the elite ID comes
                # from the strategy's ``select_parents`` for parity
                # with the lineage CSV; otherwise it's the argmax of
                # the gen's fitness vector.
                if next_selected:
                    elite_id_this_gen = next_selected[0]
                else:
                    elite_argmax = int(np.argmax(fitnesses))
                    elite_id_this_gen = gen_ids[elite_argmax]
                try:
                    elite_idx_this_gen = gen_ids.index(elite_id_this_gen)
                except ValueError:
                    logger.warning(
                        "per_gen_elites: elite_id=%s not in gen_ids; skipping snapshot for gen=%d.",
                        elite_id_this_gen,
                        gen,
                    )
                else:
                    self._append_per_gen_elite(
                        generation=gen,
                        genome_id=elite_id_this_gen,
                        params=np.asarray(
                            solutions[elite_idx_this_gen],
                            dtype=np.float32,
                        ),
                        fitness=fitnesses[elite_idx_this_gen],
                    )

                # F0 Substrate Extraction Pipeline (transgenerational only,
                # gen 0 only): load the F0 elite's captured weights, decode
                # a fresh brain, invoke ``extract_from_brain`` to compute
                # the substrate's logit_bias, save the ``.tei.pt`` artifact
                # under ``inheritance/gen-000/genome-{elite_id}.tei.pt``,
                # then GC every F0 ``.pt`` weight file (only the
                # ``.tei.pt`` is retained for the cascade). Reads
                # ``self._selected_parent_ids`` (set by the lineage block
                # above) rather than ``next_selected`` so the check works
                # even when the lineage block was skipped (it isn't,
                # since transgenerational satisfies
                # ``_inheritance_records_lineage()`` — but the indirection
                # keeps the guards independent and pyright happy).
                if self._substrate_inheritance_active() and gen == 0 and self._selected_parent_ids:
                    self._run_f0_substrate_extraction(
                        elite_id=self._selected_parent_ids[0],
                        gen_ids=gen_ids,
                        solutions=solutions,
                    )

                # Bookkeeping: prev gen for next iteration's parent_ids.
                self._prev_generation_ids = gen_ids
                self._generation += 1

                if self._generation % cfg.checkpoint_every == 0:
                    self._save_checkpoint()
                    logger.info("Checkpoint saved at generation %d", self._generation)

                # Early-stop check at the END of the body, AFTER both
                # inheritance guards (so Baldwin's _selected_parent_ids
                # is populated for resume invariants and Lamarckian's GC
                # has preserved the surviving elite checkpoint), AFTER
                # _generation += 1 (so the saved checkpoint records the
                # post-evaluation increment value), and AFTER the
                # checkpoint_every save.  The existing post-loop final
                # _save_checkpoint() handles persistence of the
                # early-stopped state — control flows out of `while` via
                # break and through the existing post-loop save site.
                # On resume, _generation < cfg.generations re-enters the
                # loop with the saturation counter intact.
                if (
                    cfg.early_stop_on_saturation is not None
                    and self._gens_without_improvement >= cfg.early_stop_on_saturation
                ):
                    # 1-indexed gen labels for human readability (matches
                    # the spec scenarios + the aggregator's gen-to-0.92
                    # convention).  ``last_improvement_gen`` is the gen
                    # index at which _last_best_fitness was last updated.
                    last_improvement_gen = self._generation - self._gens_without_improvement
                    last_improvement_label = (
                        str(last_improvement_gen)
                        if self._last_best_fitness is not None
                        else "no improvement observed"
                    )
                    logger.info(
                        "Early-stop: best_fitness has not improved for %d "
                        "generations (last improvement at gen %s)",
                        cfg.early_stop_on_saturation,
                        last_improvement_label,
                    )
                    break
        finally:
            if pool is not None:
                pool.close()
                with suppress(Exception):
                    pool.join()

        # Final results.
        result = self.optimizer.result
        # The optimiser's stored fitness is the negated value we fed it; flip back
        # to "positive = better" before writing artefacts.
        positive_best_fitness = -result.best_fitness
        positive_history = [
            {
                **h,
                "best_fitness": -h.get("best_fitness", float("inf")),
                "mean_fitness": -h.get("mean_fitness", 0.0),
            }
            for h in result.history
        ]
        result_positive = EvolutionResult(
            best_params=result.best_params,
            best_fitness=positive_best_fitness,
            generations=result.generations,
            history=positive_history,
        )

        # Always save final state on completion.
        self._save_checkpoint()
        self._write_artefacts(result_positive)

        return result_positive

    # ---- Artefacts --------------------------------------------------------

    def _write_artefacts(self, result: EvolutionResult) -> None:
        """Write best_params.json + history.csv to ``output_dir``."""
        best_path = self.output_dir / "best_params.json"
        with best_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "best_params": list(result.best_params),
                    "best_fitness": result.best_fitness,
                    "generations": result.generations,
                    "brain_type": self.encoder.brain_name
                    or (self.sim_config.brain.name if self.sim_config.brain else ""),
                    "checkpoint_version": CHECKPOINT_VERSION,
                },
                handle,
                indent=2,
            )

        history_path = self.output_dir / "history.csv"
        # history is a list[dict] with consistent keys; write CSV from the first
        # dict's keys to preserve column order.
        if result.history:
            import csv

            keys: list[str] = list(result.history[0].keys())
            with history_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                for row in result.history:
                    writer.writerow(row)
