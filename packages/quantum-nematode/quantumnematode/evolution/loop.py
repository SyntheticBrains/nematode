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

import json
import logging
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

    from quantumnematode.evolution.encoders import GenomeEncoder
    from quantumnematode.evolution.fitness import FitnessFunction
    from quantumnematode.optimizers.evolutionary import EvolutionaryOptimizer
    from quantumnematode.utils.config_loader import EvolutionConfig, SimulationConfig

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
        weight_capture_path, tei_prior_source)``.  All elements must be
        picklable.  The three trailing ``Path | None`` / tuple fields are
        used by the inheritance step:

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
    ) = args
    genome = Genome(
        params=np.asarray(params, dtype=np.float32),
        genome_id=genome_id_for(generation, index, parent_ids),
        parent_ids=parent_ids,
        generation=generation,
        birth_metadata=build_birth_metadata(sim_config),
    )
    # Only LearnedPerformanceFitness accepts the inheritance kwargs;
    # EpisodicSuccessRate's signature accepts them for ABI symmetry but
    # ignores them. Detect by signature rather than type-import to avoid
    # coupling the worker to a specific fitness class. When all three
    # kwargs are None (no-inheritance case) we drop them entirely so the
    # call shape is byte-equivalent to a frozen-weight evolution run.
    if (
        warm_start_path_override is not None
        or weight_capture_path is not None
        or tei_prior_source is not None
    ):
        return fitness.evaluate(
            genome,
            sim_config,
            encoder,
            episodes=episodes,
            seed=seed,
            warm_start_path_override=warm_start_path_override,
            weight_capture_path=weight_capture_path,
            tei_prior_source=tei_prior_source,
        )
    return fitness.evaluate(genome, sim_config, encoder, episodes=episodes, seed=seed)


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
        warm-start lookup.  Currently only :class:`LamarckianInheritance`
        returns ``"weights"`` from ``kind()``.
        """
        return self.inheritance.kind() == "weights"

    def _inheritance_records_lineage(self) -> bool:
        """Return True iff the active strategy populates the lineage CSV's `inherited_from`.

        Gates the per-generation ``select_parents`` call and the
        ``_selected_parent_ids`` update.  :class:`LamarckianInheritance`,
        :class:`BaldwinInheritance`, and
        :class:`~quantumnematode.evolution.transgenerational_inheritance.TransgenerationalInheritance`
        all return ``True`` here; only :class:`NoInheritance` returns
        ``False``.
        """
        return self.inheritance.kind() != "none"

    def _substrate_inheritance_active(self) -> bool:
        """Return True iff the active strategy uses the TEI substrate-flow pipeline.

        Gates the F0 Substrate Extraction Pipeline (F0 weight capture
        + post-eval extraction + ``.tei.pt`` save + F0 ``.pt`` GC).
        Distinct from ``_inheritance_active()`` (which gates Lamarckian
        per-child weight-IO) because the transgenerational substrate
        flow is a separate loop-level pipeline that runs ONCE at gen
        0, not a per-child operation. Currently only
        :class:`~quantumnematode.evolution.transgenerational_inheritance.TransgenerationalInheritance`
        returns ``"transgenerational"`` from ``kind()``.
        """
        return self.inheritance.kind() == "transgenerational"

    def _gc_inheritance_dir(self, generation: int, keep_ids: list[str]) -> None:
        """Garbage-collect non-survivor checkpoints in one inheritance directory.

        Deletes every file in ``inheritance/gen-{generation:03d}/`` whose
        genome ID is NOT in ``keep_ids``.  No-op when the directory does
        not exist (e.g. on the very first GC pass before any inheritance
        file has been written, or when ``generation < 0``).  Files are
        matched by the canonical ``genome-<gid>.pt`` name shape; the
        ``<gid>`` token is extracted via ``stem.removeprefix("genome-")``
        (cleaner than regex for the fixed pattern).
        """
        if generation < 0:
            return
        gen_dir = self.output_dir / "inheritance" / f"gen-{generation:03d}"
        if not gen_dir.exists():
            return
        keep = set(keep_ids)
        for path in gen_dir.glob("genome-*.pt"):
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

        # Build the probe params sequence. For now we use a minimal
        # deterministic set; the spec's "ring of probe positions" can
        # be made more sophisticated in a follow-up. Each probe is a
        # synthetic BrainParams varying the gradient strength to
        # simulate proximity to a pathogen lawn.
        probe_params = self._build_f0_probe_params()

        # Extraction seed: fixed sentinel for now. A configurable
        # ``cfg.transgenerational.extraction_seed`` Pydantic field is
        # a follow-up addition tracked in the OpenSpec change; until
        # it lands, use a stable default so the F0 telemetry pass is
        # deterministic across runs.
        extraction_seed = 424242

        substrate = extract_from_brain(
            brain=brain,
            probe_params=probe_params,
            rng_seed=extraction_seed,
            source_genome_id=elite_id,
        )

        # Save the substrate at the canonical ``.tei.pt`` path produced
        # by the strategy's ``checkpoint_path``. The path-builder is
        # the single source of truth for reader/writer alignment.
        substrate_path = self.inheritance.checkpoint_path(
            self.output_dir,
            generation=0,
            genome_id=elite_id,
        )
        if substrate_path is None:
            # Defensive: the strategy contract guarantees a non-None
            # path for transgenerational, but if a future strategy
            # variant returns None we should log and skip rather than
            # crash the run.
            logger.warning(
                "F0 substrate extraction: strategy.checkpoint_path returned "
                "None for elite_id=%s; substrate not saved.",
                elite_id,
            )
            return
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
        gen_dir = self.output_dir / "inheritance" / "gen-000"
        for path in gen_dir.iterdir():
            if path.suffix == ".pt" and not path.name.endswith(".tei.pt"):
                path.unlink(missing_ok=True)

    def _build_f0_probe_params(self) -> list:
        """Build a deterministic sequence of probe ``BrainParams`` for F0 telemetry.

        Minimal default implementation: three synthetic params with
        varying gradient strengths to simulate "near a pathogen lawn"
        sensory states. A follow-up commit may replace this with a
        ring-of-probe-positions generator that uses the env's
        STATIONARY-predator coordinates (per the spec's "deterministic
        set built relative to the lawn position" guidance).
        """
        # Imports here to avoid coupling loop module load to the agent
        # subpackage when TEI is unused.
        from quantumnematode.brain.arch import BrainParams
        from quantumnematode.env import Direction

        return [
            BrainParams(
                food_gradient_strength=0.3,
                food_gradient_direction=float(np.pi / 2),
                agent_direction=Direction.UP,
            ),
            BrainParams(
                food_gradient_strength=0.5,
                food_gradient_direction=float(np.pi),
                agent_direction=Direction.UP,
            ),
            BrainParams(
                food_gradient_strength=0.1,
                food_gradient_direction=0.0,
                agent_direction=Direction.UP,
            ),
        ]

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
        # config validator already verified that every generation in
        # [0, generations) has exactly one matching entry.
        entry = next(
            (e for e in cfg.transgenerational.lawn_schedule if e.generation == gen),
            None,
        )
        if entry is None:
            # Defensive — shouldn't be reachable given the validator,
            # but log + fall back to base config rather than crash.
            logger.warning(
                "transgenerational.lawn_schedule has no entry for gen=%d; "
                "using base sim_config unchanged.",
                gen,
            )
            return self.sim_config
        # Base the per-gen evolution copy on what workers already see
        # under non-TEI runs: ``self.sim_config.evolution`` (the raw
        # YAML-or-merged block). Falls back to ``self.evolution_config``
        # when ``sim_config.evolution`` is absent — the CLI requires
        # an ``evolution:`` block for any TEI run because the fitness
        # function does too, but the fallback keeps the helper safe to
        # call from any code path.
        base_evolution = self.sim_config.evolution or self.evolution_config
        updated_evolution = base_evolution.model_copy(
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

        Four-branch switch on the strategy's ``kind()``:

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
          and ``None`` at gen 1+. Transgenerational inheritance
          does not use per-child weight-IO inheritance paths at
          F1+; the F0 substrate is captured by a separate loop-level
          pipeline that fires AFTER F0 evaluation completes, reads
          the captured ``.pt`` files, extracts the bias via
          ``TransgenerationalMemory.extract_from_brain``, saves
          the substrate as ``.tei.pt``, and GCs the F0 ``.pt``
          files. F1+ workers receive the substrate path via a
          separate kwarg into ``fitness.evaluate`` (a follow-up
          addition), not via the per-child ``weight_capture_path``
          field. The lineage CSV's ``inherited_from`` column records
          the F0 elite ID from gen 1 onwards (same as Baldwin).
        - ``"weights"`` (Lamarckian) → returns the full
          ``(parent_warm_start, child_capture_path, parent_id)``
          tuple.  ``parent_warm_start`` is the ``Path`` to the parent's
          pre-saved checkpoint, or ``None`` if (a) gen 0, (b)
          ``assign_parent`` returned ``None``, or (c) the parent file
          is unexpectedly missing on disk (defensive fallback with a
          ``logger.warning``).
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
        # The remaining branch handles ``kind == "weights"`` (Lamarckian).
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
                # 2. Weight-IO GC guard: only weight-flow strategies
                #    (Lamarckian) write per-genome checkpoints, so only
                #    they need GC.  Phase one clears all remaining files
                #    in the previous-generation directory (keep=[]) because
                #    the gen-N children that inherited from them have just
                #    finished evaluating, so those checkpoints are no
                #    longer needed; no-op when gen is zero.  Phase two
                #    keeps only ``next_selected`` in the current-generation
                #    directory so the about-to-evaluate gen-(N+1) children
                #    can read their parent file.  Steady-state disk usage
                #    after this step is at most ``inheritance_elite_count``
                #    files, bounded over the whole run.
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
