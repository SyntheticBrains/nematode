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
    from pathlib import Path

    from quantumnematode.evolution.encoders import GenomeEncoder
    from quantumnematode.evolution.fitness import FitnessFunction
    from quantumnematode.optimizers.evolutionary import EvolutionaryOptimizer
    from quantumnematode.utils.config_loader import EvolutionConfig, SimulationConfig

logger = logging.getLogger(__name__)


# Pickle checkpoint version.  Bump if the checkpoint dict shape changes in
# a backwards-incompatible way; `_load_checkpoint` validates this and
# refuses incompatible checkpoints.  v2 (M3) adds ``selected_parent_ids``
# (list of genome IDs whose Lamarckian checkpoints survive into the next
# generation) and ``inheritance`` (the literal "none"/"lamarckian" string
# the run was launched with — used to reject mismatched-setting resumes).
CHECKPOINT_VERSION = 2


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
        weight_capture_path)``.  All elements must be picklable.  The two
        trailing ``Path | None`` fields are used by the inheritance step:
        when both are ``None`` (the no-inheritance case) the call site is
        identical to pre-M3.  ``encoder`` and ``fitness`` are class
        instances pickled by reference to their class definitions;
        concrete encoders/fitness functions in this module are top-level
        classes and pickle cleanly.

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
    ) = args
    genome = Genome(
        params=np.asarray(params, dtype=np.float32),
        genome_id=genome_id_for(generation, index, parent_ids),
        parent_ids=parent_ids,
        generation=generation,
        birth_metadata=build_birth_metadata(sim_config),
    )
    # Only LearnedPerformanceFitness accepts the inheritance kwargs;
    # EpisodicSuccessRate doesn't.  Detect by signature rather than
    # type-import to avoid coupling the worker to a specific fitness
    # class.  When both kwargs are None (no-inheritance case) we drop
    # them entirely so the call shape is identical to pre-M3.
    if warm_start_path_override is not None or weight_capture_path is not None:
        return fitness.evaluate(
            genome,
            sim_config,
            encoder,
            episodes=episodes,
            seed=seed,
            warm_start_path_override=warm_start_path_override,
            weight_capture_path=weight_capture_path,
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
        # Defaults to NoInheritance() so the loop is byte-equivalent to
        # pre-M3 when no strategy is supplied.  Use a None sentinel +
        # late-construction (rather than ``= NoInheritance()`` in the
        # signature) to avoid a shared default instance across calls.
        self.inheritance: InheritanceStrategy = (
            inheritance if inheritance is not None else NoInheritance()
        )

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

    # ---- Checkpoint / Resume ---------------------------------------------

    def _save_checkpoint(self) -> None:
        """Pickle optimiser + RNG + generation + lineage path + version."""
        # ``inheritance`` is persisted as the literal "none"/"lamarckian"
        # string (not the strategy instance) so resume can compare against
        # the resolved current value and reject mismatches.
        inheritance_value = self.evolution_config.inheritance
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "optimizer": self.optimizer,
            "generation": self._generation,
            "prev_generation_ids": self._prev_generation_ids,
            "selected_parent_ids": self._selected_parent_ids,
            "inheritance": inheritance_value,
            "rng_state": self.rng.bit_generator.state,
            "lineage_path": str(self._lineage_path),
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

        # Reject mid-run inheritance changes.  Fires BEFORE any optimiser
        # state is restored so an inadvertent --inheritance override
        # doesn't waste compute on a corrupted run.
        checkpoint_inheritance = payload.get("inheritance", "none")
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
        self._selected_parent_ids = payload.get("selected_parent_ids", [])
        self.rng.bit_generator.state = payload["rng_state"]

    # ---- Inheritance helpers --------------------------------------------

    def _inheritance_active(self) -> bool:
        """Return True iff the active strategy uses per-genome checkpoints."""
        return not isinstance(self.inheritance, NoInheritance)

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

    def _resolve_per_child_inheritance(
        self,
        child_idx: int,
        gen: int,
        gid: str,
    ) -> tuple[Path | None, Path | None, str]:
        """Compute one child's (parent_warm_start, child_capture_path, inherited_from).

        Returns ``(None, None, "")`` when the active strategy is no-op
        — the loop's per-child step then short-circuits to from-scratch
        evaluation byte-equivalently to pre-M3.

        For active strategies, returns:

        - ``parent_warm_start``: ``Path`` to the parent's pre-saved
          checkpoint, or ``None`` if (a) gen 0, (b) ``assign_parent``
          returned ``None``, or (c) the parent file is unexpectedly
          missing on disk (defensive fallback with a ``logger.warning``).
        - ``child_capture_path``: ``Path`` where THIS child writes its
          post-K-train weights (always populated when inheritance is on).
        - ``inherited_from``: parent ``genome_id`` for the lineage CSV,
          or ``""`` for the from-scratch cases above.
        """
        if not self._inheritance_active():
            return None, None, ""
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
    def run(self, *, resume_from: Path | None = None) -> EvolutionResult:  # noqa: PLR0915
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
                inheritance_on = self._inheritance_active()
                for idx, params in enumerate(solutions):
                    gid = genome_id_for(gen, idx, parent_ids)
                    gen_ids.append(gid)
                    # Per-evaluation seed derived from the parent rng so each
                    # evaluation in the population is seeded distinctly but
                    # reproducibly.
                    eval_seed = int(self.rng.integers(0, 2**31 - 1))

                    # Per-child inheritance resolution: returns (None, None, "")
                    # under no-op so the worker call shape and fitness
                    # behaviour are byte-equivalent to pre-M3.
                    parent_warm_start, child_capture_path, inherited_from = (
                        self._resolve_per_child_inheritance(idx, gen, gid)
                    )
                    inherited_from_per_child.append(inherited_from)

                    eval_args.append(
                        (
                            np.asarray(params, dtype=np.float32),
                            self.sim_config,
                            self.encoder,
                            self.fitness,
                            cfg.episodes_per_eval,
                            eval_seed,
                            gen,
                            idx,
                            parent_ids,
                            parent_warm_start,
                            child_capture_path,
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

                # Inheritance step: select the next generation's parents
                # from the just-evaluated population, then garbage-collect
                # in two phases.  Phase one clears all remaining files in
                # the previous-generation directory (keep=[]) because the
                # gen-N children that inherited from them have just
                # finished evaluating, so those checkpoints are no longer
                # needed; no-op when gen is zero.  Phase two keeps only
                # ``next_selected`` in the current-generation directory
                # so the about-to-evaluate gen-(N+1) children can read
                # their parent file.  Steady-state disk usage after this
                # step is at most ``inheritance_elite_count`` files (the
                # next_selected set in gen-N), bounded over the whole run.
                if inheritance_on:
                    next_selected = self.inheritance.select_parents(
                        gen_ids,
                        list(fitnesses),
                        gen,
                    )
                    # Drop the previous-gen elites whose children just ran.
                    self._gc_inheritance_dir(gen - 1, [])
                    # Keep only the about-to-be-parents in current gen.
                    self._gc_inheritance_dir(gen, next_selected)
                    self._selected_parent_ids = next_selected

                # Bookkeeping: prev gen for next iteration's parent_ids.
                self._prev_generation_ids = gen_ids
                self._generation += 1

                if self._generation % cfg.checkpoint_every == 0:
                    self._save_checkpoint()
                    logger.info("Checkpoint saved at generation %d", self._generation)
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
