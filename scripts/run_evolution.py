# pragma: no cover
r"""Run brain-agnostic evolution with CMA-ES or GA.

The capability spec lives at ``openspec/specs/evolution-framework/``.

Examples
--------
::

    uv run python scripts/run_evolution.py \
        --config configs/evolution/mlpppo_foraging_small.yml

    uv run python scripts/run_evolution.py \
        --config configs/evolution/lstmppo_foraging_small_klinotaxis.yml \
        --generations 50 --population 16 --parallel 4

    # Resume an interrupted run
    uv run python scripts/run_evolution.py \
        --config configs/evolution/mlpppo_foraging_small.yml \
        --resume evolution_results/<session>/checkpoint.pkl

Timing
------
Per-episode cost is ~50 ms for MLPPPO and ~100 ms for LSTMPPO+klinotaxis at
``max_steps: 1000``.  Smoke runs of either brain complete in single-digit
seconds.

The one perf footgun: ``cma_diagonal`` MUST be ``true`` for any campaign
with a genome dim >~1000 (e.g. LSTMPPO weight evolution at n≈47k).  Full-
covariance CMA-ES ``tell()`` is O(n²) and at that size becomes minutes per
generation, regardless of episode count.  The shipped LSTMPPO pilot config
sets this; if you create a new config evolving brain weights at scale, set
it on yours too.  See the ``nematode-run-evolution`` skill's "When to
enable cma_diagonal" decision table.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from quantumnematode.evolution import (
    ENCODER_REGISTRY,
    EpisodicSuccessRate,
    EvolutionLoop,
    get_encoder,
)
from quantumnematode.logging_config import configure_file_logging, logger
from quantumnematode.optimizers.evolutionary import (
    CMAESOptimizer,
    EvolutionaryOptimizer,
    GeneticAlgorithmOptimizer,
)
from quantumnematode.utils.config_loader import (
    EvolutionConfig,
    load_simulation_config,
)
from quantumnematode.utils.session import generate_session_id


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Brain-agnostic evolutionary optimization.",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override evolution.generations.",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="Override evolution.population_size.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override evolution.episodes_per_eval.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["cmaes", "ga"],
        default=None,
        help="Override evolution.algorithm.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Override evolution.sigma0 (CMA-ES initial step size).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Override evolution.parallel_workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed for the loop (per-evaluation seeds derived from this).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint.pkl file to resume from.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evolution_results",
        help="Directory to write per-session subdirectory and artefacts (default: evolution_results).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def _resolve_evolution_config(
    sim_config_evolution: EvolutionConfig | None,
    args: argparse.Namespace,
) -> EvolutionConfig:
    """Compose the effective :class:`EvolutionConfig` for this run.

    Precedence: CLI flags (when explicitly set) > YAML ``evolution:`` block
    > :class:`EvolutionConfig` defaults.
    """
    base = sim_config_evolution if sim_config_evolution is not None else EvolutionConfig()
    overrides: dict[str, object] = {}
    if args.generations is not None:
        overrides["generations"] = args.generations
    if args.population is not None:
        overrides["population_size"] = args.population
    if args.episodes is not None:
        overrides["episodes_per_eval"] = args.episodes
    if args.algorithm is not None:
        overrides["algorithm"] = args.algorithm
    if args.sigma is not None:
        overrides["sigma0"] = args.sigma
    if args.parallel is not None:
        overrides["parallel_workers"] = args.parallel
    return base.model_copy(update=overrides) if overrides else base


def _build_optimizer(
    evolution_config: EvolutionConfig,
    num_params: int,
    seed: int | None,
) -> EvolutionaryOptimizer:
    """Construct the optimiser specified by ``evolution_config.algorithm``."""
    if evolution_config.algorithm == "cmaes":
        return CMAESOptimizer(
            num_params=num_params,
            population_size=evolution_config.population_size,
            sigma0=evolution_config.sigma0,
            seed=seed,
            diagonal=evolution_config.cma_diagonal,
        )
    if evolution_config.algorithm == "ga":
        return GeneticAlgorithmOptimizer(
            num_params=num_params,
            population_size=evolution_config.population_size,
            sigma0=evolution_config.sigma0,
            elite_fraction=evolution_config.elite_fraction,
            mutation_rate=evolution_config.mutation_rate,
            crossover_rate=evolution_config.crossover_rate,
            seed=seed,
        )
    msg = f"Unknown algorithm: {evolution_config.algorithm!r}"
    raise ValueError(msg)


def _resolve_output_dir(
    output_dir_arg: str,
    resume_path: Path | None,
) -> tuple[str, Path] | None:
    """Pick the session id + output directory.

    When ``resume_path`` is given, the resumed run MUST write into the
    original session's directory so ``lineage.csv`` stays a single
    chronological history (per the evolution-framework spec scenario
    "Append mode preserves history across resume").  We derive both the
    session id and the output directory from the checkpoint's parent
    directory; a fresh session id is only minted for new runs.  Returns
    ``None`` to signal a startup error (the caller propagates exit 1).
    """
    if resume_path is None:
        session_id = generate_session_id()
        return session_id, Path(output_dir_arg) / session_id

    if not resume_path.exists():
        logger.error("Checkpoint not found: %s", resume_path)
        return None
    output_dir = resume_path.parent.resolve()
    explicit_root = Path(output_dir_arg).resolve()
    if explicit_root != output_dir.parent:
        # We always prefer the checkpoint's location to preserve lineage
        # continuity; warn the user that the explicit --output-dir is
        # being ignored on resume.
        logger.warning(
            "--output-dir %s ignored on resume; writing into the "
            "checkpoint's session directory %s instead.",
            output_dir_arg,
            output_dir,
        )
    return output_dir.name, output_dir


def main() -> int:
    """Entry point."""
    args = parse_arguments()
    logger.setLevel(args.log_level)

    sim_config = load_simulation_config(args.config)
    evolution_config = _resolve_evolution_config(sim_config.evolution, args)

    if sim_config.brain is None or sim_config.brain.name is None:
        logger.error("Config must specify brain.name.")
        return 1

    brain_name = sim_config.brain.name
    if brain_name not in ENCODER_REGISTRY:
        # Surfaces the helpful error message from get_encoder.
        try:
            get_encoder(brain_name)
        except ValueError as exc:
            logger.error(str(exc))
        return 1

    encoder = get_encoder(brain_name)
    fitness = EpisodicSuccessRate()

    resume_path = Path(args.resume) if args.resume else None
    resolved = _resolve_output_dir(args.output_dir, resume_path)
    if resolved is None:
        return 1
    session_id, output_dir = resolved
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mirror run_simulation.py: write a per-session log file under logs/ for
    # parity with the rest of the project.
    log_path = configure_file_logging(session_id)
    if log_path is not None:
        logger.info("Log file: %s", log_path)

    # Prominent startup logging (per task 7.7).
    logger.info("=" * 60)
    logger.info("Brain type:    %s", brain_name)
    logger.info("Algorithm:     %s", evolution_config.algorithm)
    logger.info("Population:    %d", evolution_config.population_size)
    logger.info("Generations:   %d", evolution_config.generations)
    logger.info("Eps per eval:  %d", evolution_config.episodes_per_eval)
    logger.info("Parallel:      %d", evolution_config.parallel_workers)
    logger.info("Output dir:    %s", output_dir)
    logger.info("=" * 60)

    # Determine genome dim by constructing a brain (call once).
    num_params = encoder.genome_dim(sim_config)
    logger.info("Genome dimension: %d", num_params)

    optimizer = _build_optimizer(evolution_config, num_params, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    log_level_int = getattr(logging, args.log_level)
    loop = EvolutionLoop(
        optimizer=optimizer,
        encoder=encoder,
        fitness=fitness,
        sim_config=sim_config,
        evolution_config=evolution_config,
        output_dir=output_dir,
        rng=rng,
        log_level=log_level_int,
    )

    result = loop.run(resume_from=resume_path)

    logger.info("=" * 60)
    logger.info("Best fitness: %.6f", result.best_fitness)
    logger.info("Generations:  %d", result.generations)
    logger.info("Artefacts:    %s/", output_dir)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
