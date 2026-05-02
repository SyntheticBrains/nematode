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
from pydantic import ValidationError
from quantumnematode.evolution import (
    EpisodicSuccessRate,
    EvolutionLoop,
    LamarckianInheritance,
    LearnedPerformanceFitness,
    NoInheritance,
    select_encoder,
)
from quantumnematode.logging_config import configure_file_logging, logger
from quantumnematode.optimizers.evolutionary import (
    CMAESOptimizer,
    EvolutionaryOptimizer,
    GeneticAlgorithmOptimizer,
    OptunaTPEOptimizer,
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
        choices=["cmaes", "ga", "tpe"],
        default=None,
        help="Override evolution.algorithm.",
    )
    parser.add_argument(
        "--inheritance",
        type=str,
        choices=["none", "lamarckian"],
        default=None,
        help=(
            "Override evolution.inheritance. 'lamarckian' enables per-genome "
            "warm-start from the prior generation's elite parent. Requires "
            "hyperparam_schema and learn_episodes_per_eval > 0; mutually "
            "exclusive with warm_start_path. See evolution-framework spec."
        ),
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
    parser.add_argument(
        "--fitness",
        type=str,
        choices=["success_rate", "learned_performance"],
        default="success_rate",
        help=(
            "Fitness function to evolve against. 'success_rate' "
            "(EpisodicSuccessRate, the default) is frozen-weight "
            "evaluation. 'learned_performance' (LearnedPerformanceFitness) "
            "runs K training episodes followed by L frozen eval episodes "
            "per genome — only valid with hyperparam_schema set."
        ),
    )
    return parser.parse_args()


def _resolve_evolution_config(
    sim_config_evolution: EvolutionConfig | None,
    args: argparse.Namespace,
) -> EvolutionConfig:
    """Compose the effective :class:`EvolutionConfig` for this run.

    Precedence: CLI flags (when explicitly set) > YAML ``evolution:`` block
    > :class:`EvolutionConfig` defaults.

    CLI overrides go through full Pydantic re-validation so invalid
    values (e.g. ``--generations -5``) are rejected with a clear error
    rather than silently bypassing field constraints.  ``model_copy
    (update=...)`` skips validation, so we round-trip through
    ``model_dump`` + ``EvolutionConfig(**merged)`` instead.
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
    if args.inheritance is not None:
        overrides["inheritance"] = args.inheritance
    if args.sigma is not None:
        overrides["sigma0"] = args.sigma
    if args.parallel is not None:
        overrides["parallel_workers"] = args.parallel
    if not overrides:
        return base
    merged = {**base.model_dump(), **overrides}
    try:
        return EvolutionConfig(**merged)
    except ValidationError as exc:
        msg = (
            f"Invalid CLI override for evolution config: {exc}.  "
            "Check --generations, --population, --episodes, --sigma, "
            "--parallel, --inheritance are within their accepted ranges."
        )
        raise SystemExit(msg) from exc


def _build_optimizer(  # noqa: PLR0913 - x0/stds/bounds are orthogonal per-optimiser inputs
    evolution_config: EvolutionConfig,
    num_params: int,
    seed: int | None,
    *,
    x0: list[float] | None = None,
    stds: list[float] | None = None,
    bounds: list[tuple[float, float]] | None = None,
) -> EvolutionaryOptimizer:
    """Construct the optimiser specified by ``evolution_config.algorithm``.

    ``x0`` seeds the optimiser's initial mean (for CMA-ES) or first
    individual (for GA).  If ``None``, both optimisers default to zeros,
    which is appropriate for weight-evolution (network weights cluster
    around zero) but invalid for hyperparameter evolution where the
    schema's bounds may sit far from the origin (e.g. log-scale
    learning_rate has bounds in [-11.5, -4.6]).  Callers that have a
    valid in-bounds starting point — typically the encoder's
    ``initial_genome().params`` — SHOULD pass it as ``x0``.

    ``stds`` (CMA-ES only) supplies per-parameter standard deviations.
    Required when genome dimensions live on materially different scales
    (e.g. mixed hyperparameter schemas where one slot has range ~7 in
    log-units and another has range ~0.1).  Currently ignored by the
    GA optimiser, which uses ``sigma0`` uniformly.

    ``bounds`` (TPE only) supplies per-parameter ``(low, high)`` ranges.
    Required by ``OptunaTPEOptimizer`` (TPE samples from a uniform-in-
    bounds prior); ignored by CMA-ES and GA which sample from
    Gaussian / mutation distributions.  Length must equal ``num_params``
    when set.
    """
    if evolution_config.algorithm == "cmaes":
        return CMAESOptimizer(
            num_params=num_params,
            x0=x0,
            population_size=evolution_config.population_size,
            sigma0=evolution_config.sigma0,
            seed=seed,
            diagonal=evolution_config.cma_diagonal,
            stds=stds,
        )
    if evolution_config.algorithm == "ga":
        return GeneticAlgorithmOptimizer(
            num_params=num_params,
            x0=x0,
            population_size=evolution_config.population_size,
            sigma0=evolution_config.sigma0,
            elite_fraction=evolution_config.elite_fraction,
            mutation_rate=evolution_config.mutation_rate,
            crossover_rate=evolution_config.crossover_rate,
            seed=seed,
        )
    if evolution_config.algorithm == "tpe":
        if bounds is None:
            msg = (
                "TPE optimiser requires per-parameter bounds, but the encoder "
                "returned None.  TPE is only supported for bounded-search "
                "configs (typically HyperparameterEncoder); weight encoders "
                "return None and should use CMA-ES or GA instead."
            )
            raise ValueError(msg)
        return OptunaTPEOptimizer(
            num_params=num_params,
            bounds=bounds,
            population_size=evolution_config.population_size,
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


def main() -> int:  # noqa: PLR0915 — sequential CLI entry point; splitting hurts readability
    """Entry point."""
    args = parse_arguments()
    logger.setLevel(args.log_level)

    sim_config = load_simulation_config(args.config)
    evolution_config = _resolve_evolution_config(sim_config.evolution, args)

    if sim_config.brain is None or sim_config.brain.name is None:
        logger.error("Config must specify brain.name.")
        return 1

    brain_name = sim_config.brain.name
    # select_encoder dispatches by hyperparam_schema presence.  When
    # set, returns HyperparameterEncoder regardless of registry
    # membership — so brains without weight encoders are still
    # reachable for hyperparameter evolution.  When None, falls back
    # to get_encoder(brain.name) which raises ValueError for brains
    # not in ENCODER_REGISTRY.
    try:
        encoder = select_encoder(sim_config)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    # --fitness flag dispatches between EpisodicSuccessRate (the
    # default, frozen-weight) and LearnedPerformanceFitness (K train +
    # L eval).  When learned_performance is selected, validate guards:
    # hyperparam_schema must be set (otherwise we'd combine a weight
    # encoder with LearnedPerformanceFitness, which would double-count
    # weights as both genome and substrate under Lamarckian inheritance —
    # the EvolutionConfig validator catches the same case from the
    # inheritance side, but this guard fires earlier and gives a more
    # focused error for the --fitness path), AND
    # learn_episodes_per_eval must be > 0.
    if args.fitness == "learned_performance":
        if sim_config.hyperparam_schema is None:
            logger.error(
                "--fitness learned_performance requires hyperparam_schema "
                "to be set in the YAML.  Combining a weight encoder with "
                "LearnedPerformanceFitness would double-count weights as "
                "both genome and substrate under Lamarckian inheritance.  "
                "For weight-evolution campaigns, use --fitness success_rate "
                "(EpisodicSuccessRate, the default).  To switch to "
                "hyperparameter evolution, author a hyperparam_schema: "
                "block in the YAML.",
            )
            return 1
        if evolution_config.learn_episodes_per_eval == 0:
            logger.error(
                "--fitness learned_performance requires "
                "evolution.learn_episodes_per_eval > 0; got 0.  Set "
                "learn_episodes_per_eval in the evolution: block.",
            )
            return 1
        fitness = LearnedPerformanceFitness()
    else:
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

    # Prominent startup logging.
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

    rng = np.random.default_rng(args.seed)

    # Seed the optimiser with a valid in-bounds starting point sampled
    # from the encoder.  Without this the optimiser's mean defaults to
    # zeros, which is invalid for hyperparameter encoders whose schema
    # bounds sit far from the origin (e.g. log-scale learning_rate).
    # For weight encoders this still produces a valid x0 (the brain's
    # actual freshly-initialised weights), which is a more principled
    # starting point than zeros.
    initial_genome = encoder.initial_genome(sim_config, rng=rng)
    x0 = list(initial_genome.params.astype(float))

    # Per-parameter standard deviations let the optimiser explore each
    # genome dimension proportionally to its bound range.  Critical for
    # hyperparameter schemas with mixed-scale dimensions; weight
    # encoders return None to keep CMA-ES's default uniform sigma.
    stds = encoder.genome_stds(sim_config)

    # Per-parameter bounds are required by TPE (which samples uniformly
    # from a bounded prior) and ignored by CMA-ES (unbounded).  Weight
    # encoders return None — TPE-on-weights isn't a supported
    # combination (n>>schema dim makes the bounded-prior assumption
    # untenable).
    bounds = encoder.genome_bounds(sim_config)

    try:
        optimizer = _build_optimizer(
            evolution_config,
            num_params,
            seed=args.seed,
            x0=x0,
            stds=stds,
            bounds=bounds,
        )
    except ValueError as exc:
        msg = (
            f"Invalid optimiser configuration: {exc}.  "
            "Check evolution.algorithm in the YAML matches the encoder "
            "(weight encoders → cmaes/ga; HyperparameterEncoder → "
            "cmaes/ga/tpe) and that any bounds in hyperparam_schema are "
            "finite with low < high."
        )
        raise SystemExit(msg) from exc

    # Construct the inheritance strategy from the resolved evolution
    # config.  Default "none" → NoInheritance() (loop is byte-equivalent
    # to pre-M3); "lamarckian" → LamarckianInheritance with the
    # configured elite count.  Validators on EvolutionConfig already
    # rejected unsafe combinations at YAML/CLI parse time.
    if evolution_config.inheritance == "lamarckian":
        inheritance = LamarckianInheritance(
            elite_count=evolution_config.inheritance_elite_count,
        )
    else:
        inheritance = NoInheritance()

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
        inheritance=inheritance,
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
