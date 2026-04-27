# pragma: no cover
r"""Wall-clock benchmark for the evolution-loop fitness eval path.

Runs one generation of the LSTMPPO+klinotaxis pilot config and prints
elapsed time per phase.  Intended as a manual reproducibility harness for
PRs that touch the per-step fitness-eval path — NOT a pytest marker
(timing-sensitive tests are flaky on shared CI hardware).

Usage::

    uv run python scripts/benchmarks/bench_evolution_smoke.py
    uv run python scripts/benchmarks/bench_evolution_smoke.py --config <path>
    uv run python scripts/benchmarks/bench_evolution_smoke.py --population 4 --episodes 2 --parallel 1

Run this BEFORE and AFTER your change and paste the timings into the PR
body so reviewers can see the speedup empirically.

Defaults are tuned for the LSTMPPO+klinotaxis brain (slowest evolvable
brain at the time of writing) so a regression there is most likely to
show.  Switch to ``--config configs/evolution/mlpppo_foraging_small.yml``
for the cheap MLPPPO baseline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Line-buffer stdout so partial progress is visible if the bench is killed
# (or its output file is read) mid-run.  Without this, prints stay buffered
# until process exit and a SIGKILL/SIGTERM mid-run leaves zero output.
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/evolution/lstmppo_foraging_small_klinotaxis.yml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Pilot config to benchmark (default: {DEFAULT_CONFIG.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cma-diagonal",
        action="store_true",
        help=(
            "Use CMA-ES diagonal mode (CMA_diagonal=True). "
            "Required for tractable wall time at genome dim >~1000 "
            "(e.g. LSTMPPO weight evolution)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run a one-generation benchmark and print wall-clock numbers."""
    args = _parse_args()

    # Validate sizing args at the CLI boundary.  Pydantic's Field(ge=1) on
    # EvolutionConfig protects the loop, but model_copy(update={...}) bypasses
    # those constraints in Pydantic v2 — so a value <= 0 here would only
    # surface as a confusing cma library error mid-run.  Fail fast instead.
    for name, value in (
        ("--population", args.population),
        ("--episodes", args.episodes),
        ("--parallel", args.parallel),
    ):
        if value <= 0:
            print(f"{name} must be >= 1, got {value}", file=sys.stderr)
            return 1

    # Import inside main so --help is fast.
    from quantumnematode.evolution import (
        ENCODER_REGISTRY,
        EpisodicSuccessRate,
        EvolutionLoop,
        get_encoder,
    )
    from quantumnematode.optimizers.evolutionary import CMAESOptimizer
    from quantumnematode.utils.config_loader import load_simulation_config

    # Quiet the loop's own logger; we want clean stdout for the timing report.
    logging.getLogger("quantumnematode.evolution.loop").setLevel(logging.WARNING)

    sim_config = load_simulation_config(str(args.config))
    if sim_config.brain is None or sim_config.brain.name is None:
        print(f"Config {args.config} is missing brain.name", file=sys.stderr)
        return 1
    brain_name = sim_config.brain.name
    if brain_name not in ENCODER_REGISTRY:
        print(
            f"No encoder for brain {brain_name!r}; registered: {sorted(ENCODER_REGISTRY)}",
            file=sys.stderr,
        )
        return 1

    encoder = get_encoder(brain_name)
    fitness = EpisodicSuccessRate()

    # Compose a minimal evolution config: 1 generation, the requested
    # population/episodes/parallel, no checkpoint.
    base_evolution = sim_config.evolution
    if base_evolution is None:
        from quantumnematode.utils.config_loader import EvolutionConfig

        base_evolution = EvolutionConfig()
    overrides: dict = {
        "generations": 1,
        "population_size": args.population,
        "episodes_per_eval": args.episodes,
        "parallel_workers": args.parallel,
        "checkpoint_every": 999,  # effectively disabled for a 1-gen run
    }
    # Only override cma_diagonal when the user passed --cma-diagonal; without
    # the flag, fall through to the YAML (or EvolutionConfig default of False).
    if args.cma_diagonal:
        overrides["cma_diagonal"] = True
    evolution_config = base_evolution.model_copy(update=overrides)

    num_params = encoder.genome_dim(sim_config)
    print(
        f"Brain: {brain_name}  Genome dim: {num_params}  "
        f"Population: {args.population}  Episodes/eval: {args.episodes}  "
        f"Parallel: {args.parallel}  cma_diagonal: {evolution_config.cma_diagonal}",
    )
    print(
        f"Total fitness evaluations this generation: "
        f"{args.population} (genomes) x {args.episodes} (episodes) "
        f"= {args.population * args.episodes} episodes",
    )

    optimizer = CMAESOptimizer(
        num_params=num_params,
        population_size=args.population,
        sigma0=evolution_config.sigma0,
        seed=args.seed,
        diagonal=evolution_config.cma_diagonal,
    )
    rng = np.random.default_rng(args.seed)

    # Project-local scratch dir so artefacts (lineage.csv, best_params.json,
    # etc.) survive the run for inspection.  Gitignored via `.bench_evolution_tmp/`
    # in the repo .gitignore so it doesn't show up as untracked.  Kept here
    # rather than tempfile.TemporaryDirectory() to preserve outputs across
    # invocations — the whole point of the bench harness is to inspect what
    # the loop produced.
    tmp_dir = PROJECT_ROOT / ".bench_evolution_tmp"
    tmp_dir.mkdir(exist_ok=True)

    loop = EvolutionLoop(
        optimizer=optimizer,
        encoder=encoder,
        fitness=fitness,
        sim_config=sim_config,
        evolution_config=evolution_config,
        output_dir=tmp_dir,
        rng=rng,
        log_level=logging.WARNING,
    )

    print("Running 1 generation...", flush=True)
    t0 = time.perf_counter()
    result = loop.run()
    elapsed = time.perf_counter() - t0

    total_episodes = args.population * args.episodes
    per_episode = elapsed / total_episodes if total_episodes else float("nan")

    print(
        f"\nGeneration completed in {elapsed:.2f} s "
        f"({per_episode:.3f} s per episode, best fitness: {result.best_fitness:.4f})",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
