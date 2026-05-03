"""Baldwin F1 (learning-acceleration) post-pilot evaluator.

For each Baldwin pilot seed:

1. Reconstruct the elite genome from ``best_params.json`` via
   ``HyperparameterEncoder.decode``.
2. Construct a schema-prior baseline genome via
   ``HyperparameterEncoder.initial_genome(sim_config, rng=np.random.default_rng(seed))``
   — a deterministic per-seed sample from the schema's prior distribution.
3. Run BOTH genomes through ``LearnedPerformanceFitness.evaluate`` with
   K' train + L eval episodes (set on a sim_config copy via
   ``model_copy(update=...)``).  Both runs use the same per-seed RNG
   seed so the only between-arm difference is the genome.
4. Append a row to ``f1_learning_acceleration.csv`` with columns
   ``seed, k_prime, episodes, elite_success_rate, baseline_success_rate,
   signal_delta`` (where ``signal_delta = elite - baseline``).

Tests learning-acceleration: does the Baldwin elite's evolved
hyperparameter genome learn faster from random init than a generic
schema-prior genome at K' = 10 episodes?  Eliminates audit findings
A2 (test now measures learning-acceleration, not random-LSTM
behaviour) and A3 (both arms include K' training, so the comparison
is apples-to-apples).

CSV is append-mode so multiple K' / L re-runs coexist for sensitivity
analysis without re-running the pilot.
"""
# pragma: no cover

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from quantumnematode.evolution.encoders import HyperparameterEncoder, build_birth_metadata
from quantumnematode.evolution.fitness import LearnedPerformanceFitness
from quantumnematode.evolution.genome import Genome
from quantumnematode.utils.config_loader import SimulationConfig, load_simulation_config


def _resolve_session(seed_dir: Path) -> Path:
    """Return the directory containing this seed's ``best_params.json``.

    Two layouts are supported:

    1. Direct: ``seed_dir/best_params.json`` (the layout written by
       ``run_evolution.py`` since logbook 014; the canonical layout
       going forward).
    2. Nested: ``seed_dir/<session_id>/best_params.json`` (older
       layouts where the loop wrote per-session subdirectories).

    Direct layout wins when both exist.  Falls back to the most
    recently-modified subdirectory if no direct file is present.
    """
    direct = seed_dir / "best_params.json"
    if direct.exists():
        return seed_dir
    sessions = [p for p in seed_dir.iterdir() if p.is_dir()]
    if not sessions:
        msg = f"No best_params.json at {direct} and no session subdirectories under {seed_dir}."
        raise FileNotFoundError(msg)
    return max(sessions, key=lambda p: p.stat().st_mtime)


def _build_sim_config_for_kprime(
    sim_config: SimulationConfig,
    *,
    k_prime: int,
    episodes: int,
) -> SimulationConfig:
    """Return a sim_config copy with the F1 evaluator's K' and L plumbed in.

    ``LearnedPerformanceFitness.evaluate`` reads K from
    ``sim_config.evolution.learn_episodes_per_eval`` and L from
    ``sim_config.evolution.eval_episodes_per_eval`` (with the
    protocol's ``episodes`` kwarg as L's fallback).  This helper
    builds a Pydantic copy with both set so the evaluator runs at
    the F1-test budget rather than the pilot's K=50/L=25.
    """
    if sim_config.evolution is None:
        msg = "F1 evaluator requires sim_config.evolution to be set (learned_performance fitness)."
        raise ValueError(msg)
    new_evolution = sim_config.evolution.model_copy(
        update={
            "learn_episodes_per_eval": k_prime,
            "eval_episodes_per_eval": episodes,
        },
    )
    return sim_config.model_copy(update={"evolution": new_evolution})


def _evaluate_one_seed(
    seed: int,
    baldwin_root: Path,
    config_path: Path,
    *,
    k_prime: int,
    episodes: int,
) -> tuple[float, float]:
    """Run the elite + schema-prior baseline at K'-train + L-eval.

    Returns ``(elite_success_rate, baseline_success_rate)`` — both in
    [0.0, 1.0].  ``signal_delta`` is left to the caller for CSV
    column composition.
    """
    seed_dir = baldwin_root / f"seed-{seed}"
    session_dir = _resolve_session(seed_dir)
    best_params_path = session_dir / "best_params.json"
    if not best_params_path.exists():  # pragma: no cover - defensive
        msg = f"No best_params.json at {best_params_path}"
        raise FileNotFoundError(msg)
    best_artefact = json.loads(best_params_path.read_text())
    best_params: list[float] = best_artefact["best_params"]

    # Load the YAML to reconstruct the schema (best_params.json doesn't
    # persist birth_metadata; HyperparameterEncoder.decode requires
    # genome.birth_metadata["param_schema"] to be populated, so we
    # rebuild it from the YAML at re-eval time per the spec scenario).
    sim_config = load_simulation_config(str(config_path))
    if sim_config.hyperparam_schema is None:
        msg = (
            f"Config {config_path} has no hyperparam_schema — cannot "
            "reconstruct an elite Genome for F1 learning-acceleration eval."
        )
        raise ValueError(msg)

    sim_config_for_kprime = _build_sim_config_for_kprime(
        sim_config,
        k_prime=k_prime,
        episodes=episodes,
    )

    encoder = HyperparameterEncoder()
    fitness_fn = LearnedPerformanceFitness()

    # Elite genome: params from best_params.json, birth metadata
    # rebuilt from YAML.
    elite_genome = Genome(
        params=np.array(best_params, dtype=np.float32),
        genome_id=f"f1_elite_seed_{seed}",
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )

    # Schema-prior baseline genome: deterministic per-seed sample from
    # the schema's prior distribution (uniform-in-bounds for floats;
    # log-uniform for log-scale floats; etc., per the encoder's
    # _sample_initial).  Per-seed seeding makes the comparison
    # apples-to-apples (same env trajectory at this seed for both
    # genomes; only the genome differs).
    baseline_genome = encoder.initial_genome(
        sim_config_for_kprime,
        rng=np.random.default_rng(seed),
    )
    baseline_genome.genome_id = f"f1_baseline_seed_{seed}"

    elite_success = fitness_fn.evaluate(
        elite_genome,
        sim_config_for_kprime,
        encoder,
        episodes=episodes,
        seed=seed,
    )
    baseline_success = fitness_fn.evaluate(
        baseline_genome,
        sim_config_for_kprime,
        encoder,
        episodes=episodes,
        seed=seed,
    )
    return elite_success, baseline_success


def main() -> int:
    """CLI entry point — F1 learning-acceleration eval per seed."""
    parser = argparse.ArgumentParser(
        description=(
            "Baldwin F1 learning-acceleration post-pilot evaluator. For each "
            "seed: take the pilot's elite genome from best_params.json, train "
            "for K' episodes, measure success rate over L frozen-eval episodes, "
            "and compare to a schema-prior baseline genome trained the same way. "
            "Writes f1_learning_acceleration.csv (append-mode for cross-K' "
            "coexistence)."
        ),
    )
    parser.add_argument(
        "--baldwin-root",
        type=Path,
        required=True,
        help=(
            "Output root of the Baldwin pilot campaign — typically "
            "evolution_results/baldwin_retry_baldwin_lstmppo_klinotaxis_predator/"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to the Baldwin pilot YAML — needed to reconstruct the "
            "hyperparam_schema for the encoder. Typically "
            "configs/evolution/baldwin_lstmppo_klinotaxis_predator_retry_pilot.yml"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46, 47, 48, 49],
        help="Seeds to evaluate (must match the pilot's seed list). Default n=8.",
    )
    parser.add_argument(
        "--k-prime",
        type=int,
        default=10,
        help=(
            "K' (train episode count) per seed. Defaults to 10 (1/5 of K=50; "
            "the F1 design's sweet spot per design Decision 3)."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=25,
        help=(
            "L (frozen-eval episode count) per seed. Defaults to 25 (matches "
            "the pilot's default L)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write/append f1_learning_acceleration.csv into.",
    )
    args = parser.parse_args()

    if args.k_prime <= 0:
        parser.error(
            f"--k-prime must be a positive integer; got {args.k_prime}. "
            "F1 learning-acceleration requires at least 1 K' training episode "
            "per seed (K' < K = 50 is recommended; design Decision 3 commits to K' = 10).",
        )
    if args.episodes <= 0:
        parser.error(
            f"--episodes must be a positive integer; got {args.episodes}. "
            "F1 evaluator requires at least 1 frozen-eval episode per seed.",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[int, int, int, float, float, float]] = []
    for seed in args.seeds:
        print(
            f"Evaluating seed {seed} at K' = {args.k_prime}, L = {args.episodes}...",
            flush=True,
        )
        elite_sr, baseline_sr = _evaluate_one_seed(
            seed,
            args.baldwin_root,
            args.config,
            k_prime=args.k_prime,
            episodes=args.episodes,
        )
        signal_delta = elite_sr - baseline_sr
        print(
            f"  seed {seed}: elite={elite_sr:.4f} baseline={baseline_sr:.4f} "
            f"delta={signal_delta:+.4f}",
            flush=True,
        )
        rows.append(
            (seed, args.k_prime, args.episodes, elite_sr, baseline_sr, signal_delta),
        )

    # Append-mode CSV so re-runs with different K' / L budgets coexist.
    # Write header only if the file doesn't yet exist.
    csv_path = args.output_dir / "f1_learning_acceleration.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                (
                    "seed",
                    "k_prime",
                    "episodes",
                    "elite_success_rate",
                    "baseline_success_rate",
                    "signal_delta",
                ),
            )
        for seed, kp, eps, elite, baseline, delta in rows:
            writer.writerow(
                (seed, kp, eps, f"{elite:.6f}", f"{baseline:.6f}", f"{delta:+.6f}"),
            )
    action = "Wrote" if write_header else "Appended"
    print(f"\n{action} {len(rows)} rows to {csv_path}")

    mean_elite = float(np.mean([r[3] for r in rows]))
    mean_baseline = float(np.mean([r[4] for r in rows]))
    mean_delta = mean_elite - mean_baseline
    print(
        f"\nF1 learning-acceleration summary at K' = {args.k_prime}, L = {args.episodes}:",
    )
    print(f"  Mean elite success rate:    {mean_elite:.4f}")
    print(f"  Mean baseline success rate: {mean_baseline:.4f}")
    print(f"  Mean signal delta:          {mean_delta:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
