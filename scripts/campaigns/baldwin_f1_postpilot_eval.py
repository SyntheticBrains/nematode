"""Baldwin F1 (innate-only) post-pilot evaluator.

Reads each Baldwin pilot seed's gen-N elite ``best_params.json``,
reconstructs the elite genome via ``HyperparameterEncoder``, and runs L
frozen-eval episodes (K=0, no learning) via ``EpisodicSuccessRate``.
Writes per-seed F1 results to ``f1_innate_only.csv``.

Tests genetic assimilation (the genome alone, without learning,
produces useful priors over policies).  See logbook 014 § Method for
the framing.
"""
# pragma: no cover

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from quantumnematode.evolution.encoders import HyperparameterEncoder, build_birth_metadata
from quantumnematode.evolution.fitness import EpisodicSuccessRate
from quantumnematode.evolution.genome import Genome
from quantumnematode.utils.config_loader import load_simulation_config


def _latest_session(seed_dir: Path) -> Path:
    """Return the most recently modified subdirectory under ``seed_dir``."""
    sessions = [p for p in seed_dir.iterdir() if p.is_dir()]
    if not sessions:
        msg = f"No session directory under {seed_dir}"
        raise FileNotFoundError(msg)
    return max(sessions, key=lambda p: p.stat().st_mtime)


def _evaluate_one_seed(
    seed: int,
    baldwin_root: Path,
    config_path: Path,
    episodes: int,
) -> tuple[float, str]:
    """Re-evaluate one Baldwin seed's elite genome with K=0 frozen-eval.

    Returns ``(success_rate, elite_genome_id)``.  ``success_rate`` is in
    [0.0, 1.0]; ``elite_genome_id`` is the seed-specific identifier (a
    synthetic "f1_elite" sentinel since ``best_params.json`` doesn't
    persist the original gen-N elite's ID, only its params and fitness).
    """
    seed_dir = baldwin_root / f"seed-{seed}"
    session_dir = _latest_session(seed_dir)
    best_params_path = session_dir / "best_params.json"
    if not best_params_path.exists():
        msg = f"No best_params.json at {best_params_path}"
        raise FileNotFoundError(msg)
    best_artefact = json.loads(best_params_path.read_text())
    best_params: list[float] = best_artefact["best_params"]

    # Load the YAML to reconstruct the schema (best_params.json doesn't
    # persist birth_metadata; HyperparameterEncoder.decode requires
    # genome.birth_metadata["param_schema"] to be populated).
    sim_config = load_simulation_config(str(config_path))
    if sim_config.hyperparam_schema is None:
        msg = (
            f"Config {config_path} has no hyperparam_schema — cannot "
            "reconstruct an elite Genome for frozen-eval."
        )
        raise ValueError(msg)

    # Synthetic Genome: params from best_params.json + birth_metadata
    # built from the YAML's hyperparam_schema (so encoder.decode can
    # patch the brain config from the genome's params).
    elite_genome_id = f"f1_elite_seed_{seed}"
    genome = Genome(
        params=np.array(best_params, dtype=np.float32),
        genome_id=elite_genome_id,
        parent_ids=[],
        generation=0,
        birth_metadata=build_birth_metadata(sim_config),
    )

    # Frozen-eval: EpisodicSuccessRate.evaluate calls encoder.decode
    # internally, so no separate brain construction is needed.
    encoder = HyperparameterEncoder()
    fitness_fn = EpisodicSuccessRate()
    success_rate = fitness_fn.evaluate(
        genome,
        sim_config,
        encoder,
        episodes=episodes,
        seed=seed,
    )
    return success_rate, elite_genome_id


def main() -> int:
    """CLI entry point — evaluate F1 innate-only success rate per seed."""
    parser = argparse.ArgumentParser(
        description=(
            "Baldwin F1 innate-only post-pilot evaluator. Reads each seed's "
            "gen-N elite hyperparam genome from best_params.json and runs L "
            "frozen-eval episodes (K=0) to test whether the genome alone "
            "produces useful priors without the K-train phase."
        ),
    )
    parser.add_argument(
        "--baldwin-root",
        type=Path,
        required=True,
        help=(
            "Output root of the Baldwin pilot campaign — typically "
            "evolution_results/m4_baldwin_lstmppo_klinotaxis_predator/"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to the Baldwin pilot YAML — needed to reconstruct the "
            "hyperparam_schema for the encoder.  Typically "
            "configs/evolution/baldwin_lstmppo_klinotaxis_predator_pilot.yml"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45],
        help="Seeds to evaluate (must match the pilot's seed list).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=25,
        help="L (frozen-eval episode count) per seed.  Defaults to the M3 L=25.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write f1_innate_only.csv into.",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error(
            f"--episodes must be a positive integer; got {args.episodes}. "
            "F1 innate-only requires at least 1 frozen-eval episode per seed.",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[int, float, str]] = []
    for seed in args.seeds:
        print(f"Evaluating seed {seed}...", flush=True)
        success_rate, elite_genome_id = _evaluate_one_seed(
            seed,
            args.baldwin_root,
            args.config,
            args.episodes,
        )
        print(
            f"  seed {seed}: success_rate={success_rate:.4f} (elite {elite_genome_id})",
            flush=True,
        )
        rows.append((seed, success_rate, elite_genome_id))

    csv_path = args.output_dir / "f1_innate_only.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("seed", "success_rate", "elite_genome_id"))
        for seed, sr, gid in rows:
            writer.writerow((seed, f"{sr:.6f}", gid))
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    mean_sr = float(np.mean([sr for _, sr, _ in rows]))
    print(f"Mean F1 innate-only success rate: {mean_sr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
