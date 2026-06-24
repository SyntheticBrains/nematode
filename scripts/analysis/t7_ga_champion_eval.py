"""Evaluate the FeedforwardGA C3 champions' full-clear success.

The GA's analogue of the PPO ``post_convergence_success_rate``.
For each seed's evolution run, decode the champion genome (``best_params.json``) into a
fresh brain and run ``EpisodicSuccessRate`` over N frozen episodes — the evolved-champion
full-clear rate the T7 cross-architecture ranking consumes. Writes
``ga_c3_results.json`` (``{seed: {full_clear_rate (percent), progress_fitness}}``).

Usage::

    uv run python scripts/analysis/t7_ga_champion_eval.py \
        --config configs/evolution/feedforwardga_small_continuous2d_combined_klinotaxis.yml \
        --runs-dir <ga-run-dir> \
        --episodes 40 --out <ga-run-dir>/ga_c3_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from quantumnematode.evolution import (
    EpisodicProgressFitness,
    EpisodicSuccessRate,
    select_encoder,
)
from quantumnematode.evolution.genome import Genome, genome_id_for
from quantumnematode.utils.config_loader import load_simulation_config

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    """Decode each seed's champion genome and evaluate its frozen full-clear rate."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="dir with resultsdir_s<seed>.txt files",
    )
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sim_config = load_simulation_config(str(args.config))
    encoder = select_encoder(sim_config)
    succ_fit = EpisodicSuccessRate()
    prog_fit = EpisodicProgressFitness()

    def genome(params: list[float]) -> Genome:
        return Genome(
            params=np.asarray(params, dtype=np.float32),
            genome_id=genome_id_for(0, 0, []),
            parent_ids=[],
            generation=0,
        )

    results: dict[str, dict] = {}
    for ptr in sorted(args.runs_dir.glob("resultsdir_s*.txt")):
        seed = int(ptr.stem.split("_s")[-1])
        rd = ptr.read_text().strip()
        bp = json.loads((REPO / "evolution_results" / rd / "best_params.json").read_text())
        params = bp["best_params"]
        # Held-out eval seed (offset from the training seed) for a frozen-policy measure.
        succ = succ_fit.evaluate(
            genome(params),
            sim_config,
            encoder,
            episodes=args.episodes,
            seed=100 + seed,
        )
        prog = prog_fit.evaluate(
            genome(params),
            sim_config,
            encoder,
            episodes=args.episodes,
            seed=100 + seed,
        )
        results[str(seed)] = {
            "full_clear_rate": round(succ * 100.0, 2),
            "progress_fitness": round(prog, 3),
        }
        print(f"  seed {seed}: full_clear {succ * 100:.1f}%  progress {prog:.3f}")

    args.out.write_text(json.dumps(results, indent=2))
    print(
        f"\nwrote {args.out}  (mean full-clear {np.mean([v['full_clear_rate'] for v in results.values()]):.1f}%)",
    )


if __name__ == "__main__":
    main()
