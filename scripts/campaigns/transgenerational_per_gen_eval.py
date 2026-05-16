"""Per-generation choice-index evaluator for transgenerational pilot/full runs.

For each (arm, seed, generation) under an output root, re-runs the
generation's elite genome (read from ``per_gen_elites.jsonl``) through
the env for ``--eval-episodes`` episodes, and counts how many steps the
agent spent inside any predator's damage radius. The choice-index is

  choice_index = 1 - (steps_inside_damage_radius / total_steps_in_episode)

Per-generation values are averaged across all eval episodes for that
generation's elite. Output goes to ``per_gen_choice_index.csv`` with
columns: ``seed, arm, generation, genome_id, episode, total_steps,
steps_inside_damage_radius, choice_index, pathogen_choice_index``
(``pathogen_choice_index`` is an alias for ``choice_index`` parallel to
the underlying ``predator_avoidance`` metric — vocabulary clarity for
artefacts).

Expected output directory layout:
  ``<root>/{tei_on,tei_off}/seed-{N}/[<session_id>/]per_gen_elites.jsonl``

The session_id subdirectory is optional — the script supports both the
direct layout (``best_params.json`` directly under ``seed-N/``) and the
older session-id-nested layout.

Usage:
  scripts/campaigns/transgenerational_per_gen_eval.py \
      --root evolution_results/m6_transgenerational \
      --config configs/evolution/transgenerational_pathogen_avoidance_lstmppo_klinotaxis.yml \
      --output-dir evaluations/m6_transgenerational
"""
# pragma: no cover

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from quantumnematode.agent.runners import StandardEpisodeRunner
from quantumnematode.evolution.encoders import (
    HyperparameterEncoder,
    build_birth_metadata,
)
from quantumnematode.evolution.fitness import _build_agent
from quantumnematode.evolution.genome import Genome
from quantumnematode.utils.config_loader import (
    configure_reward,
    create_env_from_config,
    load_simulation_config,
)

if TYPE_CHECKING:
    from quantumnematode.env.env import DynamicForagingEnvironment
    from quantumnematode.utils.config_loader import SimulationConfig

logger = logging.getLogger(__name__)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Manhattan distance — matches the env's damage-radius check semantics."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _count_path_steps_inside_damage(
    agent_path: list[tuple[int, int]],
    env: DynamicForagingEnvironment,
) -> int:
    """Post-hoc: count steps in the agent's path that lie within any predator's damage radius.

    For ``movement_pattern: stationary`` predators (the transgenerational pathogen-lawn
    substrate) this is exact — predators don't move during the
    episode, so end-state positions are valid for every step. For
    pursuit predators the count would be an approximation; the pilot
    arms both use stationary predators so this is the right tool for
    that case.
    """
    if not env.predators:
        return 0
    inside = 0
    for pos in agent_path:
        for predator in env.predators:
            pred_pos = (int(predator.position[0]), int(predator.position[1]))
            if _manhattan(pos, pred_pos) <= predator.damage_radius:
                inside += 1
                break  # Counted once per step (covers OR semantics).
    return inside


def _resolve_session_dir(seed_dir: Path) -> Path:
    """Return the directory containing this seed's artifacts.

    Two layouts are supported (same as ``baldwin_f1_postpilot_eval._resolve_session``):
    direct (``seed_dir/per_gen_elites.jsonl``) and nested (``seed_dir/<session>/...``).
    Direct wins when both exist; falls back to most-recently-modified session subdir.
    """
    direct = seed_dir / "per_gen_elites.jsonl"
    if direct.exists():
        return seed_dir
    sessions = [
        p for p in seed_dir.iterdir() if p.is_dir() and (p / "per_gen_elites.jsonl").is_file()
    ]
    if not sessions:
        msg = (
            f"No per_gen_elites.jsonl at {direct} and no session subdirectory "
            f"under {seed_dir} contains one. Has the campaign run completed for "
            f"this seed?"
        )
        raise FileNotFoundError(msg)
    return max(sessions, key=lambda p: p.stat().st_mtime)


def _read_per_gen_elites(session_dir: Path) -> list[dict]:
    """Read per-gen elite snapshots from the JSONL artifact."""
    path = session_dir / "per_gen_elites.jsonl"
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed line %d in %s: %r",
                    line_no,
                    path,
                    stripped[:120],
                )
    return rows


def _count_damage_steps_one_episode(
    *,
    brain,  # noqa: ANN001 - Brain protocol from quantumnematode.brain.arch._brain
    env: DynamicForagingEnvironment,
    sim_config: SimulationConfig,
) -> tuple[int, int]:
    """Run one episode and return ``(steps_inside, total_steps)``.

    Constructs a fresh agent on the supplied env, runs the standard
    episode runner, and post-hoc counts steps where the agent's
    position lies within any predator's damage radius. Post-hoc
    counting is exact for stationary predators (the transgenerational pathogen-lawn
    substrate); pursuit predators would require per-step
    instrumentation.
    """
    agent = _build_agent(brain, env, sim_config)
    runner = StandardEpisodeRunner()
    reward_config = configure_reward(sim_config)
    max_steps = sim_config.max_steps if sim_config.max_steps is not None else 1000
    result = runner.run(agent, reward_config, max_steps)
    agent_path = [(int(p[0]), int(p[1])) for p in result.agent_path]
    total_steps = len(agent_path)
    steps_inside = _count_path_steps_inside_damage(agent_path, env)
    return steps_inside, total_steps


def _apply_tei_substrate(
    brain,  # noqa: ANN001 - Brain protocol
    *,
    substrate_path: Path,
    decay_factor: float,
    lineage_depth: int,
) -> None:
    """Load the F0 substrate, apply decay ``lineage_depth`` times, set ``brain.tei_prior``.

    Mirrors the substrate-load logic in
    ``LearnedPerformanceFitness.evaluate`` so the offline evaluator
    measures the same substrate-biased policy that production workers
    actually ran at F1+. Without this, the F1/F2/F3 choice indices
    reflect the un-biased policy and the decision gate evaluates a
    proxy of the substrate's effect, not the substrate itself.
    """
    from quantumnematode.agent.transgenerational_memory import (
        TransgenerationalMemory,
    )
    from quantumnematode.agent.transgenerational_memory import (
        load as load_substrate,
    )

    try:
        substrate = load_substrate(substrate_path)
    except Exception as exc:
        msg = (
            f"Failed to load transgenerational substrate from "
            f"{substrate_path}: {exc}. The .tei.pt file may be corrupted "
            "or written by an incompatible schema; delete it and re-run "
            "F0 extraction."
        )
        raise RuntimeError(msg) from exc
    for _ in range(lineage_depth):
        substrate = TransgenerationalMemory.inherit_from(
            [substrate],
            decay_factor=decay_factor,
        )
    if hasattr(brain, "tei_prior"):
        brain.tei_prior = substrate.logit_bias  # type: ignore[attr-defined]
    else:
        logger.warning(
            "substrate_path set but brain type %s does not expose a "
            "tei_prior attribute; substrate is inert for this evaluation.",
            type(brain).__name__,
        )


def _find_substrate_path(session_dir: Path) -> Path | None:
    """Return the F0 ``.tei.pt`` substrate path under the session dir, or None.

    The loop writes the substrate at
    ``<session>/inheritance/gen-000/genome-<elite>.tei.pt`` per
    ``TransgenerationalInheritance.checkpoint_path``. Globs for any
    matching file — the F0 elite is expected to be the only one.
    """
    gen0_dir = session_dir / "inheritance" / "gen-000"
    if not gen0_dir.is_dir():
        return None
    matches = sorted(gen0_dir.glob("genome-*.tei.pt"))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple F0 .tei.pt files under %s — using %s.",
            gen0_dir,
            matches[0],
        )
    return matches[0]


def _eval_one_elite(  # noqa: PLR0913 - kw-only orthogonal args
    *,
    elite_row: dict,
    sim_config: SimulationConfig,
    eval_episodes: int,
    seed: int,
    substrate_path: Path | None = None,
    decay_factor: float = 0.6,
) -> list[tuple[int, int, int, float]]:
    """Re-run one generation's elite for ``eval_episodes`` episodes.

    Returns one tuple per episode: ``(episode_idx, total_steps,
    steps_inside, choice_index)``. Each episode runs against a fresh
    env so trial outcomes are independent (matching the frozen-eval
    semantics in ``LearnedPerformanceFitness``).

    When ``substrate_path`` is set AND the elite's generation > 0,
    the F0 substrate is loaded, decayed ``generation`` times (matching
    the production worker's ``lineage_depth=gen``), and set on
    ``brain.tei_prior`` before evaluation. F0 (gen 0) is unaffected —
    the F0 elite IS the substrate source, not consumer.
    """
    encoder = HyperparameterEncoder()
    genome = Genome(
        params=np.asarray(elite_row["params"], dtype=np.float32),
        genome_id=str(elite_row["genome_id"]),
        parent_ids=[],
        generation=int(elite_row["generation"]),
        birth_metadata=build_birth_metadata(sim_config),
    )
    brain = encoder.decode(genome, sim_config, seed=seed)

    # For F1+ under TEI, apply the substrate so the eval measures the
    # substrate-biased policy (matching what the production worker
    # actually ran). For F0 the brain is decoded as-is. F0's .pt weights
    # are GC'd by the substrate-extraction pipeline so we can't load
    # those at F0 — but F0 fitness in lineage.csv is the trained-policy
    # measurement.
    lineage_depth = int(elite_row["generation"])
    if substrate_path is not None and lineage_depth > 0:
        _apply_tei_substrate(
            brain,
            substrate_path=substrate_path,
            decay_factor=decay_factor,
            lineage_depth=lineage_depth,
        )

    if sim_config.environment is None:
        msg = "Evaluator requires sim_config.environment to be set."
        raise ValueError(msg)

    from quantumnematode.env.theme import Theme

    rows: list[tuple[int, int, int, float]] = []
    for ep in range(eval_episodes):
        env = create_env_from_config(
            sim_config.environment,
            seed=seed + ep,
            theme=Theme.HEADLESS,
            max_body_length=sim_config.body_length,
        )
        steps_inside, total_steps = _count_damage_steps_one_episode(
            brain=brain,
            env=env,
            sim_config=sim_config,
        )
        # Guard against total_steps == 0 (an env with max_steps=0 or
        # termination on step 0). Default choice_index to 1.0 in that
        # case — the agent spent zero of zero steps in a damage zone.
        ci = 1.0 if total_steps == 0 else 1.0 - (steps_inside / total_steps)
        rows.append((ep, total_steps, steps_inside, ci))
    return rows


def evaluate_one_seed(
    *,
    arm: str,
    seed: int,
    seed_dir: Path,
    sim_config: SimulationConfig,
    eval_episodes: int,
) -> list[tuple]:
    """Produce per-gen, per-episode rows for one (arm, seed)."""
    session_dir = _resolve_session_dir(seed_dir)
    elite_rows = _read_per_gen_elites(session_dir)

    # Resolve the F0 substrate path + decay_factor for TEI-on arms.
    # For TEI-off (control) arms, the inheritance/gen-000/ directory
    # has no .tei.pt file (the loop only writes it under
    # inheritance=transgenerational), so the substrate path is None
    # and the offline eval measures the un-biased policy — which is
    # exactly what the production worker ran for that arm too.
    substrate_path = _find_substrate_path(session_dir)
    decay_factor = 0.6  # default per the TEI YAML
    if sim_config.evolution is not None and sim_config.evolution.transgenerational is not None:
        decay_factor = float(sim_config.evolution.transgenerational.decay_factor)

    out: list[tuple] = []
    for elite in elite_rows:
        per_ep = _eval_one_elite(
            elite_row=elite,
            sim_config=sim_config,
            eval_episodes=eval_episodes,
            seed=seed,
            substrate_path=substrate_path,
            decay_factor=decay_factor,
        )
        for ep, total, inside, ci in per_ep:
            out.append(
                (
                    seed,
                    arm,
                    int(elite["generation"]),
                    str(elite["genome_id"]),
                    ep,
                    total,
                    inside,
                    ci,
                    ci,  # pathogen_choice_index alias (vocab clarity for artefacts)
                ),
            )
    return out


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Transgenerational per-generation choice-index evaluator. For each "
            "(arm, seed, generation), re-runs the generation's elite and counts "
            "steps spent inside any predator's damage radius. Writes "
            "per_gen_choice_index.csv."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help=(
            "Output root of the campaign (e.g. "
            "``evolution_results/m6_transgenerational``). Expects "
            "``<root>/{tei_on,tei_off}/seed-N/`` subdirectories."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to the transgenerational pilot YAML — used to reconstruct "
            "the hyperparam_schema for the encoder and the env config."
        ),
    )
    parser.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=["tei_on", "tei_off"],
        help="Arms to evaluate. Default: both.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45],
        help="Seeds to evaluate. Default: the campaign's full-mode seeds.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=25,
        help=(
            "Number of frozen-eval episodes per elite. Default 25 (matches "
            "the pilot YAML's eval_episodes_per_eval)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per_gen_choice_index.csv into.",
    )
    args = parser.parse_args()

    if args.eval_episodes <= 0:
        parser.error(f"--eval-episodes must be a positive integer; got {args.eval_episodes}.")

    sim_config = load_simulation_config(str(args.config))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[tuple] = []
    for arm in args.arms:
        arm_dir = args.root / arm
        if not arm_dir.is_dir():
            logger.warning("Arm directory missing: %s (skipping)", arm_dir)
            continue
        for seed in args.seeds:
            seed_dir = arm_dir / f"seed-{seed}"
            if not seed_dir.is_dir():
                logger.warning("Seed directory missing: %s (skipping)", seed_dir)
                continue
            print(f"Evaluating arm={arm} seed={seed}...", flush=True)
            try:
                seed_rows = evaluate_one_seed(
                    arm=arm,
                    seed=seed,
                    seed_dir=seed_dir,
                    sim_config=sim_config,
                    eval_episodes=args.eval_episodes,
                )
            except FileNotFoundError as exc:
                logger.warning("Skipping (arm=%s, seed=%d): %s", arm, seed, exc)
                continue
            all_rows.extend(seed_rows)

    if not all_rows:
        print("\nNo (arm, seed) directories produced data. Nothing to write.")
        return 1

    csv_path = args.output_dir / "per_gen_choice_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "seed",
                "arm",
                "generation",
                "genome_id",
                "episode",
                "total_steps",
                "steps_inside_damage_radius",
                "choice_index",
                "pathogen_choice_index",
            ),
        )
        for row in all_rows:
            seed, arm, gen, gid, ep, total, inside, ci, alias = row
            writer.writerow((seed, arm, gen, gid, ep, total, inside, f"{ci:.6f}", f"{alias:.6f}"))

    print(f"\nWrote {len(all_rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
