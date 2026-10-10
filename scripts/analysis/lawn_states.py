#!/usr/bin/env python
r"""D.1: read trained runs on patchy lawns for roaming and dwelling, and the learning gate.

Two readings, each a function the registration's analysis calls:

**The learning gate**, from training logs: each run's per-episode intake, read from the run
summary's ``Intake:`` field, averaged over the final quarter of episodes (the plateau). A learning
arm passes when its plateau beats its untrained floor's, paired by seed, with the 80% interval
above zero.

**The states**, from an evaluation:

1. Each run's final weights (or, for a floor, the seed's untrained policy) are evaluated frozen for
   held-out episodes with behaviour capture.
2. Each episode's positions, one per 5-second step, are cut into 10-second windows. A window is on a
   lawn when all three of its position samples are.
3. The states come from the model calibrated on real worms
   (:func:`quantumnematode.validation.roaming_dwelling.load_calibrated_hmm`), within each on-lawn run.

A run then reports:

- its roaming fraction on lawns;
- its mean roaming and dwelling bout durations, counting only bouts that end within an on-lawn run;
- the roaming fraction where the cell under the worm is grazed (below half density) against where
  it is not. The cell's density is recovered from the worm's intake there: intake fraction times
  density times quality.

Usage::

    uv run python scripts/analysis/lawn_states.py --logs campaigns/d1-pilot/logs \\
        --arms internal internal_frozen --seeds 9101 9102 --episodes 20 --out states.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "campaigns"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import body_control as bc  # noqa: E402  # pyright: ignore[reportMissingImports]
import body_kinematics_eval as harness  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_lawn_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]
from quantumnematode.agent import (  # noqa: E402
    DEFAULT_AGENT_BODY_LENGTH,
    DEFAULT_MAX_STEPS,
    QuantumNematodeAgent,
)
from quantumnematode.brain.weights import load_weights  # noqa: E402
from quantumnematode.env.theme import Theme  # noqa: E402
from quantumnematode.utils.config_loader import (  # noqa: E402
    configure_environment,
    configure_reward,
    configure_satiety,
    create_env_from_config,
)
from quantumnematode.utils.seeding import derive_run_seed, get_rng, set_global_seed  # noqa: E402
from quantumnematode.validation import roaming_dwelling as rd  # noqa: E402

if TYPE_CHECKING:
    from quantumnematode.report.dtypes import BehaviourStep

EVALUATION_RUN_OFFSET = 1_000_000
PLATEAU_FRACTION = 0.25
GRAZED_DENSITY = 0.5
_RUN_LINE = re.compile(r"Run:\s+(\d+)\s+Status:.*?Intake:\s+([-\d.]+)")


# ── The learning gate ───────────────────────────────────────────────────────────────────────────


def intake_plateau(log: Path) -> float | None:
    """Return a run's mean intake per episode over its final quarter of episodes, from its log."""
    if not log.is_file():
        return None
    intakes = [float(m.group(2)) for m in _RUN_LINE.finditer(log.read_text())]
    if not intakes:
        return None
    tail = max(1, int(len(intakes) * PLATEAU_FRACTION))
    return statistics.fmean(intakes[-tail:])


def plateaus(log_dirs: list[Path], arm: str, seeds: tuple[int, ...]) -> dict[int, float]:
    """Return each seed's intake plateau for one arm's runs."""
    out: dict[int, float] = {}
    for seed in seeds:
        for log_dir in log_dirs:
            value = intake_plateau(log_dir / f"{gen.stem(arm)}-seed{seed}.log")
            if value is not None:
                out[seed] = value
                break
    return out


def learning_gate(learn: dict[int, float], floor: dict[int, float]) -> dict[str, Any]:
    """Pair a learning arm's intake plateaus with its floor's: it passes with the interval above 0."""
    seeds = sorted(set(learn) & set(floor))
    test = wp.paired_seed_wilcoxon_bootstrap([learn[s] - floor[s] for s in seeds])
    return {
        "n_seeds": len(seeds),
        "learn_mean": statistics.fmean(learn[s] for s in seeds) if seeds else None,
        "floor_mean": statistics.fmean(floor[s] for s in seeds) if seeds else None,
        "test": test,
        "passes": bool(seeds) and float(test["ci_lo"]) > 0.0,
    }


# ── The states ──────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Episode:
    """One evaluated episode's per-window measures, states and the density under the worm."""

    states: np.ndarray
    density: np.ndarray


def episode_windows(
    steps: list[BehaviourStep],
    classifier: rd.GaussianClassifier,
    intake_fraction: float,
    quality: float,
) -> Episode:
    """Cut one episode's captured steps into windows; classify them; recover the cell's density.

    A window spans three position samples, two steps. It is on a lawn when all three are. Its
    density is the mean density of the cells eaten from in it, or NaN where the worm ate nothing.
    """
    x = np.array([s.x for s in steps])
    y = np.array([s.y for s in steps])
    on = np.array([bool(s.on_lawn) for s in steps])
    eaten = np.array([s.intake or 0.0 for s in steps])
    speed, angular = rd.window_measures(x, y)
    n = len(speed)
    on_window = np.array([on[2 * k : 2 * k + 3].all() for k in range(n)], dtype=bool)
    states = classifier.states(speed, angular, on_window)
    density = np.full(n, np.nan)
    for k in range(n):
        bites = eaten[2 * k + 1 : 2 * k + 3]
        bites = bites[bites > 0]
        if bites.size:
            density[k] = float(bites.mean()) / (intake_fraction * quality)
    return Episode(states=states, density=density)


def bout_durations(states: np.ndarray, state: int) -> list[float]:
    """Return the durations (seconds) of complete bouts of ``state`` inside on-lawn runs."""
    window_s = rd.STEP_SECONDS * rd.WINDOW_STEPS
    durations: list[float] = []
    for start, stop in rd.on_food_runs(states != rd.OFF_FOOD):
        run = states[start:stop]
        edges = np.flatnonzero(np.diff(run)) + 1
        bounds = np.concatenate([[0], edges, [len(run)]])
        for a, b in itertools.pairwise(bounds):
            complete = a > 0 and b < len(run)
            if complete and run[a] == state:
                durations.append((b - a) * window_s)
    return durations


def run_readings(episodes: list[Episode]) -> dict[str, Any]:
    """Pool a run's episodes into its roaming fraction, bout durations and depletion contrast."""
    states = np.concatenate([e.states for e in episodes]) if episodes else np.zeros(0, int)
    density = np.concatenate([e.density for e in episodes]) if episodes else np.zeros(0)
    on = states != rd.OFF_FOOD
    roam = [d for e in episodes for d in bout_durations(e.states, rd.ROAMING)]
    dwell = [d for e in episodes for d in bout_durations(e.states, rd.DWELLING)]
    grazed = on & (density < GRAZED_DENSITY)
    fresh = on & (density >= GRAZED_DENSITY)
    return {
        "on_lawn_windows": int(on.sum()),
        "roaming_fraction": float(np.mean(states[on] == rd.ROAMING)) if on.any() else None,
        "roaming_bout_s": statistics.fmean(roam) if roam else None,
        "dwelling_bout_s": statistics.fmean(dwell) if dwell else None,
        "n_roaming_bouts": len(roam),
        "n_dwelling_bouts": len(dwell),
        "roaming_where_grazed": float(np.mean(states[grazed] == rd.ROAMING))
        if grazed.any()
        else None,
        "roaming_where_fresh": float(np.mean(states[fresh] == rd.ROAMING)) if fresh.any() else None,
    }


def capture(config_path: Path, seed: int, weights: Path | None, episodes: int) -> tuple[Any, ...]:
    """Run ``episodes`` frozen held-out episodes with behaviour capture; return them and the lawns."""
    brain, config, sensing = harness.build_brain(config_path, seed)
    if weights is not None:
        load_weights(brain, weights)
    sensing = sensing.model_copy(update={"capture_behaviour": True})
    environment = configure_environment(config)
    agent = QuantumNematodeAgent(
        brain=brain,
        env=create_env_from_config(
            environment,
            seed=derive_run_seed(seed, EVALUATION_RUN_OFFSET),
            max_body_length=config.body_length or DEFAULT_AGENT_BODY_LENGTH,
            theme=Theme.HEADLESS,
        ),
        max_body_length=config.body_length or DEFAULT_AGENT_BODY_LENGTH,
        theme=Theme.HEADLESS,
        satiety_config=configure_satiety(config),
        sensing_config=sensing,
    )
    lawns = agent.env.foraging.lawns
    reward = configure_reward(config)
    captured: list[list[BehaviourStep]] = []
    intakes: list[float] = []
    for episode in range(episodes):
        run_seed = derive_run_seed(seed, EVALUATION_RUN_OFFSET + episode)
        set_global_seed(run_seed)
        if episode:
            agent.env.seed = run_seed
            agent.env.rng = get_rng(run_seed)
            agent.reset_environment()
            agent.reset_brain()
        result = agent.run_episode(reward, max_steps=config.max_steps or DEFAULT_MAX_STEPS)
        captured.append(list(result.behaviour or []))
        intakes.append(agent._episode_tracker.intake)
    return captured, intakes, lawns


def evaluate_run(job: tuple[str, int, list[str], int]) -> dict[str, Any]:
    """Evaluate one run's states; a floor arm reads its seed's untrained policy."""
    arm, seed, log_dirs, episodes = job
    weights = None
    if not arm.endswith("_frozen"):
        log = next(
            (
                Path(d) / f"{gen.stem(arm)}-seed{seed}.log"
                for d in log_dirs
                if (Path(d) / f"{gen.stem(arm)}-seed{seed}.log").is_file()
            ),
            None,
        )
        weights = bc.final_weights(log) if log is not None else None
        if weights is None:
            return {"arm": arm, "seed": seed, "missing": True}
    config = gen.FORAGING / f"{gen.stem(arm)}.yml"
    captured, intakes, lawns = capture(config, seed, weights, episodes)
    classifier = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm())
    quality = float(lawns.quality[0]) if lawns.quality[0] == lawns.quality[1] else float("nan")
    windows = [
        episode_windows(steps, classifier, lawns.intake_fraction, quality) for steps in captured
    ]
    return {
        "arm": arm,
        "seed": seed,
        "evaluation_intake": statistics.fmean(intakes) if intakes else None,
        "action_log_std": action_log_std(weights),
        **run_readings(windows),
    }


def action_log_std(weights: Path | None) -> list[float] | None:
    """Return the policy's learned action noise, ``log_std`` for (speed, turn), from its weights.

    The policy samples its actions, so its noise is part of the behaviour the instrument reads.
    """
    if weights is None:
        return None
    import torch

    state = torch.load(weights, map_location="cpu", weights_only=False)
    log_std = state.get("log_std") if isinstance(state, dict) else None
    if isinstance(log_std, dict):
        log_std = log_std.get("log_std")
    return [float(v) for v in log_std.tolist()] if log_std is not None else None


def main(argv: list[str] | None = None) -> int:
    """CLI: the learning gate from the logs, and the states from an evaluation."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--logs", type=Path, action="append", required=True)
    ap.add_argument("--arms", nargs="+", choices=list(gen.ARMS), default=list(gen.ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, help="write the JSON here")
    args = ap.parse_args(argv)
    seeds = tuple(args.seeds)
    gates = {
        arm: learning_gate(
            plateaus(args.logs, arm, seeds),
            plateaus(args.logs, "internal_frozen", seeds),
        )
        for arm in args.arms
        if not arm.endswith("_frozen")
    }
    jobs = [
        (arm, seed, [str(d) for d in args.logs], args.episodes)
        for arm in args.arms
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(evaluate_run, jobs))
    result = {"episodes": args.episodes, "gates": gates, "runs": runs}
    payload = json.dumps(result, indent=2, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
