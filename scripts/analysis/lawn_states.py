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

**The registered reading** (``read_panel``): the learner's on-lawn dwelling share minus its
untrained floor's, paired by seed, classified at the minimum the fourth pilot fixed, after the
learning gate. Beside it, never as verdicts: the same against the arm without internal state, bouts
of both states, and roaming where grazed against fresh.

**The knockout readings** (``read_knockouts``), for the connectome with serotonin and PDF: after the
learning gate and a dwelling gate (the intact network's dwelling share against its floor, at stage
0's minimum), each acute knockout's dwelling share against the intact network's, paired by seed and
oriented so the mutant's published direction is positive: less dwelling without serotonin (*tph-1*),
more without PDFR-1 (*pdfr-1*). An acute arm, ``modulated+<knockout>``, evaluates the intact arm's
trained weights with that modulator removed, on the intact arm's evaluation episodes. The two are
Holm-adjusted together. Beside them: the raised mutants and the satiety link against the intact
arm, the concentrations in each state, the learned pathway strengths, and stage 0's connectome.

Usage::

    uv run python scripts/analysis/lawn_states.py --logs campaigns/d1-panel/logs \\
        --seeds 2001 2002 ... 2016 --episodes 30 --out panel.json --csv per-run.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
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
import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
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
    from collections.abc import Callable

    from quantumnematode.report.dtypes import BehaviourStep

EVALUATION_RUN_OFFSET = 1_000_000
PANEL_SEEDS: tuple[int, ...] = tuple(range(2001, 2017))
# Fixed by the fourth pilot (seeds 9113-9116): 2/3 of its +0.591 dwelling-share reference.
PANEL_MINIMUM = 0.394
MDE_Z = 2.487
# The registered reading's state -> what it licenses.
VERDICTS: dict[str, str] = {
    "move_wt": "dwells",
    "move_null": "dwells_less_than_its_floor",
    "below": "dwelling_below_minimum",
    "no_move": "no_dwelling_at_minimum",
    "unresolved": "unresolved_at_this_sensitivity",
}
# B.3: the dwelling gate is classified at stage 0's registered minimum and passes on any significant
# dwelling; the speed readings' minimum is a third of the 0.028 mm/s between real worms' calibrated
# dwelling and roaming speeds. Each knockout's published direction
# is the sign of its reading minus the intact network's: on dwelling share, less without serotonin
# and more without PDFR-1; on on-lawn speed, the reverse, since roaming is the faster state.
STAGE0_DWELLING_MINIMUM = 0.278
KNOCKOUT_SPEED_MINIMUM = 0.009
KNOCKOUT_DIRECTIONS: dict[str, dict[str, int]] = {
    "dwelling": {"tph1": -1, "pdfr1": 1},
    "speed": {"tph1": 1, "pdfr1": -1},
}
KNOCKOUT_VERDICTS: dict[str, str] = {
    "move_wt": "mutant_direction",
    "move_null": "opposite_direction",
    "below": "below_minimum",
    "no_move": "no_effect_at_minimum",
    "unresolved": "unresolved_at_this_sensitivity",
}
MODULATOR_PATHWAYS: tuple[str, ...] = ("mod1", "pdfr1", "pdf_onto_nsm", "ingestion_onto_nsm")
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


def plateaus(
    log_dirs: list[Path],
    arm: str,
    seeds: tuple[int, ...],
    cell: str = "patch",
) -> dict[int, float]:
    """Return each seed's intake plateau for one arm's runs in ``cell``."""
    out: dict[int, float] = {}
    for seed in seeds:
        for log_dir in log_dirs:
            value = intake_plateau(log_dir / f"{gen.stem(arm, cell)}-seed{seed}.log")
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
        "learn": {s: learn[s] for s in seeds},
        "floor": {s: floor[s] for s in seeds},
        "learn_mean": statistics.fmean(learn[s] for s in seeds) if seeds else None,
        "floor_mean": statistics.fmean(floor[s] for s in seeds) if seeds else None,
        "test": test,
        "passes": bool(seeds) and float(test["ci_lo"]) > 0.0,
    }


# ── The states ──────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Episode:
    """One evaluated episode's per-window states, the density under the worm, and its exits."""

    states: np.ndarray
    density: np.ndarray
    exits: tuple[int, ...] = ()
    # Per window, the mean serotonin and PDF concentrations the worm's brain saw; None without them.
    concentrations: np.ndarray | None = None
    # Per window, the mean speed (mm/s); None where not recorded.
    speed: np.ndarray | None = None


def episode_windows(
    steps: list[BehaviourStep],
    classifier: rd.GaussianClassifier,
    intake_fraction: float,
    quality: float,
    concentrations: np.ndarray | None = None,
) -> Episode:
    """Cut one episode's captured steps into windows; classify them; recover the cell's density.

    A window spans three position samples, two steps. It is on a lawn when all three are. Its
    density is the mean density of the cells eaten from in it, or NaN where the worm ate nothing.
    ``concentrations``, one ``(serotonin, PDF)`` row per captured step, is averaged the same way.
    """
    x = np.array([s.x for s in steps])
    y = np.array([s.y for s in steps])
    on = np.array([bool(s.on_lawn) for s in steps])
    off = ~on
    eaten = np.array([s.intake or 0.0 for s in steps])
    speed, angular = rd.window_measures(x, y)
    n = len(speed)
    on_window = np.array([on[2 * k : 2 * k + 3].all() for k in range(n)], dtype=bool)
    off_window = np.array([off[2 * k : 2 * k + 3].all() for k in range(n)], dtype=bool)
    states = classifier.states(speed, angular, on_window)
    density = np.full(n, np.nan)
    for k in range(n):
        bites = eaten[2 * k + 1 : 2 * k + 3]
        bites = bites[bites > 0]
        if bites.size:
            density[k] = float(bites.mean()) / (intake_fraction * quality)
    per_window = None
    if concentrations is not None and len(concentrations) >= len(steps):
        per_window = np.array([concentrations[2 * k : 2 * k + 3].mean(axis=0) for k in range(n)])
    return Episode(
        states=states,
        density=density,
        exits=tuple(rd.lawn_exits(on_window, off_window)),
        concentrations=per_window,
        speed=speed,
    )


def on_lawn_speed(episodes: list[Episode]) -> float | None:
    """Return the mean speed (mm/s) over every on-lawn window, the states' continuous measure."""
    speeds = [e.speed[e.states != rd.OFF_FOOD] for e in episodes if e.speed is not None]
    pooled = np.concatenate(speeds) if speeds else np.zeros(0)
    return float(pooled.mean()) if pooled.size else None


def concentrations_by_state(episodes: list[Episode]) -> dict[str, float | None]:
    """Return the mean serotonin and PDF concentrations in on-lawn roaming and dwelling windows."""
    pairs = [(e.states, e.concentrations) for e in episodes if e.concentrations is not None]
    if not pairs:
        return {}
    states = np.concatenate([s for s, _ in pairs])
    values = np.concatenate([c for _, c in pairs])
    out: dict[str, float | None] = {}
    for name, state in (("roaming", rd.ROAMING), ("dwelling", rd.DWELLING)):
        mask = states == state
        for column, modulator in enumerate(("serotonin", "pdf")):
            out[f"{modulator}_{name}"] = float(values[mask, column].mean()) if mask.any() else None
    return out


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


def dwelling_ends(episodes: list[Episode]) -> dict[str, float | None]:
    """How often a dwelling window is followed by roaming, where its spot is grazed against fresh.

    It conditions on the worm already dwelling, so unlike roaming's share by density it is not
    confounded by dwelling being what grazes a cell.
    """
    ends: dict[str, list[bool]] = {"grazed": [], "fresh": []}
    for e in episodes:
        for k in range(len(e.states) - 1):
            if e.states[k] != rd.DWELLING or e.states[k + 1] == rd.OFF_FOOD:
                continue
            if np.isnan(e.density[k]):
                continue
            key = "grazed" if e.density[k] < GRAZED_DENSITY else "fresh"
            ends[key].append(bool(e.states[k + 1] == rd.ROAMING))
    return {f"dwell_end_{key}": float(np.mean(v)) if v else None for key, v in ends.items()}


def leaving_readings(episodes: list[Episode]) -> dict[str, Any]:
    """Pool a run's exits: their count and rate, and its band and last-window enrichment."""
    states = [e.states for e in episodes]
    exits = [list(e.exits) for e in episodes]
    n_exits = sum(len(x) for x in exits)
    on_windows = sum(int(np.sum(s != rd.OFF_FOOD)) for s in states)
    on_hours = on_windows * rd.STEP_SECONDS * rd.WINDOW_STEPS / 3600.0
    on_states = np.concatenate([s[s != rd.OFF_FOOD] for s in states]) if states else np.zeros(0)
    overall = float(np.mean(on_states == rd.ROAMING)) if on_states.size else None
    band = np.concatenate([rd.pre_exit_band(s, x) for s, x in zip(states, exits, strict=True)])
    last = np.concatenate(
        [rd.last_window_before_exits(s, x) for s, x in zip(states, exits, strict=True)],
    )
    band_share = float(np.mean(band == rd.ROAMING)) if band.size else None
    last_share = float(np.mean(last == rd.ROAMING)) if last.size else None
    return {
        "exits": n_exits,
        "exits_per_on_lawn_hour": n_exits / on_hours if on_hours else None,
        "band_windows": int(band.size),
        "band_roaming": band_share,
        "band_enrichment": None if band_share is None or overall is None else band_share - overall,
        "last_window_enrichment": None
        if last_share is None or overall is None
        else last_share - overall,
    }


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
        **dwelling_ends(episodes),
        **leaving_readings(episodes),
        **concentrations_by_state(episodes),
        "on_lawn_speed": on_lawn_speed(episodes),
    }


def capture(
    config_path: Path,
    seed: int,
    weights: Path | None,
    episodes: int,
    overrides: dict[str, object] | None = None,
) -> tuple[Any, ...]:
    """Run ``episodes`` frozen held-out episodes with behaviour capture.

    Returns the captured steps, each episode's intake, the lawns, and, for a brain with modulators,
    each episode's ``(serotonin, PDF)`` concentrations as each step saw them (otherwise None).
    ``overrides`` replaces brain-config fields for the evaluation, as an acute knockout does.
    """
    brain, config, sensing = harness.build_brain(config_path, seed, overrides)
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
    seen: list[tuple[float, float]] = []
    traced: list[np.ndarray] | None = None
    if getattr(brain, "modulator_concentrations", None) is not None:
        traced = []
        act = brain.run_brain

        def run_brain(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 - the brain's signature
            # The concentrations this step's state carries, read before the step updates them.
            seen.append(brain.modulator_concentrations)  # type: ignore[attr-defined]
            return act(*args, **kwargs)

        brain.run_brain = run_brain  # type: ignore[method-assign]
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
        if traced is not None:
            traced.append(np.array(seen, dtype=float).reshape(-1, 2))
            seen.clear()
    return captured, intakes, lawns, traced


def evaluate_run(job: tuple[str, int, list[str], int, str]) -> dict[str, Any]:
    """Evaluate one run's states; a floor arm reads its seed's untrained policy.

    An acute arm, ``<arm>+<knockout>``, reads its trained arm's weights and config with that
    knockout set at evaluation.
    """
    arm, seed, log_dirs, episodes, cell = job
    knockout = arm.partition("+")[2]
    overrides: dict[str, object] = {"modulator_knockout": knockout} if knockout else {}
    weights = None
    if not arm.endswith("_frozen"):
        name = f"{gen.stem(arm, cell)}-seed{seed}.log"
        log = next((Path(d) / name for d in log_dirs if (Path(d) / name).is_file()), None)
        weights = bc.final_weights(log) if log is not None else None
        if weights is None:
            return {"arm": arm, "seed": seed, "missing": True}
    config = gen.FORAGING / f"{gen.stem(arm, cell)}.yml"
    captured, intakes, lawns, traced = capture(config, seed, weights, episodes, overrides)
    classifier = rd.GaussianClassifier(hmm=rd.load_calibrated_hmm())
    quality = float(lawns.quality[0]) if lawns.quality[0] == lawns.quality[1] else float("nan")
    windows = [
        episode_windows(
            steps,
            classifier,
            lawns.intake_fraction,
            quality,
            None if traced is None else traced[i],
        )
        for i, steps in enumerate(captured)
    ]
    return {
        "arm": arm,
        "seed": seed,
        "evaluation_intake": statistics.fmean(intakes) if intakes else None,
        "action_log_std": action_log_std(weights),
        "modulator_strengths": modulator_strengths(weights),
        **run_readings(windows),
    }


def action_log_std(weights: Path | None) -> list[float] | None:
    """Return the policy's learned action noise, ``log_std`` for (speed, turn), from its weights.

    The policy samples its actions, so its noise is part of the behaviour the instrument reads.
    """
    if weights is None:
        return None
    import torch

    state = torch.load(weights, map_location="cpu", weights_only=True)
    return log_std_of(state)


def modulator_strengths(weights: Path | None) -> dict[str, float] | None:
    """Return a trained connectome's four learned pathway strengths, or None without the field."""
    if weights is None:
        return None
    import torch

    state = torch.load(weights, map_location="cpu", weights_only=True)
    topology = state.get("topology") if isinstance(state, dict) else None
    rho = topology.get("modulator_rho") if isinstance(topology, dict) else None
    if rho is None:
        return None
    strengths = torch.nn.functional.softplus(rho).tolist()
    return dict(zip(MODULATOR_PATHWAYS, (float(v) for v in strengths), strict=True))


def log_std_of(state: object) -> list[float] | None:
    """Find a saved policy's ``log_std``: at the top level (MLP-PPO) or in its topology (connectome)."""
    if not isinstance(state, dict):
        return None
    log_std = state.get("log_std")
    if isinstance(log_std, dict):
        log_std = log_std.get("log_std")
    if log_std is None and isinstance(state.get("topology"), dict):
        log_std = state["topology"].get("log_std")
    return [float(v) for v in log_std.tolist()] if log_std is not None else None


def _dwelling(run: dict[str, Any]) -> float | None:
    fraction = run.get("roaming_fraction")
    return None if fraction is None else 1.0 - float(fraction)


def paired(
    runs: list[dict[str, Any]],
    arm: str,
    other: str,
    key: Callable[[dict[str, Any]], float | None] = _dwelling,
) -> dict[str, Any]:
    """Pair one arm's per-seed reading with another's: the mean difference and its tests."""
    by = {(r["arm"], r["seed"]): r for r in runs if not r.get("missing")}
    seeds = sorted({s for a, s in by if a == arm} & {s for a, s in by if a == other})
    values: dict[int, float] = {}
    for s in seeds:
        a, b = key(by[(arm, s)]), key(by[(other, s)])
        if a is not None and b is not None:
            values[s] = a - b
    test = wp.paired_seed_wilcoxon_bootstrap(list(values.values())) if values else None
    return {"arm": arm, "against": other, "per_seed": values, "test": test}


def read_panel(
    gate: dict[str, Any],
    runs: list[dict[str, Any]],
    learner: str = "internal",
    floor: str = "internal_frozen",
    minimum: float = PANEL_MINIMUM,
) -> dict[str, Any]:
    """Read the gate first, then the registered reading's state at the minimum, then its verdict.

    The reading is the learner's on-lawn dwelling share minus its untrained floor's, paired by
    seed. A learner that does not beat its floor on intake leaves the panel unreadable.
    """
    out: dict[str, Any] = {"minimum": minimum, "readable": bool(gate["passes"])}
    if not out["readable"]:
        out["verdict"] = "unreadable"
        return out
    reading = paired(runs, learner, floor)
    test = reading["test"]
    per_seed = list(reading["per_seed"].values())
    q = ops.two_sided(test["wilcoxon_p"])
    state = mc.classify(test["mean_delta"], test["ci_lo"], test["ci_hi"], q, minimum)
    sd = statistics.stdev(per_seed) if len(per_seed) > 1 else float("nan")
    out |= {
        "reading": reading,
        "q": q,
        "state": state,
        "verdict": VERDICTS[state],
        "achieved_sd": sd,
        "achieved_mde": MDE_Z * sd / math.sqrt(len(per_seed)) if per_seed else None,
    }
    return out


def _speed(run: dict[str, Any]) -> float | None:
    speed = run.get("on_lawn_speed")
    return None if speed is None else float(speed)


def _intake(run: dict[str, Any]) -> float | None:
    intake = run.get("evaluation_intake")
    return None if intake is None else float(intake)


_MEASURES: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "dwelling": _dwelling,
    "speed": _speed,
}


def _oriented(
    direction: int,
    measure: str = "dwelling",
) -> Callable[[dict[str, Any]], float | None]:
    """Return a run's reading on ``measure`` times ``direction``, so a mutant's direction is positive."""
    read = _MEASURES[measure]

    def key(run: dict[str, Any]) -> float | None:
        value = read(run)
        return None if value is None else direction * value

    return key


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-adjust a family of p values: step down from the smallest, monotone, capped at 1."""
    ordered = sorted(p_values, key=p_values.__getitem__)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def dwells_significantly(dwells: dict[str, Any]) -> bool:
    """Whether a dwelling reading is significantly above its floor, at any size.

    The knockout readings need only a dwelling state for the knockouts to shift, so the gate is the
    reading's significant positive states: ``move_wt``, or ``below`` with a positive mean.
    """
    state = dwells.get("state")
    if state == "move_wt":
        return True
    reading = dwells.get("reading") or {}
    test = reading.get("test") or {}
    return state == "below" and float(test.get("mean_delta", 0.0)) > 0.0


def read_knockouts(  # noqa: PLR0913 - the gate, the runs and the registered reading
    gate: dict[str, Any],
    runs: list[dict[str, Any]],
    minimum: float,
    intact: str = "modulated",
    floor: str = "modulated_frozen",
    measure: str = "speed",
) -> dict[str, Any]:
    """Read the two gates, then each acute knockout's state at the minimum, Holm across the two.

    Each reading is the knockout's ``measure`` -- on-lawn speed (mm/s) or dwelling share -- minus
    the intact network's, paired by seed, times the mutant's published direction, so a reproduced
    mutant reads positive. The dwelling gate is read on dwelling share whatever the measure, at
    stage 0's minimum, and passes on significant dwelling of any size
    (:func:`dwells_significantly`).
    """
    dwells = read_panel(gate, runs, intact, floor, STAGE0_DWELLING_MINIMUM)
    out: dict[str, Any] = {
        "minimum": minimum,
        "measure": measure,
        "dwelling_gate": dwells,
        "readable": bool(gate["passes"]) and dwells_significantly(dwells),
    }
    if not out["readable"]:
        out["verdict"] = "unreadable"
        return out
    readings = {
        knockout: paired(
            runs,
            f"{intact}+{knockout}",
            intact,
            key=_oriented(direction, measure),
        )
        for knockout, direction in KNOCKOUT_DIRECTIONS[measure].items()
    }
    qs = holm({k: ops.two_sided(r["test"]["wilcoxon_p"]) for k, r in readings.items()})
    for knockout, reading in readings.items():
        test = reading["test"]
        state = mc.classify(test["mean_delta"], test["ci_lo"], test["ci_hi"], qs[knockout], minimum)
        per_seed = list(reading["per_seed"].values())
        sd = statistics.stdev(per_seed) if len(per_seed) > 1 else float("nan")
        out[knockout] = {
            "reading": reading,
            "direction": KNOCKOUT_DIRECTIONS[measure][knockout],
            "holm_q": qs[knockout],
            "state": state,
            "verdict": KNOCKOUT_VERDICTS[state],
            "achieved_sd": sd,
            "achieved_mde": MDE_Z * sd / math.sqrt(len(per_seed)) if per_seed else None,
        }
    return out


def modulator_beside(
    runs: list[dict[str, Any]],
    intact: str = "modulated",
) -> dict[str, Any]:
    """Return the modulator readings reported beside the verdicts, never read as one."""
    arms = {r["arm"] for r in runs}
    out: dict[str, Any] = {}
    for measure, directions in KNOCKOUT_DIRECTIONS.items():
        for knockout, direction in directions.items():
            raised = f"{intact}_{knockout}"
            if raised in arms:
                out[f"raised_{knockout}_{measure}_oriented"] = paired(
                    runs,
                    raised,
                    intact,
                    key=_oriented(direction, measure),
                )
        for knockout, direction in directions.items():
            acute = f"{intact}+{knockout}"
            if acute in arms:
                out[f"acute_{knockout}_{measure}_oriented"] = paired(
                    runs,
                    acute,
                    intact,
                    key=_oriented(direction, measure),
                )
    if f"{intact}_satiety" in arms:
        out["satiety_link_minus_intact_dwelling"] = paired(runs, f"{intact}_satiety", intact)
        out["satiety_link_minus_intact_speed"] = paired(runs, f"{intact}_satiety", intact, _speed)
    for knockout in KNOCKOUT_DIRECTIONS["speed"]:
        acute = f"{intact}+{knockout}"
        if acute in arms:
            out[f"acute_{knockout}_minus_intact_intake"] = paired(runs, acute, intact, _intake)
    out["strengths"] = {
        r["seed"]: r.get("modulator_strengths")
        for r in runs
        if r["arm"] == intact and not r.get("missing")
    }
    out["concentrations_by_state"] = {
        arm: {
            r["seed"]: {k: r.get(k) for k in MODULATOR_CSV_FIELDS[:4]}
            for r in runs
            if r["arm"] == arm and not r.get("missing")
        }
        for arm in sorted(arms)
        if arm.split("+")[0] in {intact, *(f"{intact}{r}" for r in ("_tph1", "_pdfr1", "_satiety"))}
    }
    return out


def against_panel(
    runs: list[dict[str, Any]],
    learner: str,
    panel: Path,
    arm: str,
    label: str,
) -> dict[str, Any]:
    """Pair a learner's dwelling share with another panel's arm by seed, from its readings."""
    other = [
        r | {"arm": label}
        for r in json.loads(panel.read_text())["runs"]
        if r["arm"] == arm and not r.get("missing")
    ]
    return paired([*runs, *other], learner, label)


LEAVING_VERDICTS: dict[str, str] = {
    "move_wt": "leaves_from_roaming",
    "move_null": "leaves_from_dwelling",
    "below": "coupling_below_minimum",
    "no_move": "no_coupling_at_minimum",
    "unresolved": "unresolved_at_this_sensitivity",
}


def read_leaving(  # noqa: PLR0913 - the gates and their registered thresholds
    gate: dict[str, Any],
    runs: list[dict[str, Any]],
    *,
    minimum: float,
    min_exits: int,
    min_seeds: int,
    roaming_bounds: tuple[float, float] = (0.05, 0.95),
) -> dict[str, Any]:
    """Read the gates in order, then the learner's band enrichment at the minimum.

    The learner must beat its floor on intake. A seed is read only if it leaves a lawn ``min_exits``
    times and takes both states on lawns (its roaming share within ``roaming_bounds``); other seeds
    are reported and left out, and fewer than ``min_seeds`` read seeds leave the panel unreadable.
    The reading is a one-sample one: each read seed's enrichment, against zero.
    """
    learner = [r for r in runs if r["arm"] == "internal" and not r.get("missing")]
    low, high = roaming_bounds

    def both_states(run: dict[str, Any]) -> bool:
        share = run.get("roaming_fraction")
        return share is not None and low <= share <= high

    leaving = [r for r in learner if (r.get("exits") or 0) >= min_exits and both_states(r)]
    out: dict[str, Any] = {
        "minimum": minimum,
        "min_exits": min_exits,
        "learning_gate": bool(gate["passes"]),
        "seeds_leaving": [r["seed"] for r in leaving],
        "seeds_too_few_exits": [r["seed"] for r in learner if (r.get("exits") or 0) < min_exits],
        "seeds_one_state": [r["seed"] for r in learner if not both_states(r)],
    }
    values = {
        r["seed"]: r["band_enrichment"] for r in leaving if r.get("band_enrichment") is not None
    }
    out["readable"] = out["learning_gate"] and len(values) >= min_seeds
    if not out["readable"]:
        out["verdict"] = "unreadable"
        return out
    test = wp.paired_seed_wilcoxon_bootstrap(list(values.values()))
    q = ops.two_sided(test["wilcoxon_p"])
    state = mc.classify(test["mean_delta"], test["ci_lo"], test["ci_hi"], q, minimum)
    sd = statistics.stdev(values.values()) if len(values) > 1 else float("nan")
    return out | {
        "per_seed": values,
        "test": test,
        "q": q,
        "state": state,
        "verdict": LEAVING_VERDICTS[state],
        "achieved_sd": sd,
        "achieved_mde": MDE_Z * sd / math.sqrt(len(values)),
    }


def beside(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the readings reported beside the verdict, never read as one."""

    def depletion(run: dict[str, Any]) -> float | None:
        grazed, fresh = run.get("roaming_where_grazed"), run.get("roaming_where_fresh")
        return None if grazed is None or fresh is None else grazed - fresh

    return {
        "blind_minus_floor_dwelling": paired(runs, "blind", "internal_frozen"),
        "internal_minus_blind_dwelling": paired(runs, "internal", "blind"),
        "roaming_grazed_minus_fresh": {
            arm: {r["seed"]: depletion(r) for r in runs if r["arm"] == arm and not r.get("missing")}
            for arm in ("internal", "blind")
        },
        "seeds_with_both_bouts": {
            arm: sum(
                1
                for r in runs
                if r["arm"] == arm and r.get("n_roaming_bouts") and r.get("n_dwelling_bouts")
            )
            for arm in ("internal", "blind")
        },
    }


def against_mlp(runs: list[dict[str, Any]], learner: str, mlp_panel: Path) -> dict[str, Any]:
    """Pair a learner's dwelling share with D.1's MLP's by seed, from D.1's committed readings."""
    mlp_runs = [
        r | {"arm": "mlp"}
        for r in json.loads(mlp_panel.read_text())["runs"]
        if r["arm"] == "internal" and not r.get("missing")
    ]
    return paired([*runs, *mlp_runs], learner, "mlp")


CSV_FIELDS = (
    "arm",
    "seed",
    "intake_plateau",
    "evaluation_intake",
    "on_lawn_windows",
    "roaming_fraction",
    "dwelling_share",
    "n_roaming_bouts",
    "n_dwelling_bouts",
    "roaming_bout_s",
    "dwelling_bout_s",
    "roaming_where_grazed",
    "roaming_where_fresh",
    "log_std_speed",
    "log_std_turn",
    "dwell_end_grazed",
    "dwell_end_fresh",
    "exits",
    "exits_per_on_lawn_hour",
    "band_roaming",
    "band_enrichment",
    "last_window_enrichment",
    "on_lawn_speed",
)
MODULATOR_CSV_FIELDS = (
    "serotonin_roaming",
    "serotonin_dwelling",
    "pdf_roaming",
    "pdf_dwelling",
    *(f"strength_{p}" for p in MODULATOR_PATHWAYS),
)


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """Write one row per evaluated run, with its intake plateau from training."""
    plateau: dict[tuple[str, int], float] = {}
    for arm, gate in result["gates"].items():
        plateau |= {(arm, int(s)): v for s, v in gate["learn"].items()}
        floor_arm = result.get("floor_arm", "internal_frozen")
        plateau |= {(floor_arm, int(s)): v for s, v in gate["floor"].items()}
    modulated = any(
        r.get("modulator_strengths") or r.get("serotonin_roaming") is not None
        for r in result["runs"]
    )
    fields = CSV_FIELDS + MODULATOR_CSV_FIELDS if modulated else CSV_FIELDS
    rows = []
    for run in sorted(result["runs"], key=lambda r: (r["arm"], r["seed"])):
        log_std = run.get("action_log_std") or [None, None]
        roaming = run.get("roaming_fraction")
        strengths = run.get("modulator_strengths") or {}
        rows.append(
            {
                **{k: run.get(k) for k in fields if k in run},
                **{f"strength_{p}": v for p, v in strengths.items()},
                "intake_plateau": plateau.get((run["arm"], run["seed"])),
                "dwelling_share": None if roaming is None else 1.0 - roaming,
                "log_std_speed": log_std[0],
                "log_std_turn": log_std[1],
            },
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def modulator_results(
    args: argparse.Namespace,
    gates: dict[str, Any],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the knockout readings and what is reported beside them, when the panel has them."""
    learner, floor = args.learner_arm, args.floor_arm
    acute = [arm for arm in args.arms if "+" in arm]
    if args.knockout_minimum is None or not acute or learner not in gates or floor not in args.arms:
        return {}
    out: dict[str, Any] = {
        "knockouts": read_knockouts(
            gates[learner],
            runs,
            args.knockout_minimum,
            learner,
            floor,
            args.knockout_measure,
        ),
        "modulator_beside": modulator_beside(runs, learner),
    }
    if args.compare_stage0 is not None:
        out["against_stage0"] = against_panel(
            runs,
            learner,
            args.compare_stage0,
            "connectome_e005",
            "stage0",
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI: the learning gate from the logs, and the states from an evaluation."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--logs", type=Path, action="append", required=True)
    ap.add_argument(
        "--arms",
        nargs="+",
        choices=[*gen.ARMS, *gen.CONNECTOME_ARMS, *gen.MODULATED_ARMS, *gen.ACUTE_ARMS],
        default=None,
    )
    ap.add_argument("--learner-arm", default="internal", help="the registered learner arm")
    ap.add_argument("--floor-arm", default="internal_frozen", help="its untrained floor")
    ap.add_argument("--minimum", type=float, default=PANEL_MINIMUM, help="the registered minimum")
    ap.add_argument("--compare-mlp", type=Path, help="D.1's panel.json, to pair against its MLP")
    ap.add_argument(
        "--compare-stage0",
        type=Path,
        help="stage 0's panel.json, to pair the modulated learner against its connectome",
    )
    ap.add_argument("--knockout-minimum", type=float, help="the knockouts' registered minimum")
    ap.add_argument(
        "--knockout-measure",
        choices=list(KNOCKOUT_DIRECTIONS),
        default="speed",
        help="what the knockouts are read on: on-lawn speed (mm/s) or dwelling share",
    )
    ap.add_argument("--cell", choices=list(gen.CELLS), default="patch")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, help="write the JSON here")
    ap.add_argument("--csv", type=Path, help="write one row per evaluated run here")
    ap.add_argument("--leaving-minimum", type=float, help="the leaving cell's registered minimum")
    ap.add_argument("--min-exits", type=int, help="exits a learner seed needs to be read")
    ap.add_argument("--min-seeds", type=int, help="readable seeds the leaving panel needs")
    args = ap.parse_args(argv)
    seeds = tuple(args.seeds)
    if args.arms is None:
        args.arms = list(gen.ARMS if args.cell == "patch" else gen.LEAVING_ARMS)
    gates = {
        arm: learning_gate(
            plateaus(args.logs, arm, seeds, args.cell),
            plateaus(args.logs, args.floor_arm, seeds, args.cell),
        )
        for arm in args.arms
        if not arm.endswith("_frozen") and "+" not in arm
    }
    jobs = [
        (arm, seed, [str(d) for d in args.logs], args.episodes, args.cell)
        for arm in args.arms
        for seed in seeds
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(evaluate_run, jobs))
    result: dict[str, Any] = {
        "cell": args.cell,
        "episodes": args.episodes,
        "floor_arm": args.floor_arm,
        "gates": gates,
        "runs": runs,
    }
    learner, floor = args.learner_arm, args.floor_arm
    if args.cell == "patch" and learner in gates and floor in args.arms:
        result["panel"] = read_panel(gates[learner], runs, learner, floor, args.minimum)
        if "blind" in args.arms:
            result["beside"] = beside(runs)
        if args.compare_mlp is not None:
            result["against_mlp"] = against_mlp(runs, learner, args.compare_mlp)
    result |= modulator_results(args, gates, runs)
    leaving_args = (args.leaving_minimum, args.min_exits, args.min_seeds)
    if args.cell != "patch" and "internal" in gates and None not in leaving_args:
        result["leaving"] = read_leaving(
            gates["internal"],
            runs,
            minimum=args.leaving_minimum,
            min_exits=args.min_exits,
            min_seeds=args.min_seeds,
        )
    if args.csv:
        write_csv(result, args.csv)
    payload = json.dumps(result, indent=2, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
