#!/usr/bin/env python
"""G.0: whether the gap-placement effect belongs to the wiring or to how gap junctions are scaled.

On the thermal cell at target 35, on Emmons 2024, under PPO at block V's point otherwise, the wild
type leads the gap-only null (chemical graph held, gap junctions rewired with their counts). The
gap matrix is ``count / sqrt(d_i d_j)``: a few high-count pairs couple far above 1, and the swap
moves them. This panel asks whether the lead survives without the counts, and whether moving the
count outliers alone reproduces it.

**The anchor** (seeds 9801-9816; the four ``count`` arms) fixes the registered minimum ``m`` by
D23's rule: 082's 0.0577 is carried if the anchor's lead sits in [0.058, 0.13] and the level is
readable; two-thirds of the anchor's lead if the lead excludes zero outside that range; 0.0577 again
if the anchor's interval includes zero, the panel's own ``gap(count)`` then deciding
``no_gap_effect``.

**The panel** (seeds 2401-2464). Four readings, each a paired ``wild type - null`` gap on
``auc_success``, corrected together and classified by ``mc.classify`` at ``m``: ``gap(count)``,
``gap(unit)``, ``gap(outliers_scattered)``, ``gap(outliers_held)``. ``gap(count)`` must exclude zero
above, or the panel reads ``no_gap_effect``. Then the two deciding readings, each in one of three
classes -- ``gap(unit)`` holds / fails / open, ``gap(outliers_scattered)`` reproduces / does not /
open -- give the verdict through a table that names every combination (``VERDICTS``). A deciding
level that fails its floor or saturates is open, never read.

Described beside, never verdicts: ``gap(outliers_held)``; gap-off against both count arms; the
gap-scale profile and log-count weights; settling depths 2, 3 and 6; episodes to competence; the
censoring rule's choice.

**The predictor** (Decision E) is registered before it is computed: on untrained brains, the
saturated fraction over the injected sensors, command interneurons and readout motor neurons,
null minus wild type, against each seed's committed gap.

**The table** (Decision F, D24): for every gap wiring at every panel seed, the coupling's maximum
and 99th percentile, the named neurons' total gap input, the sensor-motor gap pairs, the weight-
degree correlation, and the one- and two-hop sensor-to-motor reach.

Usage::

    uv run python scripts/analysis/gap_operating_point.py plan
    uv run python scripts/analysis/gap_operating_point.py anchor --logs <dir> --out-dir <scratch> \
        --out anchor.json
    uv run python scripts/analysis/gap_operating_point.py preflight --logs <dir> --out preflight.json
    uv run python scripts/analysis/gap_operating_point.py score --logs <dir> --anchor anchor.json \
        --out-dir <scratch> --out panel.json --csv per-seed.csv
    uv run python scripts/analysis/gap_operating_point.py predictor --source cook --out pred.json
    uv run python scripts/analysis/gap_operating_point.py table --out table.json
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quantumnematode.brain.arch.connectome_ppo import ConnectomePPOBrain, ConnectomeTopology

_HERE = Path(__file__).resolve().parent
for _path in (_HERE, _HERE.parent / "campaigns"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gate_preflight as gp  # noqa: E402  # pyright: ignore[reportMissingImports]
import generate_gap_operating_point_configs as gen  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_contrast as mc  # noqa: E402  # pyright: ignore[reportMissingImports]
import measured_prior_pilot as mp  # noqa: E402  # pyright: ignore[reportMissingImports]
import operating_point_surface as ops  # noqa: E402  # pyright: ignore[reportMissingImports]
import thermal_split as ts  # noqa: E402  # pyright: ignore[reportMissingImports]
import wiring_premise as wp  # noqa: E402  # pyright: ignore[reportMissingImports]

HALF = "ppo"
CELL = ts.CELL
SEEDS: tuple[int, ...] = tuple(range(2401, 2465))
SEEDS_32: tuple[int, ...] = SEEDS[:32]
SEEDS_16: tuple[int, ...] = SEEDS[:16]
ANCHOR_SEEDS: tuple[int, ...] = tuple(range(9801, 9817))
SMOKE_SEEDS: tuple[int, ...] = (9851, 9852)

COUNT, UNIT = "count", "unit"
SCATTERED, HELD = "outliers_scattered", "outliers_held"
GAP_OFF_VS_WT, GAP_OFF_VS_NULL = "gap_off_vs_wt", "gap_off_vs_null"
PROFILE: tuple[str, ...] = tuple(gen.PROFILE_LEVELS)
DECIDING: tuple[str, ...] = (UNIT, SCATTERED)
# The four readings corrected together.
FAMILY: tuple[str, ...] = (COUNT, UNIT, SCATTERED, HELD)
LEVELS: tuple[str, ...] = (*FAMILY, GAP_OFF_VS_WT, GAP_OFF_VS_NULL, *PROFILE)
# Levels with no frozen floor of their own: their gate reads the count level's floors, the
# depth-4, count-weight brain, so it says whether each arm learns at all and nothing finer.
FLOOR_BORROWED: tuple[str, ...] = PROFILE

PRIMARY_METRIC = ops.UNCENSORED_METRIC
BESIDE_METRIC = ops.CENSORED_METRIC
METRICS: tuple[str, ...] = (PRIMARY_METRIC, BESIDE_METRIC)

# D23's minimum. 082 registered two-thirds of the thermal split's 0.0865 on this cell on Cook 2019.
CARRIED_MINIMUM = mc.MINIMUM_FRACTION * 0.08647916666666666
CARRY_RANGE: tuple[float, float] = (0.058, 0.13)

# The longest run in the panel, in episodes; the collapse statistic applies beyond 3,000.
EPISODES = 3000


def _stem(name: str) -> str:
    stem = f"{gen.PREFIX}_{name}"
    if stem not in gen.CHILDREN:
        msg = f"{stem} is not a generated G.0 config"
        raise KeyError(msg)
    return stem


def _four(level: str) -> dict[str, str]:
    return {
        "wt_learn": _stem(f"{level}_wt_t35"),
        "wt_frozen": _stem(f"{level}_wt_frozen_t35"),
        "rn_learn": _stem(f"{level}_gaponly_t35"),
        "rn_frozen": _stem(f"{level}_gaponly_frozen_t35"),
    }


def _null(level: str) -> dict[str, str]:
    return {"rn_learn": _stem(f"{level}_t35"), "rn_frozen": _stem(f"{level}_frozen_t35")}


_COUNT_ARMS = _four(COUNT)
_WT = {"wt_learn": _COUNT_ARMS["wt_learn"], "wt_frozen": _COUNT_ARMS["wt_frozen"]}
_RN = {"rn_learn": _COUNT_ARMS["rn_learn"], "rn_frozen": _COUNT_ARMS["rn_frozen"]}
_GAP_OFF = _null("gap_off")
_FROZEN_COUNT = {"wt_frozen": _COUNT_ARMS["wt_frozen"], "rn_frozen": _COUNT_ARMS["rn_frozen"]}

# Level -> arm -> config stem, in the gate preflight's shape.
STEMS: dict[str, dict[str, str]] = {
    COUNT: _COUNT_ARMS,
    UNIT: _four(UNIT),
    SCATTERED: {**_WT, **_null(SCATTERED)},
    HELD: {**_WT, **_null(HELD)},
    GAP_OFF_VS_WT: {**_WT, **_GAP_OFF},
    GAP_OFF_VS_NULL: {
        "wt_learn": _GAP_OFF["rn_learn"],
        "wt_frozen": _GAP_OFF["rn_frozen"],
        **_RN,
    },
    **{
        level: {
            "wt_learn": _stem(f"{level}_wt_t35"),
            "rn_learn": _stem(f"{level}_gaponly_t35"),
            **_FROZEN_COUNT,
        }
        for level in PROFILE
    },
}
# The seeds each level's gap is read on, and the seeds its floors cover.
GAP_SEEDS: dict[str, tuple[int, ...]] = {
    COUNT: SEEDS,
    UNIT: SEEDS,
    SCATTERED: SEEDS_32,
    HELD: SEEDS_32,
    GAP_OFF_VS_WT: SEEDS_32,
    GAP_OFF_VS_NULL: SEEDS_32,
    **dict.fromkeys(PROFILE, SEEDS_16),
}
FLOOR_SEEDS: dict[str, tuple[int, ...]] = {
    COUNT: SEEDS,
    UNIT: SEEDS,
    SCATTERED: SEEDS_32,
    HELD: SEEDS_16,
    GAP_OFF_VS_WT: SEEDS_16,
    GAP_OFF_VS_NULL: SEEDS_16,
    **dict.fromkeys(PROFILE, SEEDS_16),
}
Requirement = dict[str, dict[str, tuple[int, ...]]]
PANEL: Requirement = {
    level: {
        arm: (FLOOR_SEEDS[level] if arm.endswith("_frozen") else GAP_SEEDS[level])
        for arm in gp.ARMS
    }
    for level in LEVELS
}
ANCHOR: Requirement = {COUNT: dict.fromkeys(gp.ARMS, ANCHOR_SEEDS)}


def runs(requirement: Requirement = PANEL) -> dict[str, tuple[int, ...]]:
    """Config stem -> every seed a requirement needs it run on."""
    out: dict[str, set[int]] = {}
    for level, arms in requirement.items():
        for arm, seeds in arms.items():
            out.setdefault(STEMS[level][arm], set()).update(seeds)
    return {stem: tuple(sorted(seeds)) for stem, seeds in sorted(out.items())}


class GapOperatingPointError(ValueError):
    """The runs on disk are not the panel this module scores."""


# ── The minimum: D23's rule ───────────────────────────────────────────────────────────────────
def anchor_minimum(lead: float, ci_lo: float, ci_hi: float, readable: bool) -> dict[str, Any]:  # noqa: FBT001
    """Fix the panel's minimum from the anchor's lead, by D23's three cases.

    An unreadable anchor (a wiring fails its floor, or both saturate) fixes nothing, and the panel
    does not launch on it.
    """
    if not readable:
        return {"case": "unreadable", "minimum": None}
    if ci_lo <= 0.0 <= ci_hi:
        return {"case": "interval_includes_zero", "minimum": CARRIED_MINIMUM}
    if CARRY_RANGE[0] <= lead <= CARRY_RANGE[1]:
        return {"case": "carried", "minimum": CARRIED_MINIMUM}
    return {"case": "refixed", "minimum": mc.MINIMUM_FRACTION * abs(lead)}


# ── The verdict map, fixed before any run ─────────────────────────────────────────────────────
HOLDS, FAILS, OPEN = "holds", "fails", "open"
REPRODUCES, DOES_NOT = "reproduces", "does_not"
_UNIT_CLASS = {"move_wt": HOLDS, "no_move": FAILS, "below": FAILS, "move_null": FAILS}
_SCATTERED_CLASS = {
    "move_wt": REPRODUCES,
    "no_move": DOES_NOT,
    "below": DOES_NOT,
    "move_null": DOES_NOT,
}
VERDICTS: dict[tuple[str, str], str] = {
    (HOLDS, DOES_NOT): "survives",
    (HOLDS, REPRODUCES): "both",
    (HOLDS, OPEN): "unresolved",
    (FAILS, DOES_NOT): "artefact",
    (FAILS, REPRODUCES): "artefact",
    (FAILS, OPEN): "artefact",
    (OPEN, DOES_NOT): "unresolved",
    (OPEN, REPRODUCES): "unresolved",
    (OPEN, OPEN): "unresolved",
}


def unit_class(state: str, readable: bool) -> str:  # noqa: FBT001
    """``gap(unit)``'s class: the lead holds without the counts, fails, or is open."""
    return _UNIT_CLASS.get(state, OPEN) if readable else OPEN


def scattered_class(state: str, readable: bool) -> str:  # noqa: FBT001
    """``gap(outliers_scattered)``'s class: moving the outliers alone reproduces a lead, or not."""
    return _SCATTERED_CLASS.get(state, OPEN) if readable else OPEN


def read_panel(
    gates: dict[str, Any],
    gaps: dict[str, dict[str, Any]],
    minimum: float,
) -> dict[str, Any]:
    """Gates, then each family reading's state, then the verdict."""
    readable = {level: mp.level_passes(gates[level]) for level in gates}
    states = {
        level: mc.classify(
            gaps[level]["gap_mean"],
            gaps[level]["test"]["ci_lo"],
            gaps[level]["test"]["ci_hi"],
            gaps[level]["test"]["bh_q"],
            minimum,
        )
        for level in FAMILY
    }
    out: dict[str, Any] = {"minimum": minimum, "level_readable": readable, "states": states}
    if not readable[COUNT]:
        out["verdict"] = "unreadable"
        out["why"] = "the count level fails a floor or saturates, so no lead is there to test"
        return out
    if float(gaps[COUNT]["test"]["ci_lo"]) <= 0.0:
        out["verdict"] = "no_gap_effect"
        out["why"] = "gap(count)'s interval does not exclude zero above: no effect on Emmons"
        return out
    classes = {
        UNIT: unit_class(states[UNIT], readable[UNIT]),
        SCATTERED: scattered_class(states[SCATTERED], readable[SCATTERED]),
    }
    out["classes"] = classes
    out["verdict"] = VERDICTS[classes[UNIT], classes[SCATTERED]]
    return out


# ── Manifest and scoring ─────────────────────────────────────────────────────────────────────
def build_manifest(log_dirs: list[Path], path: Path, requirement: Requirement) -> Path:
    """Write ``<arm> <level> <seed> <log>`` for every run a requirement names."""
    stems = {level: STEMS[level] for level in requirement}
    seen: dict[tuple[str, str, int], Path] = {}
    lines: list[str] = []
    for arm, level, seed, log in gp.evidence(stems, log_dirs):
        if seed not in requirement[level][arm]:
            continue
        prior = seen.setdefault((arm, level, seed), log)
        if prior != log:
            msg = f"{arm} {level} seed {seed} has two runs: {prior} and {log}"
            raise GapOperatingPointError(msg)
        lines.append(f"{arm} {level} {seed} {log.resolve()}")
    path.write_text("\n".join(lines) + "\n")
    return path


def require_complete(manifest: Path, requirement: Requirement) -> None:
    """Refuse a partial panel: every arm of every level on every seed it is read on."""
    have: dict[tuple[str, str], set[int]] = {}
    for raw in manifest.read_text().splitlines():
        if raw.strip():
            arm, level, seed, _ = raw.split()
            have.setdefault((arm, level), set()).add(int(seed))
    missing = [
        f"{level}/{arm}: {sorted(set(seeds) - have.get((arm, level), set()))}"
        for level, arms in requirement.items()
        for arm, seeds in arms.items()
        if set(seeds) - have.get((arm, level), set())
    ]
    if missing:
        msg = "G.0 panel is incomplete — " + "; ".join(missing)
        raise GapOperatingPointError(msg)


def anchor(log_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    """Score the anchor's count level and fix the panel's minimum by D23's rule."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(log_dirs, out_dir / "manifest-anchor.txt", ANCHOR)
    require_complete(manifest, ANCHOR)
    report = ops.score_level(manifest, HALF, COUNT, out_dir / "tmp-anchor", cell=CELL)
    gaps = {m: ops.wiring_gap(report, m) for m in METRICS}
    gates = ops.learning_gates(manifest, HALF, ANCHOR_SEEDS, COUNT, floor_level=COUNT)
    lead = gaps[PRIMARY_METRIC]
    return {
        "cell": CELL,
        "seeds": list(ANCHOR_SEEDS),
        "primary_metric": PRIMARY_METRIC,
        "gates": gates,
        "gaps": gaps,
        "readable": mp.level_passes(gates),
        "carry_range": list(CARRY_RANGE),
        "carried_minimum": CARRIED_MINIMUM,
        "rule": anchor_minimum(
            float(lead["gap_mean"]),
            float(lead["test"]["ci_lo"]),
            float(lead["test"]["ci_hi"]),
            mp.level_passes(gates),
        ),
    }


def correct(gaps: dict[str, dict[str, Any]]) -> None:
    """BH-FDR the four family readings together, per metric, every test computed first."""
    for metric in METRICS:
        ops.apply_family_correction(
            {level: {metric: {"interaction": gaps[metric][level]}} for level in FAMILY},
        )


def score(log_dirs: list[Path], out_dir: Path, minimum: float) -> dict[str, Any]:
    """Score every level, correct the family, gate each level on its floors' seeds, then read."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(log_dirs, out_dir / "manifest-gap-operating-point.txt", PANEL)
    require_complete(manifest, PANEL)
    reports = {
        level: ops.score_level(manifest, HALF, level, out_dir / f"tmp-{level}", cell=CELL)
        for level in LEVELS
    }
    gaps = {m: {level: ops.wiring_gap(reports[level], m) for level in LEVELS} for m in METRICS}
    correct(gaps)
    gates = {
        level: ops.learning_gates(manifest, HALF, FLOOR_SEEDS[level], level, floor_level=level)
        for level in LEVELS
    }
    for level in FLOOR_BORROWED:
        gates[level]["floor_borrowed_from"] = COUNT
    return {
        "cell": CELL,
        "seeds": list(SEEDS),
        "gap_seeds": {level: [GAP_SEEDS[level][0], GAP_SEEDS[level][-1]] for level in LEVELS},
        "floor_seeds": {level: [FLOOR_SEEDS[level][0], FLOOR_SEEDS[level][-1]] for level in LEVELS},
        "primary_metric": PRIMARY_METRIC,
        "beside_metric": BESIDE_METRIC,
        "gates": gates,
        "gaps": gaps,
        "censoring_rule_choice": ops.choose_metric(
            {level: ops.censoring_rates(report) for level, report in reports.items()},
        ),
        "collapse": (
            f"not read: no run here exceeds {EPISODES} episodes, the horizon below which the "
            "collapse statistic does not apply"
        ),
        "reading": read_panel(gates, gaps[PRIMARY_METRIC], minimum),
    }


def write_csv(result: dict[str, Any], path: Path) -> Path:
    """One row per panel seed: every level's plateaus and floors, and its gap on both metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["seed"]
    for level in LEVELS:
        header += [f"{level}_{c}" for c in ("wt_plateau", "wt_floor", "rn_plateau", "rn_floor")]
        header += [f"{level}_gap_{m}" for m in METRICS]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for seed in result["seeds"]:
            row: list[Any] = [seed]
            for level in LEVELS:
                for wiring in ("wt", "rn"):
                    per_seed = result["gates"][level][wiring]["per_seed"].get(seed, {})
                    row += [_fmt(per_seed.get("learn")), _fmt(per_seed.get("floor"))]
                row += [_fmt(result["gaps"][m][level]["per_seed"].get(seed)) for m in METRICS]
            writer.writerow(row)
    return path


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


# ── Building the graphs: the predictor and the table read the brain that runs ────────────────
def _brain(stem: str, seed: int) -> ConnectomePPOBrain:
    """Construct one arm at one seed, untrained, so what is measured is what runs."""
    import torch
    from quantumnematode.brain.arch.connectome_ppo import (
        ConnectomePPOBrain,
        ConnectomePPOBrainConfig,
    )
    from quantumnematode.brain.arch.dtypes import DeviceType
    from quantumnematode.utils.config_loader import load_simulation_config

    container = load_simulation_config(str(gen.THERMAL / f"{stem}.yml")).brain
    if container is None or not isinstance(container.config, ConnectomePPOBrainConfig):
        msg = f"{stem} does not carry a connectome brain config"
        raise TypeError(msg)
    torch.manual_seed(seed)
    return ConnectomePPOBrain(
        config=container.config.model_copy(update={"seed": seed}),
        device=DeviceType.CPU,
    )


# ── The predictor (Decision E) ───────────────────────────────────────────────────────────────
COMMAND_NEURONS: tuple[str, ...] = tuple(
    f"{cls}{side}" for cls in ("AVA", "AVB", "AVD", "AVE", "PVC") for side in ("L", "R")
)
MINIMUM_RHO = 0.3
ALPHA = 0.05
# Logbook 082's committed draws on Cook 2019, and the column their gap is in.
_T = gen._T
PREDICTOR_SOURCES: dict[str, dict[str, Any]] = {
    "cook": {
        "wild_type": f"{_T}_t35",
        "null": f"{_T}_rewired_gap_only_null_t35",
        "seeds": tuple(range(513, 577)),
        "outcome_csv": wp.REPO
        / "docs/experiments/logbooks/supporting/082-plastic-gaps/per-seed.csv",
        "outcome_column": "fixed_gap_auc_success",
    },
    "emmons": {
        "wild_type": _COUNT_ARMS["wt_learn"],
        "null": _COUNT_ARMS["rn_learn"],
        "seeds": SEEDS,
        "outcome_csv": wp.REPO
        / "docs/experiments/logbooks/supporting/095-gap-operating-point/per-seed.csv",
        "outcome_column": f"{COUNT}_gap_{PRIMARY_METRIC}",
    },
}


def path_indices(topology: ConnectomeTopology) -> list[int]:
    """Return the path set: injected sensors, command interneurons, readout motor neurons."""
    import sensory_motor_hops as hops  # pyright: ignore[reportMissingImports]

    names = list(topology.neuron_names)
    out = list(hops.injected_sensors(topology))
    out += [names.index(n) for n in COMMAND_NEURONS if names.index(n) not in out]
    out += [i for i in topology._motor_flat_indices.tolist() if i not in out]
    return out


def path_saturation(stem: str, seed: int) -> float:
    """Mean fraction of the path set saturated (``|h| > 0.99``) over the fixed probe inputs."""
    import sign_scale_calibration as ssc  # pyright: ignore[reportMissingImports]
    import torch

    topology = _brain(stem, seed).topology
    path = torch.tensor(path_indices(topology), dtype=torch.long)
    fractions: list[float] = []
    with torch.no_grad():
        for food, thermal in ssc.inputs():
            _, hidden = topology.forward_with_hidden(food, thermotaxis_features=thermal)
            fractions.append(float((hidden[path].abs() > ssc.SATURATED).float().mean()))
    return float(np.mean(fractions))


def predictor_class(rho: float, p: float, ci: tuple[float, float]) -> str:
    """Place the predictor in its registered state: predicts, no_prediction or unresolved."""
    if p < ALPHA and rho >= MINIMUM_RHO:
        return "predicts"
    if ci[0] > -MINIMUM_RHO and ci[1] < MINIMUM_RHO:
        return "no_prediction"
    return "unresolved"


def outcomes(path: Path, column: str) -> dict[int, float]:
    """Each seed's committed gap from a per-seed CSV."""
    with path.open() as handle:
        return {
            int(row["seed"]): float(row[column]) for row in csv.DictReader(handle) if row[column]
        }


def predictor(source: str) -> dict[str, Any]:
    """Compute the registered predictor on one source's committed draws."""
    import hop_predictor as hp  # pyright: ignore[reportMissingImports]

    spec = PREDICTOR_SOURCES[source]
    gaps = outcomes(spec["outcome_csv"], spec["outcome_column"])
    seeds = [s for s in spec["seeds"] if s in gaps]
    if len(seeds) != len(spec["seeds"]):
        msg = f"{source}: {len(spec['seeds']) - len(seeds)} seeds have no committed gap"
        raise GapOperatingPointError(msg)
    wild = {s: path_saturation(spec["wild_type"], s) for s in seeds}
    null = {s: path_saturation(spec["null"], s) for s in seeds}
    x = np.array([null[s] - wild[s] for s in seeds])
    y = np.array([gaps[s] for s in seeds])
    test = hp.registered_test(x, y)
    test.pop("verdict", None)
    ci = (float(test["ci"][0]), float(test["ci"][1]))
    test["state"] = predictor_class(float(test["rho"]), float(test["p_two_sided"]), ci)
    return {
        "source": source,
        "statistic": (
            "spearman(null - wild type saturated fraction over the sensor, command and motor set, "
            f"untrained; the committed {spec['outcome_column']})"
        ),
        "predicted_direction": "positive",
        "minimum_abs_rho": MINIMUM_RHO,
        "path": "injected sensors, command interneurons, readout motor neurons",
        "per_seed": {
            str(s): {"wild_type": wild[s], "null": null[s], "gap": gaps[s]} for s in seeds
        },
        "result": test,
    }


# ── The matched-statistics table (Decision F, D24) ───────────────────────────────────────────
TOTAL_NEURONS: tuple[str, ...] = (
    "ALA",
    "CANL",
    "CANR",
    "PVDL",
    "PVDR",
    "AVAL",
    "AVAR",
    "AVBL",
    "AVBR",
)
# The gap wirings the table measures. The depth levels share the count matrices and gap-off has
# none, so neither is listed.
TABLE_STEMS: tuple[str, ...] = tuple(
    dict.fromkeys(
        STEMS[level][arm]
        for level in (COUNT, UNIT, SCATTERED, HELD, "scale0p25", "scale0p5", "scale2", "log_count")
        for arm in ("wt_learn", "rn_learn")
    ),
)


def _spearman_or_none(x: np.ndarray, y: np.ndarray) -> float | None:
    import hop_predictor as hp  # pyright: ignore[reportMissingImports]

    if np.unique(x).size < 2 or np.unique(y).size < 2:  # a rank needs two values
        return None
    return hp.spearman(x, y)


def matched_statistics(brain: ConnectomePPOBrain) -> dict[str, Any]:
    """One built brain's coupling statistics, read off its gap matrix and propagating graph."""
    import sensory_motor_hops as hops  # pyright: ignore[reportMissingImports]

    topology = brain.topology
    gap = topology.g_gap.detach().double().numpy()
    names = list(topology.neuron_names)
    present = gap != 0
    degree = present.sum(axis=0).astype(float)
    upper = np.triu(present, k=1)
    i, j = np.nonzero(upper)
    coupling = gap[i, j]
    # Each pair's pre-normalisation weight, and from it the count the pair carries. Under unit
    # weights no count reaches the matrix, so the correlation is not defined there.
    weight = coupling * np.sqrt(degree[i] * degree[j]) / float(brain.config.gap_weight_scale)
    mode = brain.config.gap_weight_mode
    count = None if mode == "unit" else np.rint(np.expm1(weight) if mode == "log_count" else weight)
    sensors = set(hops.injected_sensors(topology))
    motors = set(topology._motor_flat_indices.tolist())
    sensor_motor = sum(1 for a, b in zip(i, j, strict=True) if {a, b} & sensors and {a, b} & motors)
    return {
        "max": float(coupling.max()),
        "p99": float(np.percentile(coupling, 99)),
        "n_pairs": len(coupling),
        "n_above_1": int((coupling > 1.0).sum()),
        "totals": {n: float(gap[names.index(n)].sum()) for n in TOTAL_NEURONS},
        "sensor_motor_pairs": sensor_motor,
        "count_degree_spearman": (
            None if count is None else _spearman_or_none(count, degree[i] * degree[j])
        ),
        "reach": hops.short_reach(topology),
    }


def table(seeds_for: dict[str, tuple[int, ...]] | None = None) -> dict[str, Any]:
    """Every gap wiring at every seed it is read on; the wild types' matrices take no seed."""
    seeds_for = seeds_for or {stem: runs()[stem] for stem in TABLE_STEMS}
    out: dict[str, Any] = {}
    for stem, seeds in seeds_for.items():
        if gen.CHILDREN[stem][1].get("wiring", "wild_type") == "wild_type":
            out[stem] = {
                "seed_independent": True,
                "stats": matched_statistics(_brain(stem, seeds[0])),
            }
            continue
        per_seed = {seed: matched_statistics(_brain(stem, seed)) for seed in seeds}
        out[stem] = {"seed_independent": False, "per_seed": per_seed, "summary": _summary(per_seed)}
    return out


def _summary(per_seed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Median and range over seeds of each scalar statistic."""
    rows = list(per_seed.values())
    flat: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, dict):
                for sub, v in value.items():
                    flat.setdefault(f"{key}.{sub}", []).append(float(v))
            elif value is not None:
                flat.setdefault(key, []).append(float(value))
    return {
        key: {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}
        for key, v in flat.items()
    }


# ── The campaign plan ─────────────────────────────────────────────────────────────────────────
def plan(requirement: Requirement = PANEL) -> dict[str, Any]:
    """Group the configs by the seeds they run on, one campaign invocation per group."""
    groups: dict[tuple[int, ...], list[str]] = {}
    for stem, seeds in runs(requirement).items():
        groups.setdefault(seeds, []).append(stem)
    return {
        "n_runs": sum(len(s) for s in runs(requirement).values()),
        "groups": [
            {"seeds": f"{seeds[0]}-{seeds[-1]}", "n_seeds": len(seeds), "configs": stems}
            for seeds, stems in sorted(groups.items(), key=lambda kv: -len(kv[0]))
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    pl = sub.add_parser("plan", help="the panel's configs, grouped by seed band")
    pl.add_argument("--anchor", action="store_true", help="the anchor pilot's plan instead")
    an = sub.add_parser("anchor", help="score the anchor and fix the minimum")
    an.add_argument("--logs", type=Path, action="append", required=True)
    an.add_argument("--out-dir", type=Path, required=True)
    pf = sub.add_parser("preflight", help="the gate preflight on the anchor's count level")
    pf.add_argument("--logs", type=Path, action="append", required=True)
    sc = sub.add_parser("score", help="score the panel")
    sc.add_argument("--logs", type=Path, action="append", required=True)
    sc.add_argument("--anchor", type=Path, required=True, help="the anchor JSON fixing m")
    sc.add_argument("--out-dir", type=Path, required=True)
    sc.add_argument("--csv", type=Path)
    pr = sub.add_parser("predictor", help="the registered saturation predictor")
    pr.add_argument("--source", choices=sorted(PREDICTOR_SOURCES), required=True)
    tb = sub.add_parser("table", help="the matched-statistics table and the shortcut audit")
    for parser in (pl, an, pf, sc, pr, tb):
        parser.add_argument("--out", type=Path, help="write the JSON here")
    args = ap.parse_args(argv)

    ok = True
    if args.command == "plan":
        result = plan(ANCHOR if args.anchor else PANEL)
    elif args.command == "anchor":
        result = anchor(args.logs, args.out_dir)
        ok = result["rule"]["minimum"] is not None
    elif args.command == "preflight":
        result = gp.preflight({COUNT: STEMS[COUNT]}, args.logs)
        ok = result["launch"]
    elif args.command == "score":
        minimum = json.loads(args.anchor.read_text())["rule"]["minimum"]
        if minimum is None:
            msg = f"{args.anchor} fixed no minimum: the anchor is unreadable"
            raise GapOperatingPointError(msg)
        result = score(args.logs, args.out_dir, float(minimum))
        if args.csv:
            write_csv(result, args.csv)
    elif args.command == "predictor":
        result = predictor(args.source)
    else:
        result = table()
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
